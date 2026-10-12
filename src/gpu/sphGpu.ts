/// <reference types="@webgpu/types" />
import { particleRadius, type Params } from '../sim/params'
import { makeKernels } from '../sim/kernels'
import { buildGrid, createGrid } from '../sim/spatialHash'
import { createState, type FrameMotion, type Obstacle, type SolidBox } from '../sim/sph'
import { rotate } from '../sim/quat'
import { MAX_BOXES, PARAMS_VEC4S, WORKGROUP, sphWgsl } from './sphShaders'

// The 3D SPH solver on the GPU (WebGPU compute). It runs a GpuScene: initial
// fluid positions and static boundary particles (built by the CPU code, so the
// tank scene starts exactly like the CPU engine), optional solid boxes, an
// optional river current and optional recycling from the outlet to the inlet.
// After setup everything runs in compute shaders.
//
// Per frame the caller runs `substeps` steps in one submission and gets a
// readback of positions, velocities and density/pressure/neighbours (plus the
// body's force) one or two frames later; rendering uses the newest readback.
// The floating body is integrated on the CPU once per frame from the GPU's
// summed force and torque (see integrateObstacle).

const BODY_FORCE_SCALE = 0.5 // fixed-point scale for the force/torque atomics
// Body boundary particles the buffer holds: an 8×8×8 box (the panel's maximum)
// at spacing 0.5 has ~2400, and resizing would invalidate the bind group.
const MAX_BODY_PARTICLES = 4096

export interface GpuScene {
  fluid: Float32Array // initial positions, stride 3
  boundary: { pos: Float32Array; psi: Float32Array } // static boundary particles, stride 3
  boxes?: SolidBox[] // static solids for the push-out (their surfaces must be in `boundary`)
  current?: { speed: number; rate: number } // pull toward +x speed `speed` at `rate` 1/s
  recycle?: { exitX: number; inletX: number; lift: number }
}

// The sandbox tank: createState's dam-break block and wall boundary.
export function tankScene(p: Params): GpuScene {
  const s = createState(p)
  return {
    fluid: s.pos.slice(0, s.n * 3),
    boundary: { pos: s.pos.slice(s.n * 3, (s.n + s.nb) * 3), psi: s.weight.slice(s.n, s.n + s.nb) },
  }
}

export interface GpuReadback {
  n: number
  pos: Float32Array // stride 4: x, y, z, 0
  vel: Float32Array // stride 4
  aux: Float32Array // stride 4: density, pressure, neighbours, 0
  bodyForce: number[] // force xyz, torque xyz, averaged per substep
}

// The solver uses six storage buffers, within the 8 per stage WebGPU guarantees,
// so the default limits suffice on any adapter.
export async function requestDevice(): Promise<GPUDevice | null> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
  return adapter ? adapter.requestDevice() : null
}

export class GpuSph {
  readonly n: number
  latest: GpuReadback | null = null
  private readonly device: GPUDevice
  private readonly params: Params
  private readonly uniform: GPUBuffer
  private readonly uniformData = new ArrayBuffer(PARAMS_VEC4S * 16)
  private readonly buffers: GPUBuffer[]
  private readonly bindGroup: GPUBindGroup
  private readonly pipelines: Record<'binCount' | 'scan' | 'scatter' | 'density' | 'accel' | 'integrate', GPUComputePipeline>
  private readonly particles: GPUBuffer // A (state) then B (cell-sorted)
  private readonly fluid: GPUBuffer // aux then v*
  private readonly cellAtomics: GPUBuffer
  private readonly boundary: GPUBuffer // walls then the body
  private readonly bodyForce: GPUBuffer
  private readonly nWall: number
  private readonly staging: { buffer: GPUBuffer; busy: boolean }[]
  private readonly numCells: number
  private readonly grid: { min: number[]; cellSize: number; cells: number[] }
  private bodyCount = 0
  private disposed = false
  private readonly boxes: SolidBox[]
  private readonly recycle: GpuScene['recycle']
  // Live-adjustable river current (speed 0, rate 0 = none).
  readonly current: { speed: number; rate: number }

  constructor(device: GPUDevice, p: Params, scene: GpuScene = tankScene(p)) {
    this.device = device
    this.params = p
    const n = scene.fluid.length / 3
    this.n = n
    if ((scene.boxes?.length ?? 0) > MAX_BOXES) throw new Error(`at most ${MAX_BOXES} solid boxes`)
    this.boxes = scene.boxes ?? []
    this.recycle = scene.recycle
    this.current = { ...(scene.current ?? { speed: 0, rate: 0 }) }

    // Static boundary particles, cell-sorted on the same grid layout the GPU uses.
    const nb = scene.boundary.psi.length
    const wallPos = scene.boundary.pos
    const wallGrid = createGrid(p.domain, p.h, 0, nb)
    buildGrid(wallGrid, wallPos, nb)
    const walls = new Float32Array(Math.max(nb, 1) * 4)
    for (let k = 0; k < nb; k++) {
      const b = wallGrid.sorted[k]
      walls.set([wallPos[b * 3], wallPos[b * 3 + 1], wallPos[b * 3 + 2], scene.boundary.psi[b]], k * 4)
    }
    this.numCells = wallGrid.numCells
    this.grid = { min: wallGrid.min, cellSize: wallGrid.cellSize, cells: Array.from(wallGrid.cellsPerAxis) }

    // Particles start at rest.
    const particles = new Float32Array(n * 8)
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) particles[i * 8 + d] = scene.fluid[i * 3 + d]
    }

    const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
    const make = (size: number, data?: ArrayBufferView) => {
      const buffer = device.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage: STORAGE })
      if (data) device.queue.writeBuffer(buffer, 0, data as GPUAllowSharedBufferSource)
      return buffer
    }
    // Layout and offsets: see the buffer table in sphShaders.ts.
    const C = this.numCells
    this.nWall = nb
    this.uniform = device.createBuffer({ size: PARAMS_VEC4S * 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST })
    this.particles = make(2 * n * 32, particles)
    this.fluid = make(2 * n * 16)
    const indices = make((n + 2 * C + 2) * 4)
    device.queue.writeBuffer(indices, (n + C + 1) * 4, wallGrid.cellStart)
    this.cellAtomics = make((2 * C + 1) * 4)
    this.boundary = make((nb + 2 * MAX_BODY_PARTICLES) * 16, walls)
    this.bodyForce = make(8 * 4)
    this.buffers = [this.uniform, this.particles, this.fluid, indices, this.cellAtomics, this.boundary, this.bodyForce]

    const layout = device.createBindGroupLayout({
      entries: this.buffers.map((_, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 0 ? 'uniform' : 'storage' },
      })),
    })
    this.bindGroup = device.createBindGroup({
      layout,
      entries: this.buffers.map((buffer, binding) => ({ binding, resource: { buffer } })),
    })
    const module = device.createShaderModule({ code: sphWgsl })
    const pipelineLayout = device.createPipelineLayout({ bindGroupLayouts: [layout] })
    const pipeline = (entryPoint: string) =>
      device.createComputePipeline({ layout: pipelineLayout, compute: { module, entryPoint } })
    this.pipelines = {
      binCount: pipeline('binCount'),
      scan: pipeline('scan'),
      scatter: pipeline('scatter'),
      density: pipeline('density'),
      accel: pipeline('accel'),
      integrate: pipeline('integrate'),
    }

    const readbackSize = n * 32 + n * 16 + 32
    this.staging = [0, 1].map(() => ({
      buffer: device.createBuffer({ size: readbackSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      busy: false,
    }))
  }

  // Uploads the body's boundary particles (tank coordinates) for the coming
  // frame, or removes the body with null. `local`/`psi` come from obstacleLattice.
  setBody(o: Obstacle | null, local: Float32Array, psi: Float32Array) {
    if (!o) {
      this.bodyCount = 0
      return
    }
    const m = psi.length
    if (m > MAX_BODY_PARTICLES) throw new Error(`body has ${m} boundary particles; the GPU buffer holds ${MAX_BODY_PARTICLES}`)
    const data = new Float32Array(m * 8)
    for (let b = 0; b < m; b++) {
      const r = rotate(o.q, [local[b * 3], local[b * 3 + 1], local[b * 3 + 2]])
      const w = o.omega
      data.set([o.center[0] + r[0], o.center[1] + r[1], o.center[2] + r[2], psi[b]], b * 8)
      data.set([
        o.vel[0] + w[1] * r[2] - w[2] * r[1],
        o.vel[1] + w[2] * r[0] - w[0] * r[2],
        o.vel[2] + w[0] * r[1] - w[1] * r[0],
        0,
      ], b * 8 + 4)
    }
    this.device.queue.writeBuffer(this.boundary, this.nWall * 16, data)
    this.bodyCount = m
  }

  // Runs `substeps` steps with the given frame motion and gravity (tank
  // coordinates) and the current body pose, then starts a readback if a staging
  // buffer is free.
  step(substeps: number, frame: FrameMotion, gravity: number[], body: Obstacle | null) {
    if (this.disposed || substeps === 0) return
    this.writeParams(frame, gravity, body)
    const encoder = this.device.createCommandEncoder()
    encoder.clearBuffer(this.bodyForce)
    const groups = Math.ceil(this.n / WORKGROUP)
    for (let k = 0; k < substeps; k++) {
      encoder.clearBuffer(this.cellAtomics)
      const pass = encoder.beginComputePass()
      pass.setBindGroup(0, this.bindGroup)
      const run = (name: keyof GpuSph['pipelines'], workgroups: number) => {
        pass.setPipeline(this.pipelines[name])
        pass.dispatchWorkgroups(workgroups)
      }
      run('binCount', groups)
      run('scan', 1)
      run('scatter', groups)
      run('density', groups)
      run('accel', groups)
      run('integrate', groups)
      pass.end()
    }
    const slot = this.staging.find((s) => !s.busy)
    if (slot) {
      encoder.copyBufferToBuffer(this.particles, 0, slot.buffer, 0, this.n * 32)
      encoder.copyBufferToBuffer(this.fluid, 0, slot.buffer, this.n * 32, this.n * 16)
      encoder.copyBufferToBuffer(this.bodyForce, 0, slot.buffer, this.n * 48, 32)
    }
    this.device.queue.submit([encoder.finish()])
    if (slot) this.readBack(slot, substeps)
  }

  private readBack(slot: { buffer: GPUBuffer; busy: boolean }, substeps: number) {
    slot.busy = true
    const n = this.n
    slot.buffer.mapAsync(GPUMapMode.READ).then(
      () => {
        if (this.disposed) return
        const data = slot.buffer.getMappedRange()
        const particles = new Float32Array(data, 0, n * 8)
        const pos = new Float32Array(n * 4)
        const vel = new Float32Array(n * 4)
        for (let i = 0; i < n; i++) {
          pos.set(particles.subarray(i * 8, i * 8 + 4), i * 4)
          vel.set(particles.subarray(i * 8 + 4, i * 8 + 8), i * 4)
        }
        const aux = new Float32Array(new Float32Array(data, n * 32, n * 4))
        const force = new Int32Array(data, n * 48, 6)
        const bodyForce = Array.from(force, (v) => v / BODY_FORCE_SCALE / substeps)
        slot.buffer.unmap()
        slot.busy = false
        this.latest = { n, pos, vel, aux, bodyForce }
      },
      () => {
        slot.busy = false
      },
    )
  }

  private writeParams(frame: FrameMotion, gravity: number[], body: Obstacle | null) {
    const p = this.params
    const f = new Float32Array(this.uniformData)
    const u = new Uint32Array(this.uniformData)
    const k = makeKernels(p.h, 3)
    const set = (slot: number, values: number[], ints = false) => {
      for (let c = 0; c < values.length; c++) (ints ? u : f)[slot * 4 + c] = values[c]
    }
    set(0, [this.n, this.numCells, body ? this.bodyCount : 0, this.nWall], true)
    set(1, [...this.grid.cells, 0], true)
    set(2, [...this.grid.min, this.grid.cellSize])
    set(3, [k.h, k.h2, k.poly6Coef, k.spikyGradCoef])
    set(4, [p.restDensity * p.spacing ** 3, p.restDensity, p.stiffness, p.useTait ? 1 : 0])
    set(5, [p.dt, p.viscosityEps, particleRadius(p), p.restitution])
    set(6, [...gravity.slice(0, 3), 0])
    set(7, [...frame.omega, 0])
    set(8, [...frame.alpha, 0])
    set(9, [...frame.pivot, 0])
    set(10, [...p.domain.min, 0])
    set(11, [...p.domain.max, 0])
    if (body) {
      set(12, [...body.center, 1])
      set(13, [...body.vel, 0])
      set(14, [...body.omega, 0])
      set(15, [...body.halfSize, BODY_FORCE_SCALE])
      for (let c = 0; c < 3; c++) set(16 + c, [...rotate(body.q, [c === 0 ? 1 : 0, c === 1 ? 1 : 0, c === 2 ? 1 : 0]), 0])
      // Bounding box of the rotated body, grown by h (fluid outside it can't touch it).
      const ext = [0, 0, 0]
      for (let c = 0; c < 3; c++) {
        const axis = rotate(body.q, [c === 0 ? 1 : 0, c === 1 ? 1 : 0, c === 2 ? 1 : 0])
        for (let d = 0; d < 3; d++) ext[d] += Math.abs(axis[d]) * body.halfSize[c]
      }
      set(19, [...body.center.map((v, d) => v - ext[d] - p.h), 0])
      set(20, [...body.center.map((v, d) => v + ext[d] + p.h), 0])
    } else {
      set(12, [0, 0, 0, 0])
      set(15, [1, 1, 1, BODY_FORCE_SCALE])
    }
    set(21, [this.current.speed, this.current.rate, 0, 0])
    const rc = this.recycle
    set(22, rc ? [rc.exitX, rc.inletX, rc.lift, 1] : [0, 0, 0, 0])
    set(23, [this.boxes.length, 0, 0, 0], true)
    this.boxes.forEach((box, b) => {
      set(24 + 2 * b, [...box.min, 0])
      set(25 + 2 * b, [...box.max, 0])
    })
    this.device.queue.writeBuffer(this.uniform, 0, this.uniformData)
  }

  dispose() {
    this.disposed = true
    for (const b of this.buffers) b.destroy()
    for (const s of this.staging) s.buffer.destroy()
  }
}

