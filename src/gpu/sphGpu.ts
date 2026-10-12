/// <reference types="@webgpu/types" />
import { particleRadius, type Params } from '../sim/params'
import { makeKernels } from '../sim/kernels'
import { buildGrid, createGrid } from '../sim/spatialHash'
import { createState, type FrameMotion, type Obstacle, type SolidBox } from '../sim/sph'
import { rotate } from '../sim/quat'
import { BODY_VEC4S, MAX_BODIES, MAX_BOXES, PARAMS_VEC4S, WORKGROUP, sphWgsl } from './sphShaders'

// The 3D SPH solver on the GPU (WebGPU compute). It runs a GpuScene: initial
// fluid positions and static boundary particles (built by the CPU code, so the
// tank scene starts exactly like the CPU engine), optional solid boxes, an
// optional river current and optional recycling from the outlet to the inlet.
// After setup everything runs in compute shaders.
//
// Per frame the caller runs `substeps` steps in one submission and gets a
// readback of positions, velocities and density/pressure/neighbours (plus the
// bodies' forces) one or two frames later; rendering uses the newest readback.
// Floating bodies are integrated on the CPU once per frame from the GPU's
// summed force and torque (see integrateObstacle).

const BODY_FORCE_SCALE = 0.5 // fixed-point scale for the force/torque atomics
// Body boundary particles the buffer holds, all bodies together: an 8×8×8 box
// (the sandbox panel's maximum) at spacing 0.5 has ~2400, a duck ~100; resizing
// would invalidate the bind group.
const MAX_BODY_PARTICLES = 4096

// A body as the GPU sees it: its pose (an Obstacle) and its boundary particles
// in the body frame with their ψ (from obstacleLattice).
export interface GpuBody {
  o: Obstacle
  local: Float32Array
  psi: Float32Array
}

export interface GpuScene {
  fluid: Float32Array // initial positions, stride 3
  boundary: { pos: Float32Array; psi: Float32Array } // static boundary particles, stride 3
  boxes?: SolidBox[] // static solids for the push-out (their surfaces must be in `boundary`)
  current?: { speed: number; rate: number } // pull toward +x speed `speed` at `rate` 1/s
  recycle?: { exitX: number; inletX: number; lift: number }
  // A winding channel along x: centreline z = centerZ + amplitude·sin(wavenumber·x),
  // fluid kept within halfWidth of it (the banks' boundary particles go in `boundary`).
  channel?: { amplitude: number; wavenumber: number; halfWidth: number; centerZ: number }
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
  bodyForce: number[] // per body: force xyz, torque xyz, averaged per substep
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
  private readonly boundary: GPUBuffer // walls then the bodies
  private readonly bodyForce: GPUBuffer
  private readonly nWall: number
  private readonly staging: { buffer: GPUBuffer; busy: boolean }[]
  private readonly numCells: number
  private readonly grid: { min: number[]; cellSize: number; cells: number[] }
  private bodyRanges: [number, number][] = [] // first particle, count, per body
  private disposed = false
  private readonly boxes: SolidBox[]
  private readonly recycle: GpuScene['recycle']
  private readonly channel: GpuScene['channel']
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
    this.channel = scene.channel
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
    this.bodyForce = make(MAX_BODIES * 6 * 4)
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

    const readbackSize = n * 32 + n * 16 + MAX_BODIES * 24
    this.staging = [0, 1].map(() => ({
      buffer: device.createBuffer({ size: readbackSize, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST }),
      busy: false,
    }))
  }

  // Uploads the bodies' boundary particles (tank coordinates) for the coming
  // frame, placed from each body's pose with velocity v + ω × r.
  setBodies(bodies: GpuBody[]) {
    if (bodies.length > MAX_BODIES) throw new Error(`at most ${MAX_BODIES} bodies`)
    const total = bodies.reduce((a, b) => a + b.psi.length, 0)
    if (total > MAX_BODY_PARTICLES) throw new Error(`${total} body particles; the GPU buffer holds ${MAX_BODY_PARTICLES}`)
    const data = new Float32Array(Math.max(total, 1) * 8)
    this.bodyRanges = []
    let k = 0
    for (const { o, local, psi } of bodies) {
      this.bodyRanges.push([k, psi.length])
      const w = o.omega
      for (let b = 0; b < psi.length; b++, k++) {
        const r = rotate(o.q, [local[b * 3], local[b * 3 + 1], local[b * 3 + 2]])
        data.set([o.center[0] + r[0], o.center[1] + r[1], o.center[2] + r[2], psi[b]], k * 8)
        data.set([
          o.vel[0] + w[1] * r[2] - w[2] * r[1],
          o.vel[1] + w[2] * r[0] - w[0] * r[2],
          o.vel[2] + w[0] * r[1] - w[1] * r[0],
          0,
        ], k * 8 + 4)
      }
    }
    if (total > 0) this.device.queue.writeBuffer(this.boundary, this.nWall * 16, data)
  }

  // Runs `substeps` steps with the given frame motion and gravity (tank
  // coordinates) and the current body pose, then starts a readback if a staging
  // buffer is free.
  step(substeps: number, frame: FrameMotion, gravity: number[], bodies: Obstacle[]) {
    if (this.disposed || substeps === 0) return
    this.writeParams(frame, gravity, bodies)
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
      encoder.copyBufferToBuffer(this.bodyForce, 0, slot.buffer, this.n * 48, MAX_BODIES * 24)
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
        const force = new Int32Array(data, n * 48, MAX_BODIES * 6)
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

  private writeParams(frame: FrameMotion, gravity: number[], bodies: Obstacle[]) {
    const p = this.params
    const f = new Float32Array(this.uniformData)
    const u = new Uint32Array(this.uniformData)
    const k = makeKernels(p.h, 3)
    const set = (slot: number, values: number[], ints = false) => {
      for (let c = 0; c < values.length; c++) (ints ? u : f)[slot * 4 + c] = values[c]
    }
    const nBodies = Math.min(bodies.length, this.bodyRanges.length)
    set(0, [this.n, this.numCells, nBodies, this.nWall], true)
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
    set(12, [BODY_FORCE_SCALE, 0, 0, 0])
    set(13, [this.current.speed, this.current.rate, 0, 0])
    const rc = this.recycle
    set(14, rc ? [rc.exitX, rc.inletX, rc.lift, 1] : [0, 0, 0, 0])
    set(15, [this.boxes.length, 0, 0, 0], true)
    const ch = this.channel
    set(16, ch ? [ch.amplitude, ch.wavenumber, ch.halfWidth, 1] : [0, 0, 0, 0])
    set(17, ch ? [ch.centerZ, 0, 0, 0] : [0, 0, 0, 0])
    this.boxes.forEach((box, b) => {
      set(18 + 2 * b, [...box.min, 0])
      set(19 + 2 * b, [...box.max, 0])
    })
    const bodySlot = 18 + 2 * MAX_BOXES
    for (let b = 0; b < nBodies; b++) {
      const o = bodies[b]
      const base = bodySlot + b * BODY_VEC4S
      const axes = [0, 1, 2].map((c) => rotate(o.q, [c === 0 ? 1 : 0, c === 1 ? 1 : 0, c === 2 ? 1 : 0]))
      // Bounding box of the rotated body, grown by h (fluid outside it can't touch it).
      const ext = [0, 1, 2].map((d) => axes.reduce((a, axis, c) => a + Math.abs(axis[d]) * o.halfSize[c], 0))
      set(base, [...o.center, 0])
      set(base + 1, [...o.vel, 0])
      set(base + 2, [...o.omega, 0])
      set(base + 3, [...o.halfSize, 0])
      for (let c = 0; c < 3; c++) set(base + 4 + c, [...axes[c], 0])
      set(base + 7, [...o.center.map((v, d) => v - ext[d] - p.h), 0])
      set(base + 8, [...o.center.map((v, d) => v + ext[d] + p.h), 0])
      set(base + 9, [...this.bodyRanges[b], 0, 0], true)
    }
    this.device.queue.writeBuffer(this.uniform, 0, this.uniformData)
  }

  dispose() {
    this.disposed = true
    for (const b of this.buffers) b.destroy()
    for (const s of this.staging) s.buffer.destroy()
  }
}

