/// <reference types="@webgpu/types" />
import { particleRadius, type Params } from '../sim/params'
import { makeKernels } from '../sim/kernels'
import { buildGrid, createGrid } from '../sim/spatialHash'
import { createState, type FrameMotion, type Obstacle, type SolidBox } from '../sim/sph'
import { rotate } from '../sim/quat'
import {
  ATTRACTOR_SLOT, BODY_SLOT, BODY_VEC4S, BOX_SLOT, FALL_SLOT, MAX_ATTRACTORS, MAX_BODIES, MAX_BOXES, MAX_FALLS, MAX_SWIRLS, MAX_ZONES, PARAMS_VEC4S, SWIRL_SLOT,
  WORKGROUP, ZONE_SLOT, sphWgsl,
} from './sphShaders'

// The 3D SPH solver on the GPU (WebGPU compute). It runs a GpuScene: initial
// fluid positions and static boundary particles (built by the CPU code, so the
// tank scene starts exactly like the CPU engine), optional solid boxes and an
// optional river course (current, banks, waterfalls, flip zones, recycling).
// After setup everything runs in compute shaders.
//
// Per frame the caller runs `substeps` steps in one submission and gets a
// readback of positions, velocities and density/pressure/neighbours (plus the
// bodies' forces) one or two frames later; rendering uses the newest readback.
// Floating bodies are integrated on the CPU once per frame from the GPU's
// summed force and torque (see integrateObstacle).

const BODY_FORCE_SCALE = 0.5 // fixed-point scale for the force/torque atomics
// Body boundary particles the buffer holds, all bodies together: an 8×8×8 box
// (the sandbox panel's maximum) at spacing 0.5 has ~2400, a duck ~100, a river
// obstacle up to ~400; resizing would invalidate the bind group.
const MAX_BODY_PARTICLES = 8192

// A body as the GPU sees it: its pose (an Obstacle) and its boundary particles
// in the body frame with their ψ (from obstacleLattice).
export interface GpuBody {
  o: Obstacle
  local: Float32Array
  psi: Float32Array
  size?: number[] // the half sizes `local` was made for (resizable bodies)
}

export interface GpuScene {
  fluid: Float32Array // initial positions, stride 3
  velocity?: Float32Array // initial velocities, stride 3 (at rest without)
  capacity?: number // room for this many particles (addParticles fills it); default: just the initial ones
  boundary: { pos: Float32Array; psi: Float32Array } // static boundary particles, stride 3
  boxes?: SolidBox[] // static solids for the push-out (their surfaces must be in `boundary`)
  current?: { speed: number; rate: number } // pull downstream to speed `speed` at `rate` 1/s
  // A river course (see course.ts): fluid kept within halfWidth of the centreline,
  // above the bed and under the flip zones' lids (their boundary particles go in
  // `boundary`); the current follows it.
  course?: GpuCourse
  // Past s = exitS, water re-enters at s = inletS, lifted by `lift` (needs a course).
  recycle?: { exitS: number; inletS: number; lift: number }
  // Whirlpools (up to MAX_SWIRLS): centre (x, z), radius, peak speed (sign = direction).
  swirls?: GpuSwirl[]
}

export interface GpuSwirl {
  x: number
  z: number
  radius: number
  strength: number
}

export interface GpuCourse {
  halfWidth: number
  ds: number // sample spacing along the centreline
  samples: Float32Array // per sample: x, z, tangent x, tangent z
  lookup: { cell: number; nx: number; nz: number; index: Uint32Array } // nearest sample per ground cell (origin 0, 0)
  top: number // bed height at the start
  falls: { s: number; drop: number }[]
  zones: { s0: number; s1: number; gravity: number; lid: number }[] // lid: height above the bed (0 = none)
}

// The course buffer's contents: samples, then the lookup as u32 bits (see sphShaders.ts).
function packCourse(c: GpuCourse | undefined): ArrayBuffer {
  if (!c) return new ArrayBuffer(16)
  const n = c.samples.length / 4
  const cells = Math.ceil(c.lookup.index.length / 4) * 4
  const data = new ArrayBuffer((n * 4 + cells) * 4)
  new Float32Array(data, 0, n * 4).set(c.samples)
  new Uint32Array(data, n * 16, cells).set(c.lookup.index)
  return data
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

// The solver uses seven storage buffers, within the 8 per stage WebGPU guarantees,
// so the default limits suffice on any adapter.
export async function requestDevice(): Promise<GPUDevice | null> {
  const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' })
  return adapter ? adapter.requestDevice() : null
}

export class GpuSph {
  n: number // particles in use
  readonly capacity: number
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
  private readonly course: GPUBuffer
  private readonly nWall: number
  private readonly staging: { buffer: GPUBuffer; busy: boolean }[]
  private readonly numCells: number
  private readonly grid: { min: number[]; cellSize: number; cells: number[] }
  private bodyRanges: [number, number][] = [] // first particle, count, per body
  private disposed = false
  private readonly boxes: SolidBox[]
  private readonly recycle: GpuScene['recycle']
  private readonly courseInfo: GpuCourse | undefined
  private swirls: GpuSwirl[]
  // Live-adjustable river current (speed 0, rate 0 = none).
  readonly current: { speed: number; rate: number }

  constructor(device: GPUDevice, p: Params, scene: GpuScene = tankScene(p)) {
    this.device = device
    this.params = p
    // Buffers are laid out for the capacity (n below); the live count is this.n.
    const n = Math.max(scene.capacity ?? 0, scene.fluid.length / 3, 1)
    this.capacity = n
    this.n = scene.fluid.length / 3
    if ((scene.boxes?.length ?? 0) > MAX_BOXES) throw new Error(`at most ${MAX_BOXES} solid boxes`)
    this.boxes = scene.boxes ?? []
    this.recycle = scene.recycle
    this.courseInfo = scene.course
    if ((scene.course?.falls.length ?? 0) > MAX_FALLS) throw new Error(`at most ${MAX_FALLS} waterfalls`)
    if ((scene.course?.zones.length ?? 0) > MAX_ZONES) throw new Error(`at most ${MAX_ZONES} flip zones`)
    this.swirls = (scene.swirls ?? []).slice(0, MAX_SWIRLS)
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

    // Particles start at rest unless given velocities.
    const particles = new Float32Array(n * 8)
    for (let i = 0; i < this.n; i++) {
      for (let d = 0; d < 3; d++) {
        particles[i * 8 + d] = scene.fluid[i * 3 + d]
        particles[i * 8 + 4 + d] = scene.velocity?.[i * 3 + d] ?? 0
      }
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
    const course = packCourse(scene.course)
    this.course = make(course.byteLength, new Uint8Array(course))
    this.buffers = [this.uniform, this.particles, this.fluid, indices, this.cellAtomics, this.boundary, this.bodyForce, this.course]

    const layout = device.createBindGroupLayout({
      entries: this.buffers.map((buffer, binding) => ({
        binding,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: binding === 0 ? 'uniform' : buffer === this.course ? 'read-only-storage' : 'storage' },
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

  // Adds particles after the live ones (positions, optional velocities, stride 3),
  // as many as there's room for; returns how many were added.
  addParticles(pos: Float32Array, vel?: Float32Array): number {
    const k = Math.min(pos.length / 3, this.capacity - this.n)
    if (k <= 0) return 0
    const data = new Float32Array(k * 8)
    for (let i = 0; i < k; i++) {
      for (let d = 0; d < 3; d++) {
        data[i * 8 + d] = pos[i * 3 + d]
        data[i * 8 + 4 + d] = vel?.[i * 3 + d] ?? 0
      }
    }
    this.device.queue.writeBuffer(this.particles, this.n * 32, data)
    this.n += k
    return k
  }

  // Removes every particle.
  clearParticles() {
    this.n = 0
  }

  // Replaces the whirlpools (up to MAX_SWIRLS), e.g. as they come and go.
  setSwirls(swirls: GpuSwirl[]) {
    this.swirls = swirls.slice(0, MAX_SWIRLS)
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
    // Read back only the live particles (the staging buffer is laid out for the
    // capacity, so the offsets stay put).
    const live = this.n
    const cap = this.capacity
    if (slot) {
      if (live > 0) {
        encoder.copyBufferToBuffer(this.particles, 0, slot.buffer, 0, live * 32)
        encoder.copyBufferToBuffer(this.fluid, 0, slot.buffer, cap * 32, live * 16)
      }
      encoder.copyBufferToBuffer(this.bodyForce, 0, slot.buffer, cap * 48, MAX_BODIES * 24)
    }
    this.device.queue.submit([encoder.finish()])
    if (slot) this.readBack(slot, substeps, live)
  }

  private readBack(slot: { buffer: GPUBuffer; busy: boolean }, substeps: number, n: number) {
    slot.busy = true
    const cap = this.capacity
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
        const aux = new Float32Array(new Float32Array(data, cap * 32, n * 4))
        const force = new Int32Array(data, cap * 48, MAX_BODIES * 6)
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
    set(0, [this.capacity, this.numCells, nBodies, this.nWall], true)
    set(19, [this.n, 0, 0, 0], true)
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
    set(12, [BODY_FORCE_SCALE, p.wallFriction ?? 0, 0, 0])
    set(13, [this.current.speed, this.current.rate, 0, 0])
    const rc = this.recycle
    set(14, rc ? [rc.exitS, rc.inletS, rc.lift, 1] : [0, 0, 0, 0])
    set(15, [this.boxes.length, 0, 0, 0], true)
    const c = this.courseInfo
    set(16, c ? [c.halfWidth, c.ds, c.samples.length / 4, 1] : [0, 0, 0, 0])
    set(17, c ? [c.lookup.cell, c.top, 0, 0] : [0, 0, 0, 0])
    set(18, c ? [c.lookup.nx, c.lookup.nz, c.falls.length, c.zones.length] : [0, 0, 0, 0], true)
    c?.falls.forEach((f, i) => set(FALL_SLOT + i, [f.s, f.drop, 0, 0]))
    c?.zones.forEach((z, i) => set(ZONE_SLOT + i, [z.s0, z.s1, z.gravity, z.lid]))
    const attractors = (p.attractors ?? []).slice(0, MAX_ATTRACTORS)
    set(ATTRACTOR_SLOT, [attractors.length, 0, 0, 0], true)
    attractors.forEach((a, i) => {
      set(ATTRACTOR_SLOT + 1 + 2 * i, [...a.center, a.gm])
      set(ATTRACTOR_SLOT + 2 + 2 * i, [a.radius, 0, 0, 0])
    })
    set(SWIRL_SLOT, [this.swirls.length, 0, 0, 0], true)
    this.swirls.forEach((w, i) => set(SWIRL_SLOT + 1 + i, [w.x, w.z, w.radius, w.strength]))
    this.boxes.forEach((box, b) => {
      set(BOX_SLOT + 2 * b, [...box.min, 0])
      set(BOX_SLOT + 1 + 2 * b, [...box.max, 0])
    })
    const bodySlot = BODY_SLOT
    for (let b = 0; b < nBodies; b++) {
      const o = bodies[b]
      const base = bodySlot + b * BODY_VEC4S
      const axes = [0, 1, 2].map((c) => rotate(o.q, [c === 0 ? 1 : 0, c === 1 ? 1 : 0, c === 2 ? 1 : 0]))
      // Bounding box of the rotated body, grown by h (fluid outside it can't touch it).
      const ext = [0, 1, 2].map((d) => axes.reduce((a, axis, c) => a + Math.abs(axis[d]) * o.halfSize[c], 0))
      set(base, [...o.center, 0])
      set(base + 1, [...o.vel, 0])
      set(base + 2, [...o.omega, 0])
      set(base + 3, [...o.halfSize, o.shape ? o.shape.bow * o.halfSize[0] : 0])
      for (let c = 0; c < 3; c++) set(base + 4 + c, [...axes[c], 0])
      set(base + 7, [...o.center.map((v, d) => v - ext[d] - p.h), 0])
      set(base + 8, [...o.center.map((v, d) => v + ext[d] + p.h), 0])
      set(base + 9, [...this.bodyRanges[b], o.shape?.kind === 'hull' ? 1 : 0, 0], true)
    }
    this.device.queue.writeBuffer(this.uniform, 0, this.uniformData)
  }

  dispose() {
    this.disposed = true
    for (const b of this.buffers) b.destroy()
    for (const s of this.staging) s.buffer.destroy()
  }
}

