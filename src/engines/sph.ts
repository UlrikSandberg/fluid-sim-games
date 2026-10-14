import type { Params } from '../sim/params'
import {
  collideBodies, createState, integrateObstacle, makeObstacle, obstacleLattice, setObstacle, step,
  type BodyShape, type FrameMotion, type Obstacle, type SimState,
} from '../sim/sph'
import { GpuSph, requestDevice, type GpuBody, type GpuScene, type GpuSwirl } from '../gpu/sphGpu'
import { MAX_BODIES } from '../gpu/sphShaders'

// Floating bodies (ducks, the sandbox's box); the rest of the GPU's body slots
// are for kinematic movers.
export const MAX_FLOATERS = 8
export const MAX_MOVERS = MAX_BODIES - MAX_FLOATERS

export type EngineKind = 'cpu' | 'gpu'

// What the 3D scene needs from a solver: advance by some substeps under the
// tank's motion, expose particle data for rendering and stats, and manage one
// floating box. `view()` has the fields the colour modes read; it is null until
// the GPU's first readback arrives.
export interface FluidEngine {
  readonly kind: EngineKind
  readonly n: number
  readonly obstacle: Obstacle | null
  step(substeps: number, frame: FrameMotion, gravity: number[]): void
  view(): SimState | null
  dropBody(rect: { center: number[]; size: number[] }): void
  removeBody(): void
  dispose(): void
}

// The CPU solver (src/sim/sph.ts), deterministic, ~0.9 µs per particle per substep.
export class CpuEngine implements FluidEngine {
  readonly kind = 'cpu'
  private readonly state: SimState
  private readonly params: Params

  constructor(params: Params) {
    this.params = params
    this.state = createState(params)
  }

  get n() {
    return this.state.n
  }

  get obstacle() {
    return this.state.obstacle
  }

  step(substeps: number, frame: FrameMotion, gravity: number[]) {
    this.params.gravity = gravity
    for (let k = 0; k < substeps; k++) step(this.state, this.params, undefined, frame)
  }

  view() {
    return this.state
  }

  dropBody(rect: { center: number[]; size: number[] }) {
    setObstacle(this.state, this.params, rect)
  }

  removeBody() {
    setObstacle(this.state, this.params, null)
  }

  dispose() {}
}

// The GPU solver (src/gpu). Bodies are integrated here on the CPU, once per
// frame, from the force and torque the GPU summed over the previous frame's
// substeps (one or two frames of lag, while a light bob takes ~2 s), then bumped
// apart where they overlap.
export class GpuEngine implements FluidEngine {
  readonly kind = 'gpu'
  // Extra constraint on each body after each step (e.g. the river's curved banks,
  // which the CPU-side rigid body doesn't otherwise know about).
  constrainBody: ((o: Obstacle) => void) | null = null
  private gpu: GpuSph
  private readonly device: GPUDevice
  private readonly params: Params
  private bodyList: GpuBody[] = []
  // Kinematic bodies: moved by their owner (setting o.center, q, vel, omega and
  // halfSize), pushing the water but not pushed back. After the floating bodies
  // on the GPU, so the floaters' force slots stay put.
  private moverList: GpuBody[] = []
  private lastReadback: object | null = null
  private cached: SimState | null = null

  private constructor(device: GPUDevice, gpu: GpuSph, params: Params) {
    this.device = device
    this.gpu = gpu
    this.params = params
  }

  // Resolves to null when this browser has no usable WebGPU adapter. Without a
  // scene, the sandbox's tank.
  static async create(params: Params, scene?: GpuScene): Promise<GpuEngine | null> {
    const device = await requestDevice()
    return device ? new GpuEngine(device, new GpuSph(device, params, scene), params) : null
  }

  // Starts over from a new scene on the same device (e.g. new static boundary
  // particles), keeping the bodies.
  replaceScene(scene: GpuScene) {
    this.gpu.dispose()
    this.gpu = new GpuSph(this.device, this.params, scene)
    this.lastReadback = null
    this.cached = null
    if (this.bodyList.length > 0 || this.moverList.length > 0) this.gpu.setBodies([...this.bodyList, ...this.moverList])
  }

  get capacity() {
    return this.gpu.capacity
  }

  // Adds particles (stride 3) if there's room; returns how many fitted.
  addParticles(pos: Float32Array, vel?: Float32Array): number {
    return this.gpu.addParticles(pos, vel)
  }

  clearParticles() {
    this.gpu.clearParticles()
  }

  // Replaces the river's whirlpools.
  setSwirls(swirls: GpuSwirl[]) {
    this.gpu.setSwirls(swirls)
  }

  // The river current (live-adjustable), when the scene has one.
  get current() {
    return this.gpu.current
  }

  get n() {
    return this.gpu.n
  }

  get bodies(): Obstacle[] {
    return this.bodyList.map((b) => b.o)
  }

  // The first body (the sandbox has at most one).
  get obstacle(): Obstacle | null {
    return this.bodyList[0]?.o ?? null
  }

  // Replaces the kinematic movers (up to MAX_MOVERS).
  setMovers(movers: Obstacle[]) {
    this.moverList = movers.slice(0, MAX_MOVERS).map((o) => ({ o, ...obstacleLattice(o, this.params) }))
  }

  get movers(): Obstacle[] {
    return this.moverList.map((m) => m.o)
  }

  // A mover whose size changed: its boundary particles are rebuilt once it has
  // grown or shrunk by a quarter spacing on some axis since they were made.
  private refreshMovers() {
    const tolerance = this.params.spacing / 4
    for (const m of this.moverList) {
      const made = m.size ?? (m.size = m.o.halfSize.slice())
      if (m.o.halfSize.some((v, d) => Math.abs(v - made[d]) > tolerance)) {
        Object.assign(m, obstacleLattice(m.o, this.params), { size: m.o.halfSize.slice() })
      }
    }
  }

  step(substeps: number, frame: FrameMotion, gravity: number[]) {
    this.params.gravity = gravity
    if (this.moverList.length > 0) this.refreshMovers()
    if ((this.bodyList.length > 0 || this.moverList.length > 0) && substeps > 0) {
      const f = this.gpu.latest?.bodyForce
      this.bodyList.forEach(({ o }, b) => {
        for (let k = 0; k < 3; k++) {
          o.force[k] = f ? f[6 * b + k] : 0
          o.torque[k] = f ? f[6 * b + 3 + k] : 0
        }
      })
      const bodies = this.bodies
      for (let k = 0; k < substeps; k++) {
        for (const o of bodies) {
          integrateObstacle(o, this.params, frame)
          this.constrainBody?.(o)
        }
        collideBodies(bodies, this.params)
      }
      this.gpu.setBodies([...this.bodyList, ...this.moverList])
    }
    this.gpu.step(substeps, frame, gravity, [...this.bodies, ...this.movers])
  }

  view(): SimState | null {
    const r = this.gpu.latest
    if (!r) return null
    if (r === this.lastReadback && this.cached) return this.cached
    // Repack stride-4 GPU data into the stride-3 arrays the renderer and colour
    // modes read.
    const n = r.n
    const pos = new Float32Array(n * 3)
    const vel = new Float32Array(n * 3)
    const density = new Float32Array(n)
    const pressure = new Float32Array(n)
    const neighborCount = new Uint16Array(n)
    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 3; d++) {
        pos[i * 3 + d] = r.pos[i * 4 + d]
        vel[i * 3 + d] = r.vel[i * 4 + d]
      }
      density[i] = r.aux[i * 4]
      pressure[i] = r.aux[i * 4 + 1]
      neighborCount[i] = r.aux[i * 4 + 2]
    }
    this.lastReadback = r
    this.cached = { dim: 3, n, pos, vel, density, pressure, neighborCount, obstacle: this.obstacle } as SimState
    return this.cached
  }

  // Adds a body (up to the GPU's limit); returns false when full. Bodies are only
  // ever appended or all removed, so the GPU's per-body force slots stay aligned.
  addBody(rect: { center: number[]; size: number[]; shape?: BodyShape; ballast?: number }): boolean {
    if (this.bodyList.length >= MAX_FLOATERS) return false
    const o = makeObstacle(this.params, rect)
    if (rect.shape) o.shape = rect.shape
    if (rect.ballast) o.ballast = rect.ballast
    this.bodyList.push({ o, ...obstacleLattice(o, this.params) })
    this.gpu.setBodies([...this.bodyList, ...this.moverList])
    return true
  }

  // The sandbox's single box: replaces any bodies.
  dropBody(rect: { center: number[]; size: number[] }) {
    this.bodyList = []
    this.addBody(rect)
  }

  removeBody() {
    this.bodyList = []
    this.gpu.setBodies(this.moverList)
  }

  dispose() {
    this.gpu.dispose()
  }
}
