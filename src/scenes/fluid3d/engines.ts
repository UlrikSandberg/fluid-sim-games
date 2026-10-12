import type { Params } from '../../sim/params'
import {
  collideBodies, createState, integrateObstacle, makeObstacle, obstacleLattice, setObstacle, step,
  type FrameMotion, type Obstacle, type SimState,
} from '../../sim/sph'
import { GpuSph, requestDevice, type GpuBody, type GpuScene, type GpuSwirl } from '../../gpu/sphGpu'
import { MAX_BODIES } from '../../gpu/sphShaders'

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
  private readonly gpu: GpuSph
  private readonly params: Params
  private bodyList: GpuBody[] = []
  private lastReadback: object | null = null
  private cached: SimState | null = null

  private constructor(gpu: GpuSph, params: Params) {
    this.gpu = gpu
    this.params = params
  }

  // Resolves to null when this browser has no usable WebGPU adapter. Without a
  // scene, the sandbox's tank.
  static async create(params: Params, scene?: GpuScene): Promise<GpuEngine | null> {
    const device = await requestDevice()
    return device ? new GpuEngine(new GpuSph(device, params, scene), params) : null
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

  step(substeps: number, frame: FrameMotion, gravity: number[]) {
    this.params.gravity = gravity
    if (this.bodyList.length > 0 && substeps > 0) {
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
      this.gpu.setBodies(this.bodyList)
    }
    this.gpu.step(substeps, frame, gravity, this.bodies)
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
  addBody(rect: { center: number[]; size: number[] }): boolean {
    if (this.bodyList.length >= MAX_BODIES) return false
    const o = makeObstacle(this.params, rect)
    this.bodyList.push({ o, ...obstacleLattice(o, this.params) })
    this.gpu.setBodies(this.bodyList)
    return true
  }

  // The sandbox's single box: replaces any bodies.
  dropBody(rect: { center: number[]; size: number[] }) {
    this.bodyList = []
    this.addBody(rect)
  }

  removeBody() {
    this.bodyList = []
    this.gpu.setBodies([])
  }

  dispose() {
    this.gpu.dispose()
  }
}
