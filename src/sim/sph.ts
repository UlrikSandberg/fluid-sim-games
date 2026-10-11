import { DIM, particleMass, type Params } from './params'
import { mulberry32 } from './rng'

// Structure-of-arrays particle state. Vector fields have stride DIM.
export interface SimState {
  n: number
  mass: number
  pos: Float32Array
  vel: Float32Array
  acc: Float32Array
  density: Float32Array
  pressure: Float32Array
  neighborCount: Uint16Array
}

export function createState(p: Params): SimState {
  const n = p.particleCount
  const s: SimState = {
    n,
    mass: particleMass(p),
    pos: new Float32Array(n * DIM),
    vel: new Float32Array(n * DIM),
    acc: new Float32Array(n * DIM),
    density: new Float32Array(n),
    pressure: new Float32Array(n),
    neighborCount: new Uint16Array(n),
  }
  seedBlock(s, p)
  return s
}

// Lay particles out on a grid in a block anchored at the domain's min corner
// (a dam break). Axis 0 is filled to `fill[0]` of the domain width, then rows
// stack along the remaining axes. A tiny seeded jitter breaks perfect symmetry.
function seedBlock(s: SimState, p: Params) {
  const { min, max } = p.domain
  const sp = p.spacing
  const fill = 0.6
  const perAxis: number[] = []
  for (let d = 0; d < DIM; d++) {
    const extent = (max[d] - min[d]) * (d === 0 ? fill : 1)
    perAxis.push(Math.max(1, Math.floor(extent / sp)))
  }
  const rand = mulberry32(p.seed)
  const jitter = 1e-3 * sp
  for (let i = 0; i < s.n; i++) {
    let idx = i
    for (let d = 0; d < DIM; d++) {
      const k = idx % perAxis[d]
      idx = Math.floor(idx / perAxis[d])
      s.pos[i * DIM + d] = min[d] + sp * (k + 0.5) + (rand() - 0.5) * jitter
    }
  }
}

// One fixed-dt substep.
// TODO: spatial hash → density → pressure (EOS) → pressure + XSPH forces.
// For now this only integrates gravity and resolves walls, to prove the loop.
export function step(s: SimState, p: Params) {
  const { n, pos, vel, acc } = s
  const dt = p.dt

  for (let i = 0; i < n; i++) {
    for (let d = 0; d < DIM; d++) acc[i * DIM + d] = p.gravity[d]
  }

  // Symplectic Euler: velocity first, then position with the new velocity.
  for (let k = 0; k < n * DIM; k++) {
    vel[k] += acc[k] * dt
    pos[k] += vel[k] * dt
  }

  resolveBoundaries(s, p)
}

// Reflect the normal velocity with restitution and clamp back inside the box.
function resolveBoundaries(s: SimState, p: Params) {
  const { n, pos, vel } = s
  const { min, max } = p.domain
  const e = p.restitution
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < DIM; d++) {
      const k = i * DIM + d
      if (pos[k] < min[d]) {
        pos[k] = min[d]
        if (vel[k] < 0) vel[k] = -vel[k] * e
      } else if (pos[k] > max[d]) {
        pos[k] = max[d]
        if (vel[k] > 0) vel[k] = -vel[k] * e
      }
    }
  }
}
