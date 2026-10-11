import { DIM, mouseAccel, mouseSpeedCap, particleMass, particleRadius, type Params } from './params.ts'
import { mulberry32 } from './rng.ts'
import { makeKernels, poly6, spikyGradMag, type Kernels } from './kernels.ts'
import { buildGrid, buildNeighbors, createGrid, type SpatialGrid } from './spatialHash.ts'

// Structure-of-arrays particle state. Vector fields have stride DIM.
export interface SimState {
  n: number
  mass: number
  pos: Float32Array
  vel: Float32Array
  velScratch: Float32Array // XSPH writes here, then swaps with vel
  acc: Float32Array
  density: Float32Array
  pressure: Float32Array
  neighborCount: Uint16Array
  grid: SpatialGrid
}

// Brush input for one step. `sign` is +1 to repel, −1 to attract.
export interface MouseInput {
  active: boolean
  pos: number[] // length DIM, world units
  sign: number
}

export function createState(p: Params): SimState {
  const n = Math.min(p.particleCount, blockCapacity(p))
  const s: SimState = {
    n,
    mass: particleMass(p),
    pos: new Float32Array(n * DIM),
    vel: new Float32Array(n * DIM),
    velScratch: new Float32Array(n * DIM),
    acc: new Float32Array(n * DIM),
    density: new Float32Array(n),
    pressure: new Float32Array(n),
    neighborCount: new Uint16Array(n),
    grid: createGrid(p.domain, p.h, n),
  }
  seedBlock(s, p)
  return s
}

// Lattice sites per axis that fit in the domain at the given spacing.
function sitesPerAxis(p: Params): number[] {
  const sites: number[] = []
  for (let d = 0; d < DIM; d++) {
    sites.push(Math.max(1, Math.floor((p.domain.max[d] - p.domain.min[d]) / p.spacing)))
  }
  return sites
}

// Most particles the initial block can hold without overlapping.
export function blockCapacity(p: Params): number {
  return sitesPerAxis(p).reduce((a, b) => a * b, 1)
}

// Lay particles out on a grid in a block anchored at the domain's min corner
// (a dam break). Axis 0 spans `fill` of the domain width (wider if the other
// axes can't hold the rest), then rows stack along the remaining axes. A tiny
// seeded jitter breaks perfect symmetry.
function seedBlock(s: SimState, p: Params) {
  const { min } = p.domain
  const sp = p.spacing
  const fill = 0.6
  const perAxis = sitesPerAxis(p)
  const otherAxes = perAxis.slice(1).reduce((a, b) => a * b, 1)
  perAxis[0] = Math.min(perAxis[0], Math.max(Math.floor(perAxis[0] * fill), Math.ceil(s.n / otherAxes)))
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
export function step(s: SimState, p: Params, mouse?: MouseInput) {
  if (s.grid.cellSize !== p.h) s.grid = createGrid(p.domain, p.h, s.n)
  buildGrid(s.grid, s.pos, s.n)
  buildNeighbors(s.grid, s.pos, s.n, p.h)

  const k = makeKernels(p.h)
  computeDensityPressure(s, p, k)
  computeAccelerations(s, p, k)
  if (mouse?.active) applyMouse(s, p, mouse)
  integrate(s, p, k)
  resolveBoundaries(s, p)
}

// ρ_i = Σ_j m W(r_ij) including self, then pressure from the EOS.
function computeDensityPressure(s: SimState, p: Params, k: Kernels) {
  const { n, mass, density, pressure, neighborCount } = s
  const { neighborStart, neighborEnd, neighborDist2 } = s.grid
  const self = mass * poly6(k, 0)

  for (let i = 0; i < n; i++) {
    let rho = self
    const start = neighborStart[i]
    const end = neighborEnd[i]
    for (let q = start; q < end; q++) rho += mass * poly6(k, neighborDist2[q])
    density[i] = rho
    neighborCount[i] = end - start
    pressure[i] = equationOfState(rho, p)
  }
}

// Linear: p = k(ρ − ρ₀). Tait: p = B((ρ/ρ₀)^7 − 1) with B = ρ₀k/7, so both have
// dp/dρ = k at rest (same c_s, same CFL limit) and Tait only differs by
// stiffening under compression. Both clamped to ≥ 0 to avoid tensile clustering.
// The 7th power is written as multiplications: Math.pow is not guaranteed
// bit-identical across JS engines.
function equationOfState(rho: number, p: Params): number {
  let pr: number
  if (p.useTait) {
    const x = rho / p.restDensity
    const x2 = x * x
    const x4 = x2 * x2
    pr = ((p.restDensity * p.stiffness) / 7) * (x4 * x2 * x - 1)
  } else {
    pr = p.stiffness * (rho - p.restDensity)
  }
  return pr > 0 ? pr : 0
}

// a_i = g − (1/ρ_i) Σ_j m (p_i + p_j)/(2ρ_j) ∇W_spiky(r_ij)
// With ∇W = −C(h − r)² r̂ and r̂ = (x_i − x_j)/r, this pushes i away from j.
function computeAccelerations(s: SimState, p: Params, k: Kernels) {
  const { n, pos, acc, mass, density, pressure } = s
  const { neighborStart, neighborEnd, neighbors, neighborDist2 } = s.grid

  for (let i = 0; i < n; i++) {
    for (let d = 0; d < DIM; d++) acc[i * DIM + d] = p.gravity[d]

    const pi = pressure[i]
    const rhoi = density[i]
    for (let q = neighborStart[i], end = neighborEnd[i]; q < end; q++) {
      const j = neighbors[q]
      const r2 = neighborDist2[q]
      if (r2 < 1e-12) continue // coincident particles: direction undefined
      const r = Math.sqrt(r2)
      const shared = (mass * (pi + pressure[j])) / (2 * density[j])
      const scale = (shared * spikyGradMag(k, r)) / (r * rhoi)
      for (let d = 0; d < DIM; d++) {
        acc[i * DIM + d] += scale * (pos[i * DIM + d] - pos[j * DIM + d])
      }
    }
  }
}

// Radial push/pull inside the brush, falloff (1 − r/R), applied as a velocity
// impulse. The impulse is scaled to the fluid's stiffness (see mouseAccel) and
// brushed particles are speed-capped (see mouseSpeedCap), so even full strength
// can neither crush the fluid nor violate CFL.
function applyMouse(s: SimState, p: Params, mouse: MouseInput) {
  const { n, pos, vel } = s
  const R = p.brushRadius
  const cap = mouseSpeedCap(p)
  const impulse = mouse.sign * mouseAccel(p) * p.dt
  for (let i = 0; i < n; i++) {
    let r2 = 0
    for (let d = 0; d < DIM; d++) {
      const dx = pos[i * DIM + d] - mouse.pos[d]
      r2 += dx * dx
    }
    if (r2 >= R * R || r2 < 1e-12) continue
    const r = Math.sqrt(r2)
    const scale = (impulse * (1 - r / R)) / r
    let v2 = 0
    for (let d = 0; d < DIM; d++) {
      vel[i * DIM + d] += scale * (pos[i * DIM + d] - mouse.pos[d])
      v2 += vel[i * DIM + d] * vel[i * DIM + d]
    }
    if (v2 > cap * cap) {
      const shrink = cap / Math.sqrt(v2)
      for (let d = 0; d < DIM; d++) vel[i * DIM + d] *= shrink
    }
  }
}

// Symplectic Euler with XSPH velocity smoothing between the velocity and
// position updates:  v_i += ε Σ_j (m/ρ_j)(v_j − v_i) W(r_ij).
// XSPH reads the old velocities and writes a scratch buffer, so the result
// does not depend on particle visiting order.
function integrate(s: SimState, p: Params, k: Kernels) {
  const { n, pos, acc, mass, density } = s
  const { neighborStart, neighborEnd, neighbors, neighborDist2 } = s.grid
  const dt = p.dt
  const eps = p.viscosityEps
  const vel = s.vel
  const out = s.velScratch

  for (let q = 0; q < n * DIM; q++) vel[q] += acc[q] * dt

  for (let i = 0; i < n; i++) {
    for (let d = 0; d < DIM; d++) out[i * DIM + d] = vel[i * DIM + d]
    for (let q = neighborStart[i], end = neighborEnd[i]; q < end; q++) {
      const j = neighbors[q]
      const w = (eps * mass * poly6(k, neighborDist2[q])) / density[j]
      for (let d = 0; d < DIM; d++) {
        out[i * DIM + d] += w * (vel[j * DIM + d] - vel[i * DIM + d])
      }
    }
  }
  s.vel = out
  s.velScratch = vel

  for (let q = 0; q < n * DIM; q++) pos[q] += out[q] * dt
}

// Reflect the normal velocity with restitution and clamp back inside the box,
// inset by the particle radius so particles touch the wall rather than straddle it.
function resolveBoundaries(s: SimState, p: Params) {
  const { n, pos, vel } = s
  const { min, max } = p.domain
  const r = particleRadius(p)
  const e = p.restitution
  for (let i = 0; i < n; i++) {
    for (let d = 0; d < DIM; d++) {
      const k = i * DIM + d
      const lo = min[d] + r
      const hi = max[d] - r
      if (pos[k] < lo) {
        pos[k] = lo
        if (vel[k] < 0) vel[k] = -vel[k] * e
      } else if (pos[k] > hi) {
        pos[k] = hi
        if (vel[k] > 0) vel[k] = -vel[k] * e
      }
    }
  }
}
