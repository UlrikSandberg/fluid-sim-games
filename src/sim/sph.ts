import { DIM, mouseAccel, mouseSpeedCap, particleMass, particleRadius, type Domain, type Params } from './params.ts'
import { mulberry32 } from './rng.ts'
import { makeKernels, poly6, spikyGradMag, type Kernels } from './kernels.ts'
import { buildGrid, buildNeighbors, createGrid, type SpatialGrid } from './spatialHash.ts'

// Structure-of-arrays particle state. Vector fields have stride DIM.
// Fluid particles are indices 0..n. Boundary particles follow at n..n+nb (walls
// first, then the obstacle): they count toward fluid density and push back
// through pressure, but are never integrated, clamped or rendered.
export interface SimState {
  n: number
  nb: number
  nWall: number
  mass: number
  weight: Float32Array // m for fluid, ψ for boundary
  pos: Float32Array
  vel: Float32Array
  velScratch: Float32Array // XSPH writes here, then swaps with vel
  acc: Float32Array // fluid only
  density: Float32Array // boundary entries stay at ρ₀
  pressure: Float32Array // fluid only
  neighborCount: Uint16Array // fluid only
  grid: SpatialGrid
  obstacle: Obstacle | null
  boundaryH: number // h the boundary layers and ψ were built for
}

// Axis-aligned rectangle filled with boundary particles. step() moves the centre
// toward `target` no faster than mouseSpeedCap, so dragging can't break CFL.
export interface Obstacle {
  center: number[]
  halfSize: number[]
  vel: number[]
  target: number[]
}

// Brush input for one step.
//  - 'radial': push (sign +1) or pull (sign −1) around `pos`.
//  - 'current': drive the fluid around `pos` along `arrow` (sign −1 reverses it).
export interface MouseInput {
  active: boolean
  kind: 'radial' | 'current'
  pos: number[] // brush centre, length DIM, world units
  sign: number
  arrow: number[] // 'current' only: the drawn vector from `pos`, world units
}

export function createState(p: Params): SimState {
  const n = Math.min(p.particleCount, blockCapacity(p))
  const mass = particleMass(p)
  const s: SimState = {
    n,
    nb: 0,
    nWall: 0,
    mass,
    weight: new Float32Array(n).fill(mass),
    pos: new Float32Array(n * DIM),
    vel: new Float32Array(n * DIM),
    velScratch: new Float32Array(n * DIM),
    acc: new Float32Array(n * DIM),
    density: new Float32Array(n),
    pressure: new Float32Array(n),
    neighborCount: new Uint16Array(n),
    grid: createGrid(p.domain, p.h, n, n),
    obstacle: null,
    boundaryH: 0,
  }
  seedBlock(s, p)
  buildBoundary(s, p)
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
  if (s.boundaryH !== p.h) buildBoundary(s, p)
  if (s.obstacle) moveObstacle(s, p, s.obstacle)
  const total = s.n + s.nb
  if (s.grid.cellSize !== p.h || s.grid.sorted.length !== total) s.grid = createGrid(p.domain, p.h, s.n, total)
  buildGrid(s.grid, s.pos, total)
  buildNeighbors(s.grid, s.pos, s.n, p.h)

  const k = makeKernels(p.h)
  computeDensityPressure(s, p, k)
  computeAccelerations(s, p, k)
  if (mouse?.active) {
    if (mouse.kind === 'current') applyCurrent(s, p, mouse)
    else applyMouse(s, p, mouse)
  }
  integrate(s, p, k)
  if (s.obstacle) pushOutOfObstacle(s, p, s.obstacle)
  resolveBoundaries(s, p)
}

// Places the obstacle (centre and full size; clamped to fit the domain), or
// removes it with null. Fluid trapped inside is pushed out.
export function setObstacle(s: SimState, p: Params, rect: { center: number[]; size: number[] } | null) {
  if (rect) {
    const { min, max } = p.domain
    const halfSize = rect.size.map((w, d) => Math.min(Math.max(w / 2, p.spacing), (max[d] - min[d]) / 2))
    const center = rect.center.slice()
    clampCenter(center, halfSize, p.domain)
    s.obstacle = { center, halfSize, vel: new Array(DIM).fill(0), target: center.slice() }
  } else {
    s.obstacle = null
  }
  buildBoundary(s, p)
  if (s.obstacle) {
    pushOutOfObstacle(s, p, s.obstacle)
    resolveBoundaries(s, p)
  }
}

// (Re)generates every boundary particle and its ψ. Runs on creation, when h
// changes (layer count and ψ depend on it) and when the obstacle is placed,
// resized or removed; dragging only translates the existing particles.
function buildBoundary(s: SimState, p: Params) {
  const k = makeKernels(p.h)
  const sp = p.spacing
  const layers = Math.ceil(p.h / sp)
  const walls = wallParticles(p.domain, sp, layers)
  const obs = s.obstacle ? obstacleParticles(s.obstacle, sp, layers) : null
  const n = s.n
  const nWall = walls.pos.length / DIM
  const nb = nWall + (obs ? obs.pos.length / DIM : 0)
  const total = n + nb

  s.pos = resized(s.pos, n * DIM, total * DIM)
  s.vel = resized(s.vel, n * DIM, total * DIM)
  s.velScratch = new Float32Array(total * DIM)
  s.density = resized(s.density, n, total)
  s.weight = resized(s.weight, n, total)
  s.density.fill(p.restDensity, n)
  s.pos.set(walls.pos, n * DIM)
  boundaryWeights(walls.pos, walls.spacing, k, p.restDensity, s.weight, n)
  if (obs && s.obstacle) {
    s.pos.set(obs.pos, (n + nWall) * DIM)
    boundaryWeights(obs.pos, obs.spacing, k, p.restDensity, s.weight, n + nWall)
    for (let i = n + nWall; i < total; i++) {
      for (let d = 0; d < DIM; d++) s.vel[i * DIM + d] = s.obstacle.vel[d]
    }
  }
  s.nb = nb
  s.nWall = nWall
  s.boundaryH = p.h
}

// New array of length `len` holding the first `keep` entries of `a`.
function resized(a: Float32Array, keep: number, len: number): Float32Array {
  const out = new Float32Array(len)
  out.set(a.subarray(0, keep))
  return out
}

interface Lattice {
  pos: number[] // stride DIM
  spacing: number[] // per axis
}

// The points of the grid coords[0] × coords[1] × … for which `keep` is true.
function latticeWhere(coords: number[][], keep: (idx: number[], x: number[]) => boolean): number[] {
  const out: number[] = []
  const idx = new Array<number>(DIM).fill(0)
  const x = new Array<number>(DIM).fill(0)
  const count = coords.reduce((a, c) => a * c.length, 1)
  for (let c = 0; c < count; c++) {
    let rem = c
    for (let d = 0; d < DIM; d++) {
      idx[d] = rem % coords[d].length
      rem = Math.floor(rem / coords[d].length)
      x[d] = coords[d][idx[d]]
    }
    if (keep(idx, x)) out.push(...x)
  }
  return out
}

// `layers` rows at the rest spacing just outside every wall, corners included.
// The first row is half a spacing out, one spacing from fluid resting on the wall.
function wallParticles(domain: Domain, sp: number, layers: number): Lattice {
  const coords: number[][] = []
  const inside: boolean[][] = []
  const spacing: number[] = []
  for (let d = 0; d < DIM; d++) {
    const lo = domain.min[d]
    const hi = domain.max[d]
    const cells = Math.max(1, Math.round((hi - lo) / sp))
    const step = (hi - lo) / cells
    const c: number[] = []
    for (let l = layers - 1; l >= 0; l--) c.push(lo - (l + 0.5) * sp)
    for (let k = 0; k < cells; k++) c.push(lo + (k + 0.5) * step)
    for (let l = 0; l < layers; l++) c.push(hi + (l + 0.5) * sp)
    coords.push(c)
    inside.push(c.map((_, k) => k >= layers && k < layers + cells))
    spacing.push(step)
  }
  return { pos: latticeWhere(coords, (idx) => idx.some((k, d) => !inside[d][k])), spacing }
}

// The obstacle's outline filled inward with `layers` rows, spacing stretched so
// the lattice exactly spans the rectangle.
function obstacleParticles(o: Obstacle, sp: number, layers: number): Lattice {
  const coords: number[][] = []
  const spacing: number[] = []
  for (let d = 0; d < DIM; d++) {
    const cells = Math.max(1, Math.round((2 * o.halfSize[d]) / sp))
    const step = (2 * o.halfSize[d]) / cells
    const c: number[] = []
    for (let k = 0; k < cells; k++) c.push(o.center[d] - o.halfSize[d] + (k + 0.5) * step)
    coords.push(c)
    spacing.push(step)
  }
  const thickness = layers * sp
  const nearOutline = (_idx: number[], x: number[]) => {
    for (let d = 0; d < DIM; d++) if (o.halfSize[d] - Math.abs(x[d] - o.center[d]) < thickness) return true
    return false
  }
  return { pos: latticeWhere(coords, nearOutline), spacing }
}

// Akinci et al. 2012: ψ_b = ρ₀ / δ_b with δ_b = Σ_k W(r_bk) over the body's own
// boundary particles, self included. δ_b is floored at the sum over a complete
// lattice of the same spacing: the layers are ≥ h thick, so a short sum only
// means b sits on a face whose missing support is fluid, not sparse sampling.
function boundaryWeights(
  x: number[], spacing: number[], k: Kernels, restDensity: number, out: Float32Array, offset: number,
) {
  const full = latticeKernelSum(k, spacing)
  const m = x.length / DIM
  for (let b = 0; b < m; b++) {
    let delta = 0
    for (let c = 0; c < m; c++) {
      let r2 = 0
      for (let d = 0; d < DIM; d++) {
        const dx = x[b * DIM + d] - x[c * DIM + d]
        r2 += dx * dx
      }
      delta += poly6(k, r2)
    }
    out[offset + b] = restDensity / Math.max(delta, full)
  }
}

// Σ W over an infinite lattice with the given per-axis spacing, centred on a site.
function latticeKernelSum(k: Kernels, spacing: number[]): number {
  const reach = spacing.map((sp) => Math.ceil(k.h / sp))
  const count = reach.reduce((a, r) => a * (2 * r + 1), 1)
  let sum = 0
  for (let c = 0; c < count; c++) {
    let rem = c
    let r2 = 0
    for (let d = 0; d < DIM; d++) {
      const side = 2 * reach[d] + 1
      const x = ((rem % side) - reach[d]) * spacing[d]
      rem = Math.floor(rem / side)
      r2 += x * x
    }
    sum += poly6(k, r2)
  }
  return sum
}

// Keeps the rectangle inside the domain (centred on an axis it doesn't fit).
function clampCenter(c: number[], halfSize: number[], domain: Domain) {
  for (let d = 0; d < DIM; d++) {
    const lo = domain.min[d] + halfSize[d]
    const hi = domain.max[d] - halfSize[d]
    c[d] = lo > hi ? (domain.min[d] + domain.max[d]) / 2 : Math.min(hi, Math.max(lo, c[d]))
  }
}

const move = new Float64Array(DIM)

// Moves the obstacle toward its target at most mouseSpeedCap · dt per step. Its
// boundary particles move with it and carry its velocity, so XSPH sees a moving wall.
function moveObstacle(s: SimState, p: Params, o: Obstacle) {
  clampCenter(o.target, o.halfSize, p.domain)
  let d2 = 0
  for (let d = 0; d < DIM; d++) {
    move[d] = o.target[d] - o.center[d]
    d2 += move[d] * move[d]
  }
  const maxStep = mouseSpeedCap(p) * p.dt
  const f = d2 > maxStep * maxStep ? maxStep / Math.sqrt(d2) : 1
  for (let d = 0; d < DIM; d++) {
    move[d] *= f
    o.center[d] += move[d]
    o.vel[d] = move[d] / p.dt
  }
  for (let i = s.n + s.nWall, end = s.n + s.nb; i < end; i++) {
    for (let d = 0; d < DIM; d++) {
      s.pos[i * DIM + d] += move[d]
      s.vel[i * DIM + d] = o.vel[d]
    }
  }
}

// Safety net like the wall clamp, and what clears the area when an obstacle is
// placed. A fluid particle inside the outline (grown by the particle radius) is
// mirrored across the nearest face, so particles at different depths don't land
// on the same spot, and its velocity into that face, relative to the obstacle,
// is reflected with restitution.
function pushOutOfObstacle(s: SimState, p: Params, o: Obstacle) {
  const { n, pos, vel } = s
  const r = particleRadius(p)
  const e = p.restitution
  for (let i = 0; i < n; i++) {
    let face = 0
    let depth = Infinity
    for (let d = 0; d < DIM; d++) {
      const dd = o.halfSize[d] + r - Math.abs(pos[i * DIM + d] - o.center[d])
      if (dd < depth) {
        depth = dd
        face = d
      }
    }
    if (depth <= 0) continue
    const k = i * DIM + face
    const sign = pos[k] >= o.center[face] ? 1 : -1
    pos[k] += sign * 2 * depth
    const vn = (vel[k] - o.vel[face]) * sign
    if (vn < 0) vel[k] = o.vel[face] - sign * vn * e
  }
}

// ρ_i = Σ_j m_j W(r_ij) including self, with ψ_b standing in for m_j on boundary
// neighbours, then pressure from the EOS.
function computeDensityPressure(s: SimState, p: Params, k: Kernels) {
  const { n, mass, weight, density, pressure, neighborCount } = s
  const { neighborStart, neighborEnd, neighbors, neighborDist2 } = s.grid
  const self = mass * poly6(k, 0)

  for (let i = 0; i < n; i++) {
    let rho = self
    const start = neighborStart[i]
    const end = neighborEnd[i]
    for (let q = start; q < end; q++) rho += weight[neighbors[q]] * poly6(k, neighborDist2[q])
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
// Boundary neighbours b mirror i's pressure (p_b = p_i, ρ_b = ρ_i): −ψ_b p_i/ρ_i² ∇W.
function computeAccelerations(s: SimState, p: Params, k: Kernels) {
  const { n, pos, acc, mass, weight, density, pressure } = s
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
      const shared = j < n ? (mass * (pi + pressure[j])) / (2 * density[j]) : (weight[j] * pi) / rhoi
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

const target = new Float64Array(DIM)

// Drives particles inside the brush toward a target velocity along the drawn
// arrow, blending v → v_target at a rate scaled by the falloff (1 − r/R). The
// target speed is the brush speed cap once the arrow spans the brush diameter,
// proportionally less for shorter arrows. The blend rate is chosen so the peak
// acceleration (rate · cap) equals mouseAccel: the same stiffness-scaled bound
// as push/pull, so a steady current can't crush fluid against a wall either.
function applyCurrent(s: SimState, p: Params, mouse: MouseInput) {
  const { n, pos, vel } = s
  const R = p.brushRadius
  const cap = mouseSpeedCap(p)
  let len2 = 0
  for (let d = 0; d < DIM; d++) len2 += mouse.arrow[d] * mouse.arrow[d]
  if (len2 < 1e-12 || cap === 0) return
  const len = Math.sqrt(len2)
  const speed = cap * Math.min(1, len / (2 * R))
  for (let d = 0; d < DIM; d++) target[d] = (mouse.sign * mouse.arrow[d] * speed) / len
  const rateDt = (mouseAccel(p) / cap) * p.dt

  for (let i = 0; i < n; i++) {
    let r2 = 0
    for (let d = 0; d < DIM; d++) {
      const dx = pos[i * DIM + d] - mouse.pos[d]
      r2 += dx * dx
    }
    if (r2 >= R * R) continue
    const blend = Math.min(1, rateDt * (1 - Math.sqrt(r2) / R))
    for (let d = 0; d < DIM; d++) vel[i * DIM + d] += blend * (target[d] - vel[i * DIM + d])
  }
}

// Symplectic Euler with XSPH velocity smoothing between the velocity and
// position updates:  v_i += ε Σ_j (m/ρ_j)(v_j − v_i) W(r_ij).
// XSPH reads the old velocities and writes a scratch buffer, so the result
// does not depend on particle visiting order. Boundary neighbours take part with
// their own (wall or obstacle) velocity and ρ₀.
function integrate(s: SimState, p: Params, k: Kernels) {
  const { n, pos, acc, weight, density } = s
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
      const w = (eps * weight[j] * poly6(k, neighborDist2[q])) / density[j]
      for (let d = 0; d < DIM; d++) {
        out[i * DIM + d] += w * (vel[j * DIM + d] - vel[i * DIM + d])
      }
    }
  }
  out.set(vel.subarray(n * DIM, (n + s.nb) * DIM), n * DIM)
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
