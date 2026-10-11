import { DIM, mouseAccel, mouseSpeedCap, obstacleMass, particleMass, particleRadius, type Domain, type Params } from './params.ts'
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

// A rectangle filled with boundary particles, stored as offsets in its own frame
// and placed in the world from its pose every step.
//  - Kinematic while grabbed, or when p.obstacleDynamic is off: step() moves the
//    centre toward `target` no faster than mouseSpeedCap, so dragging can't break CFL.
//  - Dynamic otherwise: the fluid's reaction forces, accumulated on its boundary
//    particles during a step, plus gravity move and spin it. Buoyancy and drag
//    come out of that sum; nothing models them separately.
// Orientation is a single angle, so rotation is 2D only (3D needs a quaternion
// and an inertia tensor); bodyPoint() is the one place that applies it.
export interface Obstacle {
  center: number[]
  halfSize: number[]
  angle: number
  vel: number[]
  omega: number // angular velocity
  target: number[]
  grabbed: boolean
  local: Float32Array // boundary particle offsets in the body frame, stride DIM
  force: Float64Array // fluid force on the body, accumulated over one step
  torque: number
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
  if (s.obstacle) advanceObstacle(s, p, s.obstacle)
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
    s.obstacle = {
      center,
      halfSize,
      angle: 0,
      vel: new Array(DIM).fill(0),
      omega: 0,
      target: center.slice(),
      grabbed: false,
      local: new Float32Array(0),
      force: new Float64Array(DIM),
      torque: 0,
    }
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
// resized or removed; moving it only re-places the existing particles.
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
  s.nb = nb
  s.nWall = nWall
  s.boundaryH = p.h
  if (obs && s.obstacle) {
    // ψ depends only on relative positions, so the body-frame offsets will do.
    boundaryWeights(obs.pos, obs.spacing, k, p.restDensity, s.weight, n + nWall)
    s.obstacle.local = Float32Array.from(obs.pos)
    placeBody(s, s.obstacle)
  }
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
// the lattice exactly spans the rectangle. Positions are body-frame offsets.
function obstacleParticles(o: Obstacle, sp: number, layers: number): Lattice {
  const coords: number[][] = []
  const spacing: number[] = []
  for (let d = 0; d < DIM; d++) {
    const cells = Math.max(1, Math.round((2 * o.halfSize[d]) / sp))
    const step = (2 * o.halfSize[d]) / cells
    const c: number[] = []
    for (let k = 0; k < cells; k++) c.push(-o.halfSize[d] + (k + 0.5) * step)
    coords.push(c)
    spacing.push(step)
  }
  const thickness = layers * sp
  const nearOutline = (_idx: number[], x: number[]) => {
    for (let d = 0; d < DIM; d++) if (o.halfSize[d] - Math.abs(x[d]) < thickness) return true
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

// Keeps a box with the given half-extents inside the domain (centred on an
// axis it doesn't fit).
function clampCenter(c: number[], extent: number[], domain: Domain) {
  for (let d = 0; d < DIM; d++) {
    const lo = domain.min[d] + extent[d]
    const hi = domain.max[d] - extent[d]
    c[d] = lo > hi ? (domain.min[d] + domain.max[d]) / 2 : Math.min(hi, Math.max(lo, c[d]))
  }
}

// Half-extents of the rotated rectangle's axis-aligned bounding box.
function bodyExtent(o: Obstacle): number[] {
  const c = Math.abs(Math.cos(o.angle))
  const sn = Math.abs(Math.sin(o.angle))
  return [c * o.halfSize[0] + sn * o.halfSize[1], sn * o.halfSize[0] + c * o.halfSize[1]]
}

// Body frame → world offset (rotation only). `inverse` maps world → body frame.
function bodyPoint(o: Obstacle, x: number, y: number, inverse = false): [number, number] {
  const c = Math.cos(o.angle)
  const sn = inverse ? -Math.sin(o.angle) : Math.sin(o.angle)
  return [c * x - sn * y, sn * x + c * y]
}

// True if world point `pt` lies inside the obstacle.
export function obstacleContains(o: Obstacle, pt: number[]): boolean {
  const [lx, ly] = bodyPoint(o, pt[0] - o.center[0], pt[1] - o.center[1], true)
  return Math.abs(lx) <= o.halfSize[0] && Math.abs(ly) <= o.halfSize[1]
}

// Writes the obstacle's boundary particles from its pose: x = c + R·l and
// v = v_c + ω × (R·l), so XSPH and pressure see a moving, spinning wall.
function placeBody(s: SimState, o: Obstacle) {
  const first = s.n + s.nWall
  const count = o.local.length / DIM
  for (let b = 0; b < count; b++) {
    const [rx, ry] = bodyPoint(o, o.local[b * DIM], o.local[b * DIM + 1])
    const i = (first + b) * DIM
    s.pos[i] = o.center[0] + rx
    s.pos[i + 1] = o.center[1] + ry
    s.vel[i] = o.vel[0] - o.omega * ry
    s.vel[i + 1] = o.vel[1] + o.omega * rx
  }
}

// Advances the obstacle one step, then re-places its boundary particles.
// Kinematic: toward the target at most mouseSpeedCap · dt, no spin.
// Dynamic: symplectic Euler on the fluid force accumulated in the previous step
// (one step of lag, which is fine at this dt) plus gravity. Mass is
// ρ_body · area, inertia that of a uniform rectangle.
function advanceObstacle(s: SimState, p: Params, o: Obstacle) {
  const dt = p.dt
  if (o.grabbed || !p.obstacleDynamic) {
    clampCenter(o.target, bodyExtent(o), p.domain)
    let d2 = 0
    for (let d = 0; d < DIM; d++) d2 += (o.target[d] - o.center[d]) ** 2
    const maxStep = mouseSpeedCap(p) * dt
    const f = d2 > maxStep * maxStep ? maxStep / Math.sqrt(d2) : 1
    for (let d = 0; d < DIM; d++) {
      const move = (o.target[d] - o.center[d]) * f
      o.center[d] += move
      o.vel[d] = move / dt
    }
    o.omega = 0
  } else {
    const mass = obstacleMass(p, o.halfSize)
    const inertia = (mass * ((2 * o.halfSize[0]) ** 2 + (2 * o.halfSize[1]) ** 2)) / 12
    for (let d = 0; d < DIM; d++) {
      o.vel[d] += (o.force[d] / mass + p.gravity[d]) * dt
      o.center[d] += o.vel[d] * dt
    }
    o.omega += (o.torque / inertia) * dt
    o.angle += o.omega * dt
    collideBodyWithWalls(o, p, mass, inertia)
    for (let d = 0; d < DIM; d++) o.target[d] = o.center[d]
  }
  placeBody(s, o)
  o.force.fill(0)
  o.torque = 0
}

const CONTACT_FRICTION = 0.5 // Coulomb μ between the body and the walls

// Rigid-body contact between the rectangle's corners and the domain walls. Each
// penetrating corner is pushed back out and gets a normal impulse (with
// restitution) plus a Coulomb friction impulse, both applied at the corner, so
// contact produces torque: a box landing on a corner topples flat instead of
// balancing on it. Corners and walls are visited in a fixed order (determinism).
//
// For the body, the walls are inset by one kernel radius h. Buoyancy is pressure
// on the bottom face, and a body resting flush on the floor seals the fluid out
// from under it: the particles can't represent a gap thinner than one spacing, so
// every resting body acted like a suction cup (a sunk box made light didn't move
// for seconds). The gap stands in for the thin film real water keeps under objects
// thanks to surface roughness. It is h rather than one spacing because a film
// that thin holds only a layer or two of particles: they never reach rest density,
// their pressure clamps to zero, and the seal effectively remains.
function collideBodyWithWalls(o: Obstacle, p: Params, mass: number, inertia: number) {
  const min = p.domain.min.map((v) => v + p.h)
  const max = p.domain.max.map((v) => v - p.h)
  for (const sx of [-1, 1]) {
    for (const sy of [-1, 1]) {
      for (let d = 0; d < DIM; d++) {
        for (const side of [-1, 1]) {
          const [rx, ry] = bodyPoint(o, sx * o.halfSize[0], sy * o.halfSize[1])
          const corner = o.center[d] + (d === 0 ? rx : ry)
          const depth = side < 0 ? min[d] - corner : corner - max[d]
          if (depth <= 0) continue
          // Inward wall normal n and tangent t.
          const nx = d === 0 ? -side : 0
          const ny = d === 1 ? -side : 0
          const tx = -ny
          const ty = nx
          o.center[d] -= side * depth
          const vx = o.vel[0] - o.omega * ry
          const vy = o.vel[1] + o.omega * rx
          const vn = vx * nx + vy * ny
          if (vn >= 0) continue
          const rn = rx * ny - ry * nx
          const jn = (-(1 + p.restitution) * vn) / (1 / mass + (rn * rn) / inertia)
          const rt = rx * ty - ry * tx
          const vt = vx * tx + vy * ty
          let jt = -vt / (1 / mass + (rt * rt) / inertia)
          const maxFriction = CONTACT_FRICTION * jn
          jt = Math.max(-maxFriction, Math.min(maxFriction, jt))
          o.vel[0] += (jn * nx + jt * tx) / mass
          o.vel[1] += (jn * ny + jt * ty) / mass
          o.omega += (jn * rn + jt * rt) / inertia
        }
      }
    }
  }
}

// Safety net like the wall clamp, and what clears the area when an obstacle is
// placed. Works in the body frame: a fluid particle inside the outline (grown by
// the particle radius) is mirrored across the nearest face, so particles at
// different depths don't land on the same spot, and its velocity into that face,
// relative to the body's surface velocity there, is reflected with restitution.
function pushOutOfObstacle(s: SimState, p: Params, o: Obstacle) {
  const { n, pos, vel } = s
  const r = particleRadius(p)
  const e = p.restitution
  for (let i = 0; i < n; i++) {
    const local = bodyPoint(o, pos[i * DIM] - o.center[0], pos[i * DIM + 1] - o.center[1], true)
    let face = 0
    let depth = Infinity
    for (let d = 0; d < DIM; d++) {
      const dd = o.halfSize[d] + r - Math.abs(local[d])
      if (dd < depth) {
        depth = dd
        face = d
      }
    }
    if (depth <= 0) continue
    const sign = local[face] >= 0 ? 1 : -1
    local[face] += sign * 2 * depth
    const [rx, ry] = bodyPoint(o, local[0], local[1])
    pos[i * DIM] = o.center[0] + rx
    pos[i * DIM + 1] = o.center[1] + ry
    const normal = bodyPoint(o, face === 0 ? sign : 0, face === 1 ? sign : 0)
    const relX = vel[i * DIM] - (o.vel[0] - o.omega * ry)
    const relY = vel[i * DIM + 1] - (o.vel[1] + o.omega * rx)
    const vn = relX * normal[0] + relY * normal[1]
    if (vn < 0) {
      vel[i * DIM] -= (1 + e) * vn * normal[0]
      vel[i * DIM + 1] -= (1 + e) * vn * normal[1]
    }
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
// Each boundary push on fluid i from an obstacle particle b is reacted onto the
// body (−m_i a_ib at x_b), which is where buoyancy and form drag come from.
function computeAccelerations(s: SimState, p: Params, k: Kernels) {
  const { n, pos, acc, mass, weight, density, pressure } = s
  const { neighborStart, neighborEnd, neighbors, neighborDist2 } = s.grid
  const o = s.obstacle
  const bodyStart = n + s.nWall

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
      if (o && j >= bodyStart) {
        const fx = -mass * scale * (pos[i * DIM] - pos[j * DIM])
        const fy = -mass * scale * (pos[i * DIM + 1] - pos[j * DIM + 1])
        o.force[0] += fx
        o.force[1] += fy
        o.torque += (pos[j * DIM] - o.center[0]) * fy - (pos[j * DIM + 1] - o.center[1]) * fx
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
// does not depend on particle visiting order. Obstacle boundary particles take
// part with their own velocity and ρ₀ (skin friction, so a current carries the
// body). Wall particles are skipped: free-slip walls. A no-slip floor made fluid
// drag into the gap under a resting body as if into a narrow viscous channel,
// halving how fast a sunk body made light could lift off.
function integrate(s: SimState, p: Params, k: Kernels) {
  const { n, pos, acc, mass, weight, density } = s
  const { neighborStart, neighborEnd, neighbors, neighborDist2 } = s.grid
  const o = s.obstacle
  const bodyStart = n + s.nWall
  const dt = p.dt
  const eps = p.viscosityEps
  const vel = s.vel
  const out = s.velScratch

  for (let q = 0; q < n * DIM; q++) vel[q] += acc[q] * dt

  for (let i = 0; i < n; i++) {
    for (let d = 0; d < DIM; d++) out[i * DIM + d] = vel[i * DIM + d]
    for (let q = neighborStart[i], end = neighborEnd[i]; q < end; q++) {
      const j = neighbors[q]
      if (j >= n && j < bodyStart) continue // free-slip walls
      const w = (eps * weight[j] * poly6(k, neighborDist2[q])) / density[j]
      for (let d = 0; d < DIM; d++) {
        out[i * DIM + d] += w * (vel[j * DIM + d] - vel[i * DIM + d])
      }
      // The obstacle drags fluid along (or holds it back); react that momentum
      // change onto the body as skin friction, so a current can carry it.
      if (o && j >= bodyStart) {
        const fx = (-mass * w * (vel[j * DIM] - vel[i * DIM])) / dt
        const fy = (-mass * w * (vel[j * DIM + 1] - vel[i * DIM + 1])) / dt
        o.force[0] += fx
        o.force[1] += fy
        o.torque += (pos[j * DIM] - o.center[0]) * fy - (pos[j * DIM + 1] - o.center[1]) * fx
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
