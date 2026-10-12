// Spatial dimension is not a global: a simulation's dimension is its domain's
// (dimOf). Everything downstream loops over components with stride dim, so 2D and
// 3D scenes share this solver, and nothing outside kernels.ts knows a
// dimension-specific constant.
export function dimOf(p: Params): number {
  return p.domain.min.length
}

export type Vec = number[] // length dimOf(params)

export interface Domain {
  min: Vec
  max: Vec
}

export interface Params {
  h: number // smoothing radius
  spacing: number // initial particle spacing (~0.4·h → ~20 neighbours in 2D)
  restDensity: number
  stiffness: number // k = c_s², for either EOS (see equationOfState in sph.ts)
  useTait: boolean // switch to Tait EOS once linear is stable
  viscosityEps: number // XSPH ε
  gravity: Vec
  dt: number
  maxSubsteps: number
  restitution: number
  domain: Domain
  particleCount: number
  seed: number
  brushRadius: number // mouse brush radius R (world units)
  mouseStrength: number // 0–1, fraction of the strongest safe brush (see mouseAccel)
  mouseMaxSpeed: number // speed cap for particles inside the brush (also capped by CFL)
  obstacleSize: Vec // full width/height of the draggable rectangle obstacle
  obstacleDynamic: boolean // float freely under fluid forces and gravity (else stays put)
  obstacleDensity: number // body density as a fraction of restDensity (0.5 floats half-submerged)
}

export const defaultParams: Readonly<Params> = {
  h: 1.0,
  spacing: 0.4,
  restDensity: 1000,
  // Tuned from the handoff's 2000 / 0.05 / 0.002 (see scripts/check-sim.ts):
  // k=2000 compressed the bulk ~8% and never settled. 6000 was enough while wall
  // particles were density-deficient; with boundary particles the bottom rows
  // count as bulk and read 1.05–1.06, so 8000 brings them back inside ±5%.
  // c_s = √k ≈ 89, so CFL allows dt ≤ 0.4·h/c_s ≈ 0.0045. dt=0.004 keeps ~4
  // substeps per frame at 60 fps. Don't go to 10000 at this dt: it sits exactly
  // on the CFL limit, which leaves the mouse brush a speed cap of zero.
  stiffness: 8000,
  useTait: false,
  viscosityEps: 0.1,
  gravity: [0, -9.8],
  dt: 0.004,
  maxSubsteps: 8,
  restitution: 0.3,
  domain: { min: [0, 0], max: [40, 30] },
  particleCount: 4000,
  seed: 1,
  brushRadius: 4,
  mouseStrength: 0.5,
  mouseMaxSpeed: 15,
  obstacleSize: [6, 3],
  obstacleDynamic: false,
  obstacleDensity: 0.5,
}

// Single mutable params object. The control panel writes into it, the loop reads it.
export const params: Params = structuredClone(defaultParams)

// 3D preset: same fluid, one more axis. spacing = 0.5·h gives ~33 neighbours in
// 3D, the usual target (0.4·h, right for 2D, would give ~64). The domain's length
// is what makes this a 3D simulation (dimOf).
export const defaultParams3D: Readonly<Params> = {
  ...defaultParams,
  spacing: 0.5,
  gravity: [0, -9.8, 0],
  // Sized for real time on one CPU thread: a 3D step costs ~0.87 µs per particle,
  // and 60 fps needs ~4 steps per frame, so ~4000 particles; this tank holds them
  // about 4 deep once they settle.
  domain: { min: [0, 0, 0], max: [16, 10, 8] },
  particleCount: 4000,
  obstacleSize: [4, 2, 3],
  obstacleDynamic: true, // in 3D, bodies are dropped in and float
}

// Mass is derived, never tuned: m = ρ₀ · spacing^dim.
export function particleMass(p: Params): number {
  const dim = dimOf(p)
  return p.restDensity * p.spacing ** dim
}

// Speed of sound at rest density. Both EOS variants are scaled so dp/dρ = k at ρ₀.
export function soundSpeed(p: Params): number {
  return Math.sqrt(p.stiffness)
}

// Largest stable timestep by CFL: dt ≤ 0.4 · h / c_s.
export function cflDt(p: Params): number {
  return (0.4 * p.h) / soundSpeed(p)
}

// Collision radius of a particle: half the rest spacing, so a lattice packed
// against a wall sits exactly where seeding puts it.
export function particleRadius(p: Params): number {
  return p.spacing / 2
}

// Speed cap for brushed particles: the configured cap, but never more than
//  - Mach 0.25: weakly-compressible SPH assumes |v| ≪ c_s; fluid driven into a
//    wall at speed v compresses by roughly v/c_s on impact, and
//  - what CFL allows on top of c_s (dt · (c_s + v) ≤ 0.4 · h).
// Input can't break stability.
const MOUSE_MAX_MACH = 0.25
export function mouseSpeedCap(p: Params): number {
  const c = soundSpeed(p)
  return Math.max(0, Math.min(p.mouseMaxSpeed, MOUSE_MAX_MACH * c, (0.4 * p.h) / p.dt - c))
}

// Peak brush acceleration (at the brush centre; it falls off as 1 − r/R).
// A steady push of a over the brush compresses the fluid by Δρ/ρ₀ ∝ a·R/c_s², so
// the acceleration is scaled by c_s²/R: full strength compresses by a bounded
// fraction whatever the stiffness or brush size. The constant is calibrated from
// measurement (a stationary attract brush at full strength reaches ρ/ρ₀ ≈ 1.3),
// not from the estimate, which undershoots for converging flow.
const MOUSE_MAX_COMPRESSION = 0.15
export function mouseAccel(p: Params): number {
  return (p.mouseStrength * MOUSE_MAX_COMPRESSION * soundSpeed(p) ** 2) / p.brushRadius
}

// Rigid-body mass of the obstacle: its density times its area (volume in 3D).
export function obstacleMass(p: Params, halfSize: number[]): number {
  return p.obstacleDensity * p.restDensity * halfSize.reduce((a, hs) => a * 2 * hs, 1)
}
