// Spatial dimension. Everything downstream loops over components with stride DIM,
// so switching to 3D should be a change here (plus a camera), not a rewrite.
export const DIM = 2

export type Vec = number[] // length DIM

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
}

export const defaultParams: Readonly<Params> = {
  h: 1.0,
  spacing: 0.4,
  restDensity: 1000,
  // Tuned from the handoff's 2000 / 0.05 / 0.002 (see scripts/check-sim.ts):
  // k=2000 compressed the bulk ~8% and never settled; k=6000 keeps it ~2–3%.
  // c_s = √k ≈ 77, so CFL allows dt ≤ 0.4·h/c_s ≈ 0.005. dt=0.004 halves the
  // substeps per frame (~4 at 60 fps), which is what makes real time affordable.
  stiffness: 6000,
  useTait: false,
  viscosityEps: 0.1,
  gravity: [0, -9.8],
  dt: 0.004,
  maxSubsteps: 8,
  restitution: 0.3,
  domain: { min: [0, 0], max: [40, 30] },
  particleCount: 4000,
  seed: 1,
}

// Single mutable params object. The control panel writes into it, the loop reads it.
export const params: Params = structuredClone(defaultParams)

// Mass is derived, never tuned: m = ρ₀ · spacing^DIM.
export function particleMass(p: Params): number {
  return p.restDensity * p.spacing ** DIM
}

// Speed of sound at rest density. Both EOS variants are scaled so dp/dρ = k at ρ₀.
export function soundSpeed(p: Params): number {
  return Math.sqrt(p.stiffness)
}

// Largest stable timestep by CFL: dt ≤ 0.4 · h / c_s.
export function cflDt(p: Params): number {
  return (0.4 * p.h) / soundSpeed(p)
}
