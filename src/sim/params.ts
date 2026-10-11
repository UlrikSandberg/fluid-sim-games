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
  stiffness: number // linear EOS k
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

// Single mutable params object. The control panel writes into it, the loop reads it.
export const params: Params = {
  h: 1.0,
  spacing: 0.4,
  restDensity: 1000,
  stiffness: 2000,
  useTait: false,
  viscosityEps: 0.05,
  gravity: [0, -9.8],
  dt: 0.002,
  maxSubsteps: 8,
  restitution: 0.3,
  domain: { min: [0, 0], max: [40, 30] },
  particleCount: 4000,
  seed: 1,
}

// Mass is derived, never tuned: m = ρ₀ · spacing^DIM.
export function particleMass(p: Params): number {
  return p.restDensity * p.spacing ** DIM
}
