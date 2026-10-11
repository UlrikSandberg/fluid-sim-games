// Normalization constants per dimension. This is the only place in the codebase
// that knows about dimension-specific constants.
//
//   Poly6          W(r)  = C (h² − r²)³      2D: 4/(π h⁸)     3D: 315/(64 π h⁹)
//   Spiky gradient ∇W(r) = −C (h − r)² r̂     2D: 30/(π h⁵)    3D: 45/(π h⁶)
const COEFFS: Record<number, { poly6: (h: number) => number; spikyGrad: (h: number) => number }> = {
  2: {
    poly6: (h) => 4 / (Math.PI * h ** 8),
    spikyGrad: (h) => 30 / (Math.PI * h ** 5),
  },
  3: {
    poly6: (h) => 315 / (64 * Math.PI * h ** 9),
    spikyGrad: (h) => 45 / (Math.PI * h ** 6),
  },
}

export interface Kernels {
  h: number
  h2: number
  poly6Coef: number
  spikyGradCoef: number
}

export function makeKernels(h: number, dim: number): Kernels {
  const c = COEFFS[dim]
  return { h, h2: h * h, poly6Coef: c.poly6(h), spikyGradCoef: c.spikyGrad(h) }
}

// Poly6 evaluated from squared distance (avoids a sqrt). Zero outside h.
export function poly6(k: Kernels, r2: number): number {
  if (r2 >= k.h2) return 0
  const d = k.h2 - r2
  return k.poly6Coef * d * d * d
}

// Magnitude of the Spiky gradient, C (h − r)². The direction is −r̂ (toward the
// neighbour); callers multiply by the separation vector themselves.
export function spikyGradMag(k: Kernels, r: number): number {
  if (r >= k.h) return 0
  const d = k.h - r
  return k.spikyGradCoef * d * d
}
