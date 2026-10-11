import * as THREE from 'three'
import type { Params } from '../sim/params'
import type { SimState } from '../sim/sph'

export type ColorModeId = 'density' | 'neighbors' | 'speed' | 'pressure' | 'water'

export interface ColorMode {
  id: ColorModeId
  label: string
  range: [number, number] // values outside are clamped to the end colours
  range3D?: [number, number] // if the natural scale differs in 3D (see modeRange)
  stops: string[] // sRGB, low → high; also used for the CSS legend gradient
  scalar: (s: SimState, p: Params, i: number) => number
}

// Diverging blue–grey–red (Moreland "cool to warm"), for quantities centred on a
// reference value.
const COOLWARM = ['#3b4cc0', '#8db0fe', '#dddddd', '#f49a7b', '#b40426']
// Sequential maps. Their dark ends are lifted so low values stay visible
// against the background.
const VIRIDIS = ['#440154', '#3b528b', '#21918c', '#5ec962', '#fde725']
const ICE = ['#1d3b6e', '#1f6fb2', '#3fa7d6', '#9bdcf2', '#f2fbff']
const MAGMA = ['#3b0f70', '#8c2981', '#de4968', '#fe9f6d', '#fcfdbf']
// Not a diagnostic: deep blue water that foams to white where it moves fast.
const WATER = ['#0a3a7a', '#1660b8', '#2b8ad6', '#7cc8f2', '#ffffff']

export const colorModes: ColorMode[] = [
  {
    id: 'density',
    label: 'Density ρ/ρ₀',
    range: [0.9, 1.1],
    stops: COOLWARM,
    scalar: (s, p, i) => s.density[i] / p.restDensity,
  },
  {
    id: 'neighbors',
    label: 'Neighbour count',
    // Target is ~20 in 2D, ~30–40 in 3D; keep it mid-scale.
    range: [0, 40],
    range3D: [0, 70],
    stops: VIRIDIS,
    scalar: (s, _p, i) => s.neighborCount[i],
  },
  {
    id: 'speed',
    label: 'Speed |v|',
    range: [0, 20],
    stops: ICE,
    scalar: (s, _p, i) => {
      let v2 = 0
      for (let d = 0; d < s.dim; d++) v2 += s.vel[i * s.dim + d] ** 2
      return Math.sqrt(v2)
    },
  },
  {
    id: 'pressure',
    label: 'Pressure p',
    // Upper end ≈ hydrostatic pressure at the bottom of a full box (ρ₀ g H).
    range: [0, 300000],
    stops: MAGMA,
    scalar: (s, _p, i) => s.pressure[i],
  },
  {
    id: 'water',
    label: 'Water (foam = speed)',
    range: [0, 14],
    stops: WATER,
    scalar: (s, _p, i) => {
      let v2 = 0
      for (let d = 0; d < s.dim; d++) v2 += s.vel[i * s.dim + d] ** 2
      return Math.sqrt(v2)
    },
  },
]

// The colour scale of a mode for a simulation of the given dimension.
export function modeRange(mode: ColorMode, dim: number): [number, number] {
  return dim === 3 && mode.range3D ? mode.range3D : mode.range
}

export type RenderStyle = 'particles' | 'water'

// Mutable view settings, read by the loop each frame (like `params`).
export const view = { colorMode: 'density' as ColorModeId, style: 'particles' as RenderStyle }

export function getColorMode(id: ColorModeId): ColorMode {
  return colorModes.find((m) => m.id === id) ?? colorModes[0]
}

const LUT_SIZE = 256
const luts = new Map<ColorMode, Float32Array>()

// Interpolates the stops in sRGB (matching the CSS legend), then converts to
// linear, which is what three.js expects for instance colours.
function lutFor(mode: ColorMode): Float32Array {
  let lut = luts.get(mode)
  if (lut) return lut
  lut = new Float32Array(LUT_SIZE * 3)
  const srgb = mode.stops.map((hex) => {
    const v = parseInt(hex.slice(1), 16)
    return [((v >> 16) & 255) / 255, ((v >> 8) & 255) / 255, (v & 255) / 255]
  })
  const color = new THREE.Color()
  for (let k = 0; k < LUT_SIZE; k++) {
    const t = (k / (LUT_SIZE - 1)) * (srgb.length - 1)
    const a = Math.min(Math.floor(t), srgb.length - 2)
    const f = t - a
    const rgb = srgb[a].map((c, ch) => c + (srgb[a + 1][ch] - c) * f)
    color.setRGB(rgb[0], rgb[1], rgb[2], THREE.SRGBColorSpace)
    lut[k * 3] = color.r
    lut[k * 3 + 1] = color.g
    lut[k * 3 + 2] = color.b
  }
  luts.set(mode, lut)
  return lut
}

// Writes linear RGB per particle into `out` (stride 3).
export function writeColors(mode: ColorMode, s: SimState, p: Params, out: Float32Array) {
  const lut = lutFor(mode)
  const [lo, hi] = modeRange(mode, s.dim)
  const scale = (LUT_SIZE - 1) / (hi - lo)
  for (let i = 0; i < s.n; i++) {
    let k = Math.round((mode.scalar(s, p, i) - lo) * scale)
    if (!(k >= 0)) k = 0 // also catches NaN
    else if (k > LUT_SIZE - 1) k = LUT_SIZE - 1
    out[i * 3] = lut[k * 3]
    out[i * 3 + 1] = lut[k * 3 + 1]
    out[i * 3 + 2] = lut[k * 3 + 2]
  }
}
