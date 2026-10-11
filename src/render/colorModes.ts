import * as THREE from 'three'
import type { Params } from '../sim/params'
import type { SimState } from '../sim/sph'

export type ColorModeId = 'density'

export interface ColorMode {
  id: ColorModeId
  label: string
  range: [number, number] // values outside are clamped to the end colours
  stops: string[] // sRGB, low → high; also used for the CSS legend gradient
  scalar: (s: SimState, p: Params, i: number) => number
}

// Diverging blue–grey–red (Moreland "cool to warm"), for quantities centred on a
// reference value.
const COOLWARM = ['#3b4cc0', '#8db0fe', '#dddddd', '#f49a7b', '#b40426']

export const colorModes: ColorMode[] = [
  {
    id: 'density',
    label: 'Density ρ/ρ₀',
    range: [0.9, 1.1],
    stops: COOLWARM,
    scalar: (s, p, i) => s.density[i] / p.restDensity,
  },
]

// Mutable view settings, read by the loop each frame (like `params`).
export const view = { colorMode: 'density' as ColorModeId }

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
  const [lo, hi] = mode.range
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
