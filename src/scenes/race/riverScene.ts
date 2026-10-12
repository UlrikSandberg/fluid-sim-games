import { defaultParams3D, particleRadius, type Params } from '../../sim/params'
import { boundaryWeightsFor, staticBoundary, type SolidBox } from '../../sim/sph'
import type { GpuScene } from '../../gpu/sphGpu'
import { features } from './features'

// The race course: a channel along +x that winds sideways (a sine centreline)
// and drops over a waterfall. A solid ledge fills the upstream part, so water
// runs along the upper reach, pours over the ledge's edge and continues along the
// lower reach; at the end it is recycled to the inlet on top of the ledge.

// The course's dials (the panel edits these; they apply on reset).
export interface RiverLayout {
  spacing: number // particle spacing; h = 2·spacing, dt follows h
  length: number
  width: number
  bends: number // full sine periods along the course
  bendSize: number // sideways swing of the centreline (amplitude)
  ledgeLength: number // the waterfall's lip is at x = ledgeLength
  ledgeHeight: number // the waterfall's drop
  upperDepth: number
  lowerDepth: number
}

export const defaultRiverLayout: Readonly<RiverLayout> = {
  spacing: 0.5,
  length: 60,
  width: 12, // fits 8 default ducks side by side on the start line
  bends: 1,
  bendSize: 4,
  ledgeLength: 20,
  ledgeHeight: 6,
  upperDepth: 2.5,
  lowerDepth: 3,
}

export const riverLayout: RiverLayout = structuredClone(defaultRiverLayout)

export const DEFAULT_CURRENT = { speed: 3, rate: 1.5 }
const EXIT_MARGIN = 3 // water within this of the far end is recycled

// The start line's position along the course.
export const START_X = 4

// The finish line's x: clear of the recycling zone (where water is lifted away).
export function finishX(l: RiverLayout = riverLayout): number {
  return l.length - EXIT_MARGIN - 3
}
const SIDE_MARGIN = 1 // ground between the outermost bank and the domain's side

// Geometry derived from a layout: domain, centreline, banks.
export function courseOf(l: RiverLayout) {
  const halfWidth = l.width / 2
  const amplitude = l.bends > 0 ? l.bendSize : 0
  const centerZ = SIDE_MARGIN + amplitude + halfWidth + 1
  const height = l.ledgeHeight + l.upperDepth + 6
  const wavenumber = l.bends > 0 ? (2 * Math.PI * l.bends) / l.length : 0
  return {
    domain: { min: [0, 0, 0], max: [l.length, height, 2 * centerZ] },
    halfWidth,
    amplitude,
    wavenumber,
    centerZ,
    centerline: (x: number) => centerZ + amplitude * Math.sin(wavenumber * x),
    slope: (x: number) => amplitude * wavenumber * Math.cos(wavenumber * x),
  }
}

export function riverParams(l: RiverLayout = riverLayout): Params {
  const h = 2 * l.spacing
  return {
    ...structuredClone(defaultParams3D),
    spacing: l.spacing,
    h,
    dt: 0.004 * h, // CFL: dt ≤ 0.4·h/c_s scales with h (0.004 at h = 1)
    domain: courseOf(l).domain,
  }
}

// The solids in the river: the waterfall's ledge (none when the drop is under a
// particle spacing) and this race's rocks.
export function riverBoxes(l: RiverLayout = riverLayout): SolidBox[] {
  const { domain } = courseOf(l)
  const ledge: SolidBox[] = l.ledgeHeight < l.spacing ? [] : [{ min: [0, 0, 0], max: [l.ledgeLength, l.ledgeHeight, domain.max[2]] }]
  return [...ledge, ...features.rocks]
}

// Boundary particles for both banks: `layers` rows outward from each bank,
// sampled at the rest spacing along the bank's arc length (sampling by x would
// space them out where the channel bends), from the floor to the top.
function bankParticles(p: Params, l: RiverLayout): { pos: number[]; spacing: number[] } {
  const c = courseOf(l)
  const sp = p.spacing
  const layers = Math.ceil(p.h / sp)
  const pos: number[] = []
  const top = c.domain.max[1]
  for (const side of [-1, 1]) {
    for (let layer = 0; layer < layers; layer++) {
      const offset = c.halfWidth + (layer + 0.5) * sp
      // Walk the offset curve finely, emitting a column every `sp` of arc length.
      const point = (x: number) => {
        const s = c.slope(x)
        const norm = Math.hypot(s, 1)
        return [x - (side * offset * s) / norm, c.centerline(x) + (side * offset) / norm]
      }
      let prev = point(0)
      let travelled = sp / 2
      for (let x = 0; x <= l.length; x += sp / 20) {
        const q = point(x)
        travelled += Math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
        if (travelled < sp) continue
        travelled -= sp
        if (q[0] < 0 || q[0] > l.length) continue
        for (let y = sp / 2; y < top; y += sp) pos.push(q[0], y, q[1])
      }
    }
  }
  return { pos, spacing: [sp, sp, sp] }
}

// Rest-spacing lattice inside the channel over [x0, x1] × [y0, y1], inset by the
// particle radius from everything.
function fill(p: Params, l: RiverLayout, x0: number, x1: number, y0: number, y1: number, out: number[]) {
  const c = courseOf(l)
  const r = particleRadius(p)
  for (let x = x0 + r; x <= x1 - r + 1e-6; x += p.spacing) {
    const zc = c.centerline(x)
    const limit = (c.halfWidth - r) / Math.hypot(1, c.slope(x)) // perpendicular → z extent
    for (let y = y0 + r; y <= y1 - r + 1e-6; y += p.spacing) {
      for (let z = zc - limit; z <= zc + limit + 1e-6; z += p.spacing) out.push(x, y, z)
    }
  }
}

// Particles a layout will have (for the panel): a quick count of `fill`.
export function estimateParticles(l: RiverLayout): number {
  const p = riverParams(l)
  const out: number[] = []
  fill(p, l, 0, l.ledgeLength, l.ledgeHeight, l.ledgeHeight + l.upperDepth, out)
  fill(p, l, l.ledgeLength, l.length, 0, l.lowerDepth, out)
  return out.length / 3
}

export function riverScene(p: Params, l: RiverLayout = riverLayout): GpuScene {
  const c = courseOf(l)
  const fluid: number[] = []
  fill(p, l, 0, l.ledgeLength, l.ledgeHeight, l.ledgeHeight + l.upperDepth, fluid)
  fill(p, l, l.ledgeLength, l.length, 0, l.lowerDepth, fluid)
  const boxes = riverBoxes(l)
  const walls = staticBoundary(p, boxes)
  const banks = bankParticles(p, l)
  const bankPsi = boundaryWeightsFor(p, banks.pos, banks.spacing)
  const boundary = {
    pos: new Float32Array(walls.pos.length + banks.pos.length),
    psi: new Float32Array(walls.psi.length + bankPsi.length),
  }
  boundary.pos.set(walls.pos)
  boundary.pos.set(banks.pos, walls.pos.length)
  boundary.psi.set(walls.psi)
  boundary.psi.set(bankPsi, walls.psi.length)
  return {
    fluid: Float32Array.from(fluid),
    boundary,
    boxes,
    current: { ...DEFAULT_CURRENT },
    recycle: { exitX: l.length - EXIT_MARGIN, inletX: particleRadius(p) * 2, lift: l.ledgeHeight },
    channel: { amplitude: c.amplitude, wavenumber: c.wavenumber, halfWidth: c.halfWidth, centerZ: c.centerZ },
    swirls: features.swirls,
  }
}

// The banks as polylines (x, z) for rendering: [left, right].
export function bankLines(l: RiverLayout = riverLayout): number[][][] {
  const c = courseOf(l)
  return [-1, 1].map((side) => {
    const line: number[][] = []
    for (let x = 0; x <= l.length + 1e-6; x += 0.5) {
      const s = c.slope(x)
      const norm = Math.hypot(s, 1)
      line.push([x - (side * c.halfWidth * s) / norm, c.centerline(x) + (side * c.halfWidth) / norm])
    }
    return line
  })
}
