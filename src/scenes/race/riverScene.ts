import { defaultParams3D, particleRadius, type Params } from '../../sim/params'
import { boundaryWeightsFor, type SolidBox } from '../../sim/sph'
import type { GpuScene } from '../../gpu/sphGpu'
import {
  DS, FLIP_ZONE_GRAVITY, LID_ABOVE_WATER, bedAt, courseLength, locate, type CourseGeometry, type Piece,
} from './course'

// The race course on the GPU solver: the course's pieces (course.ts) filled with
// water, with boundary particles for the bed, the waterfalls' lips, the banks,
// the ends and the flip zones' lids. Water reaching the end is recycled to the
// start, lifted by the course's total drop.

// The course's dials (the panel edits these; they apply to the next course).
export interface RiverLayout {
  spacing: number // particle spacing; h = 2·spacing, dt follows h
  width: number
  depth: number // of the water over the bed
  pieces: Piece[] // between the start and finish straights
}

export const defaultRiverLayout: Readonly<RiverLayout> = {
  spacing: 0.5,
  width: 16, // room for 8 ducks on the start line and for swirls to show
  depth: 2.5,
  pieces: [
    { kind: 'bend', angle: 45, radius: 20 },
    { kind: 'straight', length: 6 },
    { kind: 'waterfall', drop: 5 },
    { kind: 'straight', length: 10 },
    { kind: 'bend', angle: -180, radius: 11 },
    { kind: 'straight', length: 6 },
    { kind: 'flip', length: 12 },
    { kind: 'straight', length: 8 },
    { kind: 'waterfall', drop: 4 },
    { kind: 'straight', length: 10 },
  ],
}

export const riverLayout: RiverLayout = structuredClone(defaultRiverLayout) as RiverLayout

export const DEFAULT_CURRENT = { speed: 3, rate: 1.5 }
const EXIT_MARGIN = 3 // water within this of the end is recycled

// The start line's distance along the course.
export const START_S = 4

// The finish line's s: clear of the recycling zone (where water is lifted away).
export function finishS(g: CourseGeometry): number {
  return g.length - EXIT_MARGIN - 3
}

export function riverParams(g: CourseGeometry, spacing: number): Params {
  const h = 2 * spacing
  return {
    ...structuredClone(defaultParams3D),
    spacing,
    h,
    dt: 0.004 * h, // CFL: dt ≤ 0.4·h/c_s scales with h (0.004 at h = 1)
    domain: g.domain,
  }
}

const insideBox = (b: SolidBox, x: number, y: number, z: number) =>
  x > b.min[0] && x < b.max[0] && y > b.min[1] && y < b.max[1] && z > b.min[2] && z < b.max[2]

// Particles a layout will have, roughly (for the panel): channel area × depth.
export function estimateParticles(l: RiverLayout): number {
  const length = courseLength(l.pieces, l.width / 2)
  return Math.round((length * (l.width - l.spacing) * l.depth) / l.spacing ** 3)
}

// Calls `f(x, z, s, d)` for every point of the rest-spacing lattice over the
// ground (offset half a spacing).
function eachColumn(p: Params, g: CourseGeometry, f: (x: number, z: number, s: number, d: number) => void) {
  const sp = p.spacing
  for (let z = sp / 2; z < g.domain.max[2]; z += sp) {
    for (let x = sp / 2; x < g.domain.max[0]; x += sp) {
      const w = locate(g, x, z)
      f(x, z, w.s, w.d)
    }
  }
}

export function riverScene(p: Params, g: CourseGeometry, rocks: SolidBox[]): GpuScene {
  const sp = p.spacing
  const r = particleRadius(p)
  const layers = Math.ceil(p.h / sp)
  const thick = layers * sp
  const hw = g.halfWidth
  const S = g.length

  // Water: the channel from the bed up to the resting depth, clear of rocks.
  const fluid: number[] = []
  eachColumn(p, g, (x, z, s, d) => {
    if (Math.abs(d) > hw - r || s < r || s > S - r) return
    const bed = bedAt(g, s)
    for (let y = bed + r; y <= bed + g.depth - r + 1e-6; y += sp) {
      if (!rocks.some((b) => insideBox(b, x, y, z))) fluid.push(x, y, z)
    }
  })

  // Boundary particles, `layers` deep behind every surface the water can touch.
  const wall: number[] = []
  const bankTop = (s: number) => bedAt(g, Math.max(0, s - 10)) + g.depth + LID_ABOVE_WATER + 1.5
  eachColumn(p, g, (x, z, s, d) => {
    const bed = bedAt(g, Math.min(S, Math.max(0, s)))
    if (Math.abs(d) <= hw && s >= 0 && s <= S) {
      // The bed, and the solid behind a lip (down to the bed below it).
      const bottom = bedAt(g, Math.min(S, s + thick)) - thick
      for (let y = bed - sp / 2; y > bottom; y -= sp) wall.push(x, y, z)
      // A flip zone's lid.
      const zone = g.zones.find((zn) => s >= zn.s0 && s <= zn.s1)
      if (zone) {
        const lid = bed + g.depth + LID_ABOVE_WATER
        for (let k = 0; k < layers; k++) wall.push(x, lid + (k + 0.5) * sp, z)
      }
    } else if (Math.abs(d) <= hw + thick && (s < 0 ? s > -thick : s > S && s < S + thick)) {
      // The ends: a wall across the channel.
      for (let y = bed - thick + sp / 2; y < bankTop(Math.min(S, Math.max(0, s))); y += sp) wall.push(x, y, z)
    }
  })
  // Banks: `layers` rows outward from each bank, sampled every spacing along the
  // bank's own arc length (sampling by s would bunch them up on the inside of
  // bends), from below the bed to well above the water.
  for (const side of [-1, 1]) {
    for (let layer = 0; layer < layers; layer++) {
      const offset = side * (hw + (layer + 0.5) * sp)
      let travelled = sp / 2
      let prev: number[] | null = null
      for (let k = 0; k < g.n; k++) {
        const s = k * DS
        const q = [g.samples[4 * k] - g.samples[4 * k + 3] * offset, g.samples[4 * k + 1] + g.samples[4 * k + 2] * offset]
        if (prev) travelled += Math.hypot(q[0] - prev[0], q[1] - prev[1])
        prev = q
        if (travelled < sp) continue
        travelled -= sp
        const bottom = bedAt(g, Math.min(S, s + thick)) - thick
        for (let y = bottom + sp / 2; y < bankTop(s); y += sp) wall.push(q[0], y, q[1])
      }
    }
  }
  // Rocks: their sides and top, above the bed.
  for (const b of rocks) {
    const floor = b.min[1] > 0 ? b.min[1] : bedAt(g, locate(g, (b.min[0] + b.max[0]) / 2, (b.min[2] + b.max[2]) / 2).s)
    for (let x = b.min[0] + sp / 2; x < b.max[0]; x += sp) {
      for (let z = b.min[2] + sp / 2; z < b.max[2]; z += sp) {
        for (let y = floor - thick + sp / 2; y < b.max[1]; y += sp) {
          const edge = Math.min(x - b.min[0], b.max[0] - x, z - b.min[2], b.max[2] - z, b.max[1] - y)
          if (edge < thick) wall.push(x, y, z)
        }
      }
    }
  }

  const psi = boundaryWeightsFor(p, wall, [sp, sp, sp])
  return {
    fluid: Float32Array.from(fluid),
    boundary: { pos: Float32Array.from(wall), psi },
    boxes: rocks,
    current: { ...DEFAULT_CURRENT },
    course: {
      halfWidth: hw,
      ds: DS,
      samples: g.samples,
      lookup: g.lookup,
      top: g.top,
      falls: g.falls,
      zones: g.zones.map((z) => ({ ...z, gravity: FLIP_ZONE_GRAVITY, lid: g.depth + LID_ABOVE_WATER })),
    },
    recycle: { exitS: S - EXIT_MARGIN, inletS: 2 * r, lift: g.top },
  }
}
