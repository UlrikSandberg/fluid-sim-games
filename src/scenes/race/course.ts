import { MAX_FALLS, MAX_ZONES } from '../../gpu/sphShaders'

// The race course as a chain of pieces laid end to end: straights, bends (a
// U-turn is a 180° bend), waterfalls (a step down in the bed) and flip zones (a
// stretch where gravity points up, under a glass lid). A fixed start straight
// comes first and a fixed finish straight last.
//
// buildCourse turns the pieces into a centreline sampled every DS along its arc
// length s, plus a lookup grid over the ground that gives every point its nearest
// sample, so "how far down the course, how far off the centreline" is two reads
// for the GPU (see locate in sphShaders.ts) and the CPU (locate below).

export type Piece =
  | { kind: 'straight'; length: number }
  | { kind: 'bend'; angle: number; radius: number } // degrees, + turns toward +z; radius of the centreline
  | { kind: 'waterfall'; drop: number }
  | { kind: 'flip'; length: number }

export type PieceKind = Piece['kind']

export const START_LENGTH = 12 // the start straight (the gate stands on it)
export const FINISH_LENGTH = 12 // the finish straight (banner, then the outlet)
export const DS = 0.25 // centreline sample spacing
const LOOKUP_CELL = 0.5
const MARGIN = 2 // ground beyond the banks' boundary particles, to the domain's sides
export const LID_ABOVE_WATER = 3.5 // a flip zone's lid, above the resting water
export const FLIP_ZONE_GRAVITY = -0.6 // gravity in a flip zone, as a multiple of normal
const HEADROOM = 6 // domain height above the highest water

// The tightest a bend may be: the banks' boundary particles of the two legs of a
// U-turn must stay clear of each other's water.
export const minRadius = (halfWidth: number) => halfWidth + 3

// A piece's length along the centreline.
export function pieceLength(p: Piece, halfWidth: number): number {
  if (p.kind === 'straight' || p.kind === 'flip') return p.length
  if (p.kind === 'bend') return Math.max(p.radius, minRadius(halfWidth)) * Math.abs((p.angle * Math.PI) / 180)
  return 0
}

export function courseLength(pieces: Piece[], halfWidth: number): number {
  return START_LENGTH + FINISH_LENGTH + pieces.reduce((a, p) => a + pieceLength(p, halfWidth), 0)
}

export interface Fall {
  s: number // the lip
  drop: number
}

export interface Zone {
  s0: number
  s1: number
}

export interface CourseGeometry {
  halfWidth: number
  depth: number
  length: number // S, the centreline's length
  n: number // samples, at s = k·DS
  samples: Float32Array // per sample: x, z, tangent x, tangent z (domain coordinates)
  falls: Fall[]
  zones: Zone[]
  top: number // the bed's height at the start (every waterfall steps it down)
  domain: { min: number[]; max: number[] }
  lookup: { cell: number; nx: number; nz: number; index: Uint32Array } // nearest sample per ground cell
  problem: string | null // why the course can't be raced (pieces overlap, too many waterfalls…)
}

export function bedAt(g: CourseGeometry, s: number): number {
  let bed = g.top
  for (const f of g.falls) if (s >= f.s) bed -= f.drop
  return bed
}

export const waterAt = (g: CourseGeometry, s: number) => bedAt(g, s) + g.depth

export function zoneAt(g: CourseGeometry, s: number): Zone | null {
  return g.zones.find((z) => s >= z.s0 && s <= z.s1) ?? null
}

export const lidAt = (g: CourseGeometry, s: number) => waterAt(g, s) + LID_ABOVE_WATER

// Where (x, z) is relative to the course: s along it (negative before the start,
// past `length` after the end), d across it (+ toward the left normal (−tz, tx)),
// and the tangent there.
export function locate(g: CourseGeometry, x: number, z: number) {
  const { cell, nx, nz, index } = g.lookup
  const cx = Math.min(nx - 1, Math.max(0, Math.floor(x / cell)))
  const cz = Math.min(nz - 1, Math.max(0, Math.floor(z / cell)))
  const k = index[cx + cz * nx]
  const sx = g.samples[4 * k]
  const sz = g.samples[4 * k + 1]
  const tx = g.samples[4 * k + 2]
  const tz = g.samples[4 * k + 3]
  const rx = x - sx
  const rz = z - sz
  return { s: k * DS + rx * tx + rz * tz, d: -rx * tz + rz * tx, tx, tz }
}

// The point at s along the course, d across it: [x, z], and the tangent there.
export function pointAt(g: CourseGeometry, s: number, d = 0) {
  const k = Math.min(g.n - 1, Math.max(0, Math.round(s / DS)))
  const tx = g.samples[4 * k + 2]
  const tz = g.samples[4 * k + 3]
  const along = s - k * DS
  return {
    x: g.samples[4 * k] + tx * along - tz * d,
    z: g.samples[4 * k + 1] + tz * along + tx * d,
    tx,
    tz,
  }
}

// The centreline before shifting into the domain: samples every DS, the falls and
// zones along it.
function walk(pieces: Piece[], halfWidth: number) {
  const pts: number[] = []
  const falls: Fall[] = []
  const zones: Zone[] = []
  let x = 0
  let z = 0
  let heading = 0
  let s = 0
  const emit = () => pts.push(x, z, Math.cos(heading), Math.sin(heading))
  emit()
  const straight = (length: number) => {
    const steps = Math.max(1, Math.round(length / DS))
    for (let i = 0; i < steps; i++) {
      x += DS * Math.cos(heading)
      z += DS * Math.sin(heading)
      s += DS
      emit()
    }
  }
  const all: Piece[] = [{ kind: 'straight', length: START_LENGTH }, ...pieces, { kind: 'straight', length: FINISH_LENGTH }]
  for (const p of all) {
    if (p.kind === 'straight') straight(p.length)
    else if (p.kind === 'flip') {
      const s0 = s
      straight(p.length)
      zones.push({ s0, s1: s })
    } else if (p.kind === 'waterfall') {
      if (p.drop > 0) falls.push({ s, drop: p.drop })
    } else {
      const radius = Math.max(p.radius, minRadius(halfWidth))
      const turn = (p.angle * Math.PI) / 180
      const steps = Math.max(1, Math.round((radius * Math.abs(turn)) / DS))
      const dTheta = turn / steps
      // Each step is a chord DS long (so samples stay DS apart along the curve).
      for (let i = 0; i < steps; i++) {
        const mid = heading + dTheta / 2
        x += DS * Math.cos(mid)
        z += DS * Math.sin(mid)
        heading += dTheta
        s += DS
        emit()
      }
    }
  }
  return { pts, falls, zones, length: s }
}

// Two stretches of the river closer than this (centre to centre) would share
// banks; points this far apart along the course are allowed to come that close
// only in a single bend.
const clearance = (halfWidth: number) => 2 * halfWidth + 3

// Whether the course runs into itself: any two centreline points more than half
// a tight U-turn's circumference apart along it but closer than the clearance.
function overlaps(pts: number[], halfWidth: number): boolean {
  const n = pts.length / 4
  const step = Math.max(1, Math.round(1 / DS)) // check every unit
  const gap = Math.PI * minRadius(halfWidth) + 1
  const need = clearance(halfWidth) - 0.01
  for (let i = 0; i < n; i += step) {
    for (let j = i + Math.ceil(gap / DS); j < n; j += step) {
      if (Math.hypot(pts[4 * i] - pts[4 * j], pts[4 * i + 1] - pts[4 * j + 1]) < need) return true
    }
  }
  return false
}

// Whether a list of pieces makes a raceable course (without building it).
export function piecesOverlap(pieces: Piece[], width: number): boolean {
  return overlaps(walk(pieces, width / 2).pts, width / 2)
}

export function buildCourse(pieces: Piece[], width: number, depth: number): CourseGeometry {
  const halfWidth = width / 2
  const { pts, falls, zones, length } = walk(pieces, halfWidth)
  const n = pts.length / 4
  // Shift into the domain: the centreline's bounding box grown by the half width
  // and the banks.
  let minX = Infinity
  let minZ = Infinity
  let maxX = -Infinity
  let maxZ = -Infinity
  for (let k = 0; k < n; k++) {
    minX = Math.min(minX, pts[4 * k])
    maxX = Math.max(maxX, pts[4 * k])
    minZ = Math.min(minZ, pts[4 * k + 1])
    maxZ = Math.max(maxZ, pts[4 * k + 1])
  }
  // Room beside the banks for the banks' particles, and for a gate (up to 36%
  // of the width plus a margin, see movers.ts) to slide back into.
  const reach = halfWidth + Math.max(MARGIN + 1, 0.4 * width + 3)
  const samples = Float32Array.from(pts)
  for (let k = 0; k < n; k++) {
    samples[4 * k] += reach - minX
    samples[4 * k + 1] += reach - minZ
  }
  const top = falls.reduce((a, f) => a + f.drop, 0)
  const domain = {
    min: [0, 0, 0],
    max: [maxX - minX + 2 * reach, top + depth + HEADROOM, maxZ - minZ + 2 * reach],
  }

  // Nearest sample for every ground cell: each sample claims the cells around it
  // (enough for the channel and the banks); far cells (open ground) fall back to a
  // coarse search.
  const nx = Math.ceil(domain.max[0] / LOOKUP_CELL)
  const nz = Math.ceil(domain.max[2] / LOOKUP_CELL)
  const index = new Uint32Array(nx * nz)
  const best = new Float32Array(nx * nz).fill(Infinity)
  const claim = reach + 1
  const span = Math.ceil(claim / LOOKUP_CELL)
  for (let k = 0; k < n; k++) {
    const sx = samples[4 * k]
    const sz = samples[4 * k + 1]
    const cx0 = Math.floor(sx / LOOKUP_CELL)
    const cz0 = Math.floor(sz / LOOKUP_CELL)
    for (let cz = Math.max(0, cz0 - span); cz <= Math.min(nz - 1, cz0 + span); cz++) {
      for (let cx = Math.max(0, cx0 - span); cx <= Math.min(nx - 1, cx0 + span); cx++) {
        const dx = (cx + 0.5) * LOOKUP_CELL - sx
        const dz = (cz + 0.5) * LOOKUP_CELL - sz
        const d2 = dx * dx + dz * dz
        if (d2 < best[cx + cz * nx]) {
          best[cx + cz * nx] = d2
          index[cx + cz * nx] = k
        }
      }
    }
  }
  const coarse = Math.max(1, Math.round(2 / DS))
  for (let c = 0; c < nx * nz; c++) {
    if (best[c] !== Infinity) continue
    const px = ((c % nx) + 0.5) * LOOKUP_CELL
    const pz = (Math.floor(c / nx) + 0.5) * LOOKUP_CELL
    let near = Infinity
    for (let k = 0; k < n; k += coarse) {
      const d2 = (samples[4 * k] - px) ** 2 + (samples[4 * k + 1] - pz) ** 2
      if (d2 < near) {
        near = d2
        index[c] = k
      }
    }
  }

  let problem: string | null = null
  if (overlaps(pts, halfWidth)) problem = 'the river runs into itself; change a bend or add a straight'
  else if (falls.length > MAX_FALLS) problem = `at most ${MAX_FALLS} waterfalls`
  else if (zones.length > MAX_ZONES) problem = `at most ${MAX_ZONES} flip zones`

  return {
    halfWidth, depth, length, n, samples, falls, zones, top, domain,
    lookup: { cell: LOOKUP_CELL, nx, nz, index },
    problem,
  }
}

// The ground for rendering: a grid of vertices over the domain (spacing `cell`,
// row by row along x) with the river bed in the channel and grass a little above
// the water beside it, and a colour per vertex. Vertices near a bank, a lip or an
// end are moved onto it, so those are sheer walls rather than a staircase.
export function terrain(g: CourseGeometry, cell = 0.5) {
  const nx = Math.ceil(g.domain.max[0] / cell) + 1
  const nz = Math.ceil(g.domain.max[2] / cell) + 1
  const positions = new Float32Array(nx * nz * 3)
  const colors = new Float32Array(nx * nz * 3)
  // Linear RGB.
  const grass = [0.05, 0.1, 0.04]
  const bed = [0.1, 0.11, 0.13]
  const near = 1.5 * cell
  const hw = g.halfWidth
  for (let j = 0; j < nz; j++) {
    for (let i = 0; i < nx; i++) {
      const v = i + j * nx
      let x = i * cell
      let z = j * cell
      let { s, d } = locate(g, x, z)
      const inChannel = Math.abs(d) < hw && s > 0 && s < g.length
      const snap = inChannel ? -0.01 : 0.01 // just inside, or just outside
      let moved = false
      if (Math.abs(d) > hw - near && Math.abs(d) < hw + near && s > -near && s < g.length + near) {
        d = Math.sign(d) * (hw + snap)
        moved = true
      }
      if (Math.abs(d) < hw + near) {
        if (Math.abs(s) < near) [s, moved] = [-snap, true]
        else if (Math.abs(s - g.length) < near) [s, moved] = [g.length + snap, true]
        else if (inChannel) {
          const lip = g.falls.find((f) => Math.abs(s - f.s) < near)
          if (lip) [s, moved] = [s < lip.s ? lip.s - 0.01 : lip.s + 0.01, true]
        }
      }
      if (moved) ({ x, z } = pointAt(g, s, d))
      const clamped = Math.min(g.length, Math.max(0, s))
      positions.set([x, inChannel ? bedAt(g, s) : bedAt(g, clamped) + g.depth + 1, z], v * 3)
      colors.set(inChannel ? bed : grass, v * 3)
    }
  }
  return { nx, nz, positions, colors }
}

// A flip zone's lid for rendering: the outline of the channel along the zone, at
// the lid's height ([left, right] polylines of (x, y, z)).
export function zoneLid(g: CourseGeometry, z: Zone): number[][][] {
  const left: number[][] = []
  const right: number[][] = []
  for (let s = z.s0; s <= z.s1 + 1e-6; s += 0.5) {
    const y = lidAt(g, s)
    const a = pointAt(g, s, g.halfWidth)
    const b = pointAt(g, s, -g.halfWidth)
    left.push([a.x, y, a.z])
    right.push([b.x, y, b.z])
  }
  return [left, right]
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo)
const sign = () => (Math.random() < 0.5 ? -1 : 1)

export interface PieceCounts {
  length: number // target total length
  waterfalls: number
  uTurns: number
  bends: number
  flipZones: number
}

// A random course of about `c.length` with the given numbers of each feature, in
// random order with straights between (longer after a waterfall, for the splash).
// Tries random turn directions until the river doesn't run into itself; drops
// the U-turns, then the bends, if it can't.
export function randomPieces(c: PieceCounts, width: number): Piece[] {
  const halfWidth = width / 2
  for (let attempt = 0; attempt < 120; attempt++) {
    const uTurns = attempt < 80 ? c.uTurns : attempt < 100 ? Math.min(c.uTurns, 1) : 0
    const bends = attempt < 110 ? c.bends : 0
    const items: Piece[] = []
    for (let i = 0; i < c.waterfalls; i++) items.push({ kind: 'waterfall', drop: Math.round(rand(3, 6) * 2) / 2 })
    for (let i = 0; i < uTurns; i++) items.push({ kind: 'bend', angle: sign() * 180, radius: minRadius(halfWidth) })
    for (let i = 0; i < bends; i++) {
      const angle = [30, 45, 60, 90][Math.floor(Math.random() * 4)]
      items.push({ kind: 'bend', angle: sign() * angle, radius: Math.round(rand(minRadius(halfWidth) + 2, minRadius(halfWidth) + 10)) })
    }
    for (let i = 0; i < c.flipZones; i++) items.push({ kind: 'flip', length: Math.round(rand(10, 16)) })
    for (let i = items.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1))
      ;[items[i], items[j]] = [items[j], items[i]]
    }
    const pieces: Piece[] = []
    for (const item of items) {
      pieces.push(item)
      pieces.push({ kind: 'straight', length: item.kind === 'waterfall' ? 10 : 4 })
    }
    if (pieces.length === 0) pieces.push({ kind: 'straight', length: 4 })
    // Stretch the straights to the target length, a unit at a time at random.
    const straights = pieces.filter((p): p is Extract<Piece, { kind: 'straight' }> => p.kind === 'straight')
    let spare = Math.round(c.length - courseLength(pieces, halfWidth))
    while (spare > 0) {
      straights[Math.floor(Math.random() * straights.length)].length += 1
      spare--
    }
    if (!piecesOverlap(pieces, width)) return pieces
  }
  return [{ kind: 'straight', length: Math.max(4, c.length - START_LENGTH - FINISH_LENGTH) }]
}
