import type { SolidBox } from '../../sim/sph'
import { courseOf, finishX, START_X, type RiverLayout } from './riverScene'

// Per-race randomness: the course's shape and what's in the river. Generated when
// a race starts (Math.random: races are meant to differ, not to be replayed).

export interface Swirl {
  x: number
  z: number
  radius: number
  strength: number // peak tangential speed; the sign sets the direction
}

export type PickupKind = 'flip' | 'boost' | 'zeroG'

export interface Pickup {
  kind: PickupKind
  pos: number[]
  taken: boolean
}

// The panel's race dials.
export const raceSettings = {
  newCourse: true, // draw a fresh course for each race
  rocks: 3,
  swirls: 2,
  pickups: 3,
}

// What the current race has (filled by generateFeatures).
export const features = {
  rocks: [] as SolidBox[],
  swirls: [] as Swirl[],
  pickups: [] as Pickup[],
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo)
const pick = <T>(xs: readonly T[]) => xs[Math.floor(Math.random() * xs.length)]

// Water surface and bed heights at x (the upper reach sits on the ledge).
export function bedAt(l: RiverLayout, x: number) {
  return x < l.ledgeLength ? l.ledgeHeight : 0
}
export function waterAt(l: RiverLayout, x: number) {
  return x < l.ledgeLength ? l.ledgeHeight + l.upperDepth : l.lowerDepth
}

// A fresh course shape within ranges that race well; width, depths and particle
// size (the quality dial) stay as set.
export function randomizeCourse(l: RiverLayout) {
  l.length = pick([50, 55, 60, 65, 70, 75, 80])
  l.bends = pick([0.5, 1, 1.5, 2, 2.5])
  l.bendSize = Math.round(rand(2, 6) * 2) / 2
  l.ledgeHeight = Math.round(rand(3, 8) * 2) / 2
  l.ledgeLength = Math.round(rand(0.25, 0.4) * l.length)
}

// Where things may go along the course: past the start line, before the finish,
// and not on the waterfall's lip or its splash zone.
function freeX(l: RiverLayout, margin: number): number | null {
  for (let tries = 0; tries < 50; tries++) {
    const x = rand(START_X + 6, finishX(l) - margin)
    if (x > l.ledgeLength - 3 && x < l.ledgeLength + 6) continue
    return x
  }
  return null
}

// Rocks, swirls and pickups for a race on layout `l`.
export function generateFeatures(l: RiverLayout) {
  const c = courseOf(l)
  features.rocks = []
  for (let tries = 0; tries < 200 && features.rocks.length < raceSettings.rocks; tries++) {
    const x = freeX(l, 4)
    if (x == null) break
    const sx = rand(1.2, 2.4)
    const sz = rand(1.2, 2.4)
    const maxOffset = c.halfWidth - sz / 2 - 2.2 // always leave a passage beside it
    if (maxOffset <= 0) continue
    const z = c.centerline(x) + rand(-maxOffset, maxOffset)
    const tooClose = features.rocks.some((r) => Math.hypot((r.min[0] + r.max[0]) / 2 - x, (r.min[2] + r.max[2]) / 2 - z) < 5)
    if (tooClose) continue
    features.rocks.push({ min: [x - sx / 2, bedAt(l, x), z - sz / 2], max: [x + sx / 2, waterAt(l, x) + 1.2, z + sz / 2] })
  }
  features.swirls = []
  features.pickups = []
}
