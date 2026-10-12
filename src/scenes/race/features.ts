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
  for (let tries = 0; tries < 200 && features.swirls.length < raceSettings.swirls; tries++) {
    const x = freeX(l, 6)
    if (x == null) break
    const radius = Math.min(rand(2.5, 4), c.halfWidth - 0.5)
    const z = c.centerline(x) + rand(-1, 1) * Math.max(0, c.halfWidth - radius)
    const clear =
      features.rocks.every((r) => Math.hypot((r.min[0] + r.max[0]) / 2 - x, (r.min[2] + r.max[2]) / 2 - z) > radius + 1.5) &&
      features.swirls.every((w) => Math.hypot(w.x - x, w.z - z) > w.radius + radius)
    if (!clear) continue
    features.swirls.push({ x, z, radius, strength: (Math.random() < 0.5 ? -1 : 1) * rand(4, 7) })
  }
  features.pickups = []
  let zeroG = 0
  for (let tries = 0; tries < 200 && features.pickups.length < raceSettings.pickups; tries++) {
    const x = freeX(l, 5)
    if (x == null) break
    const z = c.centerline(x) + rand(-1, 1) * (c.halfWidth - 1.5)
    const clear =
      features.rocks.every((r) => Math.hypot((r.min[0] + r.max[0]) / 2 - x, (r.min[2] + r.max[2]) / 2 - z) > 3) &&
      features.pickups.every((q) => Math.hypot(q.pos[0] - x, q.pos[2] - z) > 4)
    if (!clear) continue
    // At most one Zero-G per race: it changes everything for everyone.
    const kinds: PickupKind[] = zeroG > 0 ? ['flip', 'boost'] : ['flip', 'boost', 'zeroG']
    const kind = pick(kinds)
    if (kind === 'zeroG') zeroG++
    features.pickups.push({ kind, pos: [x, waterAt(l, x) + 0.9, z], taken: false })
  }
}

export const PICKUP_INFO: Record<PickupKind, { label: string; color: number }> = {
  flip: { label: '⬆ Flip', color: 0xb07cff },
  boost: { label: '⚡ Boost', color: 0x7ee787 },
  zeroG: { label: '🌌 Zero-G', color: 0x54d6ff },
}

// How fast (1/s) a duck inside a swirl turns toward the swirl's rotation: slower
// than the water (SWIRL_RATE in the shader), it's also carried by the water.
const DUCK_SWIRL_RATE = 1.2

// The swirls' direct pull on a duck for one step of `dt` (the same profile as the
// water's, see sphShaders.ts), so a duck caught in one visibly circles.
export function swirlOnDuck(center: number[], vel: number[], dt: number) {
  for (const w of features.swirls) {
    const rx = center[0] - w.x
    const rz = center[2] - w.z
    const dist = Math.hypot(rx, rz)
    if (dist >= w.radius || dist < 1e-3) continue
    const around = [-rz / dist, rx / dist]
    const u = dist / w.radius
    const spin = w.strength * 4 * u * (1 - u)
    const along = vel[0] * around[0] + vel[2] * around[1]
    vel[0] += DUCK_SWIRL_RATE * (spin - along) * around[0] * dt
    vel[2] += DUCK_SWIRL_RATE * (spin - along) * around[1] * dt
  }
}
