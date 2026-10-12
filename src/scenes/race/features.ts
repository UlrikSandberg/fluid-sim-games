import type { SolidBox } from '../../sim/sph'
import { MAX_BOXES, MAX_SWIRLS } from '../../gpu/sphShaders'
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

// Course dials that can be drawn at random each race (the panel's 🎲 toggles).
export type RandomDial = 'length' | 'width' | 'bends' | 'bendSize' | 'ledgeHeight'

// The panel's race dials. Rocks, swirls and power-ups are per 50 units of river,
// so a longer course gets more of them at the same density.
export const raceSettings = {
  random: { length: true, width: false, bends: true, bendSize: true, ledgeHeight: true } as Record<RandomDial, boolean>,
  rocks: 2,
  swirls: 1.5,
  pickups: 2.5,
}

// How many of something a course gets, at `perFifty` per 50 units of its length,
// up to what the GPU holds (rocks share the solid-box slots with the ledge).
export function countFor(perFifty: number, l: RiverLayout, kind?: 'rocks' | 'swirls' | 'pickups'): number {
  const n = Math.round((perFifty * l.length) / 50)
  if (kind === 'rocks') return Math.min(n, MAX_BOXES - 1)
  if (kind === 'swirls') return Math.min(n, MAX_SWIRLS)
  return n
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

// Draws the course dials marked random (within ranges that race well); the rest
// keep their panel values. Depths and particle size (the quality dial) never change.
export function randomizeCourse(l: RiverLayout) {
  const r = raceSettings.random
  if (r.length) l.length = pick([60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110])
  if (r.width) l.width = pick([12, 13, 14, 15, 16, 17, 18])
  if (r.bends) l.bends = pick([0.5, 1, 1.5, 2, 2.5])
  if (r.bendSize) l.bendSize = Math.round(rand(2, 6) * 2) / 2
  if (r.ledgeHeight) l.ledgeHeight = Math.round(rand(3, 8) * 2) / 2
  // Where the waterfall is: a third of the way down, give or take.
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
  const rocks = countFor(raceSettings.rocks, l, 'rocks')
  for (let tries = 0; tries < 300 && features.rocks.length < rocks; tries++) {
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
  const swirls = countFor(raceSettings.swirls, l, 'swirls')
  for (let tries = 0; tries < 300 && features.swirls.length < swirls; tries++) {
    const x = freeX(l, 6)
    if (x == null) break
    const radius = Math.min(rand(4, 6), c.halfWidth - 0.5)
    const z = c.centerline(x) + rand(-1, 1) * Math.max(0, c.halfWidth - radius)
    const clear =
      features.rocks.every((r) => Math.hypot((r.min[0] + r.max[0]) / 2 - x, (r.min[2] + r.max[2]) / 2 - z) > radius + 1.5) &&
      features.swirls.every((w) => Math.hypot(w.x - x, w.z - z) > w.radius + radius)
    if (!clear) continue
    features.swirls.push({ x, z, radius, strength: (Math.random() < 0.5 ? -1 : 1) * rand(4, 7) })
  }
  features.pickups = []
  const pickups = countFor(raceSettings.pickups, l)
  for (let tries = 0; tries < 300 && features.pickups.length < pickups; tries++) {
    const x = freeX(l, 5)
    if (x == null) break
    const z = c.centerline(x) + rand(-1, 1) * (c.halfWidth - 1.5)
    const clear =
      features.rocks.every((r) => Math.hypot((r.min[0] + r.max[0]) / 2 - x, (r.min[2] + r.max[2]) / 2 - z) > 3) &&
      features.pickups.every((q) => Math.hypot(q.pos[0] - x, q.pos[2] - z) > 4)
    if (!clear) continue
    const kind = pick<PickupKind>(['flip', 'boost', 'zeroG'])
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
