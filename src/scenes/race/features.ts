import type { SolidBox } from '../../sim/sph'
import type { Params } from '../../sim/params'
import { MAX_BOXES, MAX_SWIRLS } from '../../gpu/sphShaders'
import { MAX_MOVERS } from '../fluid3d/engines'
import { MOVER_KINDS, makeMover, type Mover, type MoverKind } from './movers'
import { pointAt, randomPieces, waterAt, type CourseGeometry, type PieceCounts } from './course'
import { START_S, finishS, type RiverLayout } from './riverScene'

// Per-race randomness: the course's pieces and what's in the river. Math.random:
// races are meant to differ, not to be replayed.

export interface Swirl {
  x: number
  z: number
  radius: number
  peak: number // peak tangential speed; the sign sets the direction
  strength: number // the peak scaled by how far it has spun up (0 while gone)
  age: number // seconds since it appeared (negative: appears in −age seconds)
  life: number // seconds it lasts
}

export type PickupKind = 'flip' | 'boost' | 'zeroG'

export interface Pickup {
  kind: PickupKind
  pos: number[]
  taken: boolean
}

// Course dials that can be drawn at random for each new course (the panel's 🎲
// toggles). `pieces` off keeps the panel's piece list as it is.
export type RandomDial = 'pieces' | 'width' | keyof PieceCounts

// The panel's race dials. Rocks, swirls and power-ups are per 50 units of river,
// so a longer course gets more of them at the same density.
export const raceSettings = {
  random: {
    pieces: true, width: false, length: true, waterfalls: true, uTurns: true, bends: true, flipZones: true,
  } as Record<RandomDial, boolean>,
  // What a random course is made of (each drawn from its range when its 🎲 is on).
  counts: { length: 100, waterfalls: 2, uTurns: 1, bends: 2, flipZones: 1 } as PieceCounts,
  rocks: 2,
  movers: 1.5, // moving obstacles
  moverKinds: { gate: true, pulse: true, slider: true, spinner: true } as Record<MoverKind, boolean>,
  swirls: 1.5, // at a time
  pickups: 2.5,
}

// The ranges random dials are drawn from.
export const RANDOM_RANGES: Record<keyof PieceCounts | 'width', number[]> = {
  length: [70, 140],
  width: [12, 18],
  waterfalls: [1, 3],
  uTurns: [0, 2],
  bends: [0, 3],
  flipZones: [0, 2],
}

// How many of something a course gets, at `perFifty` per 50 units of its length,
// up to what the GPU holds (rocks share the solid-box slots with nothing else now,
// but keep one spare).
export function countFor(perFifty: number, length: number, kind?: 'rocks' | 'movers' | 'swirls' | 'pickups'): number {
  const n = Math.round((perFifty * length) / 50)
  if (kind === 'rocks') return Math.min(n, MAX_BOXES - 1)
  if (kind === 'movers') return Math.min(n, MAX_MOVERS)
  if (kind === 'swirls') return Math.min(n, MAX_SWIRLS)
  return n
}

// What the current race has (filled by generateFeatures).
export const features = {
  rocks: [] as SolidBox[],
  movers: [] as Mover[],
  swirls: [] as Swirl[],
  pickups: [] as Pickup[],
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo)
const randInt = ([lo, hi]: number[]) => lo + Math.floor(Math.random() * (hi - lo + 1))

// Draws the course dials marked random, then (if pieces are random) a new piece
// list from the counts. Depth and particle size (the quality dial) never change.
export function randomizeCourse(l: RiverLayout) {
  const r = raceSettings.random
  const c = raceSettings.counts
  if (r.width) l.width = randInt(RANDOM_RANGES.width)
  if (!r.pieces) return
  if (r.length) c.length = Math.round(randInt(RANDOM_RANGES.length) / 5) * 5
  if (r.waterfalls) c.waterfalls = randInt(RANDOM_RANGES.waterfalls)
  if (r.uTurns) c.uTurns = randInt(RANDOM_RANGES.uTurns)
  if (r.bends) c.bends = randInt(RANDOM_RANGES.bends)
  if (r.flipZones) c.flipZones = randInt(RANDOM_RANGES.flipZones)
  l.pieces = randomPieces(c, l.width)
}

// Where things may go along the course: past the start line, before the finish,
// and off the waterfalls' lips and splash zones and out of the flip zones.
function freeS(g: CourseGeometry, margin: number): number | null {
  for (let tries = 0; tries < 50; tries++) {
    const s = rand(START_S + 8, finishS(g) - margin)
    if (g.falls.some((f) => s > f.s - 3 - margin / 2 && s < f.s + 6)) continue
    if (g.zones.some((z) => s > z.s0 - margin && s < z.s1 + margin)) continue
    return s
  }
  return null
}

const rockCentre = (r: SolidBox) => [(r.min[0] + r.max[0]) / 2, (r.min[2] + r.max[2]) / 2]

// Whether a circle at (x, z) of radius `radius` keeps `gap` clear of the movers'
// reach (their spot along the course, and how far they get from it).
function clearOfMovers(g: CourseGeometry, x: number, z: number, radius: number, gap: number) {
  return features.movers.every((m) => {
    const c = pointAt(g, m.s, m.kind === 'gate' ? m.d * g.halfWidth : m.d)
    return Math.hypot(c.x - x, c.z - z) > m.radius + radius + gap
  })
}

// Rocks, moving obstacles, a swirl schedule and pickups for a race on course `g`.
export function generateFeatures(g: CourseGeometry, params: Params) {
  const hw = g.halfWidth
  features.rocks = []
  const rocks = countFor(raceSettings.rocks, g.length, 'rocks')
  for (let tries = 0; tries < 300 && features.rocks.length < rocks; tries++) {
    const s = freeS(g, 4)
    if (s == null) break
    const sx = rand(1.2, 2.4)
    const sz = rand(1.2, 2.4)
    const maxOffset = hw - Math.hypot(sx, sz) / 2 - 2.2 // always leave a passage beside it
    if (maxOffset <= 0) continue
    const p = pointAt(g, s, rand(-maxOffset, maxOffset))
    const tooClose = features.rocks.some((r) => Math.hypot(rockCentre(r)[0] - p.x, rockCentre(r)[1] - p.z) < 5)
    if (tooClose) continue
    // Down to the domain's floor, so the push-out never sends water out of the bottom.
    features.rocks.push({ min: [p.x - sx / 2, 0, p.z - sz / 2], max: [p.x + sx / 2, waterAt(g, s) + 1.2, p.z + sz / 2] })
  }
  features.movers = []
  const kinds = MOVER_KINDS.filter((k) => raceSettings.moverKinds[k])
  const movers = kinds.length > 0 ? countFor(raceSettings.movers, g.length, 'movers') : 0
  for (let tries = 0; tries < 300 && features.movers.length < movers; tries++) {
    const s = freeS(g, 5)
    if (s == null) break
    const kind = kinds[Math.floor(Math.random() * kinds.length)]
    const m = makeMover(params, g, kind, s, rand(-1, 1) * (hw - 3))
    if (!m) continue
    const c = pointAt(g, m.s, m.kind === 'gate' ? m.d * hw : m.d)
    const clear =
      features.rocks.every((r) => Math.hypot(rockCentre(r)[0] - c.x, rockCentre(r)[1] - c.z) > m.radius + 2.5) &&
      // Movers keep out of each other's way along the course.
      features.movers.every((o) => Math.abs(o.s - m.s) > 4 + (o.kind === 'gate' ? 1 : o.radius) + (m.kind === 'gate' ? 1 : m.radius))
    if (clear) features.movers.push(m)
  }
  features.swirls = []
  const swirls = countFor(raceSettings.swirls, g.length, 'swirls')
  for (let i = 0; i < swirls; i++) {
    const w: Swirl = { x: 0, z: 0, radius: 0, peak: 0, strength: 0, age: 0, life: 0 }
    respawn(g, w, rand(0, 4))
    features.swirls.push(w)
  }
  features.pickups = []
  const pickups = countFor(raceSettings.pickups, g.length)
  for (let tries = 0; tries < 300 && features.pickups.length < pickups; tries++) {
    const s = freeS(g, 5)
    if (s == null) break
    const p = pointAt(g, s, rand(-1, 1) * (hw - 1.5))
    const clear =
      features.rocks.every((r) => Math.hypot(rockCentre(r)[0] - p.x, rockCentre(r)[1] - p.z) > 3) &&
      clearOfMovers(g, p.x, p.z, 0, 1.5) &&
      features.pickups.every((q) => Math.hypot(q.pos[0] - p.x, q.pos[2] - p.z) > 4)
    if (!clear) continue
    const kind = (['flip', 'boost', 'zeroG'] as const)[Math.floor(Math.random() * 3)]
    features.pickups.push({ kind, pos: [p.x, waterAt(g, s) + 0.9, p.z], taken: false })
  }
}

// Swirls come and go: each lasts a few seconds (spinning up and down), is gone
// for a moment, then turns up somewhere else, so no duck stays stuck in one.
const SWIRL_LIFE = [5, 10]
const SWIRL_GAP = [1, 4]
const SWIRL_FADE = 1.2 // seconds to spin up, and to spin down

// Moves swirl `w` to a new spot, appearing in `delay` seconds.
function respawn(g: CourseGeometry, w: Swirl, delay: number) {
  w.age = -delay
  w.life = rand(SWIRL_LIFE[0], SWIRL_LIFE[1])
  w.strength = 0
  for (let tries = 0; tries < 40; tries++) {
    const s = freeS(g, 6)
    if (s == null) return
    const radius = Math.min(rand(4, 6), g.halfWidth - 0.5)
    const p = pointAt(g, s, rand(-1, 1) * Math.max(0, g.halfWidth - radius))
    const clear =
      features.rocks.every((r) => Math.hypot(rockCentre(r)[0] - p.x, rockCentre(r)[1] - p.z) > radius + 1.5) &&
      clearOfMovers(g, p.x, p.z, radius, 0.5) &&
      features.swirls.every((o) => o === w || o.age < 0 || Math.hypot(o.x - p.x, o.z - p.z) > o.radius + radius)
    if (!clear) continue
    Object.assign(w, { x: p.x, z: p.z, radius, peak: (Math.random() < 0.5 ? -1 : 1) * rand(4, 7) })
    return
  }
}

// Advances the swirls by `dt` seconds.
export function updateSwirls(g: CourseGeometry, dt: number) {
  for (const w of features.swirls) {
    w.age += dt
    if (w.age >= w.life) respawn(g, w, rand(SWIRL_GAP[0], SWIRL_GAP[1]))
    w.strength = w.peak * swirlLevel(w)
  }
}

// How spun up a swirl is: 0 (gone) to 1.
export function swirlLevel(w: Swirl): number {
  if (w.age <= 0) return 0
  return Math.min(1, w.age / SWIRL_FADE, (w.life - w.age) / SWIRL_FADE)
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
    if (w.strength === 0) continue
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
