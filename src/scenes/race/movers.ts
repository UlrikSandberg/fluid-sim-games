import { makeObstacle, type Obstacle } from '../../sim/sph'
import type { Params } from '../../sim/params'
import { rotate, type Quat } from '../../sim/quat'
import { bedAt, locate, pointAt, waterAt, type CourseGeometry } from './course'

// Moving obstacles: kinematic bodies on the GPU (they push the water, nothing
// pushes them) whose pose follows a script in time. Each lives at a spot s along
// the course, in a frame turned to the course there (x downstream, y up, z across
// toward +d).
//   gate:    a wooden wall that slides out of a bank part of the way across,
//            holds, slides back in (hidden under the bank) and rests
//   pulse:   a rock that grows and shrinks
//   slider:  a block that wanders from bank to bank
//   spinner: a bar turning about its middle

export type MoverKind = 'gate' | 'pulse' | 'slider' | 'spinner'
export const MOVER_KINDS: MoverKind[] = ['gate', 'pulse', 'slider', 'spinner']

export interface Mover {
  kind: MoverKind
  s: number
  d: number // across (the gate's bank: its sign)
  yaw: number // the course's heading at s, as a turn about +y from +x
  size: number[] // full size at rest: along, height, across
  reach: number // gate: how far out; slider: how far either side; pulse: growth; spinner: unused
  period: number // seconds per cycle
  phase: number // 0–1
  bottom: number // below the bed, so no gap shows under it
  top: number
  radius: number // how far it can reach horizontally from (s, d), for keeping things apart
  o: Obstacle // the body; its pose is set by updateMovers
}

const rand = (lo: number, hi: number) => lo + Math.random() * (hi - lo)
const smooth = (x: number) => x * x * (3 - 2 * x)

// A retracted gate's inner face sits this far inside the bank, so its boundary
// particles are out of the water's reach (h).
const GATE_TUCK = 1.3

// A mover of `kind` at s (and d, ignored for gates and spinners), or null if it
// doesn't fit there (a gate needs solid ground behind its bank to slide into).
export function makeMover(p: Params, g: CourseGeometry, kind: MoverKind, s: number, d: number): Mover | null {
  const hw = g.halfWidth
  const at = pointAt(g, s)
  const yaw = Math.atan2(-at.tz, at.tx)
  const bed = bedAt(g, s)
  const water = waterAt(g, s)
  const base = { kind, s, yaw, bottom: bed - 0.5, period: rand(6, 10), phase: Math.random() }
  let m: Omit<Mover, 'o'>
  if (kind === 'gate') {
    const reach = rand(0.28, 0.36) * 2 * hw
    const length = reach + GATE_TUCK + 0.3
    const fits = (side: number) =>
      [-0.7, 0.7].every((ds) => [hw + length / 2, hw + GATE_TUCK + length].every((dd) => {
        const q = pointAt(g, s + ds, side * dd)
        const inside = q.x > 0.5 && q.z > 0.5 && q.x < g.domain.max[0] - 0.5 && q.z < g.domain.max[2] - 0.5
        // Still beside this stretch of river, not over another one.
        return inside && Math.abs(locate(g, q.x, q.z).s - (s + ds)) < 1.5
      }))
    const sides = Math.random() < 0.5 ? [-1, 1] : [1, -1]
    const side = sides.find(fits)
    if (side == null) return null
    // Its top stays under the grass (a little above the water), so it's hidden when in.
    m = { ...base, d: side, size: [1.2, 0, length], reach, top: water + 0.85, radius: reach }
  } else if (kind === 'spinner') {
    const length = Math.min(0.55 * 2 * hw, 2 * hw - 5)
    m = { ...base, d: 0, size: [length, 0, 0.8], reach: 0, top: water + 1.1, radius: length / 2, period: rand(5, 9) * (Math.random() < 0.5 ? -1 : 1) }
  } else if (kind === 'slider') {
    const a = rand(1.5, 2.2)
    const reach = hw - a * Math.SQRT1_2 - 1.5
    m = { ...base, d: 0, size: [a, 0, a], reach, top: water + 1.2, radius: reach + a, period: rand(7, 11) }
  } else {
    const a = rand(1.6, 2.4)
    m = { ...base, d, size: [a, 0, a], reach: 0.35, top: water + 1, radius: a * 1.35 * Math.SQRT1_2 + 0.5, period: rand(4, 7) }
  }
  m.size[1] = m.top - m.bottom
  const o = makeObstacle(p, { center: [at.x, (m.top + m.bottom) / 2, at.z], size: m.size })
  o.grabbed = true // not integrated: the script moves it
  const mover = { ...m, o }
  placeMover(g, mover, 0)
  return mover
}

// Sets mover m's pose for time t (seconds).
function placeMover(g: CourseGeometry, m: Mover, t: number) {
  const hw = g.halfWidth
  const cycle = (((t / Math.abs(m.period) + m.phase) % 1) + 1) % 1
  let d = m.d
  let yaw = m.yaw
  let scale = 1
  let half = [m.size[0] / 2, m.size[1] / 2, m.size[2] / 2]
  const bottom = m.bottom
  let top = m.top
  if (m.kind === 'gate') {
    // Out over a fifth of the cycle, hold, back in, rest.
    const out = cycle < 0.2 ? smooth(cycle / 0.2) : cycle < 0.45 ? 1 : cycle < 0.65 ? 1 - smooth((cycle - 0.45) / 0.2) : 0
    d = m.d * (hw + GATE_TUCK + m.size[2] / 2 - out * (m.reach + GATE_TUCK))
  } else if (m.kind === 'slider') {
    d = m.reach * Math.sin(2 * Math.PI * cycle)
  } else if (m.kind === 'spinner') {
    yaw = m.yaw + Math.sign(m.period) * 2 * Math.PI * cycle
  } else {
    scale = 1 + m.reach * Math.sin(2 * Math.PI * cycle)
    half = [half[0] * scale, 0, half[2] * scale]
    top = waterAt(g, m.s) + 0.3 + 0.7 * scale
  }
  half[1] = (top - bottom) / 2
  const p = pointAt(g, m.s, d)
  m.o.center = [p.x, (top + bottom) / 2, p.z]
  m.o.halfSize = half
  m.o.q = [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0] as Quat
}

// Moves every mover to time t, with velocities from the step `dt` since the last
// call (for the water's and the ducks' bounces).
export function updateMovers(g: CourseGeometry, movers: Mover[], t: number, dt: number) {
  for (const m of movers) {
    const before = m.o.center.slice()
    placeMover(g, m, t)
    m.o.vel = dt > 0 ? m.o.center.map((v, k) => (v - before[k]) / dt) : [0, 0, 0]
    m.o.omega = [0, m.kind === 'spinner' ? (2 * Math.PI) / m.period : 0, 0]
  }
}

// Pushes duck o out of any mover it overlaps (the duck as a sphere of its largest
// half size against the mover's box, in the mover's frame), taking on the mover's
// velocity where they touch, plus a bounce.
export function moverConstraint(o: Obstacle, movers: Mover[]) {
  const r = Math.max(...o.halfSize)
  for (const { o: m } of movers) {
    const rel = o.center.map((v, k) => v - m.center[k])
    const local = rotate(m.q, rel, true)
    const closest = local.map((v, k) => Math.min(m.halfSize[k], Math.max(-m.halfSize[k], v)))
    const away = local.map((v, k) => v - closest[k])
    const dist = Math.hypot(...away)
    if (dist >= r) continue
    let nLocal: number[]
    let push: number
    if (dist > 1e-6) {
      nLocal = away.map((v) => v / dist)
      push = r - dist
    } else {
      // Centre inside: out through the nearest face.
      let best = Infinity
      nLocal = [0, 0, 0]
      for (let k = 0; k < 3; k++) {
        const depth = m.halfSize[k] - Math.abs(local[k])
        if (depth < best) {
          best = depth
          nLocal = [0, 0, 0]
          nLocal[k] = local[k] >= 0 ? 1 : -1
        }
      }
      push = best + r
    }
    const n = rotate(m.q, nLocal)
    for (let k = 0; k < 3; k++) o.center[k] += n[k] * push
    // The mover's velocity at the contact: v + ω × r.
    const arm = rotate(m.q, closest)
    const w = m.omega
    const surface = [
      m.vel[0] + w[1] * arm[2] - w[2] * arm[1],
      m.vel[1] + w[2] * arm[0] - w[0] * arm[2],
      m.vel[2] + w[0] * arm[1] - w[1] * arm[0],
    ]
    const vn = (o.vel[0] - surface[0]) * n[0] + (o.vel[1] - surface[1]) * n[1] + (o.vel[2] - surface[2]) * n[2]
    if (vn < 0) for (let k = 0; k < 3; k++) o.vel[k] -= 1.3 * vn * n[k]
  }
}

export const MOVER_COLORS: Record<MoverKind, number> = {
  gate: 0x8a5a2b,
  pulse: 0x6b7480,
  slider: 0x7d6b5a,
  spinner: 0xc0504d,
}
