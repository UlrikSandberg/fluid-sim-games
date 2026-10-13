import { particleRadius, type Params } from '../../sim/params'
import type { FrameMotion, Obstacle, SimState } from '../../sim/sph'
import { rotate, type Quat } from '../../sim/quat'
import { Renderer3D } from '../../render/renderer3d'
import { FoamField } from '../../render/foamField'
import { getColorMode, writeColors, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { GpuEngine } from '../../engines/sph'
import { DEFAULT_CURRENT, START_S, finishS, riverLayout, riverParams, riverScene } from './riverScene'
import {
  PICKUP_INFO, features, generateFeatures, randomizeCourse, swirlLevel, swirlOnDuck, updateSwirls, type PickupKind,
} from './features'
import {
  FLIP_ZONE_GRAVITY, bedAt, buildCourse, lidAt, locate, pointAt, terrain, waterAt, zoneAt, zoneLid, type CourseGeometry,
} from './course'
import type { BodyShape, SolidBox } from '../../sim/sph'
import { racers, type Vessel } from './racers'
import { MOVER_COLORS, moverConstraint, updateMovers } from './movers'
import { createRaceAudio } from './audio'

// idle: no ducks; setup: ducks hang at the gate while the water settles, until
// Start; countdown: 3, 2, 1; racing; finished.
export type RacePhase = 'idle' | 'setup' | 'countdown' | 'racing' | 'finished'

// A duck's place in the race: finished ducks by time, the rest by distance.
export interface Standing {
  index: number // which duck (its name and colour index)
  finishTime: number | null // seconds after the start, once across the line
  progress: number // 0 at the start line, 1 at the finish
  vessel: Vessel // actual body, not the next-race selection in the panel
  halfSize: number[]
  bow?: number
}

// A power-up going off: which duck, what, and how long ago (seconds).
export interface RaceEvent {
  duck: number
  kind: PickupKind
  age: number
}

export interface RaceStats {
  fps: number
  particles: number
  substeps: number
  ducks: number
  phase: RacePhase
  countdown: number // seconds left before the gate opens (countdown phase)
  sinceStart: number // seconds since the gate opened (racing and finished phases)
  standings: Standing[] // best first
  events: RaceEvent[] // power-ups from the last few seconds, newest first
  zeroG: number // seconds of Zero-G left (0 when gravity is on)
  course: number // counts the courses built (the panel shows the new one's pieces)
  status: 'starting' | 'running' | 'paused' | 'no-webgpu'
}

export interface RaceController {
  newCourse(): void // draw the 🎲 dials, build the course, hang the ducks at the gate
  setPaused(paused: boolean): void
  startRace(): void // setup: count down and release; finished: line up again on the same course
  lineUp(count: number): void // (re)hang `count` ducks at the gate (setup)
  dropDuck(): void // add one duck at the inlet
  clearDucks(): void
  dispose(): void
}

// The course being raced (rebuilt by newCourse).
let course: CourseGeometry = buildCourse(riverLayout.pieces, riverLayout.width, riverLayout.depth)

// Live settings the panel edits. raceParams is rebuilt for each course (spacing,
// h, dt, domain), keeping the live solver dials.
export const raceParams: Params = riverParams(course, riverLayout.spacing)
export const raceView = {
  colorMode: 'water' as ColorModeId,
  style: 'water' as RenderStyle,
  current: { ...DEFAULT_CURRENT },
  gravity: 9.8,
  duckSize: [1.6, 1, 1.2],
  duckCount: 4,
  follow: true, // camera glides after the leaders
  ride: -1, // the duck the camera rides (and the arrow keys steer), or −1
  rideView: 'first' as 'first' | 'third', // from its head, or from behind and above
  rideDistance: 8, // third person: how far behind (the mouse wheel changes it)
  steer: { turn: 0, paddle: 0 }, // held keys: turn +1 left / −1 right, paddle 0/1
}

// Steering the ridden duck (only while racing): a turn rate it's eased toward,
// and a paddle that pushes it along its heading up to a little over the current.
const TURN_RATE = 2.2 // rad/s
const PADDLE_ACCEL = 5
const PADDLE_EXTRA = 2.5 // top paddling speed over the current's

// A duck's heading on the water: its +x axis flattened, or null when it's on end.
function heading(o: Obstacle): number[] | null {
  const f = rotate(o.q, [1, 0, 0])
  const len = Math.hypot(f[0], f[2])
  return len < 0.3 ? null : [f[0] / len, f[2] / len]
}

// One step of the player's steering on duck o.
function steer(o: Obstacle, dt: number) {
  const { turn, paddle } = raceView.steer
  // A positive turn about +y swings the heading from +x toward −z: to the left,
  // seen from the duck.
  if (turn) o.omega[1] += (turn * TURN_RATE - o.omega[1]) * Math.min(1, 10 * dt)
  const h = heading(o)
  if (paddle && h && o.vel[0] * h[0] + o.vel[2] * h[1] < raceView.current.speed + PADDLE_EXTRA) {
    o.vel[0] += PADDLE_ACCEL * h[0] * dt
    o.vel[2] += PADDLE_ACCEL * h[1] * dt
  }
}

// The ride camera: from the duck's head looking ahead (the duck itself hidden),
// or third person from behind and above it, looking a little past it. Its yaw
// follows the duck's heading smoothly, so a spinning duck turns the view without
// jolting it.
export const RIDE_DISTANCE = [3, 20]
function rideView(o: Obstacle, yaw: number) {
  const f = [Math.cos(yaw), 0, Math.sin(yaw)]
  const c = o.center
  const [hx, hy] = o.halfSize
  if (raceView.rideView === 'first') {
    const eye = [c[0] + f[0] * hx * 0.9, c[1] + hy * 2 + 0.4, c[2] + f[2] * hx * 0.9]
    return { eye, target: [eye[0] + f[0] * 10, eye[1] - 1.2, eye[2] + f[2] * 10], fov: 75 }
  }
  const back = raceView.rideDistance
  return {
    eye: [c[0] - f[0] * back, c[1] + 0.2 + back * 0.5, c[2] - f[2] * back],
    target: [c[0] + f[0] * 5, c[1], c[2] + f[2] * 5],
    fov: 60,
  }
}

const MAX_RACE = 120 // seconds; ducks still out by then are ranked by distance

// Power-ups (see features.ts for where they float).
const PICKUP_REACH = 1.4 // horizontal distance from a duck's centre that grabs one
const FLIP_SCALE = -0.6 // a flipped duck's gravity, as a multiple of normal
const FLIP_SECONDS = 1.5
const BOOST_SPEED = 7 // added along the channel
const ZERO_G_SECONDS = 2
const EVENT_SECONDS = 4 // how long an event stays in the feed

const STATS_INTERVAL_MS = 250
const COUNTDOWN = 3 // seconds of simulated time before the gate opens
const LANE_GAP = 0.15 // between neighbouring ducks on the start line

// What each kind of racer is to the water, at size 1×: a duck is a box under its
// mesh with a weighted bottom; a boat is a long, low hull whose bow narrows to a
// point over its front 40%, with a keel. The ballast (centre of gravity below the
// centre, see Obstacle) keeps them upright in rough water most of the time; without
// it both spent half a race on their sides.
const DUCK_BASE = [1.6, 1, 1.2]
export const VESSELS: Record<Vessel, { size: number[]; shape?: BodyShape; ballast: number }> = {
  duck: { size: DUCK_BASE, ballast: 0.5 },
  boat: { size: [2.6, 0.8, 1.1], shape: { kind: 'hull', bow: 0.2 }, ballast: 0.9 },
}

// Racer i's hull: its vessel at the panel's size.
function vesselFor(i: number) {
  const v = VESSELS[racers.vessels[i] ?? 'duck']
  const scale = raceView.duckSize[0] / DUCK_BASE[0]
  return { size: v.size.map((s) => s * scale), shape: v.shape, ballast: v.ballast }
}

// Rebuilds raceParams for the current course, keeping the live dials.
function refreshParams() {
  const fresh = riverParams(course, riverLayout.spacing)
  const { stiffness, viscosityEps, restitution, obstacleDensity, useTait } = raceParams
  Object.assign(raceParams, fresh, { stiffness, viscosityEps, restitution, obstacleDensity, useTait })
}

// How far a duck reaches from its centre horizontally and vertically.
function reaches(o: Obstacle) {
  let side = 0
  let up = 0
  for (let k = 0; k < 3; k++) {
    const axis = rotate(o.q, [k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0])
    side = Math.max(side, Math.hypot(axis[0], axis[2]) * o.halfSize[k])
    up += Math.abs(axis[1]) * o.halfSize[k]
  }
  return { side, up }
}

// Pushes the duck back along unit vector n by `depth`, reflecting its velocity
// against n.
function pushBack(o: Obstacle, n: number[], depth: number) {
  for (let d = 0; d < 3; d++) o.center[d] += depth * n[d]
  const vn = o.vel[0] * n[0] + o.vel[1] * n[1] + o.vel[2] * n[2]
  if (vn < 0) for (let d = 0; d < 3; d++) o.vel[d] -= 1.3 * vn * n[d]
}

// Keeps a duck in the river as the GPU keeps the water: between the banks
// (measured across the course), inside the ends, above the bed (or past a lip,
// if it's nearer) and under a flip zone's lid.
function courseConstraint(g: CourseGeometry, o: Obstacle) {
  const { side, up } = reaches(o)
  const w = locate(g, o.center[0], o.center[2])
  const t = [w.tx, 0, w.tz]
  const across = [-w.tz, 0, w.tx]
  const limit = Math.max(0, g.halfWidth - side)
  if (Math.abs(w.d) > limit) pushBack(o, across.map((v) => -Math.sign(w.d) * v), Math.abs(w.d) - limit)
  if (w.s < side) pushBack(o, t, side - w.s)
  if (w.s > g.length - side) pushBack(o, t.map((v) => -v), w.s - (g.length - side))
  const bed = bedAt(g, w.s)
  const bottom = o.center[1] - up
  if (bottom < bed) {
    const lip = Math.min(...g.falls.filter((f) => f.s > w.s).map((f) => f.s))
    const past = lip + side - w.s
    if (past < bed - bottom && bottom >= bedAt(g, lip)) pushBack(o, t, past)
    else pushBack(o, [0, 1, 0], bed - bottom)
  }
  if (zoneAt(g, w.s)) {
    const lid = lidAt(g, w.s)
    if (o.center[1] + up > lid) pushBack(o, [0, -1, 0], o.center[1] + up - lid)
  }
}

// The start line: `count` ducks side by side across the channel at START_S,
// hanging just above the water and facing downstream. If they don't fit the
// channel's width at their set size, they're scaled down to fit. Also returns the
// gate bar's ends, above the ducks.
function startLine(count: number) {
  const g = course
  const at = pointAt(g, START_S)
  const across = [-at.tz, at.tx] // (x, z) unit vector across the channel
  const vessels = Array.from({ length: count }, (_, i) => vesselFor(i))
  const lane = Math.max(...vessels.map((v) => v.size[2]))
  const available = 2 * g.halfWidth - 0.6
  const needed = count * lane + (count - 1) * LANE_GAP
  const scale = Math.min(1, available / needed)
  for (const v of vessels) v.size = v.size.map((s) => s * scale)
  const water = waterAt(g, START_S)
  // Lanes are dealt out at random each race: a fixed order hands the same ducks
  // the same lanes every time, and the course's shape favours some lanes (two
  // races in a row finished in identical order before this).
  const lanes = Array.from({ length: count }, (_, i) => i)
  for (let i = count - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1))
    ;[lanes[i], lanes[j]] = [lanes[j], lanes[i]]
  }
  const racersAt = lanes.map((laneIndex, i) => {
    const offset = (laneIndex - (count - 1) / 2) * (lane * scale + LANE_GAP)
    const y = water + vessels[i].size[1] / 2 + 0.3
    return { ...vessels[i], center: [at.x + offset * across[0], y, at.z + offset * across[1]] }
  })
  const reach = g.halfWidth + 0.5
  const barY = water + Math.max(...vessels.map((v) => v.size[1])) + 0.8
  const gate = [
    [at.x - reach * across[0], barY, at.z - reach * across[1]],
    [at.x + reach * across[0], barY, at.z + reach * across[1]],
  ]
  // Facing downstream: a turn about +y from +x to the tangent.
  const yaw = Math.atan2(-at.tz, at.tx)
  const q: Quat = [Math.cos(yaw / 2), 0, Math.sin(yaw / 2), 0]
  return { racers: racersAt, gate, q }
}

// The finish banner's ends: across the channel at finishS, its bottom a little
// above the water.
function finishLine(): number[][] {
  const g = course
  const s = finishS(g)
  const reach = g.halfWidth + 0.5
  const y = waterAt(g, s) + 2.5
  const a = pointAt(g, s, -reach)
  const b = pointAt(g, s, reach)
  return [
    [a.x, y, a.z],
    [b.x, y, b.z],
  ]
}

// How far down the course each duck is.
const sOf = (o: Obstacle) => locate(course, o.center[0], o.center[2]).s

// Best first: finished ducks by finish time, then the rest by how far down the
// course they are.
function standings(bodies: Obstacle[], finishTimes: (number | null)[]): Standing[] {
  const span = finishS(course) - START_S
  return bodies
    .map((o, index) => ({
      index,
      finishTime: finishTimes[index] ?? null,
      progress: Math.min(1, Math.max(0, (sOf(o) - START_S) / span)),
      vessel: o.shape?.kind === 'hull' ? 'boat' as const : 'duck' as const,
      halfSize: [...o.halfSize],
      bow: o.shape?.bow,
    }))
    .sort((a, b) => {
      if (a.finishTime != null && b.finishTime != null) return a.finishTime - b.finishTime
      if (a.finishTime != null) return -1
      if (b.finishTime != null) return 1
      return b.progress - a.progress
    })
}

// Where the camera looks: the start line before the gate opens, then a little
// ahead of the leading pack (the first three still racing), and the finish once
// it's over.
function followPoint(bodies: Obstacle[], phase: RacePhase, finishTimes: (number | null)[]): number[] | null {
  const g = course
  const at = (s: number, y = waterAt(g, s)) => {
    const p = pointAt(g, s)
    return [p.x, y, p.z]
  }
  if (phase === 'idle' || phase === 'setup' || phase === 'countdown' || bodies.length === 0) return at(START_S + 8)
  if (phase === 'finished') return at(finishS(g))
  const racing = bodies.filter((_, i) => finishTimes[i] == null).sort((a, b) => sOf(b) - sOf(a)).slice(0, 3)
  const pack = racing.length > 0 ? racing : bodies
  const s = pack.reduce((a, o) => a + sOf(o), 0) / pack.length
  const y = pack.reduce((a, o) => a + o.center[1], 0) / pack.length
  return at(Math.min(s + 6, finishS(g)), y)
}

// Keeps a duck out of the river's solids (the ledge, rocks): treated as a sphere of
// its largest half size, pushed out from the nearest point of any box it overlaps
// (through the nearest face if its centre got inside), velocity into the solid
// reflected. The GPU keeps the water out of them; this does the same for ducks.
function solidConstraint(o: Obstacle, boxes: SolidBox[]) {
  const r = Math.max(...o.halfSize)
  for (const box of boxes) {
    const closest = o.center.map((v, d) => Math.min(box.max[d], Math.max(box.min[d], v)))
    const away = o.center.map((v, d) => v - closest[d])
    const dist = Math.hypot(...away)
    if (dist >= r) continue
    let n: number[]
    let push: number
    if (dist > 1e-6) {
      n = away.map((v) => v / dist)
      push = r - dist
    } else {
      // Centre inside: leave through the face it's nearest to.
      let best = Infinity
      n = [0, 0, 0]
      for (let d = 0; d < 3; d++) {
        for (const side of [-1, 1]) {
          const depth = side < 0 ? o.center[d] - box.min[d] : box.max[d] - o.center[d]
          if (depth < best) {
            best = depth
            n = [0, 0, 0]
            n[d] = side
          }
        }
      }
      push = best + r
    }
    for (let d = 0; d < 3; d++) o.center[d] += n[d] * push
    const vn = o.vel[0] * n[0] + o.vel[1] * n[1] + o.vel[2] * n[2]
    if (vn < 0) for (let d = 0; d < 3; d++) o.vel[d] -= 1.3 * vn * n[d]
  }
}

// The layout the current course was built from (to tell the panel whether its
// course dials have changed since), and the course's length.
export const courseState = { built: '', length: course.length, version: 0 }

// The river race's loop: the GPU solver running the course from riverScene,
// rendered with the shared 3D renderer. GPU only: a river this size isn't
// playable on the CPU solver.
export function startRaceLoop(container: HTMLElement, onStats: (s: RaceStats) => void): RaceController {
  const audio = createRaceAudio()
  let engine: GpuEngine | null = null
  let renderer: Renderer3D | null = null
  let foam: FoamField | null = null
  let generation = 0
  let paused = false
  let noWebGpu = false
  let phase: RacePhase = 'idle'
  let countdown = 0
  let sinceStart = 0
  let finishTimes: (number | null)[] = []
  let finalStandings: Standing[] | null = null
  let flipUntil: number[] = [] // per duck, race time its flip wears off (0 = none)
  let zeroGUntil = 0
  let events: { duck: number; kind: PickupKind; at: number }[] = []
  let moverClock = 0 // simulated seconds the movers have been moving
  let rider: Obstacle | null = null // the duck being ridden (and steered)
  let rideYaw: number | null = null // the ride camera's smoothed heading
  let accumulator = 0
  let last = performance.now()

  // Builds riverLayout's course (with fresh rocks, swirls and pickups) and hangs
  // the ducks at the gate once it's ready. A course whose pieces don't fit
  // together is skipped (the panel says why).
  const build = () => {
    const g = buildCourse(riverLayout.pieces, riverLayout.width, riverLayout.depth)
    if (g.problem) return
    audio.reset()
    const mine = ++generation
    engine?.dispose()
    renderer?.dispose()
    engine = null
    renderer = null
    phase = 'idle'
    finalStandings = null
    accumulator = 0
    last = performance.now()
    course = g
    courseState.built = JSON.stringify(riverLayout)
    courseState.length = g.length
    courseState.version++
    refreshParams()
    generateFeatures(g, raceParams)
    const rocks = features.rocks
    const movers = features.movers
    moverClock = 0
    void GpuEngine.create(raceParams, riverScene(raceParams, g, rocks)).then((created) => {
      if (mine !== generation) {
        created?.dispose()
        return
      }
      if (!created) {
        noWebGpu = true
        return
      }
      engine = created
      engine.constrainBody = (o) => {
        if (o === rider && !o.grabbed && phase === 'racing') steer(o, raceParams.dt)
        if (!o.grabbed) swirlOnDuck(o.center, o.vel, raceParams.dt)
        courseConstraint(g, o)
        solidConstraint(o, rocks)
        moverConstraint(o, movers)
      }
      engine.setMovers(movers.map((m) => m.o))
      // Framed for a ~50-unit stretch, not the whole course: on long rivers the
      // whole-course view made ducks and swirls tiny, and the follow camera tracks
      // the race anyway.
      const view = Math.min(Math.max(g.domain.max[0], g.domain.max[2]), 50)
      renderer = new Renderer3D(container, created.n, particleRadius(raceParams), g.domain,
        [-0.07 * view, 0.45 * view, 0.85 * view])
      renderer.showBounds(false)
      foam = new FoamField({ ...raceParams, domain: g.domain })
      renderer.addTerrain(terrain(g))
      for (const rock of rocks) {
        const bed = bedAt(g, locate(g, (rock.min[0] + rock.max[0]) / 2, (rock.min[2] + rock.max[2]) / 2).s)
        renderer.addSolid([rock.min[0], bed - 0.5, rock.min[2]], rock.max)
      }
      for (const zone of g.zones) renderer.addLid(zoneLid(g, zone))
      renderer.setFinish(finishLine())
      accumulator = 0
      lineUp(raceView.duckCount)
      renderer.follow(followPoint(engine.bodies, phase, finishTimes), true)
    })
  }

  // Hangs `count` ducks at the gate (the setup phase: the water runs, the ducks
  // wait for Start). Puts the power-ups back.
  const lineUp = (count: number) => {
    if (!engine) return
    audio.reset()
    engine.removeBody()
    const line = startLine(count)
    for (const racer of line.racers) engine.addBody(racer)
    // Held at the gate (kinematic at their spot) until the countdown ends.
    for (const o of engine.bodies) {
      o.grabbed = true
      o.q = [...line.q]
    }
    renderer?.setGate(line.gate)
    for (const pickup of features.pickups) pickup.taken = false
    phase = 'setup'
    finalStandings = null
    accumulator = 0
    last = performance.now()
    countdown = 0
    sinceStart = 0
    zeroGUntil = 0
    events = []
    finishTimes = engine.bodies.map(() => null)
    flipUntil = engine.bodies.map(() => 0)
  }

  // The first duck to reach an untaken pickup sets it off.
  const grabPickups = (bodies: Obstacle[]) => {
    for (const pickup of features.pickups) {
      if (pickup.taken) continue
      const i = bodies.findIndex(
        (o) => Math.hypot(o.center[0] - pickup.pos[0], o.center[2] - pickup.pos[2]) < PICKUP_REACH &&
          Math.abs(o.center[1] - pickup.pos[1]) < 2,
      )
      if (i < 0) continue
      pickup.taken = true
      events.push({ duck: i, kind: pickup.kind, at: sinceStart })
      const o = bodies[i]
      const w = locate(course, o.center[0], o.center[2])
      const along = [w.tx, 0, w.tz] // downstream
      if (pickup.kind === 'flip') {
        flipUntil[i] = sinceStart + FLIP_SECONDS
        for (let d = 0; d < 3; d++) o.vel[d] += 2 * along[d]
      } else if (pickup.kind === 'boost') {
        for (let d = 0; d < 3; d++) o.vel[d] += BOOST_SPEED * along[d]
        o.vel[1] += 1.5
      } else {
        // Another Zero-G while one is on extends it.
        zeroGUntil = Math.max(zeroGUntil, sinceStart) + ZERO_G_SECONDS
      }
    }
  }

  // Each duck's gravity: reversed while it's flipped and inside a flip zone (both
  // at once cancel out to normal, more or less).
  const duckGravity = (bodies: Obstacle[]) => {
    bodies.forEach((o, i) => {
      const flipped = flipUntil[i] > 0 && sinceStart < flipUntil[i]
      const inZone = zoneAt(course, sOf(o)) != null
      o.gravityScale = (flipped ? FLIP_SCALE : 1) * (inZone ? FLIP_ZONE_GRAVITY : 1)
    })
  }

  build()
  // For poking at a race from the browser console (and the headless checks).
  if (import.meta.env.DEV) Object.assign(window, { __race: { features, course: () => course, raceView, renderer: () => renderer, bodies: () => engine?.bodies ?? [] } })

  let raf = 0
  let frames = 0
  let substepTotal = 0
  let statsStart = last
  let view: SimState | null = null

  const frame = (now: number) => {
    const frameDt = Math.min(Math.max((now - last) / 1000, 0), 0.1)
    accumulator += frameDt
    last = now
    rider = engine?.bodies[raceView.ride] ?? null
    let substeps = 0
    if (paused || !engine || phase === 'finished') {
      accumulator = 0
    } else {
      substeps = Math.min(Math.floor(accumulator / raceParams.dt), raceParams.maxSubsteps)
      accumulator -= substeps * raceParams.dt
      if (substeps === raceParams.maxSubsteps) accumulator = Math.min(accumulator, raceParams.dt)
      const simulated = substeps * raceParams.dt
      engine.current.speed = raceView.current.speed
      engine.current.rate = raceView.current.rate
      updateSwirls(course, simulated)
      moverClock += simulated
      updateMovers(course, features.movers, moverClock, simulated)
      engine.setSwirls(features.swirls.filter((w) => w.strength !== 0))
      duckGravity(engine.bodies)
      const pivot = raceParams.domain.min.map((v, d) => (v + raceParams.domain.max[d]) / 2)
      const still: FrameMotion = { omega: [0, 0, 0], alpha: [0, 0, 0], pivot }
      // Zero-G switches gravity off for everything, water and ducks alike.
      const zeroG = phase === 'racing' && sinceStart < zeroGUntil
      engine.step(substeps, still, [0, zeroG ? 0 : -raceView.gravity, 0])
      // The countdown runs on simulated time, so pausing holds it.
      if (phase === 'countdown') {
        countdown -= simulated
        if (countdown <= 0) {
          for (const o of engine.bodies) {
            o.grabbed = false
            o.vel.fill(0)
          }
          phase = 'racing'
          sinceStart = 0
          renderer?.setGate(null)
        }
      } else if (phase === 'racing') {
        sinceStart += simulated
        const line = finishS(course)
        engine.bodies.forEach((o, i) => {
          if (finishTimes[i] == null && sOf(o) >= line) finishTimes[i] = sinceStart
          if (flipUntil[i] && sinceStart >= flipUntil[i]) flipUntil[i] = 0
        })
        grabPickups(engine.bodies)
        if (finishTimes.every((t) => t != null) || sinceStart >= MAX_RACE) {
          finalStandings = standings(engine.bodies, finishTimes)
          phase = 'finished'
          accumulator = 0
        }
      }
    }

    audio.update(sinceStart, engine?.bodies ?? [], finishTimes, phase === 'racing' && !paused)

    if (engine && renderer && phase !== 'finished') {
      const first = rider != null && raceView.rideView === 'first'
      renderer.setDucks(engine.bodies, first ? raceView.ride : -1)
      renderer.setMovers(features.movers.map((m) => ({ ...m.o, color: MOVER_COLORS[m.kind] })))
      renderer.setPickups(features.pickups.map((q) => ({ pos: q.pos, color: PICKUP_INFO[q.kind].color, taken: q.taken })))
      // Over the resting level: the flowing water stands a little higher.
      renderer.setSwirls(features.swirls.map((w) => ({
        x: w.x, z: w.z, radius: w.radius, strength: w.strength, level: swirlLevel(w),
        y: waterAt(course, locate(course, w.x, w.z).s) + 0.7,
      })))
      renderer.follow(raceView.follow ? followPoint(engine.bodies, phase, finishTimes) : null)
      if (rider) {
        const h = heading(rider) ?? (() => {
          const w = locate(course, rider.center[0], rider.center[2])
          return [w.tx, w.tz]
        })()
        const goal = Math.atan2(h[1], h[0])
        if (rideYaw == null) rideYaw = goal
        // Ease toward the duck's heading the short way round.
        const turn = Math.atan2(Math.sin(goal - rideYaw), Math.cos(goal - rideYaw))
        rideYaw += turn * (1 - Math.exp(-5 * frameDt))
        renderer.ride(rideView(rider, rideYaw))
      } else {
        rideYaw = null
        renderer.ride(null)
      }
      // The "Water" colour mode drawn as water: clear, with foam and spray.
      const realistic = raceView.style === 'water' && raceView.colorMode === 'water'
      view = engine.view()
      if (view) {
        renderer.updatePositions(view.pos, view.n)
        if (realistic && foam) {
          foam.update(view, frameDt, renderer.colors)
          renderer.setFoamFlow(foam.flow)
        }
        else writeColors(getColorMode(raceView.colorMode), view, raceParams, renderer.colors)
        renderer.commitColors()
      }
      renderer.render(raceView.style === 'water', realistic)
    }

    frames++
    substepTotal += substeps
    if (now - statsStart >= STATS_INTERVAL_MS) {
      onStats({
        fps: (frames * 1000) / (now - statsStart),
        particles: engine?.n ?? 0,
        substeps: substepTotal / frames,
        ducks: engine?.bodies.length ?? 0,
        phase,
        countdown,
        sinceStart,
        standings: finalStandings ?? standings(engine?.bodies ?? [], finishTimes),
        events: events
          .filter((e) => sinceStart - e.at < EVENT_SECONDS)
          .map((e) => ({ duck: e.duck, kind: e.kind, age: sinceStart - e.at }))
          .reverse(),
        zeroG: phase === 'racing' ? Math.max(0, zeroGUntil - sinceStart) : 0,
        course: courseState.version,
        status: noWebGpu ? 'no-webgpu' : !engine ? 'starting' : paused ? 'paused' : 'running',
      })
      frames = 0
      substepTotal = 0
      statsStart = now
    }
    raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)

  return {
    newCourse() {
      randomizeCourse(riverLayout) // only the dials marked random
      build()
    },
    setPaused(p) {
      paused = p
      if (p) audio.silence()
    },
    startRace() {
      if (phase === 'setup') {
        phase = 'countdown'
        countdown = COUNTDOWN
      } else if (phase !== 'countdown' && phase !== 'racing') {
        lineUp(raceView.duckCount)
      }
    },
    lineUp(count) {
      if (phase === 'setup' || phase === 'idle' || phase === 'finished') lineUp(count)
    },
    dropDuck() {
      if (phase === 'finished') return // final placements cannot acquire new entrants
      // Just downstream of the inlet, above the water, on the centreline.
      const p = pointAt(course, 3)
      const vessel = vesselFor(engine?.bodies.length ?? 0)
      const y = waterAt(course, 3) + vessel.size[1] + 1
      if (engine?.addBody({ center: [p.x, y, p.z], ...vessel })) {
        finishTimes.push(null) // a late entrant still races
        flipUntil.push(0)
      }
    },
    clearDucks() {
      audio.reset()
      engine?.removeBody()
      renderer?.setGate(null)
      phase = 'idle'
      finalStandings = null
      finishTimes = []
      flipUntil = []
    },
    dispose() {
      audio.dispose()
      generation++
      cancelAnimationFrame(raf)
      engine?.dispose()
      renderer?.dispose()
    },
  }
}
