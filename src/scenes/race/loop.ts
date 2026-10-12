import { particleRadius, type Params } from '../../sim/params'
import type { FrameMotion, Obstacle, SimState } from '../../sim/sph'
import { rotate } from '../../sim/quat'
import { Renderer3D } from '../../render/renderer3d'
import { getColorMode, writeColors, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { GpuEngine } from '../fluid3d/engines'
import {
  DEFAULT_CURRENT, bankLines, courseOf, finishX, riverBoxes, riverLayout, riverParams, riverScene,
} from './riverScene'

export type RacePhase = 'idle' | 'countdown' | 'racing' | 'finished'

// A duck's place in the race: finished ducks by time, the rest by distance.
export interface Standing {
  index: number // which duck (its name and colour index)
  finishTime: number | null // seconds after the start, once across the line
  progress: number // 0 at the start line, 1 at the finish
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
  status: 'starting' | 'running' | 'paused' | 'no-webgpu'
}

export interface RaceController {
  reset(): void // rebuild the course from riverLayout
  setPaused(paused: boolean): void
  startRace(count: number): void // clear the ducks, hang `count` at the start line, count down, release
  dropDuck(): void // add one duck at the inlet
  clearDucks(): void
  dispose(): void
}

// Live settings the panel edits. raceParams is rebuilt from the layout on reset
// (spacing, h, dt, domain), keeping the live solver dials.
export const raceParams: Params = riverParams()
export const raceView = {
  colorMode: 'water' as ColorModeId,
  style: 'water' as RenderStyle,
  current: { ...DEFAULT_CURRENT },
  gravity: 9.8,
  duckSize: [1.6, 1, 1.2],
  duckCount: 4,
  follow: true, // camera glides after the leaders
}

const MAX_RACE = 120 // seconds; ducks still out by then are ranked by distance

const STATS_INTERVAL_MS = 250
const COUNTDOWN = 3 // seconds of simulated time before the gate opens
const START_X = 4 // the start line's position along the course
const LANE_GAP = 0.15 // between neighbouring ducks on the start line

// Rebuilds raceParams for the current layout, keeping the live dials.
function refreshParams() {
  const fresh = riverParams(riverLayout)
  const { stiffness, viscosityEps, restitution, obstacleDensity, useTait } = raceParams
  Object.assign(raceParams, fresh, { stiffness, viscosityEps, restitution, obstacleDensity, useTait })
}

// Keeps the duck between the banks: its centre stays within the half width less
// its horizontal reach, measured perpendicular to the winding centreline, and
// its velocity into the bank is reflected.
function bankConstraint(o: Obstacle) {
  const c = courseOf(riverLayout)
  let reach = 0
  for (let k = 0; k < 3; k++) {
    const axis = rotate(o.q, [k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0])
    reach = Math.max(reach, Math.hypot(axis[0], axis[2]) * o.halfSize[k])
  }
  const s = c.slope(o.center[0])
  const norm = Math.hypot(s, 1)
  const across = (o.center[2] - c.centerline(o.center[0])) / norm
  const limit = Math.max(0, c.halfWidth - reach)
  if (Math.abs(across) <= limit) return
  const n = [(-Math.sign(across) * s) / norm, 0, Math.sign(across) / norm] // outward
  const excess = Math.abs(across) - limit
  o.center[0] -= excess * n[0]
  o.center[2] -= excess * n[2]
  const vn = o.vel[0] * n[0] + o.vel[2] * n[2]
  if (vn > 0) {
    o.vel[0] -= 1.3 * vn * n[0]
    o.vel[2] -= 1.3 * vn * n[2]
  }
}

// The start line: `count` ducks side by side across the channel at START_X,
// along the channel's perpendicular there, hanging just above the water and facing
// downstream. If they don't fit the channel's width at their set size, they're
// scaled down to fit. Also returns the gate bar's ends, above the ducks.
function startLine(count: number) {
  const c = courseOf(riverLayout)
  const s = c.slope(START_X)
  const norm = Math.hypot(1, s)
  const across = [-s / norm, 1 / norm] // (x, z) unit vector across the channel
  const available = 2 * c.halfWidth - 0.6
  const needed = count * raceView.duckSize[2] + (count - 1) * LANE_GAP
  const scale = Math.min(1, available / needed)
  const size = raceView.duckSize.map((v) => v * scale)
  const water = riverLayout.ledgeHeight + riverLayout.upperDepth
  const y = water + size[1] / 2 + 0.3
  const zc = c.centerline(START_X)
  const positions = Array.from({ length: count }, (_, i) => {
    const offset = (i - (count - 1) / 2) * (size[2] + LANE_GAP)
    return [START_X + offset * across[0], y, zc + offset * across[1]]
  })
  const reach = c.halfWidth + 0.5
  const barY = y + size[1] / 2 + 0.5
  const gate = [
    [START_X - reach * across[0], barY, zc - reach * across[1]],
    [START_X + reach * across[0], barY, zc + reach * across[1]],
  ]
  return { positions, size, gate }
}

// The finish banner's ends: across the channel at finishX, its bottom just above
// the lower reach's bank tops.
function finishLine(): number[][] {
  const c = courseOf(riverLayout)
  const x = finishX(riverLayout)
  const s = c.slope(x)
  const norm = Math.hypot(1, s)
  const reach = c.halfWidth + 0.5
  const y = riverLayout.lowerDepth + 3
  const zc = c.centerline(x)
  return [
    [x + (reach * s) / norm, y, zc - reach / norm],
    [x - (reach * s) / norm, y, zc + reach / norm],
  ]
}

// Best first: finished ducks by finish time, then the rest by how far down the
// course they are.
function standings(bodies: Obstacle[], finishTimes: (number | null)[]): Standing[] {
  const start = START_X
  const span = finishX(riverLayout) - start
  return bodies
    .map((o, index) => ({
      index,
      finishTime: finishTimes[index] ?? null,
      progress: Math.min(1, Math.max(0, (o.center[0] - start) / span)),
    }))
    .sort((a, b) => {
      if (a.finishTime != null && b.finishTime != null) return a.finishTime - b.finishTime
      if (a.finishTime != null) return -1
      if (b.finishTime != null) return 1
      return b.progress - a.progress
    })
}

// Where the camera looks: the start line during the countdown, then the middle of
// the leading pack (the first three still racing), and the finish once it's over.
function followPoint(bodies: Obstacle[], phase: RacePhase, finishTimes: (number | null)[]): number[] | null {
  const c = courseOf(riverLayout)
  if (phase === 'idle' || bodies.length === 0) return null
  if (phase === 'countdown') return [START_X + 8, riverLayout.ledgeHeight, c.centerline(START_X + 8)]
  if (phase === 'finished') {
    const x = finishX(riverLayout)
    return [x, riverLayout.lowerDepth, c.centerline(x)]
  }
  const racing = bodies.filter((_, i) => finishTimes[i] == null).sort((a, b) => b.center[0] - a.center[0]).slice(0, 3)
  const pack = racing.length > 0 ? racing : bodies
  const mean = [0, 1, 2].map((d) => pack.reduce((a, o) => a + o.center[d], 0) / pack.length)
  // A little ahead of the pack, but not past the course's end.
  const x = Math.min(mean[0] + 6, finishX(riverLayout))
  return [x, mean[1], c.centerline(x)]
}

// The river race's loop: the GPU solver running the course from riverScene,
// rendered with the shared 3D renderer. GPU only: a river this size isn't
// playable on the CPU solver.
export function startRaceLoop(container: HTMLElement, onStats: (s: RaceStats) => void): RaceController {
  let engine: GpuEngine | null = null
  let renderer: Renderer3D | null = null
  let generation = 0
  let paused = false
  let noWebGpu = false
  let phase: RacePhase = 'idle'
  let countdown = 0
  let sinceStart = 0
  let finishTimes: (number | null)[] = []

  const start = () => {
    const mine = ++generation
    engine?.dispose()
    renderer?.dispose()
    engine = null
    renderer = null
    phase = 'idle'
    refreshParams()
    void GpuEngine.create(raceParams, riverScene(raceParams, riverLayout)).then((created) => {
      if (mine !== generation) {
        created?.dispose()
        return
      }
      if (!created) {
        noWebGpu = true
        return
      }
      engine = created
      engine.constrainBody = bankConstraint
      const { domain } = courseOf(riverLayout)
      const size = domain.max.map((v, d) => v - domain.min[d])
      renderer = new Renderer3D(container, created.n, particleRadius(raceParams), domain,
        [-0.07 * size[0], 0.45 * size[0], 0.85 * size[0]])
      // Drawn a hair inside the domain's sides so they don't z-fight the banks' outer faces.
      for (const box of riverBoxes(riverLayout)) {
        renderer.addSolid([box.min[0], box.min[1], box.min[2] + 0.02], [box.max[0], box.max[1], box.max[2] - 0.02])
      }
      renderer.setFinish(finishLine())
      const [left, right] = bankLines(riverLayout)
      // Banks follow the water: just above the upper reach, stepping down at the
      // waterfall to just above the lower reach (the physical banks go to the top).
      const l = riverLayout
      const bankTop = (x: number) => (x < l.ledgeLength ? l.ledgeHeight + l.upperDepth : l.lowerDepth) + 1
      renderer.addBank(left, bankTop, domain.min[2])
      renderer.addBank(right, bankTop, domain.max[2])
      accumulator = 0
    })
  }
  start()

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let statsStart = last
  let view: SimState | null = null

  const frame = (now: number) => {
    accumulator += Math.min(Math.max((now - last) / 1000, 0), 0.1)
    last = now
    let substeps = 0
    if (paused || !engine) {
      accumulator = 0
    } else {
      substeps = Math.min(Math.floor(accumulator / raceParams.dt), raceParams.maxSubsteps)
      accumulator -= substeps * raceParams.dt
      if (substeps === raceParams.maxSubsteps) accumulator = Math.min(accumulator, raceParams.dt)
      engine.current.speed = raceView.current.speed
      engine.current.rate = raceView.current.rate
      const pivot = raceParams.domain.min.map((v, d) => (v + raceParams.domain.max[d]) / 2)
      const still: FrameMotion = { omega: [0, 0, 0], alpha: [0, 0, 0], pivot }
      engine.step(substeps, still, [0, -raceView.gravity, 0])
      // The countdown runs on simulated time, so pausing holds it.
      const simulated = substeps * raceParams.dt
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
        const line = finishX(riverLayout)
        engine.bodies.forEach((o, i) => {
          if (finishTimes[i] == null && o.center[0] >= line) finishTimes[i] = sinceStart
        })
        if (finishTimes.every((t) => t != null) || sinceStart >= MAX_RACE) phase = 'finished'
      }
    }

    if (engine && renderer) {
      renderer.setDucks(engine.bodies)
      renderer.follow(raceView.follow ? followPoint(engine.bodies, phase, finishTimes) : null)
      view = engine.view()
      if (view) {
        renderer.updatePositions(view.pos, view.n)
        writeColors(getColorMode(raceView.colorMode), view, raceParams, renderer.colors)
        renderer.commitColors()
      }
      renderer.render(raceView.style === 'water')
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
        standings: standings(engine?.bodies ?? [], finishTimes),
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
    reset() {
      start()
    },
    setPaused(p) {
      paused = p
    },
    startRace(count) {
      if (!engine) return
      engine.removeBody()
      const line = startLine(count)
      for (const center of line.positions) engine.addBody({ center, size: line.size })
      // Held at the gate (kinematic at their spot) until the countdown ends.
      for (const o of engine.bodies) o.grabbed = true
      renderer?.setGate(line.gate)
      phase = 'countdown'
      countdown = COUNTDOWN
      finishTimes = engine.bodies.map(() => null)
    },
    dropDuck() {
      // Just downstream of the inlet, above the upper reach, on the centreline.
      const x = 3
      const y = riverLayout.ledgeHeight + riverLayout.upperDepth + raceView.duckSize[1] + 1
      if (engine?.addBody({ center: [x, y, courseOf(riverLayout).centerline(x)], size: raceView.duckSize })) {
        finishTimes.push(null) // a late entrant still races
      }
    },
    clearDucks() {
      engine?.removeBody()
      renderer?.setGate(null)
      phase = 'idle'
      finishTimes = []
    },
    dispose() {
      generation++
      cancelAnimationFrame(raf)
      engine?.dispose()
      renderer?.dispose()
    },
  }
}
