import { particleRadius, type Params } from '../../sim/params'
import type { FrameMotion, Obstacle, SimState } from '../../sim/sph'
import { rotate } from '../../sim/quat'
import { Renderer3D } from '../../render/renderer3d'
import { getColorMode, writeColors, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { GpuEngine } from '../fluid3d/engines'
import {
  DEFAULT_CURRENT, bankLines, courseOf, riverBoxes, riverLayout, riverParams, riverScene,
} from './riverScene'

export interface RaceStats {
  fps: number
  particles: number
  substeps: number
  ducks: number
  status: 'starting' | 'running' | 'paused' | 'no-webgpu'
}

export interface RaceController {
  reset(): void // rebuild the course from riverLayout
  setPaused(paused: boolean): void
  startRace(count: number): void // clear the ducks and line up `count` at the inlet
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
}

const STATS_INTERVAL_MS = 250

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

// Starting positions for `count` ducks: rows of lanes across the channel just
// downstream of the inlet, dropped a little above the water, spaced so their
// collision spheres don't touch.
function startingGrid(count: number): number[][] {
  const c = courseOf(riverLayout)
  const size = raceView.duckSize
  const gap = 2 * Math.max(...size.map((s) => s / 2)) + 0.4
  const perRow = Math.max(1, Math.floor((2 * c.halfWidth - 1) / gap))
  const y = riverLayout.ledgeHeight + riverLayout.upperDepth + size[1] / 2 + 0.5
  const out: number[][] = []
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / perRow)
    const inRow = Math.min(perRow, count - row * perRow)
    const lane = i % perRow
    const x = 2 + gap * row
    out.push([x, y, c.centerline(x) + (lane - (inRow - 1) / 2) * gap])
  }
  return out
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

  const start = () => {
    const mine = ++generation
    engine?.dispose()
    renderer?.dispose()
    engine = null
    renderer = null
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
    }

    if (engine && renderer) {
      renderer.setDucks(engine.bodies)
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
      for (const center of startingGrid(count)) engine.addBody({ center, size: raceView.duckSize })
    },
    dropDuck() {
      // Just downstream of the inlet, above the upper reach, on the centreline.
      const x = 3
      const y = riverLayout.ledgeHeight + riverLayout.upperDepth + raceView.duckSize[1] + 1
      engine?.addBody({ center: [x, y, courseOf(riverLayout).centerline(x)], size: raceView.duckSize })
    },
    clearDucks() {
      engine?.removeBody()
    },
    dispose() {
      generation++
      cancelAnimationFrame(raf)
      engine?.dispose()
      renderer?.dispose()
    },
  }
}
