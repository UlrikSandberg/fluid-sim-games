import { params } from './sim/params'
import { createState, step, type SimState } from './sim/sph'
import { Renderer } from './render/renderer'
import { getColorMode, view, writeColors } from './render/colorModes'

export interface Stats {
  fps: number
  particles: number
  substeps: number // average substeps per frame
  densityAvg: number // ρ/ρ₀
  densityMax: number // ρ/ρ₀
  neighborsAvg: number
  paused: boolean
}

export interface SimController {
  reset(): void // rebuild particles and renderer from the current params
  setPaused(paused: boolean): void
  stepOnce(): void // advance one substep (meant for use while paused)
  dispose(): void
}

const STATS_INTERVAL_MS = 250 // ~4 Hz

// Mounts the renderer into `container` and runs the fixed-timestep loop.
// Never touches React state; stats are pushed through `onStats` at ~4 Hz.
export function startLoop(container: HTMLElement, onStats: (s: Stats) => void): SimController {
  const createRenderer = (n: number) => new Renderer(container, n, params.spacing * 0.5, params.domain)
  let state = createState(params)
  let renderer = createRenderer(state.n)
  let paused = false
  let pendingSteps = 0

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let statsStart = last

  const frame = (now: number) => {
    // Clamp so a backgrounded tab doesn't come back with a huge backlog. The rAF
    // timestamp can precede the performance.now() taken at startup, hence max(0).
    accumulator += Math.min(Math.max((now - last) / 1000, 0), 0.1)
    last = now

    let substeps = 0
    if (paused) {
      accumulator = 0
      for (; pendingSteps > 0; pendingSteps--) step(state, params)
    } else {
      while (accumulator >= params.dt && substeps < params.maxSubsteps) {
        step(state, params)
        accumulator -= params.dt
        substeps++
      }
      // Hit the cap: drop the backlog rather than spiral (sim runs slower than real time).
      if (substeps === params.maxSubsteps) accumulator = Math.min(accumulator, params.dt)
    }

    renderer.updatePositions(state.pos, state.n)
    writeColors(getColorMode(view.colorMode), state, params, renderer.colors)
    renderer.commitColors()
    renderer.render()

    frames++
    substepTotal += substeps
    if (now - statsStart >= STATS_INTERVAL_MS) {
      onStats({
        fps: (frames * 1000) / (now - statsStart),
        particles: state.n,
        substeps: substepTotal / frames,
        ...fieldStats(state),
        paused,
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
      renderer.dispose()
      state = createState(params)
      renderer = createRenderer(state.n)
      accumulator = 0
    },
    setPaused(p) {
      paused = p
    },
    stepOnce() {
      if (paused) pendingSteps++
    },
    dispose() {
      cancelAnimationFrame(raf)
      renderer.dispose()
    },
  }
}

function fieldStats(s: SimState) {
  let sum = 0
  let max = 0
  let neighbors = 0
  for (let i = 0; i < s.n; i++) {
    const r = s.density[i] / params.restDensity
    sum += r
    if (r > max) max = r
    neighbors += s.neighborCount[i]
  }
  return { densityAvg: sum / s.n, densityMax: max, neighborsAvg: neighbors / s.n }
}
