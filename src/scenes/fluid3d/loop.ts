import { defaultParams3D, particleRadius, type Params } from '../../sim/params'
import { createState, step, type SimState } from '../../sim/sph'
import { Renderer3D } from '../../render/renderer3d'
import { getColorMode, writeColors, type ColorModeId } from '../../render/colorModes'

export interface Stats3D {
  fps: number
  particles: number
  substeps: number // average substeps per frame
  stepMs: number // average solver time per substep
  densityAvg: number // ρ/ρ₀
  neighborsAvg: number
  paused: boolean
}

export interface SimController3D {
  reset(): void // rebuild particles and renderer from the current params
  setPaused(paused: boolean): void
  dispose(): void
}

// The 3D scene's own mutable params and view settings (the 2D sandbox keeps
// separate ones), so the two scenes never fight over shared state.
export const params3d: Params = structuredClone(defaultParams3D)
export const view3d = { colorMode: 'speed' as ColorModeId }

const STATS_INTERVAL_MS = 250 // ~4 Hz

// Mounts the 3D renderer into `container` and runs the fixed-timestep loop, the
// same way as the 2D sandbox. Never touches React state; stats go through
// `onStats` at ~4 Hz.
export function startLoop3D(container: HTMLElement, onStats: (s: Stats3D) => void): SimController3D {
  const createRenderer = (n: number) => new Renderer3D(container, n, particleRadius(params3d), params3d.domain)
  let state = createState(params3d)
  let renderer = createRenderer(state.n)
  let paused = false

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let stepTime = 0
  let statsStart = last

  const frame = (now: number) => {
    // The rAF timestamp can precede the performance.now() taken at startup, hence max(0).
    accumulator += Math.min(Math.max((now - last) / 1000, 0), 0.1)
    last = now

    let substeps = 0
    if (paused) {
      accumulator = 0
    } else {
      const t0 = performance.now()
      while (accumulator >= params3d.dt && substeps < params3d.maxSubsteps) {
        step(state, params3d)
        accumulator -= params3d.dt
        substeps++
      }
      stepTime += performance.now() - t0
      // Hit the cap: drop the backlog rather than spiral (sim runs slower than real time).
      if (substeps === params3d.maxSubsteps) accumulator = Math.min(accumulator, params3d.dt)
    }

    renderer.updatePositions(state.pos, state.n)
    writeColors(getColorMode(view3d.colorMode), state, params3d, renderer.colors)
    renderer.commitColors()
    renderer.render()

    frames++
    substepTotal += substeps
    if (now - statsStart >= STATS_INTERVAL_MS) {
      onStats({
        fps: (frames * 1000) / (now - statsStart),
        particles: state.n,
        substeps: substepTotal / frames,
        stepMs: substepTotal > 0 ? stepTime / substepTotal : 0,
        ...fieldStats(state, params3d),
        paused,
      })
      frames = 0
      substepTotal = 0
      stepTime = 0
      statsStart = now
    }

    raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)

  return {
    reset() {
      renderer.dispose()
      state = createState(params3d)
      renderer = createRenderer(state.n)
      accumulator = 0
    },
    setPaused(p) {
      paused = p
    },
    dispose() {
      cancelAnimationFrame(raf)
      renderer.dispose()
    },
  }
}

function fieldStats(s: SimState, p: Params) {
  let sum = 0
  let neighbors = 0
  for (let i = 0; i < s.n; i++) {
    sum += s.density[i] / p.restDensity
    neighbors += s.neighborCount[i]
  }
  return { densityAvg: sum / s.n, neighborsAvg: neighbors / s.n }
}
