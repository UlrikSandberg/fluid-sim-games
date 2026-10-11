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
}

const STATS_INTERVAL_MS = 250 // ~4 Hz

// Mounts the renderer into `container` and runs the fixed-timestep loop.
// Never touches React state; stats are pushed through `onStats` at ~4 Hz.
// Returns a cleanup function.
export function startLoop(container: HTMLElement, onStats: (s: Stats) => void): () => void {
  const state = createState(params)
  const renderer = new Renderer(container, state.n, params.spacing * 0.5, params.domain)

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let statsStart = last

  const frame = (now: number) => {
    // Clamp so a backgrounded tab doesn't come back with a huge backlog.
    accumulator += Math.min((now - last) / 1000, 0.1)
    last = now

    let substeps = 0
    while (accumulator >= params.dt && substeps < params.maxSubsteps) {
      step(state, params)
      accumulator -= params.dt
      substeps++
    }
    // Hit the cap: drop the backlog rather than spiral (sim runs slower than real time).
    if (substeps === params.maxSubsteps) accumulator = Math.min(accumulator, params.dt)

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
      })
      frames = 0
      substepTotal = 0
      statsStart = now
    }

    raf = requestAnimationFrame(frame)
  }
  raf = requestAnimationFrame(frame)

  return () => {
    cancelAnimationFrame(raf)
    renderer.dispose()
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
