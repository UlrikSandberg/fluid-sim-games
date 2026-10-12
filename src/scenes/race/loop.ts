import { particleRadius } from '../../sim/params'
import type { FrameMotion, SimState } from '../../sim/sph'
import { Renderer3D } from '../../render/renderer3d'
import { getColorMode, writeColors, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { GpuEngine } from '../fluid3d/engines'
import { RIVER, riverBoxes, riverParams, riverScene } from './riverScene'

export interface RaceStats {
  fps: number
  particles: number
  substeps: number
  status: 'starting' | 'running' | 'paused' | 'no-webgpu'
}

export interface RaceController {
  reset(): void
  setPaused(paused: boolean): void
  dropDuck(): void
  setCurrent(speed: number): void
  dispose(): void
}

export const raceView = { colorMode: 'water' as ColorModeId, style: 'water' as RenderStyle, current: RIVER.current.speed }

const STATS_INTERVAL_MS = 250
const DUCK_SIZE = [1.6, 1, 1.2]
const CAMERA_OFFSET = [-4, 26, 54] // from the course's centre: in front and above, the whole course in view

// The river race's loop: the GPU solver running the river scene (a ledge,
// a current, recycling), rendered with the shared 3D renderer. GPU only: a river
// this size isn't playable on the CPU solver.
export function startRaceLoop(container: HTMLElement, onStats: (s: RaceStats) => void): RaceController {
  const params = riverParams()
  let engine: GpuEngine | null = null
  let renderer: Renderer3D | null = null
  let generation = 0
  let paused = false
  let noWebGpu = false
  const pivot = params.domain.min.map((v, d) => (v + params.domain.max[d]) / 2)
  const still: FrameMotion = { omega: [0, 0, 0], alpha: [0, 0, 0], pivot }

  const start = () => {
    const mine = ++generation
    engine?.dispose()
    renderer?.dispose()
    engine = null
    renderer = null
    void GpuEngine.create(params, riverScene(params)).then((created) => {
      if (mine !== generation) {
        created?.dispose()
        return
      }
      if (!created) {
        noWebGpu = true
        return
      }
      engine = created
      engine.current.speed = raceView.current
      renderer = new Renderer3D(container, created.n, particleRadius(params), params.domain, CAMERA_OFFSET)
      for (const box of riverBoxes()) renderer.addSolid(box.min, box.max)
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
      substeps = Math.min(Math.floor(accumulator / params.dt), params.maxSubsteps)
      accumulator -= substeps * params.dt
      if (substeps === params.maxSubsteps) accumulator = Math.min(accumulator, params.dt)
      engine.step(substeps, still, [0, -9.8, 0])
    }

    if (engine && renderer) {
      const o = engine.obstacle
      renderer.setBody(o ? o.center : null, o ? o.halfSize : [], o ? o.q : [1, 0, 0, 0])
      view = engine.view()
      if (view) {
        renderer.updatePositions(view.pos, view.n)
        writeColors(getColorMode(raceView.colorMode), view, params, renderer.colors)
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
    dropDuck() {
      // Just upstream, above the upper reach, in the middle of the channel.
      engine?.dropBody({ center: [3, RIVER.ledgeHeight + RIVER.upperDepth + 2, RIVER.width / 2], size: DUCK_SIZE })
    },
    setCurrent(speed) {
      raceView.current = speed
      if (engine) engine.current.speed = speed
    },
    dispose() {
      generation++
      cancelAnimationFrame(raf)
      engine?.dispose()
      renderer?.dispose()
    },
  }
}
