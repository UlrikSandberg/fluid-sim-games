import { defaultParams3D, particleRadius, type Params } from '../../sim/params'
import type { SimState } from '../../sim/sph'
import { advanceTank, createTankMotion, levelTarget, nudgeTarget } from '../../sim/tankMotion'
import { Renderer3D } from '../../render/renderer3d'
import { FoamField } from '../../render/foamField'
import { getColorMode, writeColors, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { CpuEngine, GpuEngine, type EngineKind, type FluidEngine } from '../../engines/sph'

export interface Stats3D {
  fps: number
  particles: number
  substeps: number // average substeps per frame
  stepMs: number // average CPU solver time per substep (0 on the GPU, where it isn't measured)
  densityAvg: number // ρ/ρ₀
  neighborsAvg: number
  paused: boolean
  engine: EngineKind | 'starting'
}

export interface SimController3D {
  reset(): void // rebuild particles and renderer from the current params and engine
  setPaused(paused: boolean): void
  levelTank(): void
  dropBody(): void // a box at the top of the tank, sized from params3d.obstacleSize
  removeBody(): void
  hasBody(): boolean
  dispose(): void
}

export type DragMode = 'orbit' | 'tank'

// Tank presets: the small one is what the CPU solver runs at 60 fps; the large
// and XL ones are for the GPU (10× and ~40× the particles).
export const TANKS = {
  small: { size: [16, 10, 8], particles: 4000 },
  large: { size: [32, 16, 16], particles: 40000 },
  xl: { size: [48, 24, 32], particles: 150000 },
} as const
export type TankSize = keyof typeof TANKS

// The 3D scene's own mutable params and view settings (the 2D sandbox keeps
// separate ones), so the two scenes never fight over shared state.
export const params3d: Params = structuredClone(defaultParams3D)
export const view3d = {
  colorMode: 'water' as ColorModeId,
  style: 'water' as RenderStyle,
  drag: 'orbit' as DragMode,
  engine: 'gpu' as EngineKind, // falls back to the CPU without WebGPU
  tank: 'small' as TankSize,
}
// Gravity's magnitude; its direction in tank coordinates follows the tank's tilt,
// so params3d.gravity is rewritten every frame.
export const tankGravity = { value: -defaultParams3D.gravity[1] }
// Whether WebGPU worked here (null until the first attempt).
export const gpuSupport = { available: null as boolean | null }

// Selects the engine for the next reset (the GPU may still fall back to the CPU).
export function selectEngine(kind: EngineKind) {
  view3d.engine = kind
}

// Applies a tank preset to params3d (takes effect on reset).
export function applyTank(size: TankSize) {
  view3d.tank = size
  params3d.domain = { min: [0, 0, 0], max: [...TANKS[size].size] }
  params3d.particleCount = TANKS[size].particles
}

const ROTATE_PER_PIXEL = 0.008 // rad of tank rotation per pixel dragged

const STATS_INTERVAL_MS = 250 // ~4 Hz

async function createEngine(): Promise<FluidEngine> {
  if (view3d.engine === 'gpu' && gpuSupport.available !== false) {
    const gpu = await GpuEngine.create(params3d).catch((err) => {
      console.warn('WebGPU engine unavailable, using the CPU', err)
      return null
    })
    gpuSupport.available = gpu !== null
    if (gpu) return gpu
    view3d.engine = 'cpu'
  }
  return new CpuEngine(params3d)
}

// Mounts the 3D renderer into `container` and runs the fixed-timestep loop, the
// same way as the 2D sandbox. The solver is a FluidEngine (CPU or GPU), created
// asynchronously; until it's ready the frame just keeps ticking. Never touches
// React state; stats go through `onStats` at ~4 Hz.
export function startLoop3D(container: HTMLElement, onStats: (s: Stats3D) => void): SimController3D {
  let engine: FluidEngine | null = null
  let renderer: Renderer3D | null = null
  let foam: FoamField | null = null
  let generation = 0 // ignores engines that finish starting after a newer reset
  let paused = false
  const tank = createTankMotion()
  const pivot = () => params3d.domain.min.map((v, d) => (v + params3d.domain.max[d]) / 2)
  const detachDrag = attachTankDrag(container, (dx, dy) => {
    if (!renderer) return
    // Drag right: roll clockwise as seen; drag down: tip the top toward the viewer.
    const { right, forward } = renderer.cameraAxes()
    nudgeTarget(tank, forward, dx * ROTATE_PER_PIXEL)
    nudgeTarget(tank, right, dy * ROTATE_PER_PIXEL)
  })

  const start = () => {
    const mine = ++generation
    engine?.dispose()
    renderer?.dispose()
    engine = null
    renderer = null
    void createEngine().then((created) => {
      if (mine !== generation) {
        created.dispose()
        return
      }
      engine = created
      renderer = new Renderer3D(container, created.n, particleRadius(params3d), params3d.domain)
      foam = new FoamField(params3d)
      accumulator = 0
    })
  }
  start()

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let stepTime = 0
  let statsStart = last
  let view: SimState | null = null

  const frame = (now: number) => {
    // The rAF timestamp can precede the performance.now() taken at startup, hence max(0).
    const frameDt = Math.min(Math.max((now - last) / 1000, 0), 0.1)
    accumulator += frameDt
    last = now

    let substeps = 0
    if (paused || !engine) {
      accumulator = 0
    } else {
      substeps = Math.min(Math.floor(accumulator / params3d.dt), params3d.maxSubsteps)
      accumulator -= substeps * params3d.dt
      // Hit the cap: drop the backlog rather than spiral (sim runs slower than real time).
      if (substeps === params3d.maxSubsteps) accumulator = Math.min(accumulator, params3d.dt)
      // Turn the tank by the simulated time these substeps cover; they all see
      // the same frame motion and the gravity rotated into the tank.
      const motion = advanceTank(tank, substeps * params3d.dt, pivot(), tankGravity.value)
      const t0 = performance.now()
      engine.step(substeps, motion.frame, motion.gravity)
      if (engine.kind === 'cpu') stepTime += performance.now() - t0
    }

    if (engine && renderer) {
      renderer.setTankOrientation(tank.q)
      const o = engine.obstacle
      renderer.setBody(o ? o.center : null, o ? o.halfSize : [], o ? o.q : [1, 0, 0, 0])
      // The "Water" colour mode drawn as water: clear, with foam and spray.
      const realistic = view3d.style === 'water' && view3d.colorMode === 'water'
      view = engine.view()
      if (view) {
        renderer.updatePositions(view.pos, view.n)
        if (realistic && foam) {
          foam.update(view, frameDt, renderer.colors)
          renderer.setFoamFlow(foam.flow)
        }
        else writeColors(getColorMode(view3d.colorMode), view, params3d, renderer.colors)
        renderer.commitColors()
      }
      renderer.render(view3d.style === 'water', realistic)
    }

    frames++
    substepTotal += substeps
    if (now - statsStart >= STATS_INTERVAL_MS) {
      onStats({
        fps: (frames * 1000) / (now - statsStart),
        particles: engine?.n ?? 0,
        substeps: substepTotal / frames,
        stepMs: substepTotal > 0 ? stepTime / substepTotal : 0,
        ...(view ? fieldStats(view, params3d) : { densityAvg: 0, neighborsAvg: 0 }),
        paused,
        engine: engine?.kind ?? 'starting',
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
      start()
    },
    setPaused(p) {
      paused = p
    },
    levelTank() {
      levelTarget(tank)
    },
    dropBody() {
      // Centred near the top of the tank (in tank coordinates), clear of the lid.
      const { min, max } = params3d.domain
      const size = params3d.obstacleSize
      const center = min.map((lo, d) => (lo + max[d]) / 2)
      center[1] = max[1] - size[1] / 2 - params3d.h
      engine?.dropBody({ center, size })
    },
    removeBody() {
      engine?.removeBody()
    },
    hasBody() {
      return engine?.obstacle != null
    },
    dispose() {
      generation++
      cancelAnimationFrame(raf)
      detachDrag()
      engine?.dispose()
      renderer?.dispose()
    },
  }
}

// Left drag rotates the tank when the drag mode is 'tank', or with Shift held in
// either mode; otherwise the event goes on to the orbit controls. Listening in
// the capture phase on the container runs before the canvas's own listeners, so
// stopping the event there keeps the camera still while the tank turns.
function attachTankDrag(el: HTMLElement, onDrag: (dx: number, dy: number) => void): () => void {
  let dragging = -1
  let lastX = 0
  let lastY = 0
  const down = (e: PointerEvent) => {
    if (e.button !== 0 || (view3d.drag !== 'tank' && !e.shiftKey)) return
    e.stopPropagation()
    e.preventDefault()
    dragging = e.pointerId
    lastX = e.clientX
    lastY = e.clientY
    el.setPointerCapture(e.pointerId)
  }
  const move = (e: PointerEvent) => {
    if (e.pointerId !== dragging) return
    e.stopPropagation()
    onDrag(e.clientX - lastX, e.clientY - lastY)
    lastX = e.clientX
    lastY = e.clientY
  }
  const up = (e: PointerEvent) => {
    if (e.pointerId !== dragging) return
    e.stopPropagation()
    dragging = -1
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
  }
  el.addEventListener('pointerdown', down, true)
  el.addEventListener('pointermove', move, true)
  el.addEventListener('pointerup', up, true)
  el.addEventListener('pointercancel', up, true)
  return () => {
    el.removeEventListener('pointerdown', down, true)
    el.removeEventListener('pointermove', move, true)
    el.removeEventListener('pointerup', up, true)
    el.removeEventListener('pointercancel', up, true)
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
