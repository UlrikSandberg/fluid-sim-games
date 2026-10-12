import { defaultParams3D, particleRadius, type Params } from '../../sim/params'
import { createState, step, type SimState } from '../../sim/sph'
import { advanceTank, createTankMotion, levelTarget, nudgeTarget } from '../../sim/tankMotion'
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
  levelTank(): void
  dispose(): void
}

export type DragMode = 'orbit' | 'tank'

// The 3D scene's own mutable params and view settings (the 2D sandbox keeps
// separate ones), so the two scenes never fight over shared state.
export const params3d: Params = structuredClone(defaultParams3D)
export const view3d = { colorMode: 'speed' as ColorModeId, drag: 'orbit' as DragMode }
// Gravity's magnitude; its direction in tank coordinates follows the tank's tilt,
// so params3d.gravity is rewritten every frame.
export const tankGravity = { value: -defaultParams3D.gravity[1] }

const ROTATE_PER_PIXEL = 0.008 // rad of tank rotation per pixel dragged

const STATS_INTERVAL_MS = 250 // ~4 Hz

// Mounts the 3D renderer into `container` and runs the fixed-timestep loop, the
// same way as the 2D sandbox. Never touches React state; stats go through
// `onStats` at ~4 Hz.
export function startLoop3D(container: HTMLElement, onStats: (s: Stats3D) => void): SimController3D {
  const createRenderer = (n: number) => new Renderer3D(container, n, particleRadius(params3d), params3d.domain)
  let state = createState(params3d)
  let renderer = createRenderer(state.n)
  let paused = false
  const tank = createTankMotion()
  const pivot = () => params3d.domain.min.map((v, d) => (v + params3d.domain.max[d]) / 2)
  const detachDrag = attachTankDrag(container, (dx, dy) => {
    // Drag right: roll clockwise as seen; drag down: tip the top toward the viewer.
    const { right, forward } = renderer.cameraAxes()
    nudgeTarget(tank, forward, dx * ROTATE_PER_PIXEL)
    nudgeTarget(tank, right, dy * ROTATE_PER_PIXEL)
  })

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
      substeps = Math.min(Math.floor(accumulator / params3d.dt), params3d.maxSubsteps)
      accumulator -= substeps * params3d.dt
      // Hit the cap: drop the backlog rather than spiral (sim runs slower than real time).
      if (substeps === params3d.maxSubsteps) accumulator = Math.min(accumulator, params3d.dt)
      // Turn the tank by the simulated time these substeps cover; they all see
      // the same frame motion and the gravity rotated into the tank.
      const motion = advanceTank(tank, substeps * params3d.dt, pivot(), tankGravity.value)
      params3d.gravity = motion.gravity
      const t0 = performance.now()
      for (let k = 0; k < substeps; k++) step(state, params3d, undefined, motion.frame)
      stepTime += performance.now() - t0
    }

    renderer.setTankOrientation(tank.q)
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
    levelTank() {
      levelTarget(tank)
    },
    dispose() {
      cancelAnimationFrame(raf)
      detachDrag()
      renderer.dispose()
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
