import { DIM, mouseAccel, params, particleRadius } from './sim/params'
import { createState, step, type MouseInput, type SimState } from './sim/sph'
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
  brushRadius: number
  brushAccel: number // peak brush acceleration (depends on radius, so it changes with the wheel)
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
  const createRenderer = (n: number) => new Renderer(container, n, particleRadius(params), params.domain)
  let state = createState(params)
  let renderer = createRenderer(state.n)
  let paused = false
  let pendingSteps = 0

  const mouse: MouseInput = { active: false, pos: new Array(DIM).fill(0), sign: 1 }
  let hovering = false
  const detachInput = attachMouse(container, mouse, (x, y) => renderer.screenToWorld(x, y), (h) => (hovering = h))

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
      for (; pendingSteps > 0; pendingSteps--) step(state, params, mouse)
    } else {
      while (accumulator >= params.dt && substeps < params.maxSubsteps) {
        step(state, params, mouse)
        accumulator -= params.dt
        substeps++
      }
      // Hit the cap: drop the backlog rather than spiral (sim runs slower than real time).
      if (substeps === params.maxSubsteps) accumulator = Math.min(accumulator, params.dt)
    }

    renderer.updatePositions(state.pos, state.n)
    writeColors(getColorMode(view.colorMode), state, params, renderer.colors)
    renderer.commitColors()
    renderer.setBrush(hovering || mouse.active ? mouse.pos : null, params.brushRadius, mouse.active ? mouse.sign : 0)
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
        brushRadius: params.brushRadius,
        brushAccel: mouseAccel(params),
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
      detachInput()
      renderer.dispose()
    },
  }
}

const BRUSH_MIN = 0.5
const BRUSH_MAX = 15

// Left drag repels, right drag or shift+drag attracts, wheel resizes the brush.
// Writes into `mouse`, which the loop hands to every step. Returns a detach function.
function attachMouse(
  el: HTMLElement,
  mouse: MouseInput,
  toWorld: (clientX: number, clientY: number) => number[],
  setHovering: (hovering: boolean) => void,
): () => void {
  let dragButton = -1
  const track = (e: PointerEvent) => {
    const w = toWorld(e.clientX, e.clientY)
    for (let d = 0; d < w.length; d++) mouse.pos[d] = w[d]
  }
  const down = (e: PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return
    e.preventDefault()
    dragButton = e.button
    track(e)
    mouse.sign = e.button === 2 || e.shiftKey ? -1 : 1
    mouse.active = true
    el.setPointerCapture(e.pointerId)
  }
  const move = (e: PointerEvent) => {
    track(e)
    setHovering(true)
    if (mouse.active && dragButton === 0) mouse.sign = e.shiftKey ? -1 : 1
  }
  const up = (e: PointerEvent) => {
    // pointercancel reports button −1, so it always ends the drag.
    if (e.type === 'pointerup' && e.button !== dragButton) return
    mouse.active = false
    dragButton = -1
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
  }
  const leave = () => setHovering(false)
  const wheel = (e: WheelEvent) => {
    e.preventDefault()
    const r = params.brushRadius * Math.exp(-e.deltaY * 0.002)
    params.brushRadius = Math.min(BRUSH_MAX, Math.max(BRUSH_MIN, r))
  }
  const noMenu = (e: Event) => e.preventDefault()

  el.addEventListener('pointerdown', down)
  el.addEventListener('pointermove', move)
  el.addEventListener('pointerup', up)
  el.addEventListener('pointercancel', up)
  el.addEventListener('pointerleave', leave)
  el.addEventListener('wheel', wheel, { passive: false })
  el.addEventListener('contextmenu', noMenu)
  return () => {
    el.removeEventListener('pointerdown', down)
    el.removeEventListener('pointermove', move)
    el.removeEventListener('pointerup', up)
    el.removeEventListener('pointercancel', up)
    el.removeEventListener('pointerleave', leave)
    el.removeEventListener('wheel', wheel)
    el.removeEventListener('contextmenu', noMenu)
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
