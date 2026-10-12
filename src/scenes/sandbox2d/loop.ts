import { dimOf, mouseAccel, params, particleRadius } from '../../sim/params'
import { createState, obstacleAngle, obstacleContains, setObstacle, step, type MouseInput, type SimState } from '../../sim/sph'
import { Renderer } from '../../render/renderer'
import { getColorMode, view, writeColors } from '../../render/colorModes'

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
  brushMode: BrushMode
}

export type BrushMode = 'repel' | 'attract' | 'current'
export const BRUSH_MODES: readonly BrushMode[] = ['repel', 'attract', 'current']

// Mutable brush settings, like `params`: the panel and keyboard write, the loop reads.
// The mode is what a plain drag does; right drag or shift+drag does the opposite
// (for 'current', the reverse direction).
export const brush = { mode: 'repel' as BrushMode }

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

  let appliedSize = params.obstacleSize.slice()
  const placeObstacle = (center: number[] | null) => {
    appliedSize = params.obstacleSize.slice()
    setObstacle(state, params, center ? { center, size: appliedSize } : null)
  }
  const obstacle: ObstacleInput = {
    hit: (pt) => state.obstacle !== null && obstacleContains(state.obstacle, pt),
    center: () => state.obstacle?.center ?? null,
    place: (pt) => placeObstacle(pt),
    remove: () => placeObstacle(null),
    // While grabbed the body is kinematic (follows the mouse, speed-capped); on
    // release a floating body keeps that velocity, so it can be thrown.
    drag: (target) => {
      if (!state.obstacle) return
      state.obstacle.grabbed = true
      for (let d = 0; d < target.length; d++) state.obstacle.target[d] = target[d]
    },
    release: () => {
      if (state.obstacle) state.obstacle.grabbed = false
    },
  }

  const mouse: MouseInput = {
    active: false,
    kind: 'radial',
    pos: new Array(dimOf(params)).fill(0),
    sign: 1,
    arrow: new Array(dimOf(params)).fill(0),
  }
  const input = attachMouse(container, mouse, (x, y) => renderer.screenToWorld(x, y), obstacle)

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let statsStart = last

  const frame = (now: number) => {
    input.update()
    if (state.obstacle && params.obstacleSize.some((w, d) => w !== appliedSize[d])) placeObstacle(state.obstacle.center)

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
    renderer.setBrush(
      (input.hovering && !input.grabbing && !obstacle.hit(mouse.pos)) || mouse.active ? mouse.pos : null,
      params.brushRadius,
      mouse.kind === 'current' ? 'current' : mouse.sign > 0 ? 'repel' : 'attract',
      mouse.active,
    )
    renderer.setArrow(mouse.active && mouse.kind === 'current' ? mouse.pos : null, mouse.arrow, mouse.sign)
    renderer.setObstacle(state.obstacle?.center ?? null, state.obstacle?.halfSize ?? [], state.obstacle ? obstacleAngle(state.obstacle) : 0)
    renderer.render(view.style === 'water')

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
        brushMode: brush.mode,
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
      const center = state.obstacle?.center ?? null
      renderer.dispose()
      state = createState(params)
      if (center) placeObstacle(center)
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
      input.detach()
      renderer.dispose()
    },
  }
}

const BRUSH_MIN = 0.5
const BRUSH_MAX = 15

// What the mouse handler can ask of and do to the obstacle.
interface ObstacleInput {
  hit(pt: number[]): boolean
  center(): number[] | null
  place(pt: number[]): void
  remove(): void
  drag(target: number[]): void
  release(): void
}

// A plain drag applies the brush mode; right drag or shift+drag applies the
// opposite. In 'current' mode the press point anchors the brush and the drag
// draws the arrow from it. The wheel resizes the brush. Ctrl+click places the
// obstacle, a left drag starting inside it moves it, a right click on it deletes
// it. Writes into `mouse`,
// which the loop hands to every step; the loop calls update() each frame so
// switching mode or pressing shift mid-drag takes effect without moving the mouse.
function attachMouse(
  el: HTMLElement,
  mouse: MouseInput,
  toWorld: (clientX: number, clientY: number) => number[],
  obstacle: ObstacleInput,
) {
  let dragButton = -1
  let shift = false
  const cursor = mouse.pos.slice()
  const anchor = mouse.pos.slice()
  const grabOffset = mouse.pos.slice()
  const state = {
    hovering: false,
    grabbing: false,
    update() {
      if (state.grabbing) obstacle.drag(cursor.map((c, d) => c + grabOffset[d]))
      const flip = dragButton === 2 || shift
      mouse.kind = brush.mode === 'current' ? 'current' : 'radial'
      mouse.sign = (flip ? -1 : 1) * (brush.mode === 'attract' ? -1 : 1)
      const anchored = mouse.active && mouse.kind === 'current'
      for (let d = 0; d < cursor.length; d++) {
        mouse.pos[d] = anchored ? anchor[d] : cursor[d]
        mouse.arrow[d] = anchored ? cursor[d] - anchor[d] : 0
      }
    },
    detach: () => {},
  }
  const track = (e: PointerEvent) => {
    shift = e.shiftKey
    const w = toWorld(e.clientX, e.clientY)
    for (let d = 0; d < w.length; d++) cursor[d] = w[d]
  }
  const down = (e: PointerEvent) => {
    if (e.button !== 0 && e.button !== 2) return
    e.preventDefault()
    track(e)
    if (e.ctrlKey) {
      obstacle.place(cursor)
      return
    }
    const center = obstacle.hit(cursor) ? obstacle.center() : null
    if (center && e.button === 2) {
      obstacle.remove()
      return
    }
    dragButton = e.button
    if (center) {
      for (let d = 0; d < cursor.length; d++) grabOffset[d] = center[d] - cursor[d]
      state.grabbing = true
    } else {
      for (let d = 0; d < cursor.length; d++) anchor[d] = cursor[d]
      mouse.active = true
    }
    el.setPointerCapture(e.pointerId)
  }
  const move = (e: PointerEvent) => {
    track(e)
    state.hovering = true
  }
  const up = (e: PointerEvent) => {
    // pointercancel reports button −1, so it always ends the drag.
    if (e.type === 'pointerup' && e.button !== dragButton) return
    mouse.active = false
    if (state.grabbing) obstacle.release()
    state.grabbing = false
    dragButton = -1
    if (el.hasPointerCapture(e.pointerId)) el.releasePointerCapture(e.pointerId)
  }
  const leave = () => (state.hovering = false)
  const shiftKey = (e: KeyboardEvent) => (shift = e.shiftKey)
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
  window.addEventListener('keydown', shiftKey)
  window.addEventListener('keyup', shiftKey)
  state.detach = () => {
    el.removeEventListener('pointerdown', down)
    el.removeEventListener('pointermove', move)
    el.removeEventListener('pointerup', up)
    el.removeEventListener('pointercancel', up)
    el.removeEventListener('pointerleave', leave)
    el.removeEventListener('wheel', wheel)
    el.removeEventListener('contextmenu', noMenu)
    window.removeEventListener('keydown', shiftKey)
    window.removeEventListener('keyup', shiftKey)
  }
  return state
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
