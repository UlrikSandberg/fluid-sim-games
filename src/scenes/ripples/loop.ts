import * as THREE from 'three'
import { drop, maxDt, oscillate, step, type ShallowGrid } from '../../sim/shallow/shallow'
import { HeightfieldWater, type WaterLook } from '../../render/heightfieldWater'
import { createView3D } from '../../render/view3d'
import { TANK, buildTank, type PresetId } from './presets'

// The ripple tank's loop: steps the shallow-water solver in real time, turns the
// pointer into drops and a bobbing finger, drives the wave maker, and renders
// with HeightfieldWater.

export const rippleView = {
  preset: 'double' as PresetId,
  absorb: true, // sponges on the far walls (open water) or a closed pool
  waveMaker: true,
  freq: 1.2, // Hz, for the wave maker and the finger
  amplitude: 0.5, // 0…1
  look: 'real' as WaterLook,
  paused: false,
}

export interface RippleStats {
  fps: number
  steps: number // per frame
  stepMs: number
  grid: string
  speed: number // wave speed √(g·h), m/s
  wavelength: number // m, at the current frequency
}

export interface RippleController {
  reset(): void // rebuild the current preset
  dispose(): void
}

const MAX_WAVE = 0.06 // m: the wave maker's crest height at full amplitude
const FINGER_RADIUS = 0.3

export function startRippleLoop(host: HTMLElement, onStats: (s: RippleStats) => void): RippleController {
  let water: HeightfieldWater | null = null
  const view = createView3D(host, { fov: 40, near: 0.1, far: 1000, onResize: (gl) => water?.resize(gl) })
  const { gl, camera, controls } = view
  gl.toneMapping = THREE.ACESFilmicToneMapping
  // Left drag pokes the water; right drag (or Shift + left) orbits; wheel zooms.
  // On touch, one finger pokes and two orbit and zoom.
  controls.mouseButtons = { LEFT: null, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.ROTATE }
  controls.touches = { ONE: null, TWO: THREE.TOUCH.DOLLY_ROTATE }
  controls.maxPolarAngle = Math.PI * 0.47
  controls.minDistance = 4
  controls.maxDistance = 70
  controls.target.set(TANK.width / 2, TANK.depth, TANK.length / 2)
  camera.position.set(TANK.width / 2 + 2, 15, TANK.length + 13)

  const scene = new THREE.Scene()
  scene.add(new THREE.HemisphereLight(0xdfeeff, 0x6b6150, 1.2))
  const sun = new THREE.DirectionalLight(0xfff4e0, 2.2)
  scene.add(sun)

  let grid: ShallowGrid
  let level = TANK.depth
  let paddle = new Float32Array(1)
  const props = new THREE.Group() // walls, rim, deck: rebuilt with the preset
  scene.add(props)
  const materials = makeMaterials()
  let paddleMesh: THREE.Mesh

  const build = () => {
    const setup = buildTank(rippleView.preset, rippleView.absorb)
    grid = setup.grid
    level = setup.level
    paddle = new Float32Array(grid.ny)
    if (!water) {
      water = new HeightfieldWater(gl, grid)
      scene.add(water.sky, water.floor, water.water)
      sun.position.copy(water.sunDir).multiplyScalar(50)
    } else {
      water.setBed(grid)
    }
    for (const c of [...props.children]) {
      props.remove(c)
      ;(c as THREE.Mesh).geometry?.dispose()
    }
    paddleMesh = buildProps(props, grid, level, materials)
    water.update(grid, 0)
  }
  build()

  // --- Pointer: poke on press, a bobbing finger while held (moving it makes a wake). ---
  const surface = new THREE.Plane(new THREE.Vector3(0, 1, 0), -TANK.depth)
  const ray = new THREE.Raycaster()
  let finger: { x: number; y: number } | null = null
  const toTank = (e: PointerEvent) => {
    ray.setFromCamera(view.ndc(e), camera)
    const hit = ray.ray.intersectPlane(surface, new THREE.Vector3())
    if (!hit || hit.x < 0 || hit.z < 0 || hit.x > grid.nx * grid.dx || hit.z > grid.ny * grid.dx) return null
    return { x: hit.x, y: hit.z }
  }
  const onDown = (e: PointerEvent) => {
    if (e.button !== 0 || e.shiftKey) return
    const p = toTank(e)
    if (!p) return
    gl.domElement.setPointerCapture(e.pointerId)
    finger = p
    drop(grid, p.x, p.y, FINGER_RADIUS * 1.3, 0.04 + 0.12 * rippleView.amplitude)
  }
  const onMove = (e: PointerEvent) => {
    if (finger) finger = toTank(e) ?? finger
  }
  const onUp = () => (finger = null)
  // Shift turns the left button into orbiting.
  const onKey = (e: KeyboardEvent) => (controls.mouseButtons.LEFT = e.shiftKey ? THREE.MOUSE.ROTATE : null)
  gl.domElement.addEventListener('pointerdown', onDown)
  gl.domElement.addEventListener('pointermove', onMove)
  gl.domElement.addEventListener('pointerup', onUp)
  gl.domElement.addEventListener('pointercancel', onUp)
  window.addEventListener('keydown', onKey)
  window.addEventListener('keyup', onKey)

  if (import.meta.env.DEV) Object.assign(window, { __ripples: { camera, controls, grid: () => grid } })

  // --- Frame loop ---
  let raf = 0
  let last = performance.now()
  let fpsAvg = 60
  let stepMsAvg = 0
  let statsAt = 0
  let elapsed = 0
  let waveTime = 0 // how long the wave maker has run, for its gentle start
  const frame = (now: number) => {
    raf = requestAnimationFrame(frame)
    const frameDt = Math.min((now - last) / 1000, 1 / 30)
    last = now
    fpsAvg += (1 / Math.max(frameDt, 1e-3) - fpsAvg) * 0.05
    let steps = 0
    if (!rippleView.paused) {
      const t0 = performance.now()
      steps = Math.min(8, Math.ceil(frameDt / maxDt(grid)))
      const dt = frameDt / steps
      // The paddle's stroke for a crest of the chosen height (η = u·h / c), for drawing.
      const c = Math.sqrt(grid.g * level)
      const paddleAmp = (MAX_WAVE * rippleView.amplitude * c) / level
      if (rippleView.waveMaker) waveTime += frameDt
      else waveTime = 0
      // The paddle, drawn where its motion has taken it (exaggerated 4×).
      const w = 2 * Math.PI * rippleView.freq
      paddleMesh.visible = rippleView.waveMaker
      paddleMesh.position.x = 0.12 - 4 * ((paddleAmp * Math.min(1, waveTime / 2)) / w) * Math.cos(w * grid.time)
      for (let k = 0; k < steps; k++) {
        if (rippleView.waveMaker) absorbingPaddle(grid, level, MAX_WAVE * rippleView.amplitude * Math.min(1, waveTime / 2), rippleView.freq, paddle)
        else paddle.fill(0)
        if (finger) oscillate(grid, finger.x, finger.y, FINGER_RADIUS, 0.03 * rippleView.amplitude, rippleView.freq * 2, dt)
        step(grid, dt, { paddle, foamRate: 1, foamLife: 2.5 })
      }
      stepMsAvg += ((performance.now() - t0) / steps - stepMsAvg) * 0.1
      elapsed += frameDt
    }
    const w = water!
    w.update(grid, elapsed)
    w.setLook(rippleView.look, level, Math.max(0.006, 0.5 * MAX_WAVE * rippleView.amplitude))
    controls.update()
    w.render(gl, scene, camera)

    if (now - statsAt > 250) {
      statsAt = now
      const speed = Math.sqrt(grid.g * level)
      onStats({ fps: fpsAvg, steps, stepMs: stepMsAvg, grid: `${grid.nx}×${grid.ny}`, speed, wavelength: speed / rippleView.freq })
    }
  }
  raf = requestAnimationFrame(frame)

  return {
    reset: build,
    dispose() {
      cancelAnimationFrame(raf)
      gl.domElement.removeEventListener('pointerdown', onDown)
      gl.domElement.removeEventListener('pointermove', onMove)
      gl.domElement.removeEventListener('pointerup', onUp)
      gl.domElement.removeEventListener('pointercancel', onUp)
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKey)
      for (const c of props.children) {
        const m = c as THREE.Mesh
        m.geometry?.dispose()
      }
      for (const m of Object.values(materials)) m.dispose()
      water?.dispose()
      view.dispose()
    },
  }
}

// An absorbing wave maker: each row of the paddle moves to make the wave we want
// (η_in) and, at the same time, to swallow whatever comes back to it, by driving
// the water at the paddle toward 2·η_in − η (the shallow-water radiation
// condition, u = (c/h)·(2·η_in − η)). Without it, waves bouncing back from a
// wall would pile up between the wall and the paddle.
function absorbingPaddle(grid: ShallowGrid, level: number, crest: number, freq: number, out: Float32Array) {
  const c = Math.sqrt(grid.g * level)
  const wanted = crest * Math.sin(2 * Math.PI * freq * grid.time)
  for (let j = 0; j < grid.ny; j++) {
    const k = j * grid.nx
    const eta = grid.h[k] + grid.bed[k] - level
    out[j] = (c / level) * (2 * wanted - eta)
  }
}

const makeMaterials = () => ({
  stone: new THREE.MeshStandardMaterial({ color: 0x9c958a, roughness: 0.85 }),
  rim: new THREE.MeshStandardMaterial({ color: 0xe9e4d8, roughness: 0.6 }),
  deck: new THREE.MeshStandardMaterial({ color: 0xc8b89a, roughness: 0.9 }),
  paddle: new THREE.MeshStandardMaterial({ color: 0xf2c230, roughness: 0.5 }),
})
type Materials = ReturnType<typeof makeMaterials>

// The pool around the water: its walls and coping, a deck, the slit walls (one
// box per run of solid cells along each column), and the wave maker's paddle,
// which it returns.
function buildProps(group: THREE.Group, grid: ShallowGrid, level: number, materials: Materials): THREE.Mesh {
  const { nx, ny, dx, solid } = grid
  const W = nx * dx
  const L = ny * dx
  const top = level + 0.35
  const box = (w: number, h: number, d: number, x: number, y: number, z: number, mat: THREE.Material) => {
    const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), mat)
    m.position.set(x, y, z)
    group.add(m)
    return m
  }
  const t = 0.3 // wall thickness
  box(W + 2 * t, top + 0.2, t, W / 2, (top - 0.2) / 2, -t / 2, materials.rim)
  box(W + 2 * t, top + 0.2, t, W / 2, (top - 0.2) / 2, L + t / 2, materials.rim)
  box(t, top + 0.2, L, -t / 2, (top - 0.2) / 2, L / 2, materials.rim)
  box(t, top + 0.2, L, W + t / 2, (top - 0.2) / 2, L / 2, materials.rim)
  // The deck, a ring of slabs around the pool.
  const D = 8
  box(W + 2 * t + 2 * D, 0.2, D, W / 2, top - 0.1, -t - D / 2, materials.deck)
  box(W + 2 * t + 2 * D, 0.2, D, W / 2, top - 0.1, L + t + D / 2, materials.deck)
  box(D, 0.2, L + 2 * t, -t - D / 2, top - 0.1, L / 2, materials.deck)
  box(D, 0.2, L + 2 * t, W + t + D / 2, top - 0.1, L / 2, materials.deck)
  // The paddle along the left wall.
  const paddle = box(0.12, 0.6, L - 0.2, 0.12, level, L / 2, materials.paddle)
  // Slit walls.
  for (let i = 0; i < nx; i++) {
    let j = 0
    while (j < ny) {
      if (!solid[j * nx + i]) {
        j++
        continue
      }
      const j0 = j
      while (j < ny && solid[j * nx + i]) j++
      box(dx, top, (j - j0) * dx, (i + 0.5) * dx, top / 2, ((j0 + j) / 2) * dx, materials.stone)
    }
  }
  return paddle
}
