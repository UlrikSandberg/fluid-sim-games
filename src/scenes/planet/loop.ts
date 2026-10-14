import * as THREE from 'three'
import { particleRadius, type Params } from '../../sim/params'
import type { FrameMotion, SimState } from '../../sim/sph'
import { GpuEngine } from '../../engines/sph'
import { Renderer3D } from '../../render/renderer3d'
import { getColorMode, writeColors, type ColorModeId } from '../../render/colorModes'
import { defaultPlanet, makeTerrain, planetParams, planetScene, seaRadius, type PlanetLayout } from './planet'
import type { Terrain } from './terrain'

// The water planet's loop: builds the planet on the GPU solver and steps it in
// real time; draws the ground from its height map and the water as water or as
// coloured particles; and runs the tools: sculpting the ground (the mesh follows
// the brush live, the physics catches up a few times a second while you hold it
// and when you let go) and pouring water.

export type Tool = 'orbit' | 'raise' | 'lower' | 'pour'
export type Show = 'water' | 'particles' | 'none'

export const planetView = {
  layout: structuredClone(defaultPlanet) as PlanetLayout,
  gravity: 9.8, // at sea level; falls off as 1/r²
  show: 'water' as Show,
  colorMode: 'water' as ColorModeId,
  tool: 'orbit' as Tool,
  brushSize: 2, // world units across the surface
  brushStrength: 2, // units of height per second at the brush's centre
  paused: false,
}

export interface PlanetStats {
  status: 'starting' | 'running' | 'no-webgpu'
  fps: number
  water: number
  capacity: number
  rock: number
  substeps: number
  rebuildMs: number // the last stroke's physics rebuild
}

export interface PlanetController {
  rebuild(): void // a new planet from planetView.layout
  refill(): void // a fresh ocean on the current ground
  clearWater(): void
  dispose(): void
}

const STILL: FrameMotion = { omega: [0, 0, 0], alpha: [0, 0, 0], pivot: [0, 0, 0] }
const SPRAY_SHARE = 0.35 // fewer neighbours than this share of the bulk's: spray
const EXTRA_WATER = 80000 // room for poured water beyond the starting ocean
const POUR_SPEED = 2 // units/s out of the nozzle
const NOZZLE_HEIGHT = 1.2 // above the ground, the sea or a pool, whichever is highest
const SETTLED = 1.5 // water slower than this (units/s) counts as a pool, not the stream
// While sculpting, the physics catches up this often. Catching up only at the end
// of a long stroke under the sea moved thousands of particles at once, stacked
// tighter than the water's rest spacing, and the pressure flung them off as
// spray; in small steps only a thin layer moves each time.
const CATCH_UP = 0.3 // seconds

export function startPlanetLoop(container: HTMLElement, onStats: (s: PlanetStats) => void): PlanetController {
  let engine: GpuEngine | null = null
  let renderer: Renderer3D | null = null
  let params: Params | null = null
  let layout: PlanetLayout = planetView.layout
  let terrain: Terrain = makeTerrain(layout)
  let rock: THREE.Mesh<THREE.BufferGeometry, THREE.MeshLambertMaterial> | null = null
  let rockCount = 0
  let generation = 0
  let status: PlanetStats['status'] = 'starting'
  let rebuildMs = 0

  // --- The brush ring and pointer ---
  const ringGeometry = new THREE.RingGeometry(0.92, 1, 48)
  let brushRing: THREE.Mesh | null = null
  let pointer: { x: number; y: number } | null = null // over the canvas
  let held = false
  let strokeChanged = false
  let sinceCatchUp = 0
  let pourClock = 0
  const ray = new THREE.Raycaster()
  let unhook: (() => void) | null = null

  // A new planet: new ground, a fresh ocean, a new renderer.
  const build = () => {
    const mine = ++generation
    unhook?.()
    unhook = null
    engine?.dispose()
    renderer?.dispose()
    engine = null
    renderer = null
    rock = null
    brushRing = null
    status = 'starting'
    layout = structuredClone(planetView.layout)
    terrain = makeTerrain(layout)
    const p = planetParams(layout, planetView.gravity)
    params = p
    const scene = planetScene(p, layout, terrain, EXTRA_WATER)
    rockCount = scene.boundary.psi.length
    void GpuEngine.create(p, scene).then((created) => {
      if (mine !== generation) {
        created?.dispose()
        return
      }
      if (!created) {
        status = 'no-webgpu'
        return
      }
      engine = created
      const R = layout.radius
      renderer = new Renderer3D(container, created.capacity, particleRadius(p), p.domain, [R * 1.1, R * 1.3, R * 3.4])
      renderer.showBounds(false)
      renderer.showFloor(false)
      // Seen from space: black sky to reflect, and water murky enough that the
      // bottom shows only in the shallows. These oceans are very deep for their
      // planets (a quarter of the radius); clearer, and you'd see the deep sea
      // floor's horizon through the water as a ring following the planet's edge.
      renderer.setWaterStyle({ space: true, absorb: 10 })
      rock = rockMesh(terrain, layout)
      brushRing = new THREE.Mesh(
        ringGeometry.clone(),
        new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.75, depthTest: false, side: THREE.DoubleSide }),
      )
      brushRing.renderOrder = 20
      brushRing.visible = false
      renderer.addObject(rock)
      renderer.addObject(brushRing)
      renderer.addObject(stars())
      hookPointer(renderer)
      status = 'running'
      accumulator = 0
    })
  }

  // The physics again, on the current ground: keeping the water (lifted out of
  // any new rock), or with a fresh ocean.
  const rebuildPhysics = (keepWater: boolean) => {
    if (!engine || !params) return
    const t0 = performance.now()
    let water: { pos: Float32Array; vel: Float32Array } | undefined
    if (keepWater) {
      const v = engine.view()
      water = v ? { pos: v.pos.slice(0, v.n * 3), vel: v.vel.slice(0, v.n * 3) } : { pos: new Float32Array(0), vel: new Float32Array(0) }
    }
    const capacity = engine.capacity
    const scene = planetScene(params, layout, terrain, 0, water)
    scene.fluid = scene.fluid.slice(0, Math.min(scene.fluid.length, capacity * 3))
    if (scene.velocity) scene.velocity = scene.velocity.slice(0, scene.fluid.length)
    scene.capacity = capacity
    rockCount = scene.boundary.psi.length
    engine.replaceScene(scene)
    rebuildMs = performance.now() - t0
  }

  // Where the pointer's ray meets the ground, and the direction to it.
  const pick = (): { at: THREE.Vector3; dir: THREE.Vector3 } | null => {
    if (!pointer || !renderer) return null
    ray.setFromCamera(renderer.view.ndc({ clientX: pointer.x, clientY: pointer.y }), renderer.view.camera)
    const o = ray.ray.origin
    const d = ray.ray.direction
    const hit = terrain.raycast([o.x, o.y, o.z], [d.x, d.y, d.z], seaRadius(layout))
    if (!hit) return null
    const at = new THREE.Vector3(hit[0], hit[1], hit[2])
    return { at, dir: at.clone().normalize() }
  }

  const hookPointer = (r: Renderer3D) => {
    unhook?.()
    const el = r.view.gl.domElement
    const down = (e: PointerEvent) => {
      if (planetView.tool === 'orbit' || e.button !== 0 || e.shiftKey) return
      el.setPointerCapture(e.pointerId)
      held = true
      sinceCatchUp = 0
      pourClock = -1 // the first layer right away (see applyTool)
    }
    const move = (e: PointerEvent) => (pointer = { x: e.clientX, y: e.clientY })
    const up = () => {
      held = false
      if (strokeChanged) {
        strokeChanged = false
        rebuildPhysics(true)
      }
    }
    const leave = () => {
      if (!held) pointer = null
    }
    el.addEventListener('pointerdown', down)
    el.addEventListener('pointermove', move)
    el.addEventListener('pointerup', up)
    el.addEventListener('pointercancel', up)
    el.addEventListener('pointerleave', leave)
    unhook = () => {
      el.removeEventListener('pointerdown', down)
      el.removeEventListener('pointermove', move)
      el.removeEventListener('pointerup', up)
      el.removeEventListener('pointercancel', up)
      el.removeEventListener('pointerleave', leave)
    }
  }

  // Shift turns the left button into orbiting while a tool is picked.
  let shift = false
  const onKey = (e: KeyboardEvent) => (shift = e.shiftKey)
  window.addEventListener('keydown', onKey)
  window.addEventListener('keyup', onKey)

  const pourRadius = () => Math.min(planetView.brushSize / 2, 2)

  // How high the water reaches near direction `dir` (unit), within `radius`
  // across the surface: the poured pool, or the sea. Only settled water counts:
  // counting the falling stream made each layer's nozzle a step above the last,
  // and the stream climbed until it fell from far up at a hundred units a second.
  const waterTop = (dir: THREE.Vector3, radius: number): number => {
    const v = view
    if (!v) return 0
    const cos = Math.cos(Math.min(1, (radius + 1) / layout.radius))
    let top = 0
    for (let i = 0; i < v.n; i++) {
      const x = v.pos[i * 3]
      const y = v.pos[i * 3 + 1]
      const z = v.pos[i * 3 + 2]
      const d = Math.hypot(x, y, z)
      if (d <= top || (x * dir.x + y * dir.y + z * dir.z) / d <= cos) continue
      if (Math.hypot(v.vel[i * 3], v.vel[i * 3 + 1], v.vel[i * 3 + 2]) < SETTLED) top = d
    }
    return top
  }

  // Pours one layer of water: a disc of particles at the nozzle, heading down.
  // Layers leave a spacing apart in time, so they don't overlap; and the nozzle
  // rides above whatever water is already there (pouring into a pool from
  // inside it put new particles on top of old ones, and the overlap burst them
  // out into space).
  const pourLayer = (dir: THREE.Vector3, at: THREE.Vector3) => {
    if (!engine || !params) return
    const sp = params.spacing
    const surface = Math.max(at.length(), seaRadius(layout), waterTop(dir, pourRadius()))
    const nozzle = dir.clone().multiplyScalar(surface + NOZZLE_HEIGHT)
    // Two directions across the surface.
    const a = new THREE.Vector3(1, 0, 0).cross(dir)
    if (a.lengthSq() < 0.1) a.set(0, 1, 0).cross(dir)
    a.normalize()
    const b = dir.clone().cross(a)
    const radius = pourRadius()
    const pos: number[] = []
    const vel: number[] = []
    const steps = Math.floor(radius / sp)
    for (let i = -steps; i <= steps; i++) {
      for (let j = -steps; j <= steps; j++) {
        if ((i * i + j * j) * sp * sp > radius * radius) continue
        const p = nozzle.clone().addScaledVector(a, i * sp).addScaledVector(b, j * sp)
        pos.push(p.x, p.y, p.z)
        vel.push(-dir.x * POUR_SPEED, -dir.y * POUR_SPEED, -dir.z * POUR_SPEED)
      }
    }
    engine.addParticles(Float32Array.from(pos), Float32Array.from(vel))
  }

  // Shows the brush and applies the held tool for one frame.
  const applyTool = (dt: number) => {
    if (!brushRing) return
    const tool = planetView.tool
    const hit = tool === 'orbit' ? null : pick()
    brushRing.visible = hit !== null
    if (!hit) return
    // The ring: on the ground, facing out, as wide as the brush.
    brushRing.position.copy(hit.at).addScaledVector(hit.dir, 0.05)
    brushRing.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), hit.dir)
    brushRing.scale.setScalar(tool === 'pour' ? pourRadius() : planetView.brushSize / 2)
    ;(brushRing.material as THREE.MeshBasicMaterial).color.set(tool === 'raise' ? 0x9be08a : tool === 'lower' ? 0xff9b6a : 0x7cc8f2)
    if (!held) return
    if (tool === 'raise' || tool === 'lower') {
      const amount = (tool === 'raise' ? 1 : -1) * planetView.brushStrength * dt
      const touched = terrain.brush([hit.dir.x, hit.dir.y, hit.dir.z], planetView.brushSize / 2 / layout.radius, amount)
      if (touched.length > 0 && rock) {
        updateRock(rock, terrain, layout, touched)
        strokeChanged = true
      }
      sinceCatchUp += dt
      if (strokeChanged && sinceCatchUp >= CATCH_UP) {
        sinceCatchUp = 0
        strokeChanged = false
        rebuildPhysics(true)
      }
    } else if (tool === 'pour' && params) {
      // A layer every spacing / speed seconds, so each has moved a spacing on
      // before the next appears (closer, they'd overlap and burst apart).
      const every = params.spacing / POUR_SPEED
      pourClock = pourClock < 0 ? every : pourClock + dt
      if (pourClock >= every) {
        pourClock = Math.min(pourClock - every, every)
        pourLayer(hit.dir, hit.at)
      }
    }
  }

  build()

  let accumulator = 0
  let last = performance.now()
  let raf = 0
  let frames = 0
  let substepTotal = 0
  let statsStart = last
  let view: SimState | null = null

  const frame = (now: number) => {
    raf = requestAnimationFrame(frame)
    const frameDt = Math.min(Math.max((now - last) / 1000, 0), 0.1)
    accumulator += frameDt
    last = now
    if (renderer) {
      const c = renderer.view.controls
      const tools = planetView.tool !== 'orbit' && !shift
      c.mouseButtons.LEFT = tools ? null : THREE.MOUSE.ROTATE
      c.mouseButtons.RIGHT = tools ? THREE.MOUSE.ROTATE : THREE.MOUSE.PAN
      applyTool(frameDt)
    }
    let substeps = 0
    if (engine && params && !planetView.paused) {
      // The slider sets the pull at sea level: G·M = g · R².
      params.attractors![0].gm = planetView.gravity * layout.radius ** 2
      substeps = Math.min(Math.floor(accumulator / params.dt), params.maxSubsteps)
      accumulator -= substeps * params.dt
      if (substeps === params.maxSubsteps) accumulator = Math.min(accumulator, params.dt)
      if (substeps > 0) engine.step(substeps, STILL, [0, 0, 0])
    } else accumulator = 0

    if (engine && renderer && params) {
      const realistic = planetView.show === 'water' && planetView.colorMode === 'water'
      view = engine.view()
      if (view) {
        renderer.updatePositions(view.pos, view.n)
        if (realistic) writeSpray(view, params, renderer.colors)
        else writeColors(getColorMode(planetView.colorMode), view, params, renderer.colors)
        renderer.commitColors()
      }
      renderer.showFluid(planetView.show !== 'none')
      renderer.render(planetView.show === 'water', realistic)
    }

    frames++
    substepTotal += substeps
    if (now - statsStart >= 500) {
      onStats({
        status,
        fps: (frames * 1000) / (now - statsStart),
        water: view?.n ?? 0,
        capacity: engine?.capacity ?? 0,
        rock: rockCount,
        substeps: substepTotal / frames,
        rebuildMs,
      })
      frames = 0
      substepTotal = 0
      statsStart = now
    }
  }
  raf = requestAnimationFrame(frame)

  if (import.meta.env.DEV) {
    Object.assign(window, { __planet: { view: () => view, params: () => params, renderer: () => renderer, terrain: () => terrain, engine: () => engine, rebuildPhysics } })
  }

  return {
    rebuild: build,
    refill: () => rebuildPhysics(false),
    clearWater: () => engine?.clearParticles(),
    dispose() {
      generation++
      cancelAnimationFrame(raf)
      unhook?.()
      window.removeEventListener('keydown', onKey)
      window.removeEventListener('keyup', onKey)
      ringGeometry.dispose()
      engine?.dispose()
      renderer?.dispose()
    },
  }
}

// The realistic water look's colours, without foam: (0, spray, 0), spray being
// the lone particles flung clear of the rest.
function writeSpray(s: SimState, p: Params, out: Float32Array) {
  const below = SPRAY_SHARE * ((4 * Math.PI) / 3) * (p.h / p.spacing) ** 3
  for (let i = 0; i < s.n; i++) {
    out[i * 3] = 0
    out[i * 3 + 1] = s.neighborCount[i] < below ? 1 : 0
    out[i * 3 + 2] = 0
  }
}

// --- The ground mesh: one vertex per height-map point ---

// Colour by height above sea level: deep basins, shelves, beach, lowland,
// forest, rock, snow.
const HEIGHT_STOPS: [number, number][] = [
  [-6, 0x141e2a],
  [-2.4, 0x1d2a38],
  [-0.6, 0x4a5a5c],
  [-0.15, 0x8c8a70],
  [0, 0xd8c48a],
  [0.12, 0x6b9a44],
  [1, 0x3f6a2c],
  [1.6, 0x7a6a58],
  [2.2, 0xf2f2f2],
]
const stopColors = HEIGHT_STOPS.map(([, c]) => new THREE.Color(c))
const scratch = new THREE.Color()

function heightColor(h: number): THREE.Color {
  const s = HEIGHT_STOPS.findIndex(([v]) => v > h)
  if (s === -1) return scratch.copy(stopColors[stopColors.length - 1])
  if (s === 0) return scratch.copy(stopColors[0])
  const t = (h - HEIGHT_STOPS[s - 1][0]) / (HEIGHT_STOPS[s][0] - HEIGHT_STOPS[s - 1][0])
  return scratch.copy(stopColors[s - 1]).lerp(stopColors[s], t)
}

function rockMesh(t: Terrain, l: PlanetLayout): THREE.Mesh<THREE.BufferGeometry, THREE.MeshLambertMaterial> {
  const count = t.heights.length
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
  geo.setAttribute('color', new THREE.BufferAttribute(new Float32Array(count * 3), 3))
  geo.setIndex(new THREE.BufferAttribute(t.triangles(), 1))
  const mesh = new THREE.Mesh(geo, new THREE.MeshLambertMaterial({ vertexColors: true }))
  mesh.frustumCulled = false
  updateRock(mesh, t, l, null)
  return mesh
}

// Moves and recolours the given grid points (all when null) and refreshes the shading.
function updateRock(mesh: THREE.Mesh, t: Terrain, l: PlanetLayout, touched: number[] | null) {
  const pos = mesh.geometry.attributes.position as THREE.BufferAttribute
  const col = mesh.geometry.attributes.color as THREE.BufferAttribute
  const set = (k: number) => {
    const r = l.radius + t.heights[k]
    pos.setXYZ(k, t.dirs[k * 3] * r, t.dirs[k * 3 + 1] * r, t.dirs[k * 3 + 2] * r)
    const c = heightColor(t.heights[k])
    col.setXYZ(k, c.r, c.g, c.b)
  }
  if (touched) for (const k of touched) set(k)
  else for (let k = 0; k < t.heights.length; k++) set(k)
  pos.needsUpdate = true
  col.needsUpdate = true
  mesh.geometry.computeVertexNormals()
}

// A shell of faint stars far out.
function stars(): THREE.Points {
  const n = 2500
  const pos = new Float32Array(n * 3)
  let seed = 11
  const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647)
  for (let i = 0; i < n; i++) {
    const z = rand() * 2 - 1
    const a = rand() * Math.PI * 2
    const r = Math.sqrt(1 - z * z)
    pos.set([r * Math.cos(a) * 300, z * 300, r * Math.sin(a) * 300], i * 3)
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  return new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xcfd8e6, size: 1.2, sizeAttenuation: false }))
}
