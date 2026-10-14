import * as THREE from 'three'
import type { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Domain } from '../sim/params'
import { createView3D, type View3D } from './view3d'
import { WaterSurface3D } from './waterSurface3d'
import { DUCK_COLORS, disposeObject as disposeDuck, makeDuck, makeBoat } from './racerModels'
import type { FoamFlow } from './foamField'

export { DUCK_COLORS } from './racerModels'

const BACKGROUND = 0x0d1117
const ORBIT_FOV = 45

// A white three-armed spiral fading toward the rim, for whirlpool discs.
function spiralTexture(): THREE.CanvasTexture {
  const size = 256
  const canvas = document.createElement('canvas')
  canvas.width = size
  canvas.height = size
  const g = canvas.getContext('2d')!
  g.translate(size / 2, size / 2)
  g.lineCap = 'round'
  for (let arm = 0; arm < 3; arm++) {
    g.beginPath()
    for (let t = 0; t <= 1; t += 0.01) {
      const angle = arm * ((2 * Math.PI) / 3) + t * 2.6 * Math.PI
      const r = t * (size / 2 - 6)
      if (t === 0) g.moveTo(r * Math.cos(angle), r * Math.sin(angle))
      else g.lineTo(r * Math.cos(angle), r * Math.sin(angle))
    }
    g.strokeStyle = 'rgba(255, 255, 255, 1)'
    g.lineWidth = 16
    g.stroke()
  }
  const fade = g.createRadialGradient(0, 0, size * 0.3, 0, 0, size / 2)
  fade.addColorStop(0, 'rgba(0, 0, 0, 0)')
  fade.addColorStop(1, 'rgba(0, 0, 0, 1)')
  g.globalCompositeOperation = 'destination-out'
  g.fillStyle = fade
  g.fillRect(-size / 2, -size / 2, size, size)
  const texture = new THREE.CanvasTexture(canvas)
  texture.colorSpace = THREE.SRGBColorSpace
  return texture
}

// 3D counterpart of Renderer: one lit InstancedMesh of low-poly spheres, a
// perspective camera orbiting the tank, the tank outline and a floor grid for
// depth cues. Mutated imperatively each frame; positions have stride 3.
//
// The simulation runs in tank coordinates, so the particles and the outline live
// in a group that is rotated by the tank's orientation about the tank's centre;
// the camera and floor stay in the world.
export class Renderer3D {
  readonly view: View3D // renderer, camera and orbit controls, for scenes that pick or steer
  private readonly gl: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera: THREE.PerspectiveCamera
  private readonly controls: OrbitControls
  private readonly mesh: THREE.InstancedMesh<THREE.IcosahedronGeometry, THREE.MeshLambertMaterial>
  // Per-instance linear RGB (stride 3). Fill it, then call commitColors().
  readonly colors: Float32Array
  private readonly bounds: THREE.Box3Helper
  private readonly floor: THREE.GridHelper
  private readonly tank = new THREE.Group() // at the tank centre, rotated by its orientation
  private readonly body: THREE.Mesh<THREE.BoxGeometry, THREE.MeshLambertMaterial>
  private readonly surface: WaterSurface3D
  private readonly stone = new THREE.MeshLambertMaterial({ color: 0x5b6470 })
  private readonly solids: THREE.Mesh<THREE.BoxGeometry, THREE.MeshLambertMaterial>[] = []
  private readonly extras: THREE.Object3D[] = []
  private fluidHidden = false
  private readonly ground = new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide })
  private readonly glass = new THREE.MeshBasicMaterial({
    color: 0xb07cff, transparent: true, opacity: 0.22, side: THREE.DoubleSide, depthWrite: false,
  })
  private readonly rim = new THREE.LineBasicMaterial({ color: 0xd6b8ff })
  private readonly banks: (THREE.Mesh | THREE.Line)[] = [] // ground, lids
  private readonly ducks: THREE.Group[] = []
  private readonly movers: THREE.Mesh<THREE.BoxGeometry, THREE.MeshLambertMaterial>[] = []
  private readonly swirlDiscs: { mesh: THREE.Mesh<THREE.CircleGeometry, THREE.MeshBasicMaterial>; strength: number }[] = []
  private swirlTexture: THREE.CanvasTexture | null = null
  private lastFrame = performance.now()
  private readonly gems: THREE.Mesh<THREE.OctahedronGeometry, THREE.MeshBasicMaterial>[] = []
  private gemKey = ''
  private gemClock = 0
  private readonly wood = new THREE.MeshLambertMaterial({ color: 0x8a5a2b })
  private gate: THREE.Group | null = null
  private finish: THREE.Group | null = null
  private readonly checker: THREE.MeshLambertMaterial
  private readonly followGoal = new THREE.Vector3()
  private following = false
  private beforeRide: { position: THREE.Vector3; target: THREE.Vector3 } | null = null // set while riding

  // `cameraOffset`: where the camera starts relative to the domain's centre
  // (default: front-right and above, at a distance that frames the tank).
  constructor(container: HTMLElement, count: number, radius: number, domain: Domain, cameraOffset?: number[]) {
    this.view = createView3D(container, {
      fov: ORBIT_FOV,
      clearColor: BACKGROUND,
      // The water's render targets follow the canvas (a resize can come before
      // the surface exists, from the first observation).
      onResize: (gl) => this.surface?.resize(gl),
    })
    this.gl = this.view.gl
    this.camera = this.view.camera
    this.controls = this.view.controls

    this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x1a2230, 1.4))
    const sun = new THREE.DirectionalLight(0xffffff, 1.8)
    sun.position.set(0.6, 1, 0.8)
    this.scene.add(sun)
    const sunDirection = sun.position.clone()

    // Icosphere with one subdivision (80 triangles), round enough at this size;
    // the bare icosahedron (20) once there are tens of thousands of them.
    this.mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, count > 10000 ? 0 : 1), new THREE.MeshLambertMaterial(), count)
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.colors = new Float32Array(count * 3).fill(1)
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(this.colors, 3)
    this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage)
    this.mesh.frustumCulled = false
    const m = new THREE.Matrix4().makeScale(radius, radius, radius)
    for (let i = 0; i < count; i++) this.mesh.setMatrixAt(i, m)

    const [x0, y0, z0] = domain.min
    const [x1, y1, z1] = domain.max
    const center = new THREE.Vector3((x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2)
    this.bounds = new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(x0, y0, z0), new THREE.Vector3(x1, y1, z1)), 0x5a6b80)
    const inner = new THREE.Group()
    inner.position.copy(center).negate()
    // The floating body: a unit cube scaled to its half-sizes, posed in tank coordinates.
    this.body = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshLambertMaterial({ color: 0xf2c230 }))
    this.body.visible = false
    inner.add(this.mesh, this.bounds, this.body)
    this.tank.position.copy(center)
    this.tank.add(inner)
    this.scene.add(this.tank)
    this.surface = new WaterSurface3D(this.mesh, radius, sunDirection)

    // Chequered texture for the finish banner.
    const canvas = document.createElement('canvas')
    canvas.width = 64
    canvas.height = 16
    const g = canvas.getContext('2d')!
    for (let i = 0; i < 16; i++) {
      for (let j = 0; j < 4; j++) {
        g.fillStyle = (i + j) % 2 === 0 ? '#111111' : '#f5f5f5'
        g.fillRect(i * 4, j * 4, 4, 4)
      }
    }
    const texture = new THREE.CanvasTexture(canvas)
    texture.magFilter = THREE.NearestFilter
    texture.colorSpace = THREE.SRGBColorSpace
    this.checker = new THREE.MeshLambertMaterial({ map: texture, side: THREE.DoubleSide })

    // The floor sits below the tank's reach in any orientation.
    const halfDiagonal = Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2
    const floorSize = Math.max(x1 - x0, z1 - z0) * 2.5
    this.floor = new THREE.GridHelper(floorSize, Math.round(floorSize), 0x2a3442, 0x1b222c)
    this.floor.position.set(center.x, center.y - halfDiagonal - 0.5, center.z)
    this.scene.add(this.floor)

    // Orbit around the tank's centre; start looking at it from the front-right, above.
    const size = Math.max(x1 - x0, y1 - y0, z1 - z0)
    const offset = cameraOffset
      ? new THREE.Vector3(cameraOffset[0], cameraOffset[1], cameraOffset[2])
      : new THREE.Vector3(0.55, 0.45, 1).normalize().multiplyScalar(size * 1.9)
    this.camera.position.copy(center).add(offset)
    this.controls.target.copy(center)
    this.controls.maxPolarAngle = Math.PI * 0.48 // stay above the floor
    this.controls.update()
    this.surface.resize(this.gl)
  }

  // Write translations straight into the column-major instance matrices
  // (elements 12, 13, 14).
  updatePositions(pos: Float32Array, n: number) {
    const arr = this.mesh.instanceMatrix.array as Float32Array
    for (let i = 0; i < n; i++) {
      const o = i * 16 + 12
      arr[o] = pos[i * 3]
      arr[o + 1] = pos[i * 3 + 1]
      arr[o + 2] = pos[i * 3 + 2]
    }
    this.mesh.count = n
    this.mesh.instanceMatrix.needsUpdate = true
  }

  // Adds a static solid (a ledge, a rock) as a stone-coloured box, in tank coordinates.
  addSolid(min: number[], max: number[]) {
    const box = new THREE.Mesh(new THREE.BoxGeometry(max[0] - min[0], max[1] - min[1], max[2] - min[2]), this.stone)
    box.position.set((min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2)
    this.body.parent!.add(box)
    this.solids.push(box)
  }

  // Adds the ground: an nx × nz grid of vertices (row by row along x; positions
  // stride 3) with a linear RGB colour per vertex.
  addTerrain(t: { nx: number; nz: number; positions: Float32Array; colors: Float32Array }) {
    const index: number[] = []
    for (let j = 0; j + 1 < t.nz; j++) {
      for (let i = 0; i + 1 < t.nx; i++) {
        const a = i + j * t.nx
        index.push(a, a + t.nx, a + 1, a + 1, a + t.nx, a + t.nx + 1)
      }
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.BufferAttribute(t.positions, 3))
    geometry.setAttribute('color', new THREE.BufferAttribute(t.colors, 3))
    geometry.setIndex(index)
    geometry.computeVertexNormals()
    const ground = new THREE.Mesh(geometry, this.ground)
    this.body.parent!.add(ground)
    this.banks.push(ground)
  }

  // Adds a flip zone's lid: a tinted glass strip between two polylines of
  // (x, y, z) points, with a glowing rim along both edges.
  addLid([left, right]: number[][][]) {
    const positions: number[] = []
    for (let i = 0; i + 1 < left.length; i++) {
      positions.push(...left[i], ...right[i], ...left[i + 1], ...left[i + 1], ...right[i], ...right[i + 1])
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    const lid = new THREE.Mesh(geometry, this.glass)
    lid.renderOrder = 1
    this.body.parent!.add(lid)
    this.banks.push(lid)
    for (const edge of [left, right]) {
      const line = new THREE.BufferGeometry().setFromPoints(edge.map((p) => new THREE.Vector3(p[0], p[1], p[2])))
      const rim = new THREE.Line(line, this.rim)
      this.body.parent!.add(rim)
      this.banks.push(rim)
    }
  }

  // Shows or hides the domain's outline (the tank's walls; the river doesn't need it).
  showBounds(visible: boolean) {
    this.bounds.visible = visible
  }

  showFloor(visible: boolean) {
    this.floor.visible = visible
  }

  // Adds any object in tank coordinates (like the bodies); freed with the renderer.
  addObject(o: THREE.Object3D) {
    this.body.parent!.add(o)
    this.extras.push(o)
  }

  // Shows the floating body (tank coordinates, quaternion [w, x, y, z]), or hides it.
  setBody(center: number[] | null, halfSize: number[], q: number[]) {
    this.body.visible = center !== null
    if (!center) return
    this.body.position.set(center[0], center[1], center[2])
    this.body.scale.set(halfSize[0], halfSize[1], halfSize[2])
    this.body.quaternion.set(q[1], q[2], q[3], q[0])
  }

  // Shows a duck per body (tank coordinates, quaternions [w, x, y, z]); duck i
  // gets DUCK_COLORS[i]. Meshes are pooled and rebuilt only if a hull's size changes.
  // `hidden`: a duck not to draw (the one the camera is riding in first person).
  // Bodies with a hull shape are drawn as boats.
  setDucks(bodies: { center: number[]; halfSize: number[]; q: number[]; shape?: { kind: 'hull'; bow: number } }[], hidden = -1) {
    const parent = this.body.parent!
    bodies.forEach((b, i) => {
      let duck = this.ducks[i]
      const key = b.shape ? `${b.halfSize.join(',')}:hull` : b.halfSize.join(',')
      if (!duck || duck.userData.half !== key) {
        if (duck) {
          parent.remove(duck)
          disposeDuck(duck)
        }
        const color = DUCK_COLORS[i % DUCK_COLORS.length]
        duck = b.shape ? makeBoat(b.halfSize, color, b.shape.bow) : makeDuck(b.halfSize, color)
        parent.add(duck)
        this.ducks[i] = duck
      }
      duck.visible = i !== hidden
      duck.position.set(b.center[0], b.center[1], b.center[2])
      duck.quaternion.set(b.q[1], b.q[2], b.q[3], b.q[0])
    })
    for (let i = bodies.length; i < this.ducks.length; i++) this.ducks[i].visible = false
  }

  // Shows the starting gate, a wooden bar between `ends` (tank coordinates) on
  // two posts, or removes it with null.
  setGate(ends: number[][] | null) {
    if (this.gate) {
      this.gate.removeFromParent()
      this.gate.traverse((child) => {
        if (child instanceof THREE.Mesh) child.geometry.dispose()
      })
      this.gate = null
    }
    if (!ends) return
    const [a, b] = ends.map((e) => new THREE.Vector3(e[0], e[1], e[2]))
    const gate = new THREE.Group()
    const bar = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, a.distanceTo(b), 10), this.wood)
    bar.position.copy(a).add(b).multiplyScalar(0.5)
    bar.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize())
    gate.add(bar)
    for (const end of [a, b]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.18, 0.18, 4, 10), this.wood)
      post.position.set(end.x, end.y - 2, end.z)
      gate.add(post)
    }
    this.body.parent!.add(gate)
    this.gate = gate
  }

  // Shows the finish line, a chequered banner between `ends` (tank coordinates,
  // at the banner's bottom) on two posts, or removes it with null.
  setFinish(ends: number[][] | null) {
    if (this.finish) {
      this.finish.removeFromParent()
      this.finish.traverse((child) => {
        if (child instanceof THREE.Mesh) child.geometry.dispose()
      })
      this.finish = null
    }
    if (!ends) return
    const [a, b] = ends.map((e) => new THREE.Vector3(e[0], e[1], e[2]))
    const finish = new THREE.Group()
    const length = a.distanceTo(b)
    const banner = new THREE.Mesh(new THREE.PlaneGeometry(length, 1), this.checker)
    banner.position.copy(a).add(b).multiplyScalar(0.5)
    banner.position.y += 0.5
    banner.rotation.y = -Math.atan2(b.z - a.z, b.x - a.x)
    finish.add(banner)
    for (const end of [a, b]) {
      const post = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 5, 10), this.wood)
      post.position.set(end.x, end.y - 1.5, end.z)
      finish.add(post)
    }
    this.body.parent!.add(finish)
    this.finish = finish
  }

  // Shows whirlpools as spinning spiral discs lying on the water (tank
  // coordinates; y is the water surface). `level` (0–1) fades a disc in and out as
  // its swirl comes and goes. Discs are pooled; call every frame.
  setSwirls(swirls: { x: number; y: number; z: number; radius: number; strength: number; level: number }[]) {
    if (swirls.length > 0) this.swirlTexture ??= spiralTexture()
    swirls.forEach((w, i) => {
      let d = this.swirlDiscs[i]
      if (!d) {
        const material = new THREE.MeshBasicMaterial({
          map: this.swirlTexture, transparent: true, depthWrite: false, side: THREE.DoubleSide,
          color: 0xbdf3ff, blending: THREE.AdditiveBlending, // glows on the water, readable at a glancing angle
        })
        const mesh = new THREE.Mesh(new THREE.CircleGeometry(1, 48), material)
        mesh.rotation.x = -Math.PI / 2
        this.body.parent!.add(mesh)
        d = { mesh, strength: 0 }
        this.swirlDiscs.push(d)
      }
      d.strength = w.strength
      d.mesh.visible = w.level > 0
      d.mesh.position.set(w.x, w.y, w.z)
      // Grows in from 60% as it spins up.
      d.mesh.scale.setScalar(w.radius * (0.6 + 0.4 * w.level))
      d.mesh.material.opacity = w.level
    })
    for (let i = swirls.length; i < this.swirlDiscs.length; i++) this.swirlDiscs[i].mesh.visible = false
  }

  // Shows moving obstacles as boxes (tank coordinates, quaternions [w, x, y, z]),
  // one colour each. Meshes are pooled; call every frame.
  setMovers(movers: { center: number[]; halfSize: number[]; q: number[]; color: number }[]) {
    movers.forEach((m, i) => {
      let mesh = this.movers[i]
      if (!mesh) {
        mesh = new THREE.Mesh(new THREE.BoxGeometry(2, 2, 2), new THREE.MeshLambertMaterial())
        this.body.parent!.add(mesh)
        this.movers.push(mesh)
      }
      mesh.visible = true
      mesh.material.color.setHex(m.color)
      mesh.position.set(m.center[0], m.center[1], m.center[2])
      mesh.scale.set(m.halfSize[0], m.halfSize[1], m.halfSize[2])
      mesh.quaternion.set(m.q[1], m.q[2], m.q[3], m.q[0])
    })
    for (let i = movers.length; i < this.movers.length; i++) this.movers[i].visible = false
  }

  // Shows the power-ups as glowing gems (tank coordinates), spinning and bobbing;
  // taken ones are hidden. Rebuilt only when the set of pickups changes.
  setPickups(pickups: { pos: number[]; color: number; taken: boolean }[]) {
    const key = pickups.map((q) => `${q.pos.join(',')}:${q.color}`).join('|')
    if (key !== this.gemKey) {
      for (const g of this.gems) {
        g.removeFromParent()
        g.geometry.dispose()
        g.material.dispose()
      }
      this.gems.length = 0
      for (const q of pickups) {
        const gem = new THREE.Mesh(new THREE.OctahedronGeometry(0.45), new THREE.MeshBasicMaterial({ color: q.color }))
        gem.position.set(q.pos[0], q.pos[1], q.pos[2])
        gem.userData.baseY = q.pos[1]
        this.body.parent!.add(gem)
        this.gems.push(gem)
      }
      this.gemKey = key
    }
    pickups.forEach((q, i) => (this.gems[i].visible = !q.taken))
  }

  // Glides the camera after a point (tank coordinates; the race tank is unrotated)
  // keeping the current viewing angle, or stops following with null. `snap` jumps
  // there at once.
  follow(point: number[] | null, snap = false) {
    this.following = point !== null
    if (!point) return
    this.followGoal.set(point[0], point[1], point[2])
    if (snap) {
      const step = this.followGoal.clone().sub(this.controls.target)
      this.controls.target.add(step)
      this.camera.position.add(step)
    }
  }

  // Puts the camera at `eye` looking at `target` (tank coordinates) with the given
  // vertical field of view, taking it off the orbit controls; null hands it back to
  // them where it was before.
  ride(view: { eye: number[]; target: number[]; fov: number } | null) {
    if (view) {
      if (!this.beforeRide) {
        this.beforeRide = { position: this.camera.position.clone(), target: this.controls.target.clone() }
        this.controls.enabled = false
      }
      this.camera.position.set(view.eye[0], view.eye[1], view.eye[2])
      this.camera.lookAt(view.target[0], view.target[1], view.target[2])
      this.setFov(view.fov)
    } else if (this.beforeRide) {
      this.camera.position.copy(this.beforeRide.position)
      this.controls.target.copy(this.beforeRide.target)
      this.controls.enabled = true
      this.beforeRide = null
      this.setFov(ORBIT_FOV)
    }
  }

  private setFov(fov: number) {
    if (this.camera.fov === fov) return
    this.camera.fov = fov
    this.camera.updateProjectionMatrix()
  }

  // Tank orientation as a quaternion [w, x, y, z] (tank → world).
  setTankOrientation(q: number[]) {
    this.tank.quaternion.set(q[1], q[2], q[3], q[0])
  }

  // The camera's right, up and forward (view) directions in world space, so
  // screen-space drags can be turned into rotations as seen by the viewer.
  cameraAxes(): { right: number[]; up: number[]; forward: number[] } {
    const e = this.camera.matrixWorld.elements
    return { right: [e[0], e[1], e[2]], up: [e[4], e[5], e[6]], forward: [-e[8], -e[9], -e[10]] }
  }

  commitColors() {
    this.mesh.instanceColor!.needsUpdate = true
  }

  // The realistic water's surroundings (see WaterSurface3D.setStyle).
  setWaterStyle(style: { space?: boolean; absorb?: number }) {
    this.surface.setStyle(style)
  }

  // The surface flow the foam's texture moves along (realistic look).
  setFoamFlow(flow: FoamFlow) {
    this.surface.setFlow(flow)
  }

  // `water`: draw the fluid as a shaded surface (see waterSurface3d.ts) instead of
  // spheres; `realistic`: as clear water with foam, the colours being FoamField's.
  // Hides the fluid altogether (to see what's under it), or shows it again.
  showFluid(visible: boolean) {
    this.fluidHidden = !visible
  }

  render(water: boolean, realistic = false) {
    const now = performance.now()
    const dt = Math.min((now - this.lastFrame) / 1000, 0.1)
    this.lastFrame = now
    // Spin the swirl discs with their swirls. The disc is laid flat by −90° about x,
    // so its local z turns the opposite way to the water's rotation about +y.
    for (const d of this.swirlDiscs) d.mesh.rotation.z -= d.strength * 0.15 * dt
    this.gemClock += dt
    this.gems.forEach((g, i) => {
      g.rotation.y += 2 * dt
      g.position.y = g.userData.baseY + 0.2 * Math.sin(this.gemClock * 3 + i)
    })
    if (this.following && !this.beforeRide) {
      // Move the orbit target (and the camera with it) a little toward the goal each frame.
      const step = this.followGoal.clone().sub(this.controls.target).multiplyScalar(0.04)
      this.controls.target.add(step)
      this.camera.position.add(step)
    }
    if (!this.beforeRide) this.controls.update() // applies damping (and would turn a riding camera)
    this.mesh.visible = !this.fluidHidden
    if (water && !this.fluidHidden) this.surface.render(this.gl, this.scene, this.camera, this.mesh, realistic)
    else this.gl.render(this.scene, this.camera)
  }

  dispose() {
    this.surface.dispose()
    this.mesh.geometry.dispose()
    this.mesh.material.dispose()
    this.mesh.dispose()
    this.body.geometry.dispose()
    this.body.material.dispose()
    for (const s of this.solids) s.geometry.dispose()
    for (const o of this.extras) {
      o.traverse((c) => {
        const m = c as THREE.Mesh
        m.geometry?.dispose()
        const mats = m.material ? (Array.isArray(m.material) ? m.material : [m.material]) : []
        for (const mat of mats) mat.dispose()
      })
    }
    for (const b of this.banks) b.geometry.dispose()
    for (const d of this.ducks) disposeDuck(d)
    for (const m of this.movers) {
      m.geometry.dispose()
      m.material.dispose()
    }
    this.setGate(null)
    this.setFinish(null)
    for (const d of this.swirlDiscs) {
      d.mesh.geometry.dispose()
      d.mesh.material.dispose()
    }
    this.setPickups([])
    this.swirlTexture?.dispose()
    this.wood.dispose()
    this.checker.map?.dispose()
    this.checker.dispose()
    this.stone.dispose()
    this.ground.dispose()
    this.glass.dispose()
    this.rim.dispose()
    this.bounds.dispose()
    this.floor.dispose()
    this.view.dispose()
  }
}
