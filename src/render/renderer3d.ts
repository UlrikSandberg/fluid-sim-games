import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Domain } from '../sim/params'
import { WaterSurface3D } from './waterSurface3d'

const BACKGROUND = 0x0d1117

// Duck colours, one per racer.
export const DUCK_COLORS = [0xf2c230, 0xff8a3d, 0xff5d8f, 0x7ad151, 0x3fb7e8, 0xb07cff, 0xf5f5f5, 0xff4d4d]

// A rubber duck sized to a physics hull with the given half sizes, facing +x in
// its body frame: an ellipsoid body, a head, a beak, eyes and a tail. The hull
// (a box) is what the fluid sees; this is only the look.
function makeDuck(half: number[], color: number): THREE.Group {
  const [hx, hy, hz] = half
  const duck = new THREE.Group()
  const skin = new THREE.MeshLambertMaterial({ color })
  const orange = new THREE.MeshLambertMaterial({ color: 0xff8c1a })
  const black = new THREE.MeshLambertMaterial({ color: 0x111111 })
  const sphere = new THREE.SphereGeometry(1, 20, 14)
  const body = new THREE.Mesh(sphere, skin)
  body.scale.set(hx * 1.05, hy * 0.85, hz * 1.0)
  body.position.y = -0.1 * hy
  const r = 0.6 * hz
  const head = new THREE.Mesh(sphere, skin)
  head.scale.setScalar(r)
  head.position.set(0.55 * hx, 0.75 * hy + 0.3 * r, 0)
  const beak = new THREE.Mesh(new THREE.ConeGeometry(0.35 * r, 0.8 * r, 12), orange)
  beak.rotation.z = -Math.PI / 2
  beak.position.set(head.position.x + 1.1 * r, head.position.y - 0.15 * r, 0)
  const tail = new THREE.Mesh(new THREE.ConeGeometry(0.35 * hy, 0.7 * hy, 10), skin)
  tail.rotation.z = Math.PI / 2.6
  tail.position.set(-0.95 * hx, 0.3 * hy, 0)
  duck.add(body, head, beak, tail)
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(sphere, black)
    eye.scale.setScalar(0.13 * r)
    eye.position.set(head.position.x + 0.55 * r, head.position.y + 0.3 * r, side * 0.65 * r)
    duck.add(eye)
  }
  duck.userData.half = half.join(',')
  return duck
}

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

function disposeDuck(duck: THREE.Group) {
  duck.traverse((child) => {
    if (child instanceof THREE.Mesh) {
      child.geometry.dispose()
      ;(child.material as THREE.Material).dispose()
    }
  })
}

// 3D counterpart of Renderer: one lit InstancedMesh of low-poly spheres, a
// perspective camera orbiting the tank, the tank outline and a floor grid for
// depth cues. Mutated imperatively each frame; positions have stride 3.
//
// The simulation runs in tank coordinates, so the particles and the outline live
// in a group that is rotated by the tank's orientation about the tank's centre;
// the camera and floor stay in the world.
export class Renderer3D {
  private readonly gl: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.PerspectiveCamera(45, 1, 0.1, 500)
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
  private readonly ground = new THREE.MeshLambertMaterial({ color: 0x3f5a3a, side: THREE.DoubleSide })
  private readonly banks: THREE.Mesh<THREE.BufferGeometry, THREE.MeshLambertMaterial>[] = []
  private readonly ducks: THREE.Group[] = []
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
  private readonly resizeObserver: ResizeObserver
  private readonly container: HTMLElement

  // `cameraOffset`: where the camera starts relative to the domain's centre
  // (default: front-right and above, at a distance that frames the tank).
  constructor(container: HTMLElement, count: number, radius: number, domain: Domain, cameraOffset?: number[]) {
    this.container = container
    this.gl = new THREE.WebGLRenderer({ antialias: true })
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.gl.setClearColor(BACKGROUND)
    container.appendChild(this.gl.domElement)

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
    this.controls = new OrbitControls(this.camera, this.gl.domElement)
    this.controls.target.copy(center)
    this.controls.enableDamping = true
    this.controls.maxPolarAngle = Math.PI * 0.48 // stay above the floor
    this.controls.update()

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(container)
    this.resize()
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

  // Adds a river bank as a solid block of land: a wall along `line` ((x, z)
  // points) from the floor up to height(x), ground on top out to z = `outerZ`, and
  // the block's outer side and ends closed, so there's no hollow to look into.
  addBank(line: number[][], height: (x: number) => number, outerZ: number) {
    const positions: number[] = []
    const quad = (a: number[], b: number[], c: number[], d: number[]) => positions.push(...a, ...b, ...c, ...a, ...c, ...d)
    for (let i = 0; i + 1 < line.length; i++) {
      const [x0, z0] = line[i]
      const [x1, z1] = line[i + 1]
      const h0 = height(x0)
      const h1 = height(x1)
      quad([x0, 0, z0], [x1, 0, z1], [x1, h1, z1], [x0, h0, z0]) // bank wall
      quad([x0, h0, z0], [x1, h1, z1], [x1, h1, outerZ], [x0, h0, outerZ]) // ground
      quad([x0, 0, outerZ], [x1, 0, outerZ], [x1, h1, outerZ], [x0, h0, outerZ]) // outer side
    }
    for (const [x, z] of [line[0], line[line.length - 1]]) {
      const h = height(x)
      quad([x, 0, z], [x, 0, outerZ], [x, h, outerZ], [x, h, z]) // ends
    }
    const geometry = new THREE.BufferGeometry()
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3))
    geometry.computeVertexNormals()
    const bank = new THREE.Mesh(geometry, this.ground)
    this.body.parent!.add(bank)
    this.banks.push(bank)
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
  setDucks(bodies: { center: number[]; halfSize: number[]; q: number[] }[]) {
    const parent = this.body.parent!
    bodies.forEach((b, i) => {
      let duck = this.ducks[i]
      if (!duck || duck.userData.half !== b.halfSize.join(',')) {
        if (duck) {
          parent.remove(duck)
          disposeDuck(duck)
        }
        duck = makeDuck(b.halfSize, DUCK_COLORS[i % DUCK_COLORS.length])
        parent.add(duck)
        this.ducks[i] = duck
      }
      duck.visible = true
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
  // coordinates; y is the water surface), replacing any previous ones.
  setSwirls(swirls: { x: number; y: number; z: number; radius: number; strength: number }[]) {
    for (const d of this.swirlDiscs) {
      d.mesh.removeFromParent()
      d.mesh.geometry.dispose()
      d.mesh.material.dispose()
    }
    this.swirlDiscs.length = 0
    if (swirls.length === 0) return
    this.swirlTexture ??= spiralTexture()
    for (const w of swirls) {
      const material = new THREE.MeshBasicMaterial({
        map: this.swirlTexture, transparent: true, depthWrite: false, side: THREE.DoubleSide,
        color: 0xbdf3ff, blending: THREE.AdditiveBlending, // glows on the water, readable at a glancing angle
      })
      const mesh = new THREE.Mesh(new THREE.CircleGeometry(w.radius, 48), material)
      mesh.rotation.x = -Math.PI / 2
      mesh.position.set(w.x, w.y, w.z)
      this.body.parent!.add(mesh)
      this.swirlDiscs.push({ mesh, strength: w.strength })
    }
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
  // keeping the current viewing angle, or stops following with null.
  follow(point: number[] | null) {
    this.following = point !== null
    if (point) this.followGoal.set(point[0], point[1], point[2])
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

  // `water`: draw the fluid as a shaded surface (see waterSurface3d.ts) instead of spheres.
  render(water: boolean) {
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
    if (this.following) {
      // Move the orbit target (and the camera with it) a little toward the goal each frame.
      const step = this.followGoal.clone().sub(this.controls.target).multiplyScalar(0.04)
      this.controls.target.add(step)
      this.camera.position.add(step)
    }
    this.controls.update() // applies damping
    if (water) this.surface.render(this.gl, this.scene, this.camera, this.mesh)
    else this.gl.render(this.scene, this.camera)
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container
    if (w === 0 || h === 0) return
    this.gl.setSize(w, h)
    this.surface.resize(this.gl)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  dispose() {
    this.resizeObserver.disconnect()
    this.controls.dispose()
    this.surface.dispose()
    this.mesh.geometry.dispose()
    this.mesh.material.dispose()
    this.mesh.dispose()
    this.body.geometry.dispose()
    this.body.material.dispose()
    for (const s of this.solids) s.geometry.dispose()
    for (const b of this.banks) b.geometry.dispose()
    for (const d of this.ducks) disposeDuck(d)
    this.setGate(null)
    this.setFinish(null)
    this.setSwirls([])
    this.setPickups([])
    this.swirlTexture?.dispose()
    this.wood.dispose()
    this.checker.map?.dispose()
    this.checker.dispose()
    this.stone.dispose()
    this.ground.dispose()
    this.bounds.dispose()
    this.floor.dispose()
    this.gl.dispose()
    this.gl.domElement.remove()
  }
}
