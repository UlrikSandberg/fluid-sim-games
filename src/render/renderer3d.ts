import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'
import type { Domain } from '../sim/params'

const BACKGROUND = 0x0d1117

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
  private readonly resizeObserver: ResizeObserver
  private readonly container: HTMLElement

  constructor(container: HTMLElement, count: number, radius: number, domain: Domain) {
    this.container = container
    this.gl = new THREE.WebGLRenderer({ antialias: true })
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.gl.setClearColor(BACKGROUND)
    container.appendChild(this.gl.domElement)

    this.scene.add(new THREE.HemisphereLight(0xdfe9ff, 0x1a2230, 1.4))
    const sun = new THREE.DirectionalLight(0xffffff, 1.8)
    sun.position.set(0.6, 1, 0.8)
    this.scene.add(sun)

    // Icosphere with one subdivision: 80 triangles, round enough at this size.
    this.mesh = new THREE.InstancedMesh(new THREE.IcosahedronGeometry(1, 1), new THREE.MeshLambertMaterial(), count)
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

    // The floor sits below the tank's reach in any orientation.
    const halfDiagonal = Math.hypot(x1 - x0, y1 - y0, z1 - z0) / 2
    const floorSize = Math.max(x1 - x0, z1 - z0) * 2.5
    this.floor = new THREE.GridHelper(floorSize, Math.round(floorSize), 0x2a3442, 0x1b222c)
    this.floor.position.set(center.x, center.y - halfDiagonal - 0.5, center.z)
    this.scene.add(this.floor)

    // Orbit around the tank's centre; start looking at it from the front-right, above.
    const size = Math.max(x1 - x0, y1 - y0, z1 - z0)
    this.camera.position.copy(center).add(new THREE.Vector3(0.55, 0.45, 1).normalize().multiplyScalar(size * 1.9))
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

  // Shows the floating body (tank coordinates, quaternion [w, x, y, z]), or hides it.
  setBody(center: number[] | null, halfSize: number[], q: number[]) {
    this.body.visible = center !== null
    if (!center) return
    this.body.position.set(center[0], center[1], center[2])
    this.body.scale.set(halfSize[0], halfSize[1], halfSize[2])
    this.body.quaternion.set(q[1], q[2], q[3], q[0])
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

  render() {
    this.controls.update() // applies damping
    this.gl.render(this.scene, this.camera)
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container
    if (w === 0 || h === 0) return
    this.gl.setSize(w, h)
    this.camera.aspect = w / h
    this.camera.updateProjectionMatrix()
  }

  dispose() {
    this.resizeObserver.disconnect()
    this.controls.dispose()
    this.mesh.geometry.dispose()
    this.mesh.material.dispose()
    this.mesh.dispose()
    this.body.geometry.dispose()
    this.body.material.dispose()
    this.bounds.dispose()
    this.floor.dispose()
    this.gl.dispose()
    this.gl.domElement.remove()
  }
}
