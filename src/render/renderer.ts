import * as THREE from 'three'
import { DIM, type Domain } from '../sim/params'
import { WaterSurface } from './waterSurface'

// Pads a DIM-length vector out to a THREE.Vector3.
function toVec3(v: number[]): THREE.Vector3 {
  return new THREE.Vector3(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0)
}

const BRUSH_COLORS = { repel: 0xff9b54, attract: 0x54d6ff, current: 0x7ee787 }
const BACKGROUND = 0x0d1117

// One InstancedMesh of circles, mutated imperatively each frame.
export class Renderer {
  private readonly gl: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.OrthographicCamera()
  private readonly mesh: THREE.InstancedMesh
  // Per-instance linear RGB (stride 3). Fill it, then call commitColors().
  readonly colors: Float32Array
  private readonly bounds: THREE.Box3Helper
  private readonly brush: THREE.LineLoop<THREE.BufferGeometry, THREE.LineBasicMaterial>
  private readonly arrow: THREE.LineSegments<THREE.BufferGeometry, THREE.LineBasicMaterial>
  private readonly obstacle: THREE.LineLoop<THREE.BufferGeometry, THREE.LineBasicMaterial>
  private readonly surface: WaterSurface
  private readonly resizeObserver: ResizeObserver
  private readonly container: HTMLElement
  private readonly domain: Domain

  constructor(container: HTMLElement, count: number, radius: number, domain: Domain) {
    this.container = container
    this.domain = domain

    this.gl = new THREE.WebGLRenderer({ antialias: true })
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.gl.setClearColor(BACKGROUND)
    container.appendChild(this.gl.domElement)

    this.camera.position.set(0, 0, 10)
    this.camera.near = 0.1
    this.camera.far = 100

    const geometry = new THREE.CircleGeometry(1, 10)
    const material = new THREE.MeshBasicMaterial({ color: 0xffffff })
    this.mesh = new THREE.InstancedMesh(geometry, material, count)
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    this.colors = new Float32Array(count * 3).fill(1)
    this.mesh.instanceColor = new THREE.InstancedBufferAttribute(this.colors, 3)
    this.mesh.instanceColor.setUsage(THREE.DynamicDrawUsage)
    // Instances move every frame; a cached bounding sphere would go stale.
    this.mesh.frustumCulled = false
    const m = new THREE.Matrix4().makeScale(radius, radius, radius)
    for (let i = 0; i < count; i++) this.mesh.setMatrixAt(i, m)
    this.scene.add(this.mesh)
    this.surface = new WaterSurface(this.mesh, BACKGROUND)

    const box = new THREE.Box3(toVec3(domain.min), toVec3(domain.max))
    this.bounds = new THREE.Box3Helper(box, 0x3a4552)
    this.scene.add(this.bounds)

    // Unit circle outline, scaled and moved to show the mouse brush.
    const brushPoints: THREE.Vector3[] = []
    for (let k = 0; k < 64; k++) {
      const a = (k / 64) * Math.PI * 2
      brushPoints.push(new THREE.Vector3(Math.cos(a), Math.sin(a), 1))
    }
    this.brush = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints(brushPoints),
      new THREE.LineBasicMaterial({ transparent: true, opacity: 0.7 }),
    )
    this.brush.visible = false
    this.scene.add(this.brush)

    // Current-mode arrow: shaft plus two arrowhead strokes (3 segments).
    const arrowGeometry = new THREE.BufferGeometry()
    arrowGeometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6 * 3), 3))
    this.arrow = new THREE.LineSegments(arrowGeometry, new THREE.LineBasicMaterial({ color: BRUSH_COLORS.current }))
    this.arrow.frustumCulled = false
    this.arrow.visible = false
    this.scene.add(this.arrow)

    // Unit square outline, scaled and moved to show the obstacle.
    this.obstacle = new THREE.LineLoop(
      new THREE.BufferGeometry().setFromPoints([
        new THREE.Vector3(-1, -1, 1),
        new THREE.Vector3(1, -1, 1),
        new THREE.Vector3(1, 1, 1),
        new THREE.Vector3(-1, 1, 1),
      ]),
      new THREE.LineBasicMaterial({ color: 0xc9d1d9 }),
    )
    this.obstacle.visible = false
    this.scene.add(this.obstacle)

    this.resizeObserver = new ResizeObserver(() => this.resize())
    this.resizeObserver.observe(container)
    this.resize()
  }

  // Write translations straight into the column-major instance matrices
  // (elements 12, 13, 14), looping over DIM components.
  updatePositions(pos: Float32Array, n: number) {
    const arr = this.mesh.instanceMatrix.array as Float32Array
    for (let i = 0; i < n; i++) {
      const o = i * 16 + 12
      for (let d = 0; d < DIM; d++) arr[o + d] = pos[i * DIM + d]
    }
    this.mesh.count = n
    this.mesh.instanceMatrix.needsUpdate = true
  }

  commitColors() {
    this.mesh.instanceColor!.needsUpdate = true
  }

  // Shows the brush outline at `pos` (or hides it when null), coloured by what it
  // does: faint while hovering (what a drag would do), solid while applied.
  setBrush(pos: number[] | null, radius: number, kind: keyof typeof BRUSH_COLORS, active: boolean) {
    this.brush.visible = pos !== null
    if (!pos) return
    this.brush.position.set(pos[0], pos[1], 0)
    this.brush.scale.set(radius, radius, 1)
    this.brush.material.color.set(BRUSH_COLORS[kind])
    this.brush.material.opacity = active ? 0.9 : 0.4
  }

  // Draws the current-mode arrow from `from` along `vec` (reversed when sign < 0),
  // or hides it when `from` is null.
  setArrow(from: number[] | null, vec: number[], sign: number) {
    const len = Math.hypot(vec[0], vec[1])
    this.arrow.visible = from !== null && len > 1e-6
    if (!from || !this.arrow.visible) return
    const ux = (sign * vec[0]) / len
    const uy = (sign * vec[1]) / len
    const tx = from[0] + ux * len
    const ty = from[1] + uy * len
    const head = Math.min(0.35 * len, 1.5)
    const c = Math.cos(0.45)
    const sn = Math.sin(0.45)
    const pts = this.arrow.geometry.getAttribute('position') as THREE.BufferAttribute
    pts.setXYZ(0, from[0], from[1], 1)
    pts.setXYZ(1, tx, ty, 1)
    pts.setXYZ(2, tx, ty, 1)
    pts.setXYZ(3, tx - head * (ux * c - uy * sn), ty - head * (uy * c + ux * sn), 1)
    pts.setXYZ(4, tx, ty, 1)
    pts.setXYZ(5, tx - head * (ux * c + uy * sn), ty - head * (uy * c - ux * sn), 1)
    pts.needsUpdate = true
  }

  // Shows the obstacle outline, or hides it when `center` is null.
  setObstacle(center: number[] | null, halfSize: number[], angle: number) {
    this.obstacle.visible = center !== null
    if (!center) return
    this.obstacle.position.set(center[0], center[1], 0)
    this.obstacle.scale.set(halfSize[0], halfSize[1], 1)
    this.obstacle.rotation.z = angle
  }

  // Client (CSS pixel) coordinates → world coordinates on the z = 0 plane.
  // 2D only: in 3D this becomes a ray cast against the scene.
  screenToWorld(clientX: number, clientY: number): number[] {
    const rect = this.gl.domElement.getBoundingClientRect()
    const u = (clientX - rect.left) / rect.width
    const v = (clientY - rect.top) / rect.height
    return [
      this.camera.left + u * (this.camera.right - this.camera.left),
      this.camera.top + v * (this.camera.bottom - this.camera.top),
    ]
  }

  // `water`: draw the fluid as a smooth surface (see waterSurface.ts) instead of
  // dots; the overlays (box, brush, arrow, obstacle) are drawn on top either way.
  render(water: boolean) {
    if (!water) {
      this.gl.render(this.scene, this.camera)
      return
    }
    this.surface.render(this.gl, this.camera, this.mesh.count)
    this.mesh.visible = false
    this.gl.autoClear = false
    this.gl.render(this.scene, this.camera)
    this.gl.autoClear = true
    this.mesh.visible = true
  }

  // Frame the domain exactly (plus a small margin), letterboxing to the container aspect.
  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container
    if (w === 0 || h === 0) return
    this.gl.setSize(w, h)
    this.surface.resize(this.gl)

    const { min, max } = this.domain
    const pad = 0.5
    const boxW = max[0] - min[0] + 2 * pad
    const boxH = max[1] - min[1] + 2 * pad
    const cx = (min[0] + max[0]) / 2
    const cy = (min[1] + max[1]) / 2
    const aspect = w / h
    let halfW: number
    let halfH: number
    if (aspect > boxW / boxH) {
      halfH = boxH / 2
      halfW = halfH * aspect
    } else {
      halfW = boxW / 2
      halfH = halfW / aspect
    }
    this.camera.left = cx - halfW
    this.camera.right = cx + halfW
    this.camera.top = cy + halfH
    this.camera.bottom = cy - halfH
    this.camera.updateProjectionMatrix()
  }

  dispose() {
    this.resizeObserver.disconnect()
    this.mesh.geometry.dispose()
    ;(this.mesh.material as THREE.Material).dispose()
    this.mesh.dispose()
    this.bounds.dispose()
    this.brush.geometry.dispose()
    this.brush.material.dispose()
    this.arrow.geometry.dispose()
    this.arrow.material.dispose()
    this.obstacle.geometry.dispose()
    this.obstacle.material.dispose()
    this.surface.dispose()
    this.gl.dispose()
    this.gl.domElement.remove()
  }
}
