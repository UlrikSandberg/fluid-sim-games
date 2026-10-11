import * as THREE from 'three'
import { DIM, type Domain } from '../sim/params'

// Pads a DIM-length vector out to a THREE.Vector3.
function toVec3(v: number[]): THREE.Vector3 {
  return new THREE.Vector3(v[0] ?? 0, v[1] ?? 0, v[2] ?? 0)
}

// One InstancedMesh of circles, mutated imperatively each frame.
export class Renderer {
  private readonly gl: THREE.WebGLRenderer
  private readonly scene = new THREE.Scene()
  private readonly camera = new THREE.OrthographicCamera()
  private readonly mesh: THREE.InstancedMesh
  private readonly bounds: THREE.Box3Helper
  private readonly resizeObserver: ResizeObserver
  private readonly container: HTMLElement
  private readonly domain: Domain

  constructor(container: HTMLElement, count: number, radius: number, domain: Domain) {
    this.container = container
    this.domain = domain

    this.gl = new THREE.WebGLRenderer({ antialias: true })
    this.gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
    this.gl.setClearColor(0x0d1117)
    container.appendChild(this.gl.domElement)

    this.camera.position.set(0, 0, 10)
    this.camera.near = 0.1
    this.camera.far = 100

    const geometry = new THREE.CircleGeometry(1, 10)
    const material = new THREE.MeshBasicMaterial({ color: 0x4fa3ff })
    this.mesh = new THREE.InstancedMesh(geometry, material, count)
    this.mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage)
    // Instances move every frame; a cached bounding sphere would go stale.
    this.mesh.frustumCulled = false
    const m = new THREE.Matrix4().makeScale(radius, radius, radius)
    for (let i = 0; i < count; i++) this.mesh.setMatrixAt(i, m)
    this.scene.add(this.mesh)

    const box = new THREE.Box3(toVec3(domain.min), toVec3(domain.max))
    this.bounds = new THREE.Box3Helper(box, 0x3a4552)
    this.scene.add(this.bounds)

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

  render() {
    this.gl.render(this.scene, this.camera)
  }

  // Frame the domain exactly (plus a small margin), letterboxing to the container aspect.
  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container
    if (w === 0 || h === 0) return
    this.gl.setSize(w, h)

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
    this.gl.dispose()
    this.gl.domElement.remove()
  }
}
