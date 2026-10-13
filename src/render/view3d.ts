import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

// The part of every 3D scene that isn't about its physics: a WebGL renderer in
// the container, a perspective camera with orbit controls, and keeping both
// sized to the container. Scenes add their own content and render.

export interface View3DOptions {
  fov?: number
  near?: number
  far?: number
  clearColor?: THREE.ColorRepresentation
  // Called after the renderer and camera take a new size (render targets that
  // follow the canvas resize here).
  onResize?: (gl: THREE.WebGLRenderer, width: number, height: number) => void
}

export interface View3D {
  readonly gl: THREE.WebGLRenderer
  readonly camera: THREE.PerspectiveCamera
  readonly controls: OrbitControls
  // The pointer's position in normalized device coordinates (for raycasting).
  ndc(e: { clientX: number; clientY: number }): THREE.Vector2
  dispose(): void
}

export function createView3D(container: HTMLElement, opts: View3DOptions = {}): View3D {
  const gl = new THREE.WebGLRenderer({ antialias: true })
  gl.setPixelRatio(Math.min(window.devicePixelRatio, 2))
  if (opts.clearColor !== undefined) gl.setClearColor(opts.clearColor)
  container.appendChild(gl.domElement)

  const camera = new THREE.PerspectiveCamera(opts.fov ?? 45, 1, opts.near ?? 0.1, opts.far ?? 500)
  const controls = new OrbitControls(camera, gl.domElement)
  controls.enableDamping = true

  const resize = () => {
    const { clientWidth: w, clientHeight: h } = container
    if (w === 0 || h === 0) return
    gl.setSize(w, h)
    camera.aspect = w / h
    camera.updateProjectionMatrix()
    opts.onResize?.(gl, w, h)
  }
  const observer = new ResizeObserver(resize)
  observer.observe(container)
  resize()

  return {
    gl,
    camera,
    controls,
    ndc(e) {
      const r = gl.domElement.getBoundingClientRect()
      return new THREE.Vector2(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1)
    },
    dispose() {
      observer.disconnect()
      controls.dispose()
      gl.dispose()
      gl.domElement.remove()
    },
  }
}
