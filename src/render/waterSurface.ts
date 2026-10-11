import * as THREE from 'three'

// Screen-space fluid surface ("metaballs"):
//  1. Every particle is drawn as a soft blob, weight w = (1 − r²/R²)³, additively
//     into a half-float target: alpha accumulates Σw, rgb accumulates Σ w·colour.
//     Overlapping blobs merge into one smooth field.
//  2. A full-screen pass thresholds the field into a solid, antialiased body,
//     divides out the weights to get a smoothly blended colour, and brightens a
//     thin band just inside the surface.
// The blobs reuse the particle mesh's instance buffers, so nothing is uploaded twice.

// Blob radius as a multiple of the particle radius, and the field threshold. With
// spacing = 2·radius, the interior sums to Σw ≈ 1.2 and a lone particle peaks at 1,
// so 0.4 keeps splashes visible as droplets about the size of a particle.
const BLOB_SCALE = 2.5
const THRESHOLD = 0.4

const blobVertex = /* glsl */ `
  uniform float blobScale;
  varying vec2 vOffset;
  varying vec3 vColor;
  void main() {
    vOffset = position.xy;
    #ifdef USE_INSTANCING_COLOR
      vColor = instanceColor;
    #else
      vColor = vec3(1.0);
    #endif
    gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(position.xy * blobScale, 0.0, 1.0);
  }
`

const blobFragment = /* glsl */ `
  varying vec2 vOffset;
  varying vec3 vColor;
  void main() {
    float r2 = dot(vOffset, vOffset);
    if (r2 >= 1.0) discard;
    float w = 1.0 - r2;
    w = w * w * w;
    gl_FragColor = vec4(vColor * w, w);
  }
`

const compositeVertex = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`

const compositeFragment = /* glsl */ `
  uniform sampler2D tField;
  uniform vec3 background;
  uniform float threshold;
  varying vec2 vUv;
  void main() {
    vec4 field = texture2D(tField, vUv);
    float w = field.a;
    vec3 color = field.rgb / max(w, 1e-5);
    float aa = fwidth(w) * 0.75 + 1e-4;
    float inside = smoothstep(threshold - aa, threshold + aa, w);
    // Brighter band just inside the surface, fading out with depth into the field.
    float rim = 1.0 - smoothstep(threshold, threshold + 0.35, w);
    color = mix(color, min(color * 1.35 + 0.1, vec3(1.0)), rim * 0.8);
    gl_FragColor = vec4(mix(background, color, inside), 1.0);
    #include <colorspace_fragment>
  }
`

export class WaterSurface {
  private readonly blobScene = new THREE.Scene()
  private readonly blobs: THREE.InstancedMesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly composite: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly compositeScene = new THREE.Scene()
  private readonly compositeCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
  private readonly target: THREE.WebGLRenderTarget
  private readonly size = new THREE.Vector2()

  constructor(particles: THREE.InstancedMesh, background: THREE.ColorRepresentation) {
    const material = new THREE.ShaderMaterial({
      vertexShader: blobVertex,
      fragmentShader: blobFragment,
      uniforms: { blobScale: { value: BLOB_SCALE } },
      transparent: true,
      depthTest: false,
      depthWrite: false,
      // Plain sums for both colour and weight (AdditiveBlending would scale rgb by alpha).
      blending: THREE.CustomBlending,
      blendEquation: THREE.AddEquation,
      blendSrc: THREE.OneFactor,
      blendDst: THREE.OneFactor,
      blendSrcAlpha: THREE.OneFactor,
      blendDstAlpha: THREE.OneFactor,
    })
    this.blobs = new THREE.InstancedMesh(new THREE.PlaneGeometry(2, 2), material, particles.count)
    this.blobs.instanceMatrix = particles.instanceMatrix
    this.blobs.instanceColor = particles.instanceColor
    this.blobs.frustumCulled = false
    this.blobScene.add(this.blobs)

    this.target = new THREE.WebGLRenderTarget(1, 1, {
      type: THREE.HalfFloatType,
      depthBuffer: false,
      minFilter: THREE.LinearFilter,
      magFilter: THREE.LinearFilter,
    })
    this.composite = new THREE.Mesh(
      new THREE.PlaneGeometry(2, 2),
      new THREE.ShaderMaterial({
        vertexShader: compositeVertex,
        fragmentShader: compositeFragment,
        uniforms: {
          tField: { value: this.target.texture },
          background: { value: new THREE.Color(background) }, // linear working space
          threshold: { value: THRESHOLD },
        },
        depthTest: false,
        depthWrite: false,
      }),
    )
    this.composite.frustumCulled = false
    this.compositeScene.add(this.composite)
  }

  // Matches the field target to the canvas's drawing-buffer size.
  resize(gl: THREE.WebGLRenderer) {
    gl.getDrawingBufferSize(this.size)
    this.target.setSize(this.size.x, this.size.y)
  }

  // Draws the fluid body to the screen (replacing the clear). Overlays go on top.
  render(gl: THREE.WebGLRenderer, camera: THREE.Camera, count: number) {
    this.blobs.count = count
    const clearColor = gl.getClearColor(new THREE.Color())
    const clearAlpha = gl.getClearAlpha()
    gl.setRenderTarget(this.target)
    gl.setClearColor(0x000000, 0)
    gl.clear()
    gl.render(this.blobScene, camera)
    gl.setRenderTarget(null)
    gl.setClearColor(clearColor, clearAlpha)
    gl.render(this.compositeScene, this.compositeCamera)
  }

  dispose() {
    this.blobs.geometry.dispose()
    this.blobs.material.dispose()
    this.composite.geometry.dispose()
    this.composite.material.dispose()
    this.target.dispose()
  }
}
