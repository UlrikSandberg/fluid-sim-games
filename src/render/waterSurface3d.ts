import * as THREE from 'three'

// Screen-space fluid rendering for the 3D scene (after Green, "Screen Space Fluid
// Rendering", GDC 2010):
//  1. The rest of the scene (floor, tank outline, bodies) renders to a texture
//     with its depth.
//  2. Particles render as camera-facing sphere impostors into a float target
//     holding the view depth of the nearest sphere surface.
//  3. They also add up thickness (how much water lies behind each pixel) and
//     thickness-weighted colour, so the colour modes still read on the surface.
//  4. A narrow-range filter smooths the depth (Truong & Yuksel, "A Narrow-Range
//     Filter for Screen-Space Fluid Rendering", 2018): a Gaussian in which samples
//     far behind the centre are clamped to just behind it and samples in front of
//     it are skipped. The spheres' bumps flatten into one surface and silhouettes
//     stay sharp, without the streaks a separable bilateral (depth-weighted) blur
//     leaves along its axes.
//  5. A full-screen pass rebuilds normals from the smoothed depth and shades:
//     thick water is the colour mode's colour, thin water shows the refracted
//     scene through it (Beer–Lambert), a Fresnel term mixes in a sky reflection,
//     a sun adds a highlight, and a depth test against the scene lets bodies
//     hide, or be hidden by, the water.
// The impostors share the particle mesh's instance buffers and live in its
// group (so the tank's tilt applies); layers keep each pass to its own objects.

const DEPTH_LAYER = 1
const THICKNESS_LAYER = 2
// Tuned by eye in the 3D tank: smaller blurs leave the wall-packed particle
// lattice visible as stripes, and the sun highlight then sparkles on every bump.
const IMPOSTOR_SCALE = 1.6 // impostor radius / particle radius, closing the gaps between spheres
const BLUR_WORLD_RADIUS = 1.0 // smoothing radius in world units (converted to pixels per fragment)
const BLUR_DEPTH_RANGE = 0.5 // world units: samples further behind are clamped, further in front skipped
const BLUR_ITERATIONS = 3
const MAX_BLUR_PIXELS = 32 // kernel half-width cap per pass (the loop bound in the shader)

const impostorVertex = /* glsl */ `
  uniform float radius;
  varying vec2 vOffset;
  varying vec3 vCenter;
  varying vec3 vColor;
  void main() {
    vec4 center = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vCenter = center.xyz;
    vOffset = position.xy;
    #ifdef USE_INSTANCING_COLOR
      vColor = instanceColor;
    #else
      vColor = vec3(1.0);
    #endif
    gl_Position = projectionMatrix * (center + vec4(position.xy * radius, 0.0, 0.0));
  }
`

// Writes the view depth of the sphere's front surface, and the matching
// hardware depth so the nearest sphere wins.
const depthFragment = /* glsl */ `
  uniform float radius;
  uniform mat4 uProjection;
  varying vec2 vOffset;
  varying vec3 vCenter;
  void main() {
    float r2 = dot(vOffset, vOffset);
    if (r2 >= 1.0) discard;
    vec3 p = vCenter + vec3(vOffset * radius, sqrt(1.0 - r2) * radius);
    vec4 clip = uProjection * vec4(p, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    gl_FragColor = vec4(-p.z, 0.0, 0.0, 1.0);
  }
`

// Adds the chord length through the sphere (thickness) and colour weighted by it.
const thicknessFragment = /* glsl */ `
  uniform float radius;
  varying vec2 vOffset;
  varying vec3 vColor;
  void main() {
    float r2 = dot(vOffset, vOffset);
    if (r2 >= 1.0) discard;
    float t = 2.0 * sqrt(1.0 - r2) * radius;
    gl_FragColor = vec4(vColor * t, t);
  }
`

const fullscreenVertex = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`

// One direction of the narrow-range filter on view depth (0 = no fluid).
const blurFragment = /* glsl */ `
  #define MAX_BLUR ${MAX_BLUR_PIXELS}
  uniform sampler2D tDepth;
  uniform vec2 direction; // one texel along the blur axis
  uniform float worldRadius;
  uniform float focalPixels; // pixels per world unit at depth 1
  uniform float depthRange;
  varying vec2 vUv;
  void main() {
    float d0 = texture2D(tDepth, vUv).r;
    if (d0 <= 0.0) { gl_FragColor = vec4(0.0); return; }
    float sigma = max(worldRadius * focalPixels / d0, 0.5) / 2.0;
    float sum = 0.0;
    float weights = 0.0;
    for (int i = -MAX_BLUR; i <= MAX_BLUR; i++) {
      float x = float(i);
      if (abs(x) > 2.0 * sigma) continue;
      float d = texture2D(tDepth, vUv + direction * x).r;
      if (d <= 0.0) d = d0 + depthRange; // empty: treat as just behind, like a far sample
      if (d < d0 - depthRange) continue; // a nearer layer: not part of this surface
      d = min(d, d0 + depthRange);       // a farther layer: clamp
      float w = exp(-x * x / (2.0 * sigma * sigma));
      sum += d * w;
      weights += w;
    }
    gl_FragColor = vec4(sum / weights, 0.0, 0.0, 1.0);
  }
`

const compositeFragment = /* glsl */ `
  #include <packing>
  uniform sampler2D tScene;
  uniform sampler2D tSceneDepth;
  uniform sampler2D tFluid;
  uniform sampler2D tThickness;
  uniform vec2 texel;
  uniform float tanHalfFov;
  uniform float aspect;
  uniform float near;
  uniform float far;
  uniform vec3 lightDir; // view space, toward the light
  uniform mat3 viewToWorld;
  varying vec2 vUv;

  vec3 viewPosition(vec2 uv, float depth) {
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * tanHalfFov * aspect * depth, ndc.y * tanHalfFov * depth, -depth);
  }

  // Derivative of the surface along one screen axis, from whichever neighbour is
  // closer in depth (so silhouettes don't smear normals across a depth jump).
  vec3 derivative(vec2 uv, float depth, vec2 step) {
    vec3 p = viewPosition(uv, depth);
    float dPlus = texture2D(tFluid, uv + step).r;
    float dMinus = texture2D(tFluid, uv - step).r;
    vec3 forward = viewPosition(uv + step, dPlus) - p;
    vec3 backward = p - viewPosition(uv - step, dMinus);
    if (dPlus <= 0.0) return backward;
    if (dMinus <= 0.0) return forward;
    return abs(forward.z) < abs(backward.z) ? forward : backward;
  }

  void main() {
    vec3 scene = texture2D(tScene, vUv).rgb;
    float depth = texture2D(tFluid, vUv).r;
    float sceneDepth = -perspectiveDepthToViewZ(texture2D(tSceneDepth, vUv).r, near, far);
    if (depth <= 0.0 || sceneDepth < depth - 0.05) {
      gl_FragColor = vec4(scene, 1.0);
      #include <colorspace_fragment>
      return;
    }

    vec3 n = normalize(cross(derivative(vUv, depth, vec2(texel.x, 0.0)), derivative(vUv, depth, vec2(0.0, texel.y))));
    vec3 view = normalize(-viewPosition(vUv, depth));
    if (dot(n, view) < 0.0) n = -n;

    vec4 thick = texture2D(tThickness, vUv);
    float thickness = thick.a;
    vec3 albedo = thick.rgb / max(thickness, 1e-4);

    // Body colour: thin water shows the refracted scene, tinted by absorption;
    // thick water shows the lit albedo.
    vec2 refractUv = clamp(vUv + n.xy * 0.035 * min(thickness, 3.0), vec2(0.0), vec2(1.0));
    vec3 behind = texture2D(tScene, refractUv).rgb;
    vec3 transmitted = behind * exp(-thickness * (1.0 - albedo) * 0.9);
    float diffuse = 0.35 + 0.65 * max(dot(n, lightDir), 0.0);
    float opacity = 1.0 - exp(-thickness * 0.45);
    vec3 body = mix(transmitted, albedo * diffuse, opacity);

    // Sky reflection with Fresnel, plus a sun highlight.
    vec3 r = viewToWorld * reflect(-view, n);
    vec3 sky = mix(vec3(0.03, 0.05, 0.08), vec3(0.45, 0.62, 0.85), clamp(r.y * 0.5 + 0.5, 0.0, 1.0));
    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, view), 0.0), 5.0);
    float specular = pow(max(dot(n, normalize(lightDir + view)), 0.0), 80.0) * 0.6;

    // Soft edge where only a sliver of water covers the pixel.
    float coverage = smoothstep(0.0, 0.25, thickness);
    vec3 color = mix(body, sky, fresnel) + specular;
    gl_FragColor = vec4(mix(scene, color, coverage), 1.0);
    #include <colorspace_fragment>
  }
`

function fullscreenMaterial(fragmentShader: string, uniforms: Record<string, THREE.IUniform>) {
  return new THREE.ShaderMaterial({ vertexShader: fullscreenVertex, fragmentShader, uniforms, depthTest: false, depthWrite: false })
}

export class WaterSurface3D {
  private readonly depthMesh: THREE.InstancedMesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly thicknessMesh: THREE.InstancedMesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly sceneTarget: THREE.WebGLRenderTarget
  private readonly depthTarget: THREE.WebGLRenderTarget
  private readonly blurTargets: [THREE.WebGLRenderTarget, THREE.WebGLRenderTarget]
  private readonly thicknessTarget: THREE.WebGLRenderTarget
  private readonly quadScene = new THREE.Scene()
  private readonly quad: THREE.Mesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1)
  private readonly blurMaterial: THREE.ShaderMaterial
  private readonly compositeMaterial: THREE.ShaderMaterial
  private readonly size = new THREE.Vector2()
  private readonly sunWorld: THREE.Vector3

  // `particles` must already be in the group the fluid lives in; the impostor
  // meshes are added next to it. `sun` points toward the light (world space).
  constructor(particles: THREE.InstancedMesh, radius: number, sun: THREE.Vector3) {
    this.sunWorld = sun.clone().normalize()
    const geometry = new THREE.PlaneGeometry(2, 2)
    const impostorRadius = radius * IMPOSTOR_SCALE
    const makeImpostors = (material: THREE.ShaderMaterial, layer: number) => {
      const mesh = new THREE.InstancedMesh(geometry, material, particles.count)
      mesh.instanceMatrix = particles.instanceMatrix
      mesh.instanceColor = particles.instanceColor
      mesh.frustumCulled = false
      mesh.layers.set(layer)
      particles.parent!.add(mesh)
      return mesh
    }
    this.depthMesh = makeImpostors(
      new THREE.ShaderMaterial({
        vertexShader: impostorVertex,
        fragmentShader: depthFragment,
        uniforms: { radius: { value: impostorRadius }, uProjection: { value: new THREE.Matrix4() } },
      }),
      DEPTH_LAYER,
    )
    this.thicknessMesh = makeImpostors(
      new THREE.ShaderMaterial({
        vertexShader: impostorVertex,
        fragmentShader: thicknessFragment,
        uniforms: { radius: { value: impostorRadius } },
        depthTest: false,
        depthWrite: false,
        transparent: true,
        blending: THREE.CustomBlending,
        blendEquation: THREE.AddEquation,
        blendSrc: THREE.OneFactor,
        blendDst: THREE.OneFactor,
        blendSrcAlpha: THREE.OneFactor,
        blendDstAlpha: THREE.OneFactor,
      }),
      THICKNESS_LAYER,
    )

    const float = { type: THREE.FloatType, minFilter: THREE.NearestFilter, magFilter: THREE.NearestFilter }
    this.sceneTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthTexture: new THREE.DepthTexture(1, 1) })
    this.depthTarget = new THREE.WebGLRenderTarget(1, 1, { ...float, depthBuffer: true })
    this.blurTargets = [
      new THREE.WebGLRenderTarget(1, 1, { ...float, depthBuffer: false }),
      new THREE.WebGLRenderTarget(1, 1, { ...float, depthBuffer: false }),
    ]
    this.thicknessTarget = new THREE.WebGLRenderTarget(1, 1, { type: THREE.HalfFloatType, depthBuffer: false })

    this.blurMaterial = fullscreenMaterial(blurFragment, {
      tDepth: { value: null },
      direction: { value: new THREE.Vector2() },
      worldRadius: { value: BLUR_WORLD_RADIUS },
      focalPixels: { value: 1 },
      depthRange: { value: BLUR_DEPTH_RANGE },
    })
    this.compositeMaterial = fullscreenMaterial(compositeFragment, {
      tScene: { value: this.sceneTarget.texture },
      tSceneDepth: { value: this.sceneTarget.depthTexture },
      tFluid: { value: null },
      tThickness: { value: this.thicknessTarget.texture },
      texel: { value: new THREE.Vector2() },
      tanHalfFov: { value: 1 },
      aspect: { value: 1 },
      near: { value: 0.1 },
      far: { value: 100 },
      lightDir: { value: new THREE.Vector3() },
      viewToWorld: { value: new THREE.Matrix3() },
    })
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial)
    this.quad.frustumCulled = false
    this.quadScene.add(this.quad)
  }

  resize(gl: THREE.WebGLRenderer) {
    gl.getDrawingBufferSize(this.size)
    for (const t of [this.sceneTarget, this.depthTarget, ...this.blurTargets, this.thicknessTarget]) {
      t.setSize(this.size.x, this.size.y)
    }
  }

  // Renders the scene with the fluid drawn as a surface. `particles` is hidden
  // for the scene pass (the impostors replace it).
  render(gl: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, particles: THREE.InstancedMesh) {
    this.depthMesh.count = particles.count
    this.thicknessMesh.count = particles.count
    const clearColor = gl.getClearColor(new THREE.Color())
    const clearAlpha = gl.getClearAlpha()

    // 1. Everything but the fluid, with depth.
    particles.visible = false
    gl.setRenderTarget(this.sceneTarget)
    gl.clear()
    gl.render(scene, camera)
    particles.visible = true

    // 2–3. Fluid depth and thickness, each pass seeing only its own layer.
    const layers = camera.layers.mask
    gl.setClearColor(0x000000, 0)
    this.depthMesh.material.uniforms.uProjection.value.copy(camera.projectionMatrix)
    camera.layers.set(DEPTH_LAYER)
    gl.setRenderTarget(this.depthTarget)
    gl.clear()
    gl.render(scene, camera)
    camera.layers.set(THICKNESS_LAYER)
    gl.setRenderTarget(this.thicknessTarget)
    gl.clear()
    gl.render(scene, camera)
    camera.layers.mask = layers

    // 4. Bilateral blur, alternating horizontal and vertical passes.
    const tanHalfFov = Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2)
    this.quad.material = this.blurMaterial
    this.blurMaterial.uniforms.focalPixels.value = this.size.y / (2 * tanHalfFov)
    let source = this.depthTarget
    for (let i = 0; i < BLUR_ITERATIONS * 2; i++) {
      const target = this.blurTargets[i % 2]
      this.blurMaterial.uniforms.tDepth.value = source.texture
      this.blurMaterial.uniforms.direction.value.set(i % 2 === 0 ? 1 / this.size.x : 0, i % 2 === 0 ? 0 : 1 / this.size.y)
      gl.setRenderTarget(target)
      gl.render(this.quadScene, this.quadCamera)
      source = target
    }

    // 5. Shade to the screen.
    const u = this.compositeMaterial.uniforms
    u.tFluid.value = source.texture
    u.texel.value.set(1 / this.size.x, 1 / this.size.y)
    u.tanHalfFov.value = tanHalfFov
    u.aspect.value = camera.aspect
    u.near.value = camera.near
    u.far.value = camera.far
    u.lightDir.value.copy(this.sunWorld).transformDirection(camera.matrixWorldInverse)
    u.viewToWorld.value.setFromMatrix4(camera.matrixWorld)
    this.quad.material = this.compositeMaterial
    gl.setRenderTarget(null)
    gl.setClearColor(clearColor, clearAlpha)
    gl.render(this.quadScene, this.quadCamera)
  }

  dispose() {
    this.depthMesh.removeFromParent()
    this.thicknessMesh.removeFromParent()
    this.depthMesh.geometry.dispose()
    this.depthMesh.material.dispose()
    this.thicknessMesh.material.dispose()
    this.blurMaterial.dispose()
    this.compositeMaterial.dispose()
    this.quad.geometry.dispose()
    for (const t of [this.sceneTarget, this.depthTarget, ...this.blurTargets, this.thicknessTarget]) t.dispose()
  }
}
