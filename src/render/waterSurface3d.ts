import * as THREE from 'three'
import type { FoamFlow } from './foamField'

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
//
// In the realistic look (the "Water" colour mode) the particle colours carry
// (foam, spray, air) from foamField.ts instead of a colour. Then the water is
// clear and absorbs light with depth (red first), so you see what's under and
// behind it; the foam rides along with the depth through the smoothing and is
// drawn as lacy white patches; and spray particles are left out of the surface
// (they made blue blobs) and drawn afterwards as droplets.

const DEPTH_LAYER = 1
const THICKNESS_LAYER = 2
const SPRAY_LAYER = 3
const SPRAY_SCALE = 0.55 // droplet radius / particle radius
// Tuned by eye in the 3D tank: smaller blurs leave the wall-packed particle
// lattice visible as stripes, and the sun highlight then sparkles on every bump.
const IMPOSTOR_SCALE = 1.6 // impostor radius / particle radius, closing the gaps between spheres
const BLUR_WORLD_RADIUS = 1.4 // smoothing radius in world units (converted to pixels per fragment)
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
// In the realistic look it also carries the sphere's foam, and skips spray.
const depthFragment = /* glsl */ `
  uniform float radius;
  uniform mat4 uProjection;
  uniform float realistic;
  varying vec2 vOffset;
  varying vec3 vCenter;
  varying vec3 vColor;
  void main() {
    if (realistic > 0.5 && vColor.g > 0.5) discard;
    float r2 = dot(vOffset, vOffset);
    if (r2 >= 1.0) discard;
    vec3 p = vCenter + vec3(vOffset * radius, sqrt(1.0 - r2) * radius);
    vec4 clip = uProjection * vec4(p, 1.0);
    gl_FragDepth = clip.z / clip.w * 0.5 + 0.5;
    gl_FragColor = vec4(-p.z, realistic > 0.5 ? vColor.r : 0.0, 0.0, 1.0);
  }
`

// Adds the chord length through the sphere (thickness) and colour weighted by it.
const thicknessFragment = /* glsl */ `
  uniform float radius;
  uniform float realistic;
  varying vec2 vOffset;
  varying vec3 vColor;
  void main() {
    if (realistic > 0.5 && vColor.g > 0.5) discard;
    float r2 = dot(vOffset, vOffset);
    if (r2 >= 1.0) discard;
    float t = 2.0 * sqrt(1.0 - r2) * radius;
    // (In the realistic look the colour's blue is the churned-in air, so the
    // weighted sum averages it through the depth.)
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
    float foam = 0.0; // the realistic look's foam, smoothed with the surface
    float weights = 0.0;
    for (int i = -MAX_BLUR; i <= MAX_BLUR; i++) {
      float x = float(i);
      if (abs(x) > 2.0 * sigma) continue;
      vec2 sampleDF = texture2D(tDepth, vUv + direction * x).rg;
      float d = sampleDF.r;
      if (d <= 0.0) d = d0 + depthRange; // empty: treat as just behind, like a far sample
      if (d < d0 - depthRange) continue; // a nearer layer: not part of this surface
      d = min(d, d0 + depthRange);       // a farther layer: clamp
      float w = exp(-x * x / (2.0 * sigma * sigma));
      sum += d * w;
      foam += sampleDF.g * w;
      weights += w;
    }
    gl_FragColor = vec4(sum / weights, foam / weights, 0.0, 1.0);
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
  uniform vec3 cameraWorld;
  uniform float realistic;
  uniform float time;
  // The surface flow (FoamField's grid, tank coordinates) and world → tank.
  uniform sampler2D tFlow;
  uniform float hasFlow;
  uniform vec2 flowOrigin;
  uniform float flowCell;
  uniform vec2 flowCount;
  uniform mat4 toTank;
  varying vec2 vUv;

  // Clear water: absorption per world unit (red goes first) and the colour
  // light scattered back from deep inside turns.
  const vec3 ABSORB = vec3(0.30, 0.075, 0.045);
  const vec3 INSCATTER = vec3(0.015, 0.16, 0.26);

  float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
  }
  float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    vec2 u = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash12(i), hash12(i + vec2(1, 0)), u.x), mix(hash12(i + vec2(0, 1)), hash12(i + vec2(1, 1)), u.x), u.y);
  }
  // Distances to the nearest and second-nearest of a jittered grid of points
  // (Voronoi). F2 − F1 is small along the borders between cells: a network of
  // threads, like the lace foam leaves on water.
  vec2 voronoi(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float f1 = 9.0;
    float f2 = 9.0;
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        vec2 o = vec2(float(x), float(y));
        vec2 c = o + vec2(hash12(i + o), hash12(i + o + 17.0)) - f;
        float d = dot(c, c);
        if (d < f1) {
          f2 = f1;
          f1 = d;
        } else if (d < f2) {
          f2 = d;
        }
      }
    }
    return sqrt(vec2(f1, f2));
  }
  // Distance to the edge of the nearest small bubble (negative inside); each
  // swells on its own clock and pops.
  float bubbleEdge(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    float best = 9.0;
    for (int y = -1; y <= 1; y++) {
      for (int x = -1; x <= 1; x++) {
        vec2 o = vec2(float(x), float(y));
        float h = hash12(i + o);
        vec2 c = o + vec2(h, hash12(i + o + 17.0)) - f;
        float life = fract(time * (0.35 + 0.5 * h) + h * 7.0);
        best = min(best, length(c) - (0.08 + 0.3 * life));
      }
    }
    return best;
  }
  // How much of the surface the foam covers here: where it's thin, a lace of
  // threads (wider as it thickens); where it's thick, a solid layer dotted
  // with small popping bubbles.
  float foamCover(float foam, vec2 p) {
    float patches = vnoise(p * 0.7) * 0.6 + vnoise(p * 2.3) * 0.4;
    float f = foam * (0.5 + 0.8 * patches);
    // Warped cells, so the lace is irregular rather than a honeycomb, and only
    // some of its threads showing, so it's broken.
    vec2 warp = vec2(vnoise(p * 1.3), vnoise(p * 1.3 + 5.2)) - 0.5;
    vec2 v = voronoi(p * 2.2 + warp * 1.6);
    float threads = smoothstep(0.4, 0.75, vnoise(p * 3.7 + 11.0) * 0.8 + f * 0.6);
    float lace = (1.0 - smoothstep(0.015, 0.05 + 0.25 * f, v.y - v.x)) * threads * smoothstep(0.25, 0.5, f);
    float sheet = smoothstep(0.45, 0.8, f);
    float holes = smoothstep(0.03, -0.03, bubbleEdge(p * 11.0)) * 0.45;
    return max(lace, sheet * (1.0 - holes));
  }
  // The foam's texture carried along by the flow v: two copies, each dragged
  // for a short cycle and then reset, crossfaded so a reset never shows (a
  // flow map), so the pattern travels with the water without stretching.
  float flowingFoam(float foam, vec2 p, vec2 v) {
    const float CYCLE = 0.7;
    float ph = fract(time / CYCLE);
    float a = foamCover(foam, p - v * ph * CYCLE);
    float b = foamCover(foam, p - v * fract(ph + 0.5) * CYCLE + vec2(7.31, 3.17));
    return mix(a, b, abs(1.0 - 2.0 * ph));
  }

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
    float diffuse = 0.35 + 0.65 * max(dot(n, lightDir), 0.0);
    vec3 r = viewToWorld * reflect(-view, n);
    float fresnel = 0.02 + 0.98 * pow(1.0 - max(dot(n, view), 0.0), 5.0);
    float coverage = smoothstep(0.0, 0.25, thickness); // soft edge where only a sliver covers the pixel
    vec3 color;

    if (realistic > 0.5) {
      // Clear water: the scene behind, bent by the surface and dimmed by the
      // water the light crosses, plus light scattered back from inside it.
      vec2 refractUv = clamp(vUv + n.xy * 0.03 * min(thickness, 6.0), vec2(0.0), vec2(1.0));
      vec3 behind = texture2D(tScene, refractUv).rgb;
      vec3 transmit = exp(-ABSORB * thickness);
      vec3 body = behind * transmit + INSCATTER * (1.0 - transmit) * (0.5 + 0.5 * diffuse);
      // Churned-in air (the thickness pass averaged it through the depth) turns
      // the water milky white.
      float air = smoothstep(0.3, 1.0, thick.b / max(thickness, 1e-4));
      body = mix(body, vec3(0.80, 0.88, 0.92) * (0.6 + 0.4 * diffuse), air * 0.4);
      vec3 sky = mix(vec3(0.05, 0.07, 0.10), vec3(0.55, 0.70, 0.90), clamp(r.y * 0.5 + 0.5, 0.0, 1.0));
      float glint = pow(max(dot(n, normalize(lightDir + view)), 0.0), 400.0) * 1.2;
      float sheen = pow(max(dot(n, normalize(lightDir + view)), 0.0), 40.0) * 0.15;
      color = mix(body, sky, fresnel) + glint + sheen;
      // Foam, in the water's own frame (the tank's), moving with its flow.
      float foam = texture2D(tFluid, vUv).g;
      if (foam > 0.01) {
        vec3 world = viewToWorld * viewPosition(vUv, depth) + cameraWorld;
        vec3 p = (toTank * vec4(world, 1.0)).xyz;
        // The pattern lies in the plane the surface mostly faces, so it doesn't
        // smear into streaks on steep faces; there it drifts down.
        vec3 nt = abs(mat3(toTank) * (viewToWorld * n));
        vec2 flowUv = ((p.xz - flowOrigin) / flowCell + 0.5) / flowCount;
        vec2 v = hasFlow > 0.5 ? texture2D(tFlow, flowUv).rg : vec2(0.0);
        float cover = nt.y > 0.6
          ? flowingFoam(foam, p.xz, v)
          : flowingFoam(foam, nt.x > nt.z ? p.zy : p.xy, vec2(0.0, -2.0));
        vec3 foamColor = vec3(0.93, 0.96, 1.0) * (0.55 + 0.45 * diffuse);
        color = mix(color, foamColor, cover);
        coverage = max(coverage, cover);
      }
    } else {
      // Colour modes: thin water shows the refracted scene, tinted by
      // absorption; thick water shows the lit colour.
      vec3 albedo = thick.rgb / max(thickness, 1e-4);
      vec2 refractUv = clamp(vUv + n.xy * 0.035 * min(thickness, 3.0), vec2(0.0), vec2(1.0));
      vec3 behind = texture2D(tScene, refractUv).rgb;
      vec3 transmitted = behind * exp(-thickness * (1.0 - albedo) * 0.9);
      float opacity = 1.0 - exp(-thickness * 0.45);
      vec3 body = mix(transmitted, albedo * diffuse, opacity);
      vec3 sky = mix(vec3(0.03, 0.05, 0.08), vec3(0.45, 0.62, 0.85), clamp(r.y * 0.5 + 0.5, 0.0, 1.0));
      float specular = pow(max(dot(n, normalize(lightDir + view)), 0.0), 80.0) * 0.6;
      color = mix(body, sky, fresnel) + specular;
    }
    gl_FragColor = vec4(mix(scene, color, coverage), 1.0);
    #include <colorspace_fragment>
  }
`

// Spray: the lone particles, as small soft white droplets drawn over the
// composite, hidden behind the scene and behind the water's surface.
const sprayVertex = /* glsl */ `
  uniform float radius;
  varying vec2 vOffset;
  void main() {
    vec4 center = modelViewMatrix * instanceMatrix * vec4(0.0, 0.0, 0.0, 1.0);
    vOffset = position.xy;
    #ifdef USE_INSTANCING_COLOR
      bool spray = instanceColor.g > 0.5;
    #else
      bool spray = false;
    #endif
    gl_Position = spray ? projectionMatrix * (center + vec4(position.xy * radius, 0.0, 0.0)) : vec4(2.0, 2.0, 2.0, 1.0);
  }
`

const sprayFragment = /* glsl */ `
  #include <packing>
  uniform sampler2D tSceneDepth;
  uniform sampler2D tFluid;
  uniform vec2 resolution;
  uniform float near;
  uniform float far;
  varying vec2 vOffset;
  void main() {
    float r2 = dot(vOffset, vOffset);
    if (r2 >= 1.0) discard;
    vec2 uv = gl_FragCoord.xy / resolution;
    float here = -perspectiveDepthToViewZ(gl_FragCoord.z, near, far);
    float scene = -perspectiveDepthToViewZ(texture2D(tSceneDepth, uv).r, near, far);
    float water = texture2D(tFluid, uv).r;
    if (here > scene || (water > 0.0 && here > water + 0.1)) discard;
    float a = (1.0 - r2) * 0.85;
    gl_FragColor = vec4(vec3(0.92, 0.95, 1.0), a);
    #include <colorspace_fragment>
  }
`

function fullscreenMaterial(fragmentShader: string, uniforms: Record<string, THREE.IUniform>) {
  return new THREE.ShaderMaterial({ vertexShader: fullscreenVertex, fragmentShader, uniforms, depthTest: false, depthWrite: false })
}

export class WaterSurface3D {
  private readonly depthMesh: THREE.InstancedMesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly thicknessMesh: THREE.InstancedMesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
  private readonly sprayMesh: THREE.InstancedMesh<THREE.PlaneGeometry, THREE.ShaderMaterial>
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
  private flowTexture: THREE.DataTexture | null = null
  private flowFilterSet = false
  private readonly startedAt = performance.now()

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
        uniforms: { radius: { value: impostorRadius }, uProjection: { value: new THREE.Matrix4() }, realistic: { value: 0 } },
      }),
      DEPTH_LAYER,
    )
    this.thicknessMesh = makeImpostors(
      new THREE.ShaderMaterial({
        vertexShader: impostorVertex,
        fragmentShader: thicknessFragment,
        uniforms: { radius: { value: impostorRadius }, realistic: { value: 0 } },
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
    this.sprayMesh = makeImpostors(
      new THREE.ShaderMaterial({
        vertexShader: sprayVertex,
        fragmentShader: sprayFragment,
        uniforms: {
          radius: { value: radius * SPRAY_SCALE },
          tSceneDepth: { value: this.sceneTarget.depthTexture },
          tFluid: { value: null },
          resolution: { value: new THREE.Vector2(1, 1) },
          near: { value: 0.1 },
          far: { value: 100 },
        },
        transparent: true,
        depthTest: false,
        depthWrite: false,
      }),
      SPRAY_LAYER,
    )

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
      cameraWorld: { value: new THREE.Vector3() },
      realistic: { value: 0 },
      time: { value: 0 },
      tFlow: { value: null },
      hasFlow: { value: 0 },
      flowOrigin: { value: new THREE.Vector2() },
      flowCell: { value: 1 },
      flowCount: { value: new THREE.Vector2(1, 1) },
      toTank: { value: new THREE.Matrix4() },
    })
    this.quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.compositeMaterial)
    this.quad.frustumCulled = false
    this.quadScene.add(this.quad)
  }

  // The foam field's surface flow, uploaded as a texture each frame.
  setFlow(flow: FoamFlow) {
    if (!this.flowTexture || this.flowTexture.image.width !== flow.nx || this.flowTexture.image.height !== flow.nz) {
      this.flowTexture?.dispose()
      this.flowTexture = new THREE.DataTexture(flow.data, flow.nx, flow.nz, THREE.RGFormat, THREE.FloatType)
      this.flowTexture.wrapS = this.flowTexture.wrapT = THREE.ClampToEdgeWrapping
      this.flowFilterSet = false
    }
    this.flowTexture.image.data = flow.data
    this.flowTexture.needsUpdate = true
    const u = this.compositeMaterial.uniforms
    u.tFlow.value = this.flowTexture
    u.hasFlow.value = 1
    u.flowOrigin.value.set(flow.x0, flow.z0)
    u.flowCell.value = flow.cell
    u.flowCount.value.set(flow.nx, flow.nz)
  }

  resize(gl: THREE.WebGLRenderer) {
    gl.getDrawingBufferSize(this.size)
    for (const t of [this.sceneTarget, this.depthTarget, ...this.blurTargets, this.thicknessTarget]) {
      t.setSize(this.size.x, this.size.y)
    }
  }

  // Renders the scene with the fluid drawn as a surface. `particles` is hidden
  // for the scene pass (the impostors replace it). `realistic`: the particle
  // colours are (foam, spray, 0) from FoamField, not colours.
  render(gl: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera, particles: THREE.InstancedMesh, realistic = false) {
    this.depthMesh.count = particles.count
    this.thicknessMesh.count = particles.count
    this.sprayMesh.count = particles.count
    const look = realistic ? 1 : 0
    this.depthMesh.material.uniforms.realistic.value = look
    this.thicknessMesh.material.uniforms.realistic.value = look
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
    u.cameraWorld.value.setFromMatrixPosition(camera.matrixWorld)
    u.realistic.value = look
    u.time.value = (performance.now() - this.startedAt) / 1000
    u.toTank.value.copy(particles.matrixWorld).invert()
    if (this.flowTexture && !this.flowFilterSet) {
      // Smooth flow where float textures can be filtered; blocky otherwise.
      const linear = gl.extensions.has('OES_texture_float_linear')
      this.flowTexture.magFilter = this.flowTexture.minFilter = linear ? THREE.LinearFilter : THREE.NearestFilter
      this.flowTexture.needsUpdate = true
      this.flowFilterSet = true
    }
    this.quad.material = this.compositeMaterial
    gl.setRenderTarget(null)
    gl.setClearColor(clearColor, clearAlpha)
    gl.render(this.quadScene, this.quadCamera)

    // 6. Spray droplets on top.
    if (realistic) {
      const s = this.sprayMesh.material.uniforms
      s.tFluid.value = source.texture
      s.resolution.value.copy(this.size)
      s.near.value = camera.near
      s.far.value = camera.far
      const autoClear = gl.autoClear
      gl.autoClear = false
      camera.layers.set(SPRAY_LAYER)
      gl.render(scene, camera)
      camera.layers.mask = layers
      gl.autoClear = autoClear
    }
  }

  dispose() {
    this.depthMesh.removeFromParent()
    this.thicknessMesh.removeFromParent()
    this.sprayMesh.removeFromParent()
    this.sprayMesh.material.dispose()
    this.depthMesh.geometry.dispose()
    this.depthMesh.material.dispose()
    this.thicknessMesh.material.dispose()
    this.blurMaterial.dispose()
    this.compositeMaterial.dispose()
    this.flowTexture?.dispose()
    this.quad.geometry.dispose()
    for (const t of [this.sceneTarget, this.depthTarget, ...this.blurTargets, this.thicknessTarget]) t.dispose()
  }
}
