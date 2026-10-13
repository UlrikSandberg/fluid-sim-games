import * as THREE from 'three'
import type { ShallowGrid } from '../sim/shallow/shallow'

// Renders a height-field water simulation (src/sim/shallow) so you can see into
// it. Each frame:
//   1. the scene without the water (floor, walls, anything under the surface) is
//      drawn into an off-screen target, colour and depth;
//   2. the scene is drawn again with the water on top. The water's shader looks
//      up what's behind it in that target, bent by the surface's slope
//      (refraction), and tints it by how far the light travels through the water
//      (absorption); it mixes in the sky's reflection by the viewing angle
//      (Fresnel), adds the sun's glint, and draws the simulation's foam.
// The floor's shader brightens and darkens it where the curved surface focuses
// or spreads the sunlight (caustics).
// The simulation reaches the GPU as one RGBA float texture per frame:
// surface elevation, depth, foam, bed.

export type WaterLook = 'real' | 'heights'

export interface WaterStyle {
  absorb: THREE.Vector3 // per metre, red / green / blue (red goes first)
  scatter: THREE.Color // the colour deep water fades to
  caustics: number // strength of the focusing
}

export const POOL_WATER: WaterStyle = {
  absorb: new THREE.Vector3(0.7, 0.16, 0.1),
  scatter: new THREE.Color(0x06506e),
  caustics: 3,
}

// The sky, shared by the dome and the water's reflections so they agree.
const SKY_GLSL = /* glsl */ `
uniform vec3 uSunDir;
vec3 skyColor(vec3 d) {
  float y = clamp(d.y, -0.2, 1.0);
  vec3 horizon = vec3(0.62, 0.76, 0.92);
  vec3 zenith = vec3(0.12, 0.33, 0.72);
  vec3 c = mix(horizon, zenith, pow(max(y, 0.0), 0.6));
  c = mix(c, vec3(0.42, 0.45, 0.48), smoothstep(0.0, -0.2, y)); // below the horizon
  float s = max(dot(d, uSunDir), 0.0);
  c += vec3(1.0, 0.9, 0.7) * (pow(s, 8.0) * 0.25 + pow(s, 250.0) * 2.0);
  return c;
}
`

const NOISE_GLSL = /* glsl */ `
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
`

// Field lookups shared by the water and the floor.
const FIELD_GLSL = /* glsl */ `
uniform sampler2D uField;
uniform vec2 uTexel; // one cell, in uv
uniform float uCell; // one cell, in metres
float eta(vec2 uv) { return texture2D(uField, uv).r; }
vec3 fieldNormal(vec2 uv) {
  float l = eta(uv - vec2(uTexel.x, 0.0));
  float r = eta(uv + vec2(uTexel.x, 0.0));
  float d = eta(uv - vec2(0.0, uTexel.y));
  float u = eta(uv + vec2(0.0, uTexel.y));
  return normalize(vec3(l - r, 2.0 * uCell, d - u));
}
float laplacian(vec2 uv) {
  float c = eta(uv);
  return (eta(uv - vec2(uTexel.x, 0.0)) + eta(uv + vec2(uTexel.x, 0.0)) + eta(uv - vec2(0.0, uTexel.y)) + eta(uv + vec2(0.0, uTexel.y)) - 4.0 * c) / (uCell * uCell);
}
`

const WATER_VERT = /* glsl */ `
uniform sampler2D uField;
varying vec2 vUv;
varying vec3 vWorld;
varying float vDepth;
void main() {
  vUv = uv;
  vec4 f = texture2D(uField, uv);
  vec3 p = position;
  // Dry cells tuck the surface under the bed, out of sight.
  p.y = f.g > 0.002 ? f.r : f.a - 0.05;
  vDepth = f.g;
  vec4 world = modelMatrix * vec4(p, 1.0);
  vWorld = world.xyz;
  gl_Position = projectionMatrix * viewMatrix * world;
}
`

const WATER_FRAG = /* glsl */ `
#include <packing>
${FIELD_GLSL}
${SKY_GLSL}
${NOISE_GLSL}
uniform sampler2D uScene;
uniform sampler2D uSceneDepth;
uniform vec2 uResolution;
uniform float uNear;
uniform float uFar;
uniform vec3 uAbsorb;
uniform vec3 uScatter;
uniform float uTime;
uniform float uLook; // 0 real, 1 heights
uniform float uRest;
uniform float uScale;
varying vec2 vUv;
varying vec3 vWorld;
varying float vDepth;

float viewZAt(vec2 uv) { return perspectiveDepthToViewZ(texture2D(uSceneDepth, uv).x, uNear, uFar); }

void main() {
  if (vDepth < 0.002) discard;
  vec3 n = fieldNormal(vUv);
  vec3 V = normalize(cameraPosition - vWorld);
  float cosT = max(dot(n, V), 0.0);
  float fresnel = 0.02 + 0.98 * pow(1.0 - cosT, 5.0);

  // Reflection: the sky, and the sun's glint.
  vec3 R = reflect(-V, n);
  vec3 refl = skyColor(R);
  float glint = pow(max(dot(R, uSunDir), 0.0), 900.0) * 12.0;

  // Refraction: what's behind, displaced by the slope, dimmed by the water it
  // passes through. If the displaced sample lands on something in front of the
  // water, fall back to looking straight through.
  vec2 suv = gl_FragCoord.xy / uResolution;
  float waterZ = perspectiveDepthToViewZ(gl_FragCoord.z, uNear, uFar);
  vec2 ruv = suv + n.xz * 0.25 * clamp(vDepth, 0.0, 1.5) / max(1.0, -waterZ * 0.08);
  float behindZ = viewZAt(ruv);
  if (behindZ > waterZ) {
    ruv = suv;
    behindZ = viewZAt(suv);
  }
  float travel = max(0.0, waterZ - behindZ);
  vec3 transmit = exp(-uAbsorb * travel);
  vec3 refr = texture2D(uScene, ruv).rgb * transmit + uScatter * (1.0 - transmit) * 0.6;

  vec3 col = mix(refr, refl, fresnel) + glint * vec3(1.0, 0.95, 0.85);

  // Foam: the simulation's foam, broken up into patches and bubbles.
  float foam = texture2D(uField, vUv).b;
  if (foam > 0.01) {
    float breakup = vnoise(vWorld.xz * 5.0 + uTime * 0.2) * 0.6 + vnoise(vWorld.xz * 19.0 - uTime * 0.4) * 0.4;
    float f = smoothstep(0.25, 0.65, foam * (0.45 + breakup));
    float lit = 0.75 + 0.25 * max(dot(n, uSunDir), 0.0);
    col = mix(col, vec3(0.93, 0.96, 0.97) * lit, f * 0.95);
  }

  // Heights: crests red, troughs blue, so interference patterns stand out.
  if (uLook > 0.5) {
    float e = clamp((eta(vUv) - uRest) / uScale, -1.0, 1.0);
    vec3 c = e > 0.0 ? mix(vec3(0.95), vec3(0.85, 0.15, 0.1), e) : mix(vec3(0.95), vec3(0.1, 0.3, 0.85), -e);
    col = c * (0.55 + 0.45 * max(dot(n, uSunDir), 0.0)) + glint * 0.3;
  }

  gl_FragColor = vec4(col, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

const FLOOR_VERT = /* glsl */ `
varying vec3 vWorld;
varying vec3 vNormal;
void main() {
  vec4 world = modelMatrix * vec4(position, 1.0);
  vWorld = world.xyz;
  vNormal = normalize(mat3(modelMatrix) * normal);
  gl_Position = projectionMatrix * viewMatrix * world;
}
`

const FLOOR_FRAG = /* glsl */ `
${FIELD_GLSL}
${NOISE_GLSL}
uniform vec3 uSunDir;
uniform vec2 uSize; // the field's extent in metres
uniform float uCaustics;
varying vec3 vWorld;
varying vec3 vNormal;
void main() {
  // Pool tiles: 30 cm squares, slightly varied, with grout lines.
  vec2 t = vWorld.xz / 0.3;
  vec2 g = abs(fract(t) - 0.5);
  float grout = smoothstep(0.455, 0.48, max(g.x, g.y));
  vec3 tile = vec3(0.80, 0.90, 0.92) * (0.93 + 0.07 * hash12(floor(t)));
  vec3 base = mix(tile, vec3(0.55, 0.64, 0.66), grout);

  vec2 uv = vWorld.xz / uSize;
  vec4 f = texture2D(uField, uv);
  // Caustics: a crest (negative curvature) focuses light on the floor below it.
  float wet = smoothstep(0.0, 0.05, f.g);
  float focus = 1.0 / clamp(1.0 + uCaustics * f.g * laplacian(uv), 0.3, 4.0);
  float sun = max(dot(normalize(vNormal), uSunDir), 0.0);
  float light = 0.35 + 0.75 * sun * mix(1.0, focus, wet);
  gl_FragColor = vec4(base * light, 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

const SKY_VERT = /* glsl */ `
varying vec3 vDir;
void main() {
  vDir = normalize((modelMatrix * vec4(position, 0.0)).xyz);
  vec4 p = projectionMatrix * viewMatrix * modelMatrix * vec4(position, 1.0);
  gl_Position = p.xyww; // always at the far plane
}
`

const SKY_FRAG = /* glsl */ `
${SKY_GLSL}
varying vec3 vDir;
void main() {
  gl_FragColor = vec4(skyColor(normalize(vDir)), 1.0);
  #include <tonemapping_fragment>
  #include <colorspace_fragment>
}
`

export class HeightfieldWater {
  readonly water: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>
  readonly floor: THREE.Mesh<THREE.BufferGeometry, THREE.ShaderMaterial>
  readonly sky: THREE.Mesh<THREE.SphereGeometry, THREE.ShaderMaterial>
  readonly sunDir = new THREE.Vector3(-0.45, 0.75, 0.5).normalize()
  private readonly field: THREE.DataTexture
  private readonly data: Float32Array
  private readonly target: THREE.WebGLRenderTarget
  private readonly nx: number
  private readonly ny: number

  constructor(gl: THREE.WebGLRenderer, grid: ShallowGrid, style: WaterStyle = POOL_WATER) {
    const { nx, ny, dx } = grid
    this.nx = nx
    this.ny = ny
    this.data = new Float32Array(nx * ny * 4)
    this.field = new THREE.DataTexture(this.data, nx, ny, THREE.RGBAFormat, THREE.FloatType)
    const linear = gl.extensions.has('OES_texture_float_linear')
    this.field.magFilter = this.field.minFilter = linear ? THREE.LinearFilter : THREE.NearestFilter
    this.field.wrapS = this.field.wrapT = THREE.ClampToEdgeWrapping

    const size = new THREE.Vector2(1, 1)
    gl.getDrawingBufferSize(size)
    this.target = new THREE.WebGLRenderTarget(size.x, size.y, {
      type: THREE.HalfFloatType,
      depthTexture: new THREE.DepthTexture(size.x, size.y),
    })

    const shared = {
      uField: { value: this.field },
      uTexel: { value: new THREE.Vector2(1 / nx, 1 / ny) },
      uCell: { value: dx },
      uSunDir: { value: this.sunDir },
    }
    this.water = new THREE.Mesh(
      gridGeometry(nx, ny, dx),
      new THREE.ShaderMaterial({
        vertexShader: WATER_VERT,
        fragmentShader: WATER_FRAG,
        uniforms: {
          ...shared,
          uScene: { value: this.target.texture },
          uSceneDepth: { value: this.target.depthTexture },
          uResolution: { value: size.clone() },
          uNear: { value: 0.1 },
          uFar: { value: 500 },
          uAbsorb: { value: style.absorb },
          uScatter: { value: style.scatter },
          uTime: { value: 0 },
          uLook: { value: 0 },
          uRest: { value: 0 },
          uScale: { value: 0.04 },
        },
      }),
    )
    this.water.frustumCulled = false
    this.floor = new THREE.Mesh(
      bedGeometry(grid),
      new THREE.ShaderMaterial({
        vertexShader: FLOOR_VERT,
        fragmentShader: FLOOR_FRAG,
        uniforms: { ...shared, uSize: { value: new THREE.Vector2(nx * dx, ny * dx) }, uCaustics: { value: style.caustics } },
      }),
    )
    this.sky = new THREE.Mesh(
      new THREE.SphereGeometry(400, 32, 16),
      new THREE.ShaderMaterial({ vertexShader: SKY_VERT, fragmentShader: SKY_FRAG, uniforms: { uSunDir: shared.uSunDir }, side: THREE.BackSide, depthWrite: false }),
    )
    this.sky.frustumCulled = false
  }

  // Copies the simulation into the field texture.
  update(grid: ShallowGrid, time: number) {
    const { h, bed, foam, solid, rest } = grid
    const d = this.data
    for (let k = 0; k < this.nx * this.ny; k++) {
      // Inside a wall, the surface carries on at rest (hidden by the wall), so
      // the water meets the wall without a seam.
      const s = solid[k]
      d[k * 4] = s ? rest[k] : h[k] + bed[k]
      d[k * 4 + 1] = s ? 1 : h[k]
      d[k * 4 + 2] = s ? 0 : foam[k]
      d[k * 4 + 3] = bed[k]
    }
    this.field.needsUpdate = true
    this.water.material.uniforms.uTime.value = time
  }

  // `scale`: the height (m) that shows as full red or blue in the heights look.
  setLook(look: WaterLook, restLevel: number, scale = 0.04) {
    const u = this.water.material.uniforms
    u.uLook.value = look === 'heights' ? 1 : 0
    u.uRest.value = restLevel
    u.uScale.value = scale
  }

  // The bed changed (a new set-up): rebuild the floor's shape.
  setBed(grid: ShallowGrid) {
    this.floor.geometry.dispose()
    this.floor.geometry = bedGeometry(grid)
  }

  resize(gl: THREE.WebGLRenderer) {
    const size = new THREE.Vector2()
    gl.getDrawingBufferSize(size)
    this.target.setSize(size.x, size.y)
    this.water.material.uniforms.uResolution.value.copy(size)
  }

  // Both passes: everything but the water into the target, then all of it.
  render(gl: THREE.WebGLRenderer, scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    const u = this.water.material.uniforms
    u.uNear.value = camera.near
    u.uFar.value = camera.far
    this.water.visible = false
    gl.setRenderTarget(this.target)
    gl.render(scene, camera)
    gl.setRenderTarget(null)
    this.water.visible = true
    gl.render(scene, camera)
  }

  dispose() {
    this.water.geometry.dispose()
    this.water.material.dispose()
    this.floor.geometry.dispose()
    this.floor.material.dispose()
    this.sky.geometry.dispose()
    this.sky.material.dispose()
    this.field.dispose()
    this.target.depthTexture?.dispose()
    this.target.dispose()
  }
}

// A grid of vertices at the cell centres, uv addressing the field texture; the
// outermost ones are pushed out to the grid's edge, to meet the walls.
function gridGeometry(nx: number, ny: number, dx: number): THREE.BufferGeometry {
  const pos = new Float32Array(nx * ny * 3)
  const uv = new Float32Array(nx * ny * 2)
  const edge = (i: number, n: number) => (i === 0 ? 0 : i === n - 1 ? n : i + 0.5) * dx
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i
      pos.set([edge(i, nx), 0, edge(j, ny)], k * 3)
      uv.set([(i + 0.5) / nx, (j + 0.5) / ny], k * 2)
    }
  }
  const index: number[] = []
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx - 1; i++) {
      const a = j * nx + i
      index.push(a, a + nx, a + 1, a + 1, a + nx, a + nx + 1)
    }
  }
  const geo = new THREE.BufferGeometry()
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3))
  geo.setAttribute('uv', new THREE.BufferAttribute(uv, 2))
  geo.setIndex(index)
  return geo
}

// The bed as a surface, edge to edge of the grid.
function bedGeometry(grid: ShallowGrid): THREE.BufferGeometry {
  const { nx, ny, dx, bed } = grid
  const geo = new THREE.PlaneGeometry(nx * dx, ny * dx, nx - 1, ny - 1)
  geo.rotateX(-Math.PI / 2)
  geo.translate((nx * dx) / 2, 0, (ny * dx) / 2)
  const p = geo.attributes.position as THREE.BufferAttribute
  for (let k = 0; k < p.count; k++) {
    const i = Math.min(nx - 1, Math.max(0, Math.floor(p.getX(k) / dx)))
    const j = Math.min(ny - 1, Math.max(0, Math.floor(p.getZ(k) / dx)))
    p.setY(k, bed[j * nx + i])
  }
  geo.computeVertexNormals()
  return geo
}
