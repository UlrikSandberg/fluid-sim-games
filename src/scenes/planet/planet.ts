import { defaultParams3D, particleRadius, type Params } from '../../sim/params.ts'
import { boundaryWeightsFor } from '../../sim/sph.ts'
import { Terrain } from './terrain.ts'

// A water planet: a rocky ball whose ground is a sculptable height map
// (terrain.ts), centred on the origin, holding an ocean by its own gravity.
// Gravity is one body at the centre (Newton: G·M/r², strength set so it's `g`
// at sea level), so "down" is toward the centre everywhere and gets weaker with
// height. The rock is boundary particles (like the river's bed), the water the
// SPH fluid. Pure, so the checks can build one headless.

export interface PlanetLayout {
  radius: number // sea level
  ocean: number // the deepest sea floor, below sea level (generated planets)
  relief: number // the highest mountains, above sea level (generated planets)
  seed: number
  spacing: number // particle spacing; h = 2·spacing
  blank: boolean // a smooth ball just above sea level, to sculpt from scratch
}

export const defaultPlanet: Readonly<PlanetLayout> = {
  radius: 10,
  ocean: 2.4,
  relief: 1.6,
  seed: 7,
  spacing: 0.5,
  blank: false,
}

const MARGIN = 5 // room above the highest ground for splashes, inside the domain's walls
const COAST = 0.12 // the noise value at the coastline: about 30% of the surface is land
// The rock is a band of boundary particles this deep (in smoothing lengths,
// measured straight down), so even under a steep coast water can't leak through.
const ROCK_DEPTH = 3
// The rock's particles fill the lattice sites under its surface, so its top
// layer lands anywhere up to a spacing below the true surface, and water came to
// rest up to half a spacing inside the drawn rock. Filling up to this far above
// the surface puts the surface the water rests on where the rock is drawn.
const ROCK_LIFT = 0.5 // in spacings
// The lattice settles about this much as the water finds its packing, so a new
// ocean is seeded this much above sea level and settles to it.
const SETTLE = 0.4
const BLANK_HEIGHT = 0.3 // a blank ball's ground, just above the sea
// Water this far inside the rock's surface counts as buried when the rock is
// rebuilt. Settled water rests up to half a spacing into it (the lattice's top
// layer sits below the surface), and moving that disturbed the water: every
// rebuild flung a few hundred particles off as spray.
const BURIED = 1 // in spacings

// How far the brush may sculpt: from deep basins to tall peaks.
export const sculptLimits = (l: PlanetLayout) => ({ min: -0.35 * l.radius, max: 0.3 * l.radius })

// --- Generated terrain: fractal noise over the sphere ---

function hash3(x: number, y: number, z: number, seed: number): number {
  let h = Math.imul(x, 374761393) ^ Math.imul(y, 668265263) ^ Math.imul(z, 2147483647) ^ Math.imul(seed, 1274126177)
  h = Math.imul(h ^ (h >>> 13), 1274126177)
  return ((h ^ (h >>> 16)) >>> 0) / 4294967296
}

function noise3(x: number, y: number, z: number, seed: number): number {
  const ix = Math.floor(x)
  const iy = Math.floor(y)
  const iz = Math.floor(z)
  const s = (t: number) => t * t * (3 - 2 * t)
  const fx = s(x - ix)
  const fy = s(y - iy)
  const fz = s(z - iz)
  const lerp = (a: number, b: number, t: number) => a + (b - a) * t
  const c = (dx: number, dy: number, dz: number) => hash3(ix + dx, iy + dy, iz + dz, seed)
  return (
    lerp(
      lerp(lerp(c(0, 0, 0), c(1, 0, 0), fx), lerp(c(0, 1, 0), c(1, 1, 0), fx), fy),
      lerp(lerp(c(0, 0, 1), c(1, 0, 1), fx), lerp(c(0, 1, 1), c(1, 1, 1), fx), fy),
      fz,
    ) *
      2 -
    1
  )
}

// The generated ground's height above sea level in direction `dir` (unit
// vector). Land rises gently from the coast to mountains inland (up to
// `relief`); off the coast the sea floor drops quickly to the deep ocean (down to
// `ocean`).
export function generatedHeight(l: PlanetLayout, dir: number[]): number {
  if (l.blank) return BLANK_HEIGHT
  let v = 0
  let amp = 0.55
  let f = 1.6
  for (let k = 0; k < 4; k++) {
    v += amp * noise3(dir[0] * f + 11, dir[1] * f + 23, dir[2] * f + 37, l.seed + k)
    f *= 2.1
    amp *= 0.5
  }
  if (v > COAST) return l.relief * Math.min(1, (v - COAST) / 0.4) ** 1.4
  return -l.ocean * Math.min(1, (COAST - v) / 0.35) ** 0.8
}

// A planet's terrain, generated, on a grid about as fine as its particles.
export function makeTerrain(l: PlanetLayout): Terrain {
  const n = Math.min(160, Math.max(48, Math.ceil((l.radius * Math.PI) / 2 / l.spacing) + 1))
  const { min, max } = sculptLimits(l)
  return new Terrain(n, min, max, (dir) => generatedHeight(l, dir))
}

export const seaRadius = (l: PlanetLayout) => l.radius
export const groundRadius = (l: PlanetLayout, t: Terrain, dir: number[]) => l.radius + t.heightAt(dir[0], dir[1], dir[2])

// --- The simulation ---

export function planetParams(l: PlanetLayout, g: number): Params {
  const h = 2 * l.spacing
  const half = l.radius + sculptLimits(l).max + MARGIN
  return {
    ...structuredClone(defaultParams3D),
    spacing: l.spacing,
    h,
    dt: 0.004 * h,
    domain: { min: [-half, -half, -half], max: [half, half, half] },
    gravity: [0, 0, 0],
    // The ground drags on water touching it. Free-slip, a thin film poured on
    // dry land never slowed: it raced right round the planet, met itself on
    // the far side and splashed off into space.
    wallFriction: 1.2,
    // Inside the deepest rock the pull eases off to zero at the centre (nothing
    // should get there).
    attractors: [{ center: [0, 0, 0], gm: g * l.radius * l.radius, radius: l.radius + sculptLimits(l).min }],
  }
}

// What the GPU solver starts from (a GpuScene: fluid, velocities, capacity, and
// static boundary particles with their ψ), kept free of the GPU module so this
// runs in Node.
export interface PlanetScene {
  fluid: Float32Array
  velocity?: Float32Array
  capacity: number
  boundary: { pos: Float32Array; psi: Float32Array }
}

// Every lattice point in the shell between radii r0 and r1, in rows along x,
// skipping the hollow middle.
function shellLattice(sp: number, r0: number, r1: number, f: (x: number, y: number, z: number, d: number) => void) {
  const start = (a: number) => Math.ceil((a - sp / 2) / sp) * sp + sp / 2 // first lattice value ≥ a
  for (let z = start(-r1); z < r1; z += sp) {
    for (let y = start(-r1); y < r1; y += sp) {
      const yz = y * y + z * z
      if (yz > r1 * r1) continue
      const xOut = Math.sqrt(r1 * r1 - yz)
      const xIn = yz < r0 * r0 ? Math.sqrt(r0 * r0 - yz) : 0
      for (let x = start(-xOut); x <= xOut; x += sp) {
        if (Math.abs(x) < xIn) {
          x = start(xIn) - sp // jump over the hollow
          continue
        }
        f(x, y, z, Math.hypot(x, y, z))
      }
    }
  }
}

// The rock (boundary particles in a band under the ground) and the water:
// `water` (positions and velocities, stride 3) carried over from before, any of
// it inside the new rock moved out on top of its column; or, without, a fresh ocean
// filling the low ground up to sea level. Room for `extra` more water.
export function planetScene(p: Params, l: PlanetLayout, t: Terrain, extra: number, water?: { pos: Float32Array; vel: Float32Array }): PlanetScene {
  const sp = p.spacing
  const r = particleRadius(p)
  const thick = ROCK_DEPTH * p.h
  const lift = ROCK_LIFT * sp
  const sea = seaRadius(l)
  const inner = l.radius + t.min - thick - sp
  const outer = Math.max(l.radius + t.max + lift, sea + SETTLE) + sp
  const wall: number[] = []
  const fluid: number[] = []
  shellLattice(sp, inner, outer, (x, y, z, d) => {
    const top = l.radius + t.heightAt(x, y, z) + lift
    if (d <= top && d > top - lift - thick) wall.push(x, y, z)
    else if (!water && d > top + r && d <= sea + SETTLE) fluid.push(x, y, z)
  })
  const boundary = { pos: Float32Array.from(wall), psi: boundaryWeightsFor(p, wall, [sp, sp, sp]) }
  if (!water) return { fluid: Float32Array.from(fluid), capacity: fluid.length / 3 + extra, boundary }
  // Water the new rock buried goes on top of its own column of water (columns
  // being directions binned a spacing wide), a spacing apart: dropped straight
  // onto the rock's surface it would land inside the water already there, and
  // the overlap would fling it off as spray.
  const pos = Float32Array.from(water.pos)
  const vel = Float32Array.from(water.vel)
  const count = pos.length / 3
  const column = (x: number, y: number, z: number) => {
    const d = Math.hypot(x, y, z) || 1
    const k = (l.radius + t.max) / sp // column width ≈ a spacing at the highest ground
    return `${Math.round((x / d) * k)},${Math.round((y / d) * k)},${Math.round((z / d) * k)}`
  }
  const buried: number[] = []
  const tops = new Map<string, number>()
  for (let i = 0; i < count; i++) {
    const x = pos[i * 3]
    const y = pos[i * 3 + 1]
    const z = pos[i * 3 + 2]
    const d = Math.hypot(x, y, z)
    const top = l.radius + t.heightAt(x, y, z) + lift
    if (d < top - BURIED * sp) buried.push(i)
    else {
      const key = column(x, y, z)
      tops.set(key, Math.max(tops.get(key) ?? 0, d))
    }
  }
  for (const i of buried) {
    const x = pos[i * 3]
    const y = pos[i * 3 + 1]
    const z = pos[i * 3 + 2]
    const d = Math.hypot(x, y, z) || 1
    const key = column(x, y, z)
    const ground = l.radius + t.heightAt(x, y, z) + lift + r
    const to = Math.max(ground, (tops.get(key) ?? 0) + sp)
    tops.set(key, to)
    pos.set([(x / d) * to, (y / d) * to, (z / d) * to], i * 3)
    vel.set([0, 0, 0], i * 3)
  }
  return { fluid: pos, velocity: vel, capacity: count + extra, boundary }
}

// Roughly how many particles a layout makes, rock and water, before building it
// (for the panel), from the generated heights in sampled directions.
export function estimateParticles(l: PlanetLayout): { rock: number; water: number } {
  const sp = l.spacing
  const band = ROCK_DEPTH * 2 * sp + ROCK_LIFT * sp
  let rock = 0
  let water = 0
  const N = 600
  for (let k = 0; k < N; k++) {
    const z = 1 - (2 * (k + 0.5)) / N
    const s = Math.sqrt(1 - z * z)
    const h = generatedHeight(l, [s * Math.cos(k * 2.39996), z, s * Math.sin(k * 2.39996)])
    const ground = l.radius + h
    rock += (ground ** 3 - (ground - band) ** 3) / 3 // the band's volume per steradian
    if (h < 0) water += (l.radius ** 3 - ground ** 3) / 3
  }
  const perSample = (4 * Math.PI) / N / sp ** 3
  return { rock: Math.round(rock * perSample), water: Math.round(water * perSample) }
}

// Where the ocean's surface is in direction `dir`: the farthest fluid particle
// within `cone` (cosine) of it. For the checks.
export function surfaceRadius(pos: Float32Array, n: number, dir: number[], cone = 0.995): number {
  let best = 0
  for (let i = 0; i < n; i++) {
    const x = pos[i * 3]
    const y = pos[i * 3 + 1]
    const z = pos[i * 3 + 2]
    const d = Math.hypot(x, y, z)
    if (d > best && (x * dir[0] + y * dir[1] + z * dir[2]) / d > cone) best = d
  }
  return best
}
