// Archimedes check for the dynamic obstacle: a box dropped onto a settled pool
// must float level with draft = (ρ_body/ρ₀) · height, and a box denser than the
// fluid must sink to the floor (resting h above it: the contact gap that keeps a
// film of fluid underneath) and lie flat. That sunk box, made light, must then
// lift off promptly rather than stick like a suction cup. Nothing in the solver
// models buoyancy directly; it has to come out of the fluid's pressure forces on
// the body's boundary particles.
// Usage: node scripts/buoyancy.ts [--3d]
import { defaultParams, defaultParams3D, dimOf, particleRadius, type Params } from '../src/sim/params.ts'
import { createState, setObstacle, step, type Obstacle, type SimState } from '../src/sim/sph.ts'
import { rotate } from '../src/sim/quat.ts'

const is3D = process.argv.includes('--3d')
const preset = is3D
  ? { params: defaultParams3D, size: [4, 2, 3], dropAt: [8, 8, 4], settle: 15, float: 20 }
  : { params: defaultParams, size: [6, 3], dropAt: [20, 22], settle: 40, float: 30 }
const AVERAGE_LAST = 5 // seconds averaged at the end, to smooth out bobbing
const LIFT_SECONDS = 8
const LIFT_MIN_FRACTION = 0.8 // of the way from resting on the floor to floating

const base: Params = { ...structuredClone(preset.params), obstacleDynamic: true }
const DIM = dimOf(base)
const settled = createState(base)
for (let k = 0; k < preset.settle / base.dt; k++) step(settled, base)

// The body's lowest point, and how far its most-vertical axis is from vertical
// (0 when it lies flat on a face).
function pose(o: Obstacle) {
  let below = 0
  let mostVertical = 0
  for (let k = 0; k < DIM; k++) {
    const axis = rotate(o.q, [k === 0 ? 1 : 0, k === 1 ? 1 : 0, k === 2 ? 1 : 0])
    below += Math.abs(axis[1]) * o.halfSize[k]
    mostVertical = Math.max(mostVertical, Math.abs(axis[1]))
  }
  return { bottom: o.center[1] - below, tilt: Math.acos(Math.min(1, mostVertical)) }
}

// Fluid surface height away from the box: mean of the top particle per column
// (1×1 cells in x, and z in 3D) at least 1 from the walls and more than 2 units
// beyond the box's footprint, plus the particle radius (centres sit half a
// spacing below the surface).
function surfaceLevel(s: SimState, p: Params, o: Obstacle): number {
  const tops = new Map<number, number>()
  const { min, max } = p.domain
  for (let i = 0; i < s.n; i++) {
    let key = 0
    let nearWall = false
    let farFromBox = false
    for (const d of DIM === 3 ? [0, 2] : [0]) {
      const x = s.pos[i * DIM + d]
      if (x < min[d] + 1 || x > max[d] - 1) nearWall = true
      if (Math.abs(x - o.center[d]) >= o.halfSize[d] + 2) farFromBox = true
      key = key * 1000 + Math.floor(x - min[d])
    }
    if (!nearWall && farFromBox) tops.set(key, Math.max(tops.get(key) ?? -Infinity, s.pos[i * DIM + 1]))
  }
  const heights = [...tops.values()]
  return heights.reduce((a, b) => a + b, 0) / heights.length + particleRadius(p)
}

const results: boolean[] = []
const check = (name: string, ok: boolean, detail: string) => {
  results.push(ok)
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
}

console.log(`${DIM}D, box ${preset.size.join('×')}`)
const floatingBottom = new Map<number, number>()
for (const ratio of [0.25, 0.5, 0.75, 1.5]) {
  const p: Params = { ...structuredClone(base), obstacleDensity: ratio }
  const s = structuredClone(settled)
  setObstacle(s, p, { center: preset.dropAt, size: preset.size })
  const o = s.obstacle!
  let sumBottom = 0
  let sumTilt = 0
  let sumSurface = 0
  let samples = 0
  const steps = Math.round(preset.float / p.dt)
  const averageFrom = steps - Math.round(AVERAGE_LAST / p.dt)
  for (let k = 0; k < steps; k++) {
    step(s, p)
    if (k >= averageFrom && k % 25 === 0) {
      const { bottom, tilt } = pose(o)
      sumBottom += bottom
      sumTilt += tilt
      sumSurface += surfaceLevel(s, p, o)
      samples++
    }
  }
  const bottom = sumBottom / samples
  const tilt = sumTilt / samples
  const surface = sumSurface / samples
  const tiltText = `${((tilt * 180) / Math.PI).toFixed(1)}°`
  if (ratio < 1) {
    floatingBottom.set(ratio, bottom)
    const draft = surface - bottom
    const expected = ratio * preset.size[1]
    check(
      `floats at ρ_body/ρ₀ = ${ratio}`,
      Math.abs(draft - expected) <= p.spacing && tilt < 0.1 && bottom > 1,
      `draft ${draft.toFixed(2)} vs Archimedes ${expected.toFixed(2)} (±${p.spacing}), tilt ${tiltText}, bottom at y=${bottom.toFixed(2)}`,
    )
  } else {
    const restsAt = p.domain.min[1] + p.h
    check(
      `sinks and rests flat at ρ_body/ρ₀ = ${ratio}`,
      Math.abs(bottom - restsAt) < 0.1 && tilt < 0.1,
      `lowest point at y=${bottom.toFixed(2)} (contact gap puts it at ${restsAt.toFixed(2)}), tilt ${tiltText}`,
    )

    // Now make it light. It used to stay sealed to the floor for seconds. Its
    // centre should get most of the way up to where the 0.25 ρ₀ box floated
    // (measured at the centre: a single snapshot can catch it mid-wobble).
    p.obstacleDensity = 0.25
    const startY = o.center[1]
    for (let k = 0; k < LIFT_SECONDS / p.dt; k++) step(s, p)
    const rose = o.center[1] - startY
    const needed = floatingBottom.get(0.25)! - restsAt
    check(
      `lifts off the floor when made light (0.25 ρ₀)`,
      rose >= LIFT_MIN_FRACTION * needed,
      `rose ${rose.toFixed(2)} of the ${needed.toFixed(2)} to floating height in ${LIFT_SECONDS} s (≥ ${LIFT_MIN_FRACTION * 100}%)`,
    )
  }
}

process.exit(results.every(Boolean) ? 0 : 1)
