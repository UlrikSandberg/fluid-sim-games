// Headless checks for the 3D preset: the same solver as 2D, so the same claims
// should hold with one more axis. Usage: node scripts/acceptance3d.ts
//   1. Kernels integrate to 1 in 3D (the dimension-specific constants are right).
//   2. A dam break settles with bulk ρ/ρ₀ within ±5% and ~30–40 neighbours.
//   3. No particle ever leaves the domain.
//   4. Determinism: two runs from the same seed give bit-identical positions.
//   5. Tank tilted 20° (gravity rotated in tank coordinates): the surface settles
//      level in the world, not level with the tank.
//   6. Tank spinning about the vertical at ω (rotating-frame forces): the surface
//      settles into Newton's paraboloid, curvature ω²/2g.
//   7. Swooshing: the tank chases a target tumbling faster than its spin cap for
//      10 s. Nothing diverges or escapes, and once levelled the water resettles.
import { createHash } from 'node:crypto'
import { advanceTank, createTankMotion, levelTarget, nudgeTarget } from '../src/sim/tankMotion.ts'
import { defaultParams3D, dimOf, type Params } from '../src/sim/params.ts'
import { makeKernels, poly6 } from '../src/sim/kernels.ts'
import { createState, step, type FrameMotion, type SimState } from '../src/sim/sph.ts'

const p: Params = structuredClone(defaultParams3D)
const DIM = dimOf(p)
const results: boolean[] = []
const check = (name: string, ok: boolean, detail: string) => {
  results.push(ok)
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
}

// 1. Riemann sum of Poly6 and of Spiky's W = (C/3)(h − r)³ over [−h, h]³.
{
  const k = makeKernels(p.h, DIM)
  const res = 120
  const cell = (2 * p.h) / res
  let poly = 0
  let spiky = 0
  for (let a = 0; a < res; a++) {
    for (let b = 0; b < res; b++) {
      for (let c = 0; c < res; c++) {
        const x = -p.h + (a + 0.5) * cell
        const y = -p.h + (b + 0.5) * cell
        const z = -p.h + (c + 0.5) * cell
        const r2 = x * x + y * y + z * z
        poly += poly6(k, r2)
        const r = Math.sqrt(r2)
        if (r < p.h) spiky += (k.spikyGradCoef / 3) * (p.h - r) ** 3
      }
    }
  }
  const vol = cell ** 3
  check('3D kernels normalised', Math.abs(poly * vol - 1) < 1e-3 && Math.abs(spiky * vol - 1) < 1e-3,
    `∫poly6 = ${(poly * vol).toFixed(4)}, ∫spiky = ${(spiky * vol).toFixed(4)}`)
}

let outside = 0
function run(s: SimState, seconds: number) {
  for (let k = 0; k < seconds / p.dt; k++) {
    step(s, p)
    for (let i = 0; i < s.n * DIM; i++) {
      const x = s.pos[i]
      const d = i % DIM
      if (!(x >= p.domain.min[d] && x <= p.domain.max[d])) outside++
    }
  }
}

// 2–3. Dam break, settle.
const s = createState(p)
run(s, 20)
const nbr = Float64Array.from(s.neighborCount).sort()
const median = nbr[Math.floor(nbr.length / 2)]
let sum = 0
let n = 0
let within = 0
for (let i = 0; i < s.n; i++) {
  if (s.neighborCount[i] < 0.9 * median) continue
  const r = s.density[i] / p.restDensity
  sum += r
  n++
  if (Math.abs(r - 1) <= 0.05) within++
}
check('bulk density within ±5%', Math.abs(sum / n - 1) <= 0.05 && within / n >= 0.9,
  `bulk mean ρ/ρ₀ ${(sum / n).toFixed(3)}, ${((100 * within) / n).toFixed(0)}% of bulk within ±5%`)
check('neighbour count ~30–40', median >= 28 && median <= 40, `bulk median ${median}`)
check('never left the domain', outside === 0, `${outside} out-of-domain samples`)

// 4. Determinism, in-process (scripts/determinism.ts does 2D across processes).
const hash = () => {
  const t = createState(p)
  for (let k = 0; k < 500; k++) step(t, p)
  return createHash('sha256').update(new Uint8Array(t.pos.buffer, 0, t.n * DIM * 4)).digest('hex')
}
const a = hash()
const b = hash()
check('deterministic (500 steps, two runs)', a === b, a.slice(0, 16))

const g = 9.8
const pivot = p.domain.min.map((v, d) => (v + p.domain.max[d]) / 2)

// 5. Static tilt about z: world up is u = −ĝ, and e ⊥ ĝ runs across the tilted
// surface. Top particle per 1-wide band of e, away from the walls.
{
  const th = (20 * Math.PI) / 180
  const tilted: Params = { ...structuredClone(p), gravity: [-g * Math.sin(th), -g * Math.cos(th), 0] }
  const t = createState(tilted)
  for (let k = 0; k < 20 / p.dt; k++) step(t, tilted)
  const across = (i: number) => Math.cos(th) * t.pos[i * 3] - Math.sin(th) * t.pos[i * 3 + 1]
  const up = (i: number) => Math.sin(th) * t.pos[i * 3] + Math.cos(th) * t.pos[i * 3 + 1]
  let lo = Infinity
  let hi = -Infinity
  for (let i = 0; i < t.n; i++) {
    lo = Math.min(lo, across(i))
    hi = Math.max(hi, across(i))
  }
  const tops = new Map<number, number>()
  for (let i = 0; i < t.n; i++) {
    const w = across(i)
    if (w < lo + 1.5 || w > hi - 1.5) continue
    tops.set(Math.floor(w), Math.max(tops.get(Math.floor(w)) ?? -Infinity, up(i)))
  }
  const heights = [...tops.values()]
  const spread = Math.max(...heights) - Math.min(...heights)
  const wrong = (p.domain.max[0] - p.domain.min[0]) * Math.sin(th)
  check('tilted tank: surface level in the world', spread <= 2 * p.spacing,
    `world-height spread ${spread.toFixed(2)} (≤ ${(2 * p.spacing).toFixed(2)}; level with the tank would be ~${wrong.toFixed(1)})`)
}

// 6. Spin about the vertical through the tank's centre. Fit h = h₀ + c·r² to the
// top particle per radial band, inside the tank's inscribed circle; averaged over
// the last 5 s, since the surface still sloshes a little (src/gpu/gpuChecks.ts
// measures the GPU the same way).
{
  const w = 0.8
  const frame: FrameMotion = { omega: [0, w, 0], alpha: [0, 0, 0], pivot }
  const t = createState(p)
  const band = 0.5
  const inscribed = Math.min(p.domain.max[0] - p.domain.min[0], p.domain.max[2] - p.domain.min[2]) / 2
  const curvature = () => {
    const tops = new Map<number, number>()
    for (let i = 0; i < t.n; i++) {
      const r = Math.hypot(t.pos[i * 3] - pivot[0], t.pos[i * 3 + 2] - pivot[2])
      const b = Math.floor(r / band)
      tops.set(b, Math.max(tops.get(b) ?? -Infinity, t.pos[i * 3 + 1]))
    }
    let m = 0
    let sx = 0
    let sy = 0
    let sxx = 0
    let sxy = 0
    for (const [b, h] of tops) {
      const r = (b + 0.5) * band
      if (r > inscribed - band) continue
      m++
      sx += r * r
      sy += h
      sxx += r ** 4
      sxy += r * r * h
    }
    return (m * sxy - sx * sy) / (m * sxx - sx * sx)
  }
  for (let k = 0; k < 25 / p.dt; k++) step(t, p, undefined, frame)
  const samples: number[] = []
  for (let n = 0; n < 10; n++) {
    samples.push(curvature())
    for (let k = 0; k < 0.5 / p.dt; k++) step(t, p, undefined, frame)
  }
  const c = samples.reduce((a, b) => a + b, 0) / samples.length
  const spread = Math.max(...samples) - Math.min(...samples)
  const expected = (w * w) / (2 * g)
  check('spinning tank: Newton\'s paraboloid', Math.abs(c / expected - 1) <= 0.15,
    `mean curvature ${c.toFixed(4)} (samples spread ${spread.toFixed(4)}) vs ω²/2g = ${expected.toFixed(4)} (±15%)`)
}

// 7. Swoosh: the target tumbles about a skewed axis at 3 rad/s (above MAX_SPIN),
// reversing every 2 s, applied once per 60 Hz frame like a user's drag.
{
  const t = createState(p)
  const tank = createTankMotion()
  const sp: Params = structuredClone(p)
  const frameSteps = 4
  let escaped = 0
  let nonFinite = 0
  let maxRho = 0
  let maxSpeed = 0
  const runFrames = (seconds: number, drive: (time: number) => void) => {
    for (let f = 0; f < seconds / (frameSteps * p.dt); f++) {
      drive(f * frameSteps * p.dt)
      const motion = advanceTank(tank, frameSteps * p.dt, pivot, g)
      sp.gravity = motion.gravity
      for (let k = 0; k < frameSteps; k++) {
        step(t, sp, undefined, motion.frame)
        for (let i = 0; i < t.n; i++) {
          let v2 = 0
          for (let d = 0; d < DIM; d++) {
            const x = t.pos[i * DIM + d]
            const v = t.vel[i * DIM + d]
            if (!Number.isFinite(x) || !Number.isFinite(v)) nonFinite++
            if (!(x >= p.domain.min[d] && x <= p.domain.max[d])) escaped++
            v2 += v * v
          }
          maxSpeed = Math.max(maxSpeed, Math.sqrt(v2))
          maxRho = Math.max(maxRho, t.density[i] / p.restDensity)
        }
      }
    }
  }
  runFrames(10, (time) => {
    const dir = Math.floor(time / 2) % 2 === 0 ? 1 : -1
    nudgeTarget(tank, [1, 0.4, 0.7], dir * 3 * frameSteps * p.dt)
  })
  check('swooshing stays bounded', nonFinite === 0 && escaped === 0 && maxRho < 2,
    `max ρ/ρ₀ ${maxRho.toFixed(2)}, max |v| ${maxSpeed.toFixed(1)}, escaped ${escaped}, non-finite ${nonFinite}`)
  levelTarget(tank)
  runFrames(20, () => {})
  let sum = 0
  let m = 0
  const nbrs = Float64Array.from(t.neighborCount).sort()
  const med = nbrs[Math.floor(nbrs.length / 2)]
  for (let i = 0; i < t.n; i++) {
    if (t.neighborCount[i] < 0.9 * med) continue
    sum += t.density[i] / p.restDensity
    m++
  }
  const spin = Math.hypot(...tank.omega)
  check('levels and resettles after swooshing', Math.abs(sum / m - 1) <= 0.05 && spin < 1e-3,
    `bulk mean ρ/ρ₀ ${(sum / m).toFixed(3)}, tank spin ${spin.toExponential(1)} rad/s`)
}

process.exit(results.every(Boolean) ? 0 : 1)
