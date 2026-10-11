// Headless Phase 1 acceptance checks (everything except frame rate, which needs a
// real browser, and the colour modes, which are visual):
//   1. A dam break settles to a flat surface with bulk ρ/ρ₀ within ±5%.
//   2. Neighbour count reads ~20 in the bulk.
//   3. No particle ever leaves the domain (checked after every step).
//   4. Full-strength mouse dragging (repel, attract, then a current pumped into a
//      wall) does not blow up, and the fluid settles again afterwards.
// Usage: node scripts/acceptance.ts
import { DIM, defaultParams, type Params } from '../src/sim/params.ts'
import { createState, step, type MouseInput, type SimState } from '../src/sim/sph.ts'

const p: Params = structuredClone(defaultParams)
const results: boolean[] = []
const check = (name: string, ok: boolean, detail: string) => {
  results.push(ok)
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
}

let outsideEver = 0
let nonFiniteEver = 0
let maxRhoEver = 0
let maxSpeedEver = 0
let spikeSamples = 0 // particle-steps with ρ/ρ₀ > SPIKE
let samples = 0
const SPIKE = 1.5

function run(s: SimState, seconds: number, mouse?: (t: number) => MouseInput) {
  const steps = Math.round(seconds / p.dt)
  for (let k = 0; k < steps; k++) {
    step(s, p, mouse?.(k * p.dt))
    for (let i = 0; i < s.n; i++) {
      let v2 = 0
      for (let d = 0; d < DIM; d++) {
        const x = s.pos[i * DIM + d]
        const v = s.vel[i * DIM + d]
        if (!Number.isFinite(x) || !Number.isFinite(v)) nonFiniteEver++
        if (!(x >= p.domain.min[d] && x <= p.domain.max[d])) outsideEver++
        v2 += v * v
      }
      maxSpeedEver = Math.max(maxSpeedEver, Math.sqrt(v2))
      const rho = s.density[i] / p.restDensity
      maxRhoEver = Math.max(maxRhoEver, rho)
      if (rho > SPIKE) spikeSamples++
    }
    samples += s.n
  }
}

function median(values: ArrayLike<number>): number {
  const sorted = Float64Array.from(values).sort()
  return sorted[Math.floor(sorted.length / 2)]
}

// Bulk = particles with a near-full neighbourhood (away from the surface and walls).
function bulkDensity(s: SimState) {
  const nbrMedian = median(s.neighborCount)
  let sum = 0
  let n = 0
  let within = 0
  for (let i = 0; i < s.n; i++) {
    if (s.neighborCount[i] < 0.9 * nbrMedian) continue
    const r = s.density[i] / p.restDensity
    sum += r
    n++
    if (Math.abs(r - 1) <= 0.05) within++
  }
  return { mean: sum / n, fractionWithin: within / n, nbrMedian }
}

// Spread of the highest particle per column, ignoring the columns touching the
// side walls (wall layers stack there until boundary particles exist).
function surfaceSpread(s: SimState): number {
  const cols = 40
  const { min, max } = p.domain
  const w = (max[0] - min[0]) / cols
  const top = new Float64Array(cols).fill(-Infinity)
  for (let i = 0; i < s.n; i++) {
    const c = Math.min(cols - 1, Math.floor((s.pos[i * DIM] - min[0]) / w))
    top[c] = Math.max(top[c], s.pos[i * DIM + 1])
  }
  const inner = Array.from(top).slice(1, -1).filter(Number.isFinite)
  return Math.max(...inner) - Math.min(...inner)
}

const s = createState(p)

// 1–3: dam break, then settle. 60 s rather than 40: boundary-particle walls push
// back conservatively, so the first slosh mode decays slower than it did against
// the old dissipative clamp.
run(s, 60)
const settled = bulkDensity(s)
const spread = surfaceSpread(s)
check(
  'bulk density within ±5%',
  Math.abs(settled.mean - 1) <= 0.05 && settled.fractionWithin >= 0.9,
  `bulk mean ρ/ρ₀ ${settled.mean.toFixed(3)}, ${(settled.fractionWithin * 100).toFixed(0)}% of bulk within ±5%`,
)
check('flat surface', spread <= 2.5 * p.spacing, `surface spread ${spread.toFixed(2)} (≤ ${(2.5 * p.spacing).toFixed(2)}, 2.5 spacings)`)
check('neighbour count ~20', settled.nbrMedian >= 17 && settled.nbrMedian <= 23, `bulk median ${settled.nbrMedian}`)

// 4: full-strength mouse. Repel and attract sweeping through the pool, then a
// current anchored mid-pool, pumping at full speed into the right wall while its
// direction slowly swings up and down.
p.mouseStrength = 1 // slider maximum
p.mouseMaxSpeed = 40 // slider maximum; CFL still caps the effective value
p.brushRadius = 6
const { min, max } = p.domain
const cx = (min[0] + max[0]) / 2
const sweep = (sign: number) => (t: number): MouseInput => ({
  active: true,
  kind: 'radial',
  sign,
  pos: [cx + 0.4 * (max[0] - min[0]) * Math.sin(2 * t), 7 + 4 * Math.cos(2.6 * t)],
  arrow: [0, 0],
})
const pump = (t: number): MouseInput => ({
  active: true,
  kind: 'current',
  sign: 1,
  pos: [cx, 6],
  arrow: [3 * p.brushRadius * Math.cos(0.5 * Math.sin(t)), 3 * p.brushRadius * Math.sin(0.5 * Math.sin(t))],
})
const damBreakMaxSpeed = maxSpeedEver
maxRhoEver = 0
maxSpeedEver = 0
spikeSamples = 0
samples = 0
run(s, 8, sweep(1))
run(s, 8, sweep(-1))
run(s, 8, pump)
// "Doesn't blow up" means no divergence: everything stays finite, density spikes
// are rare and brief (fast sweeps slam fluid into walls, where Phase 1 boundary
// handling is weakest) with a hard ceiling, and speeds stay comparable to the
// dam break's own splashes.
const spikeFraction = spikeSamples / samples
check(
  'full-strength mouse stays bounded',
  nonFiniteEver === 0 && maxRhoEver < 2.5 && spikeFraction < 1e-3 && maxSpeedEver < 2 * damBreakMaxSpeed,
  `max ρ/ρ₀ ${maxRhoEver.toFixed(2)} (< 2.5), ρ/ρ₀ > ${SPIKE} in ${(spikeFraction * 100).toFixed(3)}% of particle-steps (< 0.1%), ` +
    `max |v| ${maxSpeedEver.toFixed(1)} (< 2× dam break's ${damBreakMaxSpeed.toFixed(1)}), non-finite ${nonFiniteEver}`,
)

p.mouseStrength = defaultParams.mouseStrength
p.mouseMaxSpeed = defaultParams.mouseMaxSpeed
run(s, 40)
const after = bulkDensity(s)
check(
  'resettles after mouse',
  Math.abs(after.mean - 1) <= 0.05 && surfaceSpread(s) <= 2.5 * p.spacing,
  `bulk mean ρ/ρ₀ ${after.mean.toFixed(3)}, surface spread ${surfaceSpread(s).toFixed(2)}`,
)

check('never left the domain', outsideEver === 0 && nonFiniteEver === 0, `${outsideEver} out-of-domain samples over the whole run`)

process.exit(results.every(Boolean) ? 0 : 1)
