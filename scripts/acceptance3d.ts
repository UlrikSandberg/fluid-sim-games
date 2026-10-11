// Headless checks for the 3D preset: the same solver as 2D, so the same claims
// should hold with one more axis. Usage: node scripts/acceptance3d.ts
//   1. Kernels integrate to 1 in 3D (the dimension-specific constants are right).
//   2. A dam break settles with bulk ρ/ρ₀ within ±5% and ~30–40 neighbours.
//   3. No particle ever leaves the domain.
//   4. Determinism: two runs from the same seed give bit-identical positions.
import { createHash } from 'node:crypto'
import { defaultParams3D, dimOf, type Params } from '../src/sim/params.ts'
import { makeKernels, poly6 } from '../src/sim/kernels.ts'
import { createState, step, type SimState } from '../src/sim/sph.ts'

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

process.exit(results.every(Boolean) ? 0 : 1)
