// Headless sanity check for the solver: kernel normalization, then a dam break
// with periodic stats. Usage: node scripts/check-sim.ts [seconds] [key=value ...]
import { DIM, params } from '../src/sim/params.ts'
import { makeKernels, poly6 } from '../src/sim/kernels.ts'
import { createState, step, type SimState } from '../src/sim/sph.ts'

function checkKernels() {
  // Integrate over a DIM-cube [-h, h]^DIM on a fine grid; both kernels should integrate to 1.
  const h = params.h
  const k = makeKernels(h)
  const spikyCoef = k.spikyGradCoef / 3 // W = (C/3)(h − r)³ has gradient magnitude C(h − r)²
  const res = DIM === 2 ? 1000 : 120
  const cell = (2 * h) / res
  const idx = new Array(DIM).fill(0)
  let sumPoly6 = 0
  let sumSpiky = 0
  for (let c = 0; c < res ** DIM; c++) {
    let rem = c
    let r2 = 0
    for (let d = 0; d < DIM; d++) {
      idx[d] = rem % res
      rem = Math.floor(rem / res)
      const x = -h + (idx[d] + 0.5) * cell
      r2 += x * x
    }
    sumPoly6 += poly6(k, r2)
    const r = Math.sqrt(r2)
    if (r < h) sumSpiky += spikyCoef * (h - r) ** 3
  }
  const vol = cell ** DIM
  console.log(`kernels (DIM=${DIM}): ∫poly6 = ${(sumPoly6 * vol).toFixed(4)}, ∫spiky = ${(sumSpiky * vol).toFixed(4)}`)
}

function percentile(sorted: Float64Array, q: number) {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))]
}

function report(s: SimState, t: number, msPerStep: number) {
  const n = s.n
  const rho = new Float64Array(n)
  let nbrSum = 0
  let maxSpeed = 0
  let nan = 0
  let outside = 0
  for (let i = 0; i < n; i++) {
    rho[i] = s.density[i] / params.restDensity
    nbrSum += s.neighborCount[i]
    let v2 = 0
    for (let d = 0; d < DIM; d++) {
      const x = s.pos[i * DIM + d]
      const v = s.vel[i * DIM + d]
      if (!Number.isFinite(x) || !Number.isFinite(v)) nan++
      if (x < params.domain.min[d] || x > params.domain.max[d]) outside++
      v2 += v * v
    }
    maxSpeed = Math.max(maxSpeed, Math.sqrt(v2))
  }
  rho.sort()

  // Bulk = particles with a near-full neighbourhood (not at the surface or a wall).
  let bulkSum = 0
  let bulkN = 0
  const nbrSorted = Float64Array.from(s.neighborCount).sort()
  const nbrMedian = percentile(nbrSorted, 0.5)
  for (let i = 0; i < n; i++) {
    if (s.neighborCount[i] >= 0.9 * nbrMedian) {
      bulkSum += s.density[i] / params.restDensity
      bulkN++
    }
  }

  // Surface flatness: spread of the highest particle per column (DIM=2 only).
  let flat = ''
  if (DIM === 2) {
    const cols = 40
    const w = (params.domain.max[0] - params.domain.min[0]) / cols
    const top = new Float64Array(cols).fill(-Infinity)
    for (let i = 0; i < n; i++) {
      const c = Math.min(cols - 1, Math.floor((s.pos[i * DIM] - params.domain.min[0]) / w))
      top[c] = Math.max(top[c], s.pos[i * DIM + 1])
    }
    const tops = Array.from(top).filter(Number.isFinite)
    flat = ` surface y∈[${Math.min(...tops).toFixed(2)}, ${Math.max(...tops).toFixed(2)}]`
  }

  console.log(
    `t=${t.toFixed(2)}s ρ/ρ₀ p5=${percentile(rho, 0.05).toFixed(3)} p50=${percentile(rho, 0.5).toFixed(3)}` +
      ` p95=${percentile(rho, 0.95).toFixed(3)} max=${rho[n - 1].toFixed(3)} bulk=${(bulkSum / bulkN).toFixed(3)}` +
      ` nbrs avg=${(nbrSum / n).toFixed(1)} med=${nbrMedian}` +
      ` |v|max=${maxSpeed.toFixed(2)}${flat} nan=${nan} out=${outside} ${msPerStep.toFixed(2)}ms/step`,
  )
}

// Overrides as key=value, e.g. `stiffness=4000 dt=0.004 useTait=true`.
for (const arg of process.argv.slice(3)) {
  const [key, value] = arg.split('=')
  if (!(key in params)) throw new Error(`unknown param ${key}`)
  ;(params as unknown as Record<string, unknown>)[key] = JSON.parse(value)
}

checkKernels()

const seconds = Number(process.argv[2] ?? 10)
const steps = Math.round(seconds / params.dt)
const reportEvery = Math.round(1 / params.dt)
const s = createState(params)
let t0 = performance.now()
for (let i = 1; i <= steps; i++) {
  step(s, params)
  if (i % reportEvery === 0 || i === 1) {
    const now = performance.now()
    report(s, i * params.dt, (now - t0) / (i === 1 ? 1 : reportEvery))
    t0 = now
  }
}
