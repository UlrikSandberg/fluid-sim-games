/// <reference types="@webgpu/types" />
// Physics and performance checks for the GPU solver. They need a browser with
// WebGPU, so they run in the page (scripts/gpu-check.mjs drives them through the
// dev server); the same claims as scripts/acceptance3d.ts and buoyancy.ts --3d,
// checked statistically because the GPU isn't bit-reproducible.
import { defaultParams3D, particleRadius, type Params } from '../sim/params'
import { advanceTank, createTankMotion } from '../sim/tankMotion'
import type { FrameMotion } from '../sim/sph'
import { GpuSph, requestDevice, type GpuReadback } from './sphGpu'
import { GpuEngine } from '../scenes/fluid3d/engines'

export interface CheckResult {
  name: string
  ok: boolean
  detail: string
}

const g = 9.8
const pivotOf = (p: Params) => p.domain.min.map((v, d) => (v + p.domain.max[d]) / 2)
const still = (p: Params): FrameMotion => ({ omega: [0, 0, 0], alpha: [0, 0, 0], pivot: pivotOf(p) })
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => r(null)))

// Runs `seconds` of simulated time in 25-substep batches and returns a fresh readback.
async function run(device: GPUDevice, sim: GpuSph, p: Params, seconds: number, frame: FrameMotion, gravity: number[]) {
  const batches = Math.round(seconds / p.dt / 25)
  for (let b = 0; b < batches; b++) {
    sim.step(25, frame, gravity, [])
    if (b % 8 === 7) await device.queue.onSubmittedWorkDone()
  }
  return fresh(device, sim, frame, gravity)
}

// One more step, then wait for its readback to land.
async function fresh(device: GPUDevice, sim: GpuSph, frame: FrameMotion, gravity: number[]): Promise<GpuReadback> {
  await device.queue.onSubmittedWorkDone()
  const before = sim.latest
  for (let tries = 0; tries < 200; tries++) {
    sim.step(1, frame, gravity, [])
    await device.queue.onSubmittedWorkDone()
    await new Promise((r) => setTimeout(r, 5))
    if (sim.latest && sim.latest !== before) return sim.latest
  }
  throw new Error('no readback from the GPU')
}

function bulkStats(r: GpuReadback, p: Params) {
  const nbrs = Array.from({ length: r.n }, (_, i) => r.aux[i * 4 + 2]).sort((a, b) => a - b)
  const median = nbrs[Math.floor(r.n / 2)]
  let sum = 0
  let m = 0
  let within = 0
  for (let i = 0; i < r.n; i++) {
    if (r.aux[i * 4 + 2] < 0.9 * median) continue
    const rho = r.aux[i * 4] / p.restDensity
    sum += rho
    m++
    if (Math.abs(rho - 1) <= 0.05) within++
  }
  return { mean: sum / m, within: within / m, median }
}

// Curvature c of h = h₀ + c·r² fitted to the top particle per radial band about
// the tank's vertical axis, inside the tank's inscribed circle.
function paraboloidCurvature(r: GpuReadback, p: Params): number {
  const pivot = pivotOf(p)
  const band = 0.5
  const inscribed = Math.min(p.domain.max[0] - p.domain.min[0], p.domain.max[2] - p.domain.min[2]) / 2
  const tops = new Map<number, number>()
  for (let i = 0; i < r.n; i++) {
    const rad = Math.hypot(r.pos[i * 4] - pivot[0], r.pos[i * 4 + 2] - pivot[2])
    const b = Math.floor(rad / band)
    tops.set(b, Math.max(tops.get(b) ?? -Infinity, r.pos[i * 4 + 1]))
  }
  let m = 0
  let sx = 0
  let sy = 0
  let sxx = 0
  let sxy = 0
  for (const [b, h] of tops) {
    const rad = (b + 0.5) * band
    if (rad > inscribed - band) continue
    m++
    sx += rad * rad
    sy += h
    sxx += rad ** 4
    sxy += rad * rad * h
  }
  return (m * sxy - sx * sy) / (m * sxx - sx * sx)
}

export async function runGpuChecks(log: (line: string) => void = console.log): Promise<CheckResult[]> {
  const results: CheckResult[] = []
  const check = (name: string, ok: boolean, detail: string) => {
    results.push({ name, ok, detail })
    log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
  }
  const device = await requestDevice()
  if (!device) {
    check('WebGPU available', false, 'no adapter')
    return results
  }
  const p: Params = structuredClone(defaultParams3D)
  const down = [0, -g, 0]

  // 1. Settle: bulk density and neighbours.
  {
    const sim = new GpuSph(device, p)
    const r = await run(device, sim, p, 20, still(p), down)
    const { mean, within, median } = bulkStats(r, p)
    check('bulk density within ±5%', Math.abs(mean - 1) <= 0.05 && within >= 0.9,
      `bulk mean ρ/ρ₀ ${mean.toFixed(3)}, ${(within * 100).toFixed(0)}% within ±5%`)
    check('neighbour count ~30–40', median >= 28 && median <= 40, `bulk median ${median}`)
    let outside = 0
    for (let i = 0; i < r.n; i++) {
      for (let d = 0; d < 3; d++) {
        const x = r.pos[i * 4 + d]
        if (!(x >= p.domain.min[d] && x <= p.domain.max[d])) outside++
      }
    }
    check('stays in the domain', outside === 0, `${outside} out-of-domain coordinates`)
    sim.dispose()
  }

  // 2. Tilted 20° about z: the surface settles level in the world.
  {
    const th = (20 * Math.PI) / 180
    const gravity = [-g * Math.sin(th), -g * Math.cos(th), 0]
    const sim = new GpuSph(device, p)
    const r = await run(device, sim, p, 20, still(p), gravity)
    const across = (i: number) => Math.cos(th) * r.pos[i * 4] - Math.sin(th) * r.pos[i * 4 + 1]
    const up = (i: number) => Math.sin(th) * r.pos[i * 4] + Math.cos(th) * r.pos[i * 4 + 1]
    let lo = Infinity
    let hi = -Infinity
    for (let i = 0; i < r.n; i++) {
      lo = Math.min(lo, across(i))
      hi = Math.max(hi, across(i))
    }
    const tops = new Map<number, number>()
    for (let i = 0; i < r.n; i++) {
      const w = across(i)
      if (w < lo + 1.5 || w > hi - 1.5) continue
      tops.set(Math.floor(w), Math.max(tops.get(Math.floor(w)) ?? -Infinity, up(i)))
    }
    const heights = [...tops.values()]
    const spread = Math.max(...heights) - Math.min(...heights)
    check('tilted tank: surface level in the world', spread <= 2 * p.spacing, `world-height spread ${spread.toFixed(2)} (≤ ${2 * p.spacing})`)
    sim.dispose()
  }

  // 3. Spinning about the vertical: Newton's paraboloid. The surface sloshes as
  // it spins up, so the curvature is averaged over the last 5 s rather than read
  // from one snapshot (which, run to run, lands on different slosh phases).
  {
    const w = 0.8
    const frame: FrameMotion = { omega: [0, w, 0], alpha: [0, 0, 0], pivot: pivotOf(p) }
    const sim = new GpuSph(device, p)
    let r = await run(device, sim, p, 25, frame, down)
    const samples: number[] = []
    for (let k = 0; k < 10; k++) {
      samples.push(paraboloidCurvature(r, p))
      r = await run(device, sim, p, 0.5, frame, down)
    }
    const c = samples.reduce((a, b) => a + b, 0) / samples.length
    const spread = Math.max(...samples) - Math.min(...samples)
    const expected = (w * w) / (2 * g)
    check('spinning tank: Newton\'s paraboloid', Math.abs(c / expected - 1) <= 0.15,
      `mean curvature ${c.toFixed(4)} (samples spread ${spread.toFixed(4)}) vs ω²/2g = ${expected.toFixed(4)} (±15%)`)
    sim.dispose()
  }

  // 4. Buoyancy through the real engine (body integrated per frame from the
  // GPU's summed force), paced at the display rate like the app.
  {
    const engine = (await GpuEngine.create(p))!
    const tank = createTankMotion()
    const frames = async (seconds: number, each?: () => void) => {
      for (let f = 0; f < seconds / (4 * p.dt); f++) {
        const motion = advanceTank(tank, 4 * p.dt, pivotOf(p), g)
        engine.step(4, motion.frame, motion.gravity)
        each?.()
        await nextFrame()
      }
    }
    await frames(8)
    const size = [4, 2, 3]
    engine.dropBody({ center: [8, 8, 4], size })
    p.obstacleDensity = 0.5
    await frames(15)
    let draft = 0
    let samples = 0
    await frames(5, () => {
      const view = engine.view()
      const o = engine.obstacle
      if (!view || !o) return
      // Far-field surface: top particle per 1×1 column away from the box and walls.
      const tops = new Map<number, number>()
      for (let i = 0; i < view.n; i++) {
        const x = view.pos[i * 3]
        const z = view.pos[i * 3 + 2]
        const far = Math.abs(x - o.center[0]) >= o.halfSize[0] + 2 || Math.abs(z - o.center[2]) >= o.halfSize[2] + 2
        if (!far || x < 1 || x > 15 || z < 1 || z > 7) continue
        const key = Math.floor(x) * 100 + Math.floor(z)
        tops.set(key, Math.max(tops.get(key) ?? -Infinity, view.pos[i * 3 + 1]))
      }
      const heights = [...tops.values()]
      const surface = heights.reduce((a, b) => a + b, 0) / heights.length + particleRadius(p)
      draft += surface - (o.center[1] - o.halfSize[1])
      samples++
    })
    draft /= samples
    check('floats at ρ_body/ρ₀ = 0.5', Math.abs(draft - 1) <= p.spacing, `draft ${draft.toFixed(2)} vs Archimedes 1.00 (±${p.spacing})`)
    engine.dispose()
  }

  // 5. Throughput at growing particle counts: GPU time per substep.
  for (const [n, size] of [[4000, [16, 10, 8]], [40000, [32, 16, 16]], [100000, [40, 20, 24]], [200000, [48, 24, 32]]] as const) {
    const q: Params = { ...structuredClone(defaultParams3D), particleCount: n, domain: { min: [0, 0, 0], max: [...size] } }
    const sim = new GpuSph(device, q)
    await run(device, sim, q, 0.4, still(q), down) // warm up past the first collapse
    const steps = 100
    const t0 = performance.now()
    for (let b = 0; b < steps / 25; b++) sim.step(25, still(q), down, [])
    await device.queue.onSubmittedWorkDone()
    const ms = (performance.now() - t0) / steps
    log(`     ${n.toLocaleString()} particles: ${ms.toFixed(3)} ms/substep → ${Math.floor(16.7 / 4.2 / ms) >= 1 ? 'real time at 60 fps' : `${(1000 / (ms * 4.2)).toFixed(0)} fps of real time`}`)
    sim.dispose()
  }
  return results
}
