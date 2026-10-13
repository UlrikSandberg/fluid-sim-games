// Headless checks for the shallow-water solver (src/sim/shallow):
//   1. A closed tank keeps its water: drops and sloshing change the volume by
//      nothing but rounding.
//   2. Waves travel at √(g·h): a pulse in 1 m of water crosses 8 m in 8/3.13 s.
//   3. A lake over a bumpy bed, at rest, stays at rest (the solver is "well
//      balanced"), including where the bumps stick out of the water.
//   4. A dam break on a dry bed: no negative depths, no NaNs, and the depth
//      profile matches Ritter's exact solution.
//   5. The ripple tank's sponges soak waves up rather than reflecting them.
//   6. Speed: a ripple-tank step on the CPU.
// Usage: node scripts/shallow.ts
import { createGrid, drop, fillTo, maxDt, step, totalVolume } from '../src/sim/shallow/shallow.ts'
import { buildTank, TANK } from '../src/scenes/ripples/presets.ts'

const results: boolean[] = []
const check = (name: string, ok: boolean, detail: string) => {
  results.push(ok)
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
}

// 1. Volume.
{
  const s = createGrid(120, 80, 0.1)
  fillTo(s, 1)
  const v0 = totalVolume(s)
  drop(s, 4, 4, 0.4, 0.2)
  drop(s, 8, 3, 0.3, -0.15)
  for (let k = 0; k < 2000; k++) step(s, maxDt(s), { advect: true, foamRate: 1 })
  const err = Math.abs(totalVolume(s) - v0) / v0
  check('closed tank keeps its water', err < 1e-5, `relative volume change ${err.toExponential(1)} after 2000 steps`)
}

// 2. Wave speed: a long thin channel, a bump at x = 2 m, watch x = 10 m.
{
  const dx = 0.05
  const s = createGrid(300, 4, dx)
  fillTo(s, 1)
  for (let i = 0; i < s.nx; i++) for (let j = 0; j < s.ny; j++) s.h[j * s.nx + i] += 0.01 * Math.exp(-((((i + 0.5) * dx - 2) / 0.3) ** 2))
  const watch = Math.round(10 / dx)
  let tPeak = 0
  let peak = 0
  const dt = maxDt(s, 0.3)
  while (s.time < 5) {
    step(s, dt)
    const e = s.h[watch] - 1
    if (e > peak) {
      peak = e
      tPeak = s.time
    }
  }
  const c = 8 / tPeak
  const expect = Math.sqrt(9.81 * 1)
  check('waves travel at √(g·h)', Math.abs(c / expect - 1) < 0.03, `measured ${c.toFixed(3)} m/s, theory ${expect.toFixed(3)} m/s`)
}

// 3. Lake at rest over bumps, some above the water line.
{
  const s = createGrid(80, 80, 0.1)
  for (let j = 0; j < 80; j++) for (let i = 0; i < 80; i++) s.bed[j * 80 + i] = 0.6 + 0.6 * Math.sin(i * 0.2) * Math.cos(j * 0.15)
  fillTo(s, 1)
  for (let k = 0; k < 1000; k++) step(s, maxDt(s), { advect: true })
  let vmax = 0
  for (const x of s.u) vmax = Math.max(vmax, Math.abs(x))
  for (const x of s.v) vmax = Math.max(vmax, Math.abs(x))
  check('lake at rest stays at rest', vmax < 1e-4, `largest velocity ${vmax.toExponential(1)} m/s after 1000 steps (float32 rounding)`)
}

// 4. Dam break onto a dry bed.
{
  const dx = 0.05
  const s = createGrid(400, 3, dx)
  for (let j = 0; j < 3; j++) for (let i = 0; i < 400; i++) s.h[j * 400 + i] = (i + 0.5) * dx < 5 ? 1 : 0
  let bad = 0
  while (s.time < 1.5) {
    step(s, maxDt(s, 0.3), { advect: true })
    for (const x of s.h) if (!(x >= 0)) bad++
  }
  // Compare the depth profile with Ritter's exact solution behind the dam's site.
  const c = Math.sqrt(9.81)
  let err = 0
  let n = 0
  for (let i = 0; i < 400; i++) {
    const x = (i + 0.5) * dx
    const xi = (x - 5) / s.time
    const exact = xi < -c ? 1 : xi > 2 * c ? 0 : (2 * c - xi) ** 2 / (9 * 9.81)
    err += (s.h[400 + i] - exact) ** 2
    n++
  }
  const rms = Math.sqrt(err / n)
  check('dam break stays sane', bad === 0, `${bad} negative or NaN depths`)
  check('dam break matches Ritter', rms < 0.01, `rms depth error ${(rms * 1000).toFixed(1)} mm against the exact solution at t = ${s.time.toFixed(2)} s`)
}

// 5. Sponges: a drop's ripples die out in the open tank, but ring on in a closed one.
{
  const energy = (absorb: boolean) => {
    const { grid: s } = buildTank('pool', absorb)
    drop(s, 12, 8, 0.4, 0.15)
    const dt = maxDt(s)
    while (s.time < 20) step(s, dt)
    let e = 0
    for (let k = 0; k < s.h.length; k++) e += (s.h[k] + s.bed[k] - TANK.depth) ** 2
    return Math.sqrt(e / s.h.length)
  }
  const open = energy(true)
  const closed = energy(false)
  check('sponges absorb waves', open < closed * 0.25, `rms ripple after 20 s: ${(open * 1000).toFixed(2)} mm open vs ${(closed * 1000).toFixed(2)} mm closed`)
}

// 6. Speed.
{
  const { grid: s } = buildTank('double', true)
  const dt = maxDt(s)
  for (let k = 0; k < 20; k++) step(s, dt, { paddle: 0.1, foamRate: 1 })
  const t0 = performance.now()
  for (let k = 0; k < 100; k++) step(s, dt, { paddle: 0.1, foamRate: 1 })
  const ms = (performance.now() - t0) / 100
  const perFrame = Math.ceil(1 / 60 / dt)
  check('ripple tank step is fast', ms * perFrame < 10, `${s.nx}×${s.ny} grid: ${ms.toFixed(2)} ms/step, ${perFrame} steps per frame`)
}

process.exit(results.every(Boolean) ? 0 : 1)
