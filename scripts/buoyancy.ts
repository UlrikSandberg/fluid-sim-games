// Archimedes check for the dynamic obstacle: a box dropped onto a settled pool
// must float level with draft = (ρ_body/ρ₀) · height, and a box denser than the
// fluid must sink to the floor (resting h above it: the contact gap that keeps a
// film of fluid underneath). That sunk box, made light, must then lift off
// promptly rather than stick like a suction cup. Nothing in the solver models
// buoyancy directly;
// it has to come out of the fluid's pressure forces on the body's boundary
// particles. Usage: node scripts/buoyancy.ts
import { dimOf, defaultParams, particleRadius, type Params } from '../src/sim/params.ts'
import { createState, setObstacle, step, type SimState } from '../src/sim/sph.ts'

const SIZE = [6, 3]
const DROP_AT = [20, 22] // above the pool, which settles at ~14.5
const FLOAT_SECONDS = 30
const AVERAGE_LAST = 5 // seconds averaged at the end, to smooth out bobbing
const LIFT_SECONDS = 8
const LIFT_MIN_RISE = 6

const base: Params = { ...structuredClone(defaultParams), obstacleDynamic: true }
const DIM = dimOf(base)
const settled = createState(base)
for (let k = 0; k < 40 / base.dt; k++) step(settled, base)

// Fluid surface height away from the box: mean of the top particle per column,
// plus the particle radius (particle centres sit half a spacing below the surface).
function surfaceLevel(s: SimState, p: Params, boxX: number, halfWidth: number): number {
  const cols = 40
  const { min, max } = p.domain
  const w = (max[0] - min[0]) / cols
  const top = new Float64Array(cols).fill(-Infinity)
  for (let i = 0; i < s.n; i++) {
    const x = s.pos[i * DIM]
    top[Math.min(cols - 1, Math.floor((x - min[0]) / w))] = Math.max(top[Math.min(cols - 1, Math.floor((x - min[0]) / w))], s.pos[i * DIM + 1])
  }
  const far = Array.from(top).filter((y, c) => {
    const cx = min[0] + (c + 0.5) * w
    return Number.isFinite(y) && Math.abs(cx - boxX) > halfWidth + 2 && c > 0 && c < cols - 1
  })
  return far.reduce((a, b) => a + b, 0) / far.length + particleRadius(p)
}

const results: boolean[] = []
const check = (name: string, ok: boolean, detail: string) => {
  results.push(ok)
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
}

for (const ratio of [0.25, 0.5, 0.75, 1.5]) {
  const p: Params = { ...structuredClone(base), obstacleDensity: ratio }
  const s = structuredClone(settled)
  setObstacle(s, p, { center: DROP_AT, size: SIZE })
  const o = s.obstacle!
  let sumY = 0
  let sumAngle = 0
  let sumSurface = 0
  let samples = 0
  const steps = Math.round(FLOAT_SECONDS / p.dt)
  const averageFrom = steps - Math.round(AVERAGE_LAST / p.dt)
  for (let k = 0; k < steps; k++) {
    step(s, p)
    if (k >= averageFrom && k % 25 === 0) {
      sumY += o.center[1]
      sumAngle += o.angle
      sumSurface += surfaceLevel(s, p, o.center[0], o.halfSize[0])
      samples++
    }
  }
  const cy = sumY / samples
  const angle = sumAngle / samples
  const surface = sumSurface / samples
  // Lowest point of the (possibly tilted) box.
  const bottom = cy - (Math.abs(Math.sin(angle)) * o.halfSize[0] + Math.abs(Math.cos(angle)) * o.halfSize[1])
  const tilt = `${((angle * 180) / Math.PI).toFixed(1)}°`
  if (ratio < 1) {
    const draft = surface - bottom
    const expected = ratio * SIZE[1]
    check(
      `floats at ρ_body/ρ₀ = ${ratio}`,
      Math.abs(draft - expected) <= p.spacing && Math.abs(angle) < 0.1 && bottom > 1,
      `draft ${draft.toFixed(2)} vs Archimedes ${expected.toFixed(2)} (±${p.spacing}), tilt ${tilt}, bottom at y=${bottom.toFixed(2)}`,
    )
  } else {
    // Resting flat means a multiple of 90°: lying on a long side or (less likely) a short one.
    const offFlat = Math.abs(angle - (Math.PI / 2) * Math.round(angle / (Math.PI / 2)))
    const restsAt = p.domain.min[1] + p.h
    check(
      `sinks and rests flat at ρ_body/ρ₀ = ${ratio}`,
      Math.abs(bottom - restsAt) < 0.1 && offFlat < 0.1,
      `lowest point at y=${bottom.toFixed(2)} (contact gap puts it at ${restsAt.toFixed(2)}), tilt ${tilt}`,
    )

    // Now make it light. It used to stay sealed to the floor for seconds.
    p.obstacleDensity = 0.25
    const startY = o.center[1]
    for (let k = 0; k < LIFT_SECONDS / p.dt; k++) step(s, p)
    const rise = o.center[1] - startY
    check(
      `lifts off the floor when made light (0.25 ρ₀)`,
      rise >= LIFT_MIN_RISE,
      `rose ${rise.toFixed(2)} in ${LIFT_SECONDS} s (≥ ${LIFT_MIN_RISE})`,
    )
  }
}

process.exit(results.every(Boolean) ? 0 : 1)
