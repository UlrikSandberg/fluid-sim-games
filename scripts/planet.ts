// Headless checks for gravity toward bodies and the water planet's build (the
// planet's physics runs on the GPU: npm run test:gpu checks it settles):
//   1. A body pulls G·M/r² toward its centre: twice as far, a quarter the pull.
//   2. Inside its radius the pull falls linearly to zero at the centre (shell
//      theorem), so nothing feels an infinite pull.
//   3. Pulls add up, with the uniform part: two equal bodies cancel halfway.
//   4. The CPU solver uses it: a particle released above a body falls toward it.
//   5. The planet: gravity at sea level is what was asked for; the water lies
//      between the ground and sea level; about a third of the surface is land.
//   6. Sculpting: the brush raises the ground under it and nowhere else; a ray
//      from space lands on the ground (picking).
//   7. Rebuilding the rock after a stroke keeps the water, lifting out any that
//      the new rock buried.
// Usage: node scripts/planet.ts
import { defaultParams3D, gravityAt, type Params } from '../src/sim/params.ts'
import { createState, step } from '../src/sim/sph.ts'
import { defaultPlanet, groundRadius, makeTerrain, planetParams, planetScene, seaRadius } from '../src/scenes/planet/planet.ts'

const results: boolean[] = []
const check = (name: string, ok: boolean, detail: string) => {
  results.push(ok)
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}  — ${detail}`)
}
const len = (v: number[]) => Math.hypot(...v)

// 1–3. The law.
{
  const p = { gravity: [0, 0, 0], attractors: [{ center: [0, 0, 0], gm: 100, radius: 2 }] }
  const at5 = gravityAt(p, [5, 0, 0])
  const at10 = gravityAt(p, [10, 0, 0])
  check('inverse square', Math.abs(len(at5) / len(at10) - 4) < 1e-9 && Math.abs(len(at5) - 100 / 25) < 1e-9 && at5[0] < 0,
    `|g| ${len(at5)} at r = 5, ${len(at10)} at r = 10, pointing at the centre`)
  const at1 = gravityAt(p, [0, 1, 0])
  const at2 = gravityAt(p, [0, 2, 0])
  const atCentre = gravityAt(p, [0, 0, 0])
  check('linear inside the body', Math.abs(len(at1) / len(at2) - 0.5) < 1e-9 && len(atCentre) === 0 && Math.abs(len(at2) - 100 / 4) < 1e-9,
    `|g| ${len(at1)} at r = 1, ${len(at2)} at the surface (r = 2), ${len(atCentre)} at the centre`)
  const two = { gravity: [0, -1, 0], attractors: [{ center: [-5, 0, 0], gm: 50, radius: 1 }, { center: [5, 0, 0], gm: 50, radius: 1 }] }
  const mid = gravityAt(two, [0, 0, 0])
  check('pulls add up', Math.abs(mid[0]) < 1e-12 && mid[1] === -1, `halfway between equal bodies: (${mid.map((v) => +v.toFixed(12)).join(', ')}), just the uniform part`)
}

// 4. The CPU solver: one particle released above a body, in a box with no
// uniform gravity, falls toward it at G·M/r².
{
  const p: Params = {
    ...structuredClone(defaultParams3D),
    particleCount: 1,
    gravity: [0, 0, 0],
    domain: { min: [0, 0, 0], max: [20, 20, 20] },
    attractors: [{ center: [10, 2, 10], gm: 40, radius: 1 }],
  }
  const s = createState(p)
  s.pos.set([10, 12, 10])
  const steps = 25
  for (let k = 0; k < steps; k++) step(s, p)
  const t = steps * p.dt
  const fell = 12 - s.pos[1]
  const expect = 0.5 * (40 / 100) * t * t // ½·(GM/r²)·t², r ≈ 10 over this short a time
  check('CPU solver falls toward the body', Math.abs(fell / expect - 1) < 0.05 && Math.abs(s.pos[0] - 10) < 1e-6,
    `fell ${fell.toExponential(3)} in ${t.toFixed(2)} s, ½·(GM/r²)·t² = ${expect.toExponential(3)}`)
}

// 5. The planet.
{
  const l = defaultPlanet
  const t = makeTerrain(l)
  const p = planetParams(l, 9.8)
  const atSea = len(gravityAt(p, [0, seaRadius(l), 0]))
  check('planet gravity at sea level', Math.abs(atSea - 9.8) < 1e-9, `${atSea.toFixed(3)} at r = ${seaRadius(l)}`)
  const s = planetScene(p, l, t, 500)
  let wrong = 0
  for (let i = 0; i < s.fluid.length / 3; i++) {
    const x = [s.fluid[i * 3], s.fluid[i * 3 + 1], s.fluid[i * 3 + 2]]
    const d = len(x)
    if (d < groundRadius(l, t, x.map((v) => v / d)) || d > seaRadius(l) + 0.5) wrong++
  }
  check('water between the ground and sea level', wrong === 0 && s.capacity === s.fluid.length / 3 + 500,
    `${s.fluid.length / 3} particles (room for ${s.capacity}), ${wrong} misplaced; ${s.boundary.pos.length / 3} rock particles`)
  let land = 0
  const N = 4000
  for (let k = 0; k < N; k++) {
    const z = 1 - (2 * (k + 0.5)) / N
    const r = Math.sqrt(1 - z * z)
    if (groundRadius(l, t, [r * Math.cos(k * 2.39996), z, r * Math.sin(k * 2.39996)]) > seaRadius(l)) land++
  }
  check('about a third is land', land / N > 0.2 && land / N < 0.45, `${Math.round((100 * land) / N)}% of the surface`)

  // 6. Sculpting: the brush raises the ground under it, most at the centre, and
  // nothing outside its radius; a ray from space lands on the ground.
  const up = [0, 1, 0]
  const side = [Math.sin(0.5), Math.cos(0.5), 0] // 0.5 rad away
  const h0 = t.heightAt(0, 1, 0)
  const s0 = t.heightAt(side[0], side[1], side[2])
  const near = [Math.sin(0.05), Math.cos(0.05), 0]
  const n0 = t.heightAt(near[0], near[1], near[2])
  t.brush(up, 0.2, 1)
  const raised = t.heightAt(0, 1, 0) - h0
  const nearRaised = t.heightAt(near[0], near[1], near[2]) - n0
  check('brush raises under it, not beyond', raised > 0.95 && nearRaised > 0 && nearRaised < raised && t.heightAt(side[0], side[1], side[2]) === s0,
    `+${raised.toFixed(3)} at the centre, +${nearRaised.toFixed(3)} a quarter of the way out, unchanged outside`)
  const hit = t.raycast([0, 40, 0], [0, -1, 0], seaRadius(l))
  check('a ray lands on the ground', hit !== null && Math.abs(hit[1] - (seaRadius(l) + t.heightAt(0, 1, 0))) < 1e-3,
    hit ? `hit at r = ${hit[1].toFixed(3)}, ground at ${(seaRadius(l) + t.heightAt(0, 1, 0)).toFixed(3)}` : 'missed')

  // 7. Rebuilding after a stroke keeps the water. Water the new rock buried
  // moves out onto its column: onto the rock, or on top of the water already
  // there, a spacing up (not into it).
  const sp = p.spacing
  const dirA = up
  const dirB = [Math.sin(0.3), Math.cos(0.3), 0]
  const gA = groundRadius(l, t, dirA)
  const gB = groundRadius(l, t, dirB)
  const at = (d: number[], r: number) => d.map((v) => v * r)
  const water = Float32Array.from([
    ...at(dirA, gA - 1), // buried, nothing above it
    30, 0, 0, // far away, untouched
    ...at(dirB, gB - 1), // buried, under…
    ...at(dirB, gB + 2), // …this, already in its column
  ])
  const kept = planetScene(p, l, t, 10, { pos: water, vel: new Float32Array(12).fill(1) })
  const rA = Math.hypot(kept.fluid[0], kept.fluid[1], kept.fluid[2])
  const rB = Math.hypot(kept.fluid[6], kept.fluid[7], kept.fluid[8])
  check(
    'rebuild moves buried water out',
    kept.fluid.length === 12 && kept.capacity === 14 && rA > gA && rA <= gA + sp + 1e-3 && kept.fluid[3] === 30 && Math.abs(rB - (gB + 2 + sp)) < 1e-3 && kept.velocity![0] === 0,
    `onto the rock: r = ${rA.toFixed(2)} (ground ${gA.toFixed(2)}); under water: r = ${rB.toFixed(2)}, a spacing over the water above (${(gB + 2).toFixed(2)})`,
  )
}

process.exit(results.every(Boolean) ? 0 : 1)
