// Determinism test: same seed, 1000 steps, no mouse input → identical checksum of
// the position array across two independent runs (separate processes, so no
// shared JIT or module state). A different seed must give a different checksum.
// Usage: node scripts/determinism.ts
import { spawnSync } from 'node:child_process'
import { createHash } from 'node:crypto'
import { fileURLToPath } from 'node:url'
import { defaultParams } from '../src/sim/params.ts'
import { createState, step } from '../src/sim/sph.ts'

const STEPS = 1000

// SHA-256 over the raw bytes of the Float32 position array: bit-exact comparison.
function checksum(seed: number): string {
  const p = { ...structuredClone(defaultParams), seed }
  const s = createState(p)
  for (let i = 0; i < STEPS; i++) step(s, p)
  return createHash('sha256').update(new Uint8Array(s.pos.buffer, s.pos.byteOffset, s.pos.byteLength)).digest('hex')
}

function runChild(seed: number): string {
  const self = fileURLToPath(import.meta.url)
  const r = spawnSync(process.execPath, [self, '--child', String(seed)], { encoding: 'utf8' })
  if (r.status !== 0) throw new Error(`child run failed:\n${r.stderr}`)
  return r.stdout.trim()
}

const childIdx = process.argv.indexOf('--child')
if (childIdx !== -1) {
  console.log(checksum(Number(process.argv[childIdx + 1])))
} else {
  const seed = defaultParams.seed
  const a = runChild(seed)
  const b = runChild(seed)
  const other = runChild(seed + 1)
  console.log(`seed ${seed}, ${STEPS} steps`)
  console.log(`  run 1: ${a}`)
  console.log(`  run 2: ${b}`)
  console.log(`  seed ${seed + 1}: ${other}`)
  const same = a === b
  const differs = other !== a
  console.log(`${same ? 'PASS' : 'FAIL'} identical checksum across runs`)
  console.log(`${differs ? 'PASS' : 'FAIL'} different seed gives a different checksum`)
  process.exit(same && differs ? 0 : 1)
}
