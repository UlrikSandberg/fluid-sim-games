# Duck Game: SPH fluid sandbox

The foundation for a 3D duck-racing game in which ducks float down a river driven by a
real fluid simulation, used as a verifiably fair randomizer. **Phase 1** is a 2D
weakly-compressible SPH sandbox in a box. It is written so that moving to 3D is a
parameter change (`DIM` in `src/sim/params.ts`), not a rewrite.

## Running

```sh
npm install
npm run dev            # http://localhost:5173
npm run build          # typecheck + static build in dist/
npm test               # determinism + headless acceptance checks (~1 min)
npm run check -- 20 stiffness=8000 dt=0.003   # headless dam break with param overrides
```

## Controls

| Input | Action |
| --- | --- |
| Left drag | Repel |
| Right drag, or Shift + drag | Attract |
| Mouse wheel | Brush radius |
| `1`–`4` | Colour mode: density ρ/ρ₀, neighbour count, speed, pressure |
| Space | Pause / resume |

The control panel edits solver parameters live without restarting the simulation.
Particle count, spacing and seed apply on **Reset**; the button turns amber while
those changes are pending.

## Layout

```text
src/sim/       solver: plain TypeScript over typed arrays, no DOM, runs in Node
  params.ts      DIM, the mutable params object, derived values (mass, CFL, brush caps)
  kernels.ts     Poly6 + Spiky; the only place with dimension-specific constants
  spatialHash.ts counting-sort cell grid, deterministic neighbour lists
  sph.ts         state arrays and step()
src/render/    three.js: one InstancedMesh, orthographic camera, colour modes
src/ui/        React: control panel, HUD, legend; never re-renders per frame
src/loop.ts    fixed-timestep loop, mouse input, glue between sim and renderer
scripts/       determinism.ts, acceptance.ts, check-sim.ts (tuning)
```

## Phase 1 notes

### Parameters changed from the handoff, and why

| Parameter | Handoff | Now | Why |
| --- | --- | --- | --- |
| `stiffness` | 2000 | 6000 | At 2000 the bulk stayed ~8% compressed and was still sloshing after 20 s. At 6000 it settles to ρ/ρ₀ ≈ 1.02–1.03. |
| `dt` | 0.002 | 0.004 | At 0.002, real time needs 500 steps/s (~2 ms each), which is the whole CPU. 0.004 halves that and is still under CFL: c_s = √k ≈ 77, so the limit is 0.4·h/c_s ≈ 0.0052. |
| `viscosityEps` | 0.05 | 0.1 | Settles noticeably faster. 0.3 is flatter still but looks syrupy. |

`h = 1`, `spacing = 0.4` and the derived mass were right as given: the initial lattice
reads ρ/ρ₀ = 0.994 with 20 neighbours.

### Other decisions worth knowing

- **`stiffness` means c_s² for both equations of state.** Tait uses `B = ρ₀·k/7`, so
  switching EOS keeps the same speed of sound and CFL limit. Tait only differs by
  stiffening under compression. Used directly as `B`, the same number would be about
  7× softer.
- **The mouse brush is scaled to the fluid.** Strength is a 0–1 fraction. The peak
  acceleration is `strength · 0.15 · c_s² / R`, because compression under a steady
  push goes as `a·R/c_s²`. Brushed particles are also capped at Mach 0.25 and at the
  CFL limit. With a raw acceleration, full strength crushed the fluid to 3× rest
  density.
- **Determinism:** the `.ts` extensions in `src/sim` imports exist so Node can run the
  solver directly. `Math.pow` is avoided in the solver because engines don't guarantee
  bit-identical results.

### Acceptance status

| Criterion | Status |
| --- | --- |
| 4000 particles at 60 fps | **Not verified on target hardware.** About 2 ms per step in Node, at ~4.2 steps per 60 fps frame, so ~8.5 ms of sim per frame on an Apple-silicon laptop. Half of that is the neighbour search. Check the HUD on a mid-range laptop. |
| Settles flat, bulk ρ/ρ₀ within ±5% | Pass. Bulk mean 1.023, 98% of bulk particles within ±5%, surface spread 0.55 after 40 s. |
| No particle leaves the domain | Pass. Checked after every step through the dam break, the mouse stress test and the resettle. |
| Neighbour count ~20 in the bulk | Pass. Bulk median 18. |
| Full-strength mouse doesn't blow up | Pass, with a caveat. A fast full-strength repel sweep gives brief spikes up to ρ/ρ₀ 1.94 (dam break alone: 1.44). Particles above 1.5 are 0.009% of particle-steps, mostly against walls. Nothing goes non-finite, and the fluid resettles. |
| Four colour modes with legends | Pass (checked in a browser). |
| Determinism (same seed, 1000 steps, two runs) | Pass. Bit-identical SHA-256 of the position array across separate processes. |

### Known limitations (Phase 2 material)

- **Density deficiency at walls.** Wall particles are missing half their kernel
  support, so they crowd to about twice the normal density along each wall (about 200
  particles in the floor layer where a normal row holds about 100), followed by a gap.
  Boundary particles fix this. They are also the mechanism the duck's buoyancy needs.
- **Wall slamming is the worst case for the mouse.** Most stress-test density spikes
  happen against walls, for the same reason.
- **Performance headroom is thin.** The next levers are processing each pair once
  instead of twice, or Position Based Fluids, which allows much larger timesteps.
