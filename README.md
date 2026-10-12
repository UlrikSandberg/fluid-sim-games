# Duck Game: SPH fluid sandbox

The foundation for a 3D duck-racing game in which ducks float down a river driven by a
real fluid simulation, used as a verifiably fair randomizer. One weakly-compressible
SPH solver drives two scenes, picked from a start screen:

- **2D Sandbox** (`#/2d`): a side-view tank with mouse forces, a floating rigid body
  and water rendering.
- **3D Fluid** (`#/3d`): the same solver in a 3D tank with an orbit camera.

A simulation's dimension is its domain's (`dimOf(params)`), so 3D was a preset plus a
renderer, not a rewrite.

## Running

```sh
npm install
npm run dev            # http://localhost:5173 (start screen; the sandbox is #/2d)
npm run build          # typecheck + static build in dist/
npm test               # determinism, acceptance, buoyancy, 3D checks (~5 min)
npm run check -- 20 stiffness=8000 dt=0.003   # headless dam break with param overrides
npm run check -- --3d 10                      # the same, with the 3D preset
```

## Controls (2D sandbox)

| Input | Action |
| --- | --- |
| Left drag (or touch) | Apply the brush mode: repel, attract, or current |
| `B`, or the panel's Drag toggle | Cycle brush mode |
| Right drag, or Shift + drag | The opposite of the brush mode (reverse flow in current mode) |
| Mouse wheel | Brush radius |
| Ctrl + click | Place the rectangle obstacle (replaces an existing one) |
| Left drag inside the obstacle | Move it (speed-capped like the brush); a floating one keeps its velocity on release, so you can throw it |
| Right click on the obstacle | Delete it |
| `1`–`5` | Colour mode: density ρ/ρ₀, neighbour count, speed, pressure, water (foam = speed) |
| `V`, or the panel's Draw toggle | Particles (diagnostic dots) or Water (smooth surface) |
| Space | Pause / resume |

In **current** mode, pressing anchors the brush and dragging draws an arrow from
the anchor. While held, fluid in the brush is driven along the arrow. A longer arrow
gives a faster current, reaching full speed once it spans the brush diameter.

The control panel edits solver parameters live without restarting the simulation.
Particle count, spacing and seed apply on **Reset**; the button turns amber while
those changes are pending.

## 3D Fluid scene

Drag to orbit, right-drag to pan, wheel to zoom; `1`–`5` colour modes, Space pauses.
**Rotate the tank** to slosh the water: switch the drag mode with `T` (or the panel),
or Shift+drag in orbit mode; `L` levels it again. Drag right rolls the tank clockwise
as you see it; drag down tips its top toward you.

The water isn't pushed by moving walls: the solver works in tank coordinates, where
gravity rotates with the tilt and every particle feels the rotating frame's inertial
forces (Euler −α×r, centrifugal −ω×(ω×r), Coriolis −2ω×v). For a rigid tank that is
exact, and it needs no boundary regeneration. The tank chases your drag with spin
capped at 2 rad/s and spin-up at 8 rad/s² (`src/sim/tankMotion.ts`), so even a wild
swoosh moves the walls through the water at about Mach 0.2 at most. `npm run test:3d`
checks it against two exact answers: a tank tilted 20° settles with a world-level
surface (spread 0.16), and one spinning at 0.8 rad/s forms Newton's paraboloid
(curvature 0.0329 vs ω²/2g = 0.0327). Ten seconds of swooshing stays bounded and
resettles.
The preset (`defaultParams3D`) uses spacing 0.5·h, which gives ~32 neighbours in 3D,
the usual 30–40 target (2D's 0.4·h would give ~64). A 3D step costs ~0.9 µs per
particle on one CPU thread, so the default of 4000 particles in a 16×10×8 tank is
what fits 60 fps. `npm run test:3d` checks the 3D kernel constants, settling
(bulk ρ/ρ₀ 1.006), neighbours (median 32), the domain and determinism.

**Floating boxes:** the panel's Floating box section drops a box at the top of the
tank (size from the sliders, density live). It is the same rigid body as in 2D, now
with a quaternion orientation, 3-vector angular velocity and torque, a box inertia
tensor with Euler's equations (I ω̇ = τ − ω × Iω), and corner contacts that use the
full tensor and rub along the actual sliding direction. In a tilting tank it also
feels the rotating-frame forces. `npm run test:buoyancy3d`: drafts 0.45 / 1.00 /
1.57 against Archimedes' 0.50 / 1.00 / 1.50, a 1.5 ρ₀ box sinks and lies flat, and
made light it lifts off.

**Water rendering** (`V` toggles it against the diagnostic spheres; on by default):
screen-space fluid rendering in `src/render/waterSurface3d.ts`. Particles render as
sphere impostors into a depth target and a thickness target; a narrow-range filter
(Truong & Yuksel 2018) smooths the depth into one surface without the axis streaks of
a separable bilateral blur; a final pass rebuilds normals and shades. Thick water takes
the colour mode's colour (`5`: blue, foaming white with speed), thin water shows the
refracted scene through it, a Fresnel term mixes in sky reflection, and a depth test
lets the floating box sit in the water. Known limits: faint vertical streaks can show
inside tall breaking waves, and silhouettes stay slightly scalloped by the spheres.

Not in 3D yet: resizing the tank, and enough speed for river-sized particle counts
(Web Workers / WebGPU, or Position Based Fluids for bigger timesteps).

## Layout

```text
src/App.tsx    hash router: start screen at #/, each scene at its own route
src/scenes/    registry.ts (the scene list), Home.tsx
  sandbox2d/   the 2D sandbox: Sandbox2D.tsx, ControlPanel.tsx, loop.ts
  fluid3d/     the 3D scene: Fluid3D.tsx (with its panel), loop.ts
src/sim/       solver: plain TypeScript over typed arrays, no DOM, runs in Node
  params.ts      2D and 3D presets, dimOf, derived values (mass, CFL, brush caps)
  kernels.ts     Poly6 + Spiky; the only place with dimension-specific constants
  spatialHash.ts counting-sort cell grids (fluid, boundary), unique-pair list
  sph.ts         state arrays, boundary particles, the obstacle body, and step()
src/render/    renderer.ts (2D, orthographic), renderer3d.ts (perspective, orbit),
               colour modes, waterSurface.ts (2D metaballs),
               waterSurface3d.ts (3D screen-space fluid)
src/ui/        shared React pieces: HUD, legend, panel controls
scripts/       determinism.ts, acceptance.ts, buoyancy.ts, acceptance3d.ts,
               check-sim.ts (tuning)
```

## Phase 1 notes

### Parameters changed from the handoff, and why

| Parameter | Handoff | Now | Why |
| --- | --- | --- | --- |
| `stiffness` | 2000 | 8000 | At 2000 the bulk stayed ~8% compressed and was still sloshing after 20 s. 6000 was enough with the wall clamp; with boundary particles the bottom rows count as bulk and read 1.05–1.06, and 8000 brings 100% of the bulk inside ±5% (mean 1.022). |
| `dt` | 0.002 | 0.004 | At 0.002, real time needs 500 steps/s (~2 ms each), which is the whole CPU. 0.004 halves that and is still under CFL: c_s = √k ≈ 89, so the limit is 0.4·h/c_s ≈ 0.0045. At k = 10000 the limit equals dt exactly and the mouse brush's speed cap drops to zero. |
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
  density. Current mode blends velocities toward the arrow's target velocity at a
  rate chosen so its peak acceleration matches the same bound.
- **Determinism:** the `.ts` extensions in `src/sim` imports exist so Node can run the
  solver directly. `Math.pow` is avoided in the solver because engines don't guarantee
  bit-identical results.

### Acceptance status

| Criterion | Status |
| --- | --- |
| 4000 particles at 60 fps | **Not verified on target hardware.** About 1.35 ms per step in Node, at ~4.2 steps per 60 fps frame, so ~5.7 ms of sim per frame on an Apple-silicon laptop. Half of that is the neighbour search. Check the HUD on a mid-range laptop. |
| Settles flat, bulk ρ/ρ₀ within ±5% | Pass. Bulk mean 1.023, 98% of bulk particles within ±5%, surface spread 0.55 after 40 s. |
| No particle leaves the domain | Pass. Checked after every step through the dam break, the mouse stress test and the resettle. |
| Neighbour count ~20 in the bulk | Pass. Bulk median 18. |
| Full-strength mouse doesn't blow up | Pass, with a caveat. A fast full-strength repel sweep gives brief spikes up to ρ/ρ₀ 1.94 (dam break alone: 1.44). Particles above 1.5 are 0.009% of particle-steps, mostly against walls. Nothing goes non-finite, and the fluid resettles. |
| Four colour modes with legends | Pass (checked in a browser). |
| Determinism (same seed, 1000 steps, two runs) | Pass. Bit-identical SHA-256 of the position array across separate processes. |

### Known limitations (Phase 2 material)

- **Density deficiency at walls: fixed by boundary particles.** With the old wall
  clamp, wall particles were missing half their kernel support and crowded to about
  twice the normal density (about 200 particles in the floor layer where a normal row
  holds about 100), followed by a gap. Static boundary particles behind each wall
  (Akinci-style ψ weights, mirrored pressure) bring the floor layer back to about 110
  with even row spacing. The clamp stays as a safety net.
- **Floating bodies (the duck's mechanics).** With the panel's Body toggle set
  to Floats, the obstacle is a rigid body. Every boundary push on the fluid (pressure)
  and every velocity change it causes (XSPH) is reacted back onto it as a force
  and torque, then integrated with gravity. Nothing models buoyancy explicitly, yet
  `npm run test:buoyancy` shows boxes at 0.25, 0.5 and 0.75 ρ₀ floating within 0.12
  of Archimedes' draft, and a 1.5 ρ₀ box sinking and resting flat. Wall contact uses
  per-corner impulses with friction, so a box landing on a corner topples. The
  orientation is a quaternion shared with 3D; in 2D every torque points along z.
- **Bodies rest `h` above the walls, and walls are free-slip.** Buoyancy is pressure
  on the bottom face, and particles can't represent a gap thinner than one spacing.
  So a body resting flush on the floor sealed the fluid out and acted like a suction
  cup: a sunk box made light didn't move for seconds. The `h` gap stands in for the
  thin film real water keeps under objects. Dropping XSPH friction against the walls
  (the floating body keeps it) doubled how fast fluid flows back into that gap.
  A sunk box made light now rises about 13 units in 8 s. A short lag remains; that
  part is real (water has to flow in sideways under a wide box before it can lift).
- **Performance headroom is thin.** Neighbours are a flat list of unique pairs,
  found by scanning only forward in the sorted grid, and each solver pass visits a
  pair once and updates both ends. That took a step from 2.1 to 1.35 ms (1.55×).
  The pair search is now about half the step; the remaining waste is wall
  particles scanning each other, which a separate static wall grid would remove.
  Beyond that: Position Based Fluids for much larger timesteps.
