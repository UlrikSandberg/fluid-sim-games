# Duck Game: SPH fluid sandbox

The foundation for a 3D duck-racing game in which ducks float down a river driven by a
real fluid simulation, used as a verifiably fair randomizer. One weakly-compressible
SPH solver drives three scenes, and a shallow-water solver a fourth, picked from a start
screen:

- **2D Sandbox** (`#/2d`): a side-view tank with mouse forces, a floating rigid body
  and water rendering.
- **3D Sandbox** (`#/3d`): the same solver in a 3D tank, on the CPU or the GPU.
- **River Race** (`#/race`): the game, in progress. A river with a waterfall on the GPU.
- **Ripple Tank** (`#/ripples`): waves on a height field (shallow-water equations): a
  pool you poke, a wave maker, slits, and an underwater lens.

A simulation's dimension is its domain's (`dimOf(params)`), so 3D was a preset plus a
renderer, not a rewrite.

## Running

```sh
npm install
npm run dev            # http://localhost:5173 (start screen; the sandbox is #/2d)
npm run build          # typecheck + static build in dist/
npm test               # determinism, acceptance, buoyancy, 3D, shallow-water checks (~5 min)
npm run test:shallow   # just the shallow-water checks (seconds)
npm run check -- 20 stiffness=8000 dt=0.003   # headless dam break with param overrides
npm run check -- --3d 10                      # the same, with the 3D preset
npm run test:gpu       # GPU solver checks in headless Chrome (needs `npm run dev` running)
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
| `1`–`5` | Colour mode: density ρ/ρ₀, neighbour count, speed, pressure, water (foam = speed; in 3D water rendering, clear water with foam) |
| `V`, or the panel's Draw toggle | Particles (diagnostic dots) or Water (smooth surface) |
| Space | Pause / resume |

In **current** mode, pressing anchors the brush and dragging draws an arrow from
the anchor. While held, fluid in the brush is driven along the arrow. A longer arrow
gives a faster current, reaching full speed once it spans the brush diameter.

The control panel edits solver parameters live without restarting the simulation.
Particle count, spacing and seed apply on **Reset**; the button turns amber while
those changes are pending.

## 3D Sandbox scene

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
(curvature 0.0346 averaged over 5 s, vs ω²/2g = 0.0327; the ~6% excess is the
measurement: particle centres sit just under the surface, which still wobbles). Ten seconds of swooshing stays bounded and
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
a separable bilateral blur; a final pass rebuilds normals and shades. With colour mode
`5` (Water) it's drawn as clear water: what's behind and beneath shows through, bent by
the surface and dimmed by the depth it crosses (red first, Beer–Lambert), with light
scattered back from inside, a Fresnel sky reflection and the sun's glint. Foam lives in
a 2D grid over the surface (`src/render/foamField.ts`, since the GPU reorders its
particles each step and they can't carry it). Churning water makes it. That means a
particle moving differently from the water around it (a plunge pool, the flow round a
rock), or the local flow being slammed to a stop (a wave landing, a splash on a wall).
Fast but smooth water (a river, the sheet over a waterfall's lip) makes none. Foam
drifts with the surface flow and fades over ~2 s, and shows only where a particle is at
a free surface. A free surface is judged by a shortage of neighbours, so walls and lids
count and it works upside down in flip zones. The shader draws it as an irregular,
broken lace of threads, a solid layer where it's thick, and small bubbles that swell
and pop. The texture travels with the surface flow: the foam grid's velocities go to the
GPU as a texture, and the pattern is dragged along in two crossfaded cycles (a flow map),
so it moves with the water without stretching. Fresh churn also clouds the water briefly (bubbles, gone in
a third of a second). Lone particles (few neighbours) are spray: left out
of the surface, where they made blobs, and drawn as white droplets. The other colour
modes keep the opaque, colour-mode shading. A depth test lets the floating box sit in
the water. Known limits: faint vertical streaks can show inside tall breaking waves, and
silhouettes stay slightly scalloped by the spheres.

**GPU engine (WebGPU, the default where available):** the same physics as compute
shaders in `src/gpu/` (WGSL in `sphShaders.ts`). Each substep is six dispatches:
count particles per cell (atomics), prefix-sum the counts, scatter a cell-sorted copy,
then density, acceleration and integration, each gathering from the 3×3×3 neighbouring
cells. Walls are a static, pre-sorted boundary grid built by the CPU code; the floating
box's force and torque are summed with fixed-point atomics and the box is integrated
on the CPU once per frame. Positions come back to the existing renderer through a
double-buffered async readback. Everything fits in six storage buffers, inside
WebGPU's guaranteed 8 per shader stage.

It is not bit-reproducible (atomics order the grid), which the game doesn't need, so
`npm run test:gpu` checks it statistically against the CPU solver's results: bulk
ρ/ρ₀ 1.006 (CPU 1.006), median 32 neighbours (32), a tilted tank level to 0.12 (0.16),
Newton's paraboloid 0.035 vs 0.0327 (CPU 0.0346, both averaged over 5 s; a single
snapshot flaked past the ±15% bar on the GPU, whose runs differ), and a 0.5 ρ₀ box floating at draft
0.92 vs 1.00. Throughput on an Apple-silicon laptop: 0.15 ms per substep at 4,000
particles, 0.34 ms at 40,000, 1.45 ms at 200,000 (the CPU needs ~180 ms for that).
The full app (solver, readback and rendering) holds 60 fps at 200,000 particles.
The panel's Engine toggle switches CPU/GPU, and the Tank presets (Small 4k, Large 40k,
XL 150k) restart the simulation.

Not in 3D yet: resizing the tank.

## River Race scene (in progress)

A river built from pieces, on the GPU solver (`src/scenes/race/`). The default
course (a bend, a waterfall, a U-turn, a flip zone, a second waterfall) has about
39k particles at particle size 0.5 and runs at 60 fps on an M5 Pro; raise the
particle size for weaker GPUs.

**Before the start:** a new course is built with the ducks already hanging side by
side at the gate, while the water settles and runs under them. Look around, change
the racers, roll another course (`N`), then press **Start race** (`S`): a 3, 2, 1
countdown (on simulated time, so pausing holds it), then the gate opens. After the
results, **Race again here** (`S`) lines up on the same course; **New course** (`N`)
rolls another.

**Ducks and boats:** each racer floats in a 🦆 duck or a ⛵ boat (the toggle beside
its name, remembered with the names). A duck is a 1.6 × 1 × 1.2 box under its mesh.
A boat is a 2.6 × 0.8 × 1.1 hull whose sides curve in to a pointed bow over its
front 40% (`BodyShape` in `sph.ts`). The water sees that shape: its boundary
particles follow the hull, the shader's push-out narrows toward the bow, and its
mass uses the hull's volume. Both carry ballast (centre of gravity below the
centre: half the half height for a duck, 0.9 for a boat's keel), a righting torque
from gravity. Without it, ducks and boats both spent about half a race past 90°; with
it they ride mostly within 25°, except in flip zones, where reversed gravity
turns them upside down.

**Ride a duck:** `C` cycles the camera: orbit → **first person** from a racer's head
(that duck isn't drawn) → **third person** from behind and above it (the mouse wheel
moves it nearer or farther) → orbit. Racers → Ride picks the duck. The view turns
with the duck's heading, eased so a spinning duck doesn't jolt it. While racing, the
arrow keys steer the ridden duck: ←/→ turn it (2.2 rad/s), ↑ paddles along its
heading (up to 2.5 faster than the current), so it's a nudge, not a motor.

**Pieces** (`course.ts`): a fixed start straight, then the panel's pieces in order,
then a fixed finish straight:
- **Straight** (length).
- **Bend** (turn −180°…180°, radius); a 180° bend is a **U-turn**. The tightest radius
  is the half width + 3, so the two legs' banks stay clear of each other's water.
- **Waterfall** (drop): a step down in the bed, with a sheer lip.
- **Flip zone** (length): gravity is reversed (×−0.6) for water and ducks under a
  violet glass lid 3.5 above the water, so the river lifts off its bed and runs along
  the lid, then pours back down where the zone ends.

The panel lists the pieces with a dial each, ↑ ↓ to reorder, ✕ to remove and buttons to
add more; it warns (and won't build) when the river runs into itself. Editing a
piece turns the random pieces 🎲 off, so you can roll a course, keep it and tweak it.

**Dials** (the River panel):
- **Course** (applies to the next course; New course turns amber while changes are
  pending): particle size (the main particle-count dial; the panel estimates the
  count), width 🎲, depth, and either your pieces or random pieces 🎲 made of a target
  length 🎲 and numbers of waterfalls 🎲, U-turns 🎲, bends 🎲 and flip zones 🎲. Each
  🎲 that's on draws its dial at random for every new course; off uses your value.
  A random course shuffles its features with straights between them (longer after a
  waterfall, for the splash), stretches the straights to the length, and retries
  turn directions until the river doesn't overlap itself.
- **Water** (live): current speed and pull, gravity, stiffness, XSPH ε, and dt with
  its CFL limit.
- **Race** (each course): rocks, swirls and power-ups per 50 units of river.
- **Racers:** how many (1–8; re-hangs the gate before a race), their names, who does
  the chore, follow camera.
- **Ducks:** Add one (`D`), Clear, density (live), size (next line-up).

**What makes it a river** (solver features in `src/gpu`):
- **Course lookup:** the centreline is sampled every 0.25 along its length, and a
  grid over the ground (cells of 0.5) stores each cell's nearest sample, so the
  shader finds any particle's distance along the course (s) and across it (d) with
  two reads and that sample's tangent line. Everything below works in (s, d), so a
  U-turn is no harder than a straight.
- **Banks, bed, lips, ends, lids:** boundary particles (banks along their own arc
  length, the bed and the solid behind each lip on a lattice) plus a push-out in the
  shader: out through the banks, up out of the bed or downstream past a lip,
  whichever is nearer, and down from a flip zone's lid. Ducks get the same rules on
  the CPU (`courseConstraint`).
- **Current:** each particle's velocity along the course tangent relaxes toward a
  target speed. It stands in for a real river's slope and bed friction; a plain
  downhill push would accelerate forever with free-slip walls.
- **Recycling:** water near the end re-enters at the start, lifted by the course's
  total drop and keeping its place and velocity relative to the channel, so the river
  never drains.
- **Ground:** a height grid with the vertices near a bank, lip or end moved onto it,
  so those render as sheer walls rather than a staircase.

**Ducks:** up to 8 rigid bodies on the GPU (`MAX_BODIES`), each with its own range
of boundary particles and its own force/torque atomics; the shader loops over the
bodies near each particle. Each duck's physics is a box hull, the shape the
Archimedes tests cover. The rubber-duck look (body, head, beak, eyes, tail, one
colour per racer) is only a mesh. On the CPU, each duck is integrated from its GPU
forces, kept between the banks (`constrainBody`), and bumped off the others
(`collideBodies`: two boxes overlap when their centres are closer than the sum of
their reaches along the line between them, which is exact face-on, so ducks can sit
side by side; overlaps are separated by mass and bounced with restitution). An 8-duck race runs at 60 fps; the ducks jostle over the
waterfall and spread out along the lower reach.

**Race rules:** each duck has a name (the panel's Racers section, next to its colour;
remembered in this browser). A chequered finish line spans the river near the end;
ducks are timed across it, and a live leaderboard ranks finished ducks by time and
the rest by distance down the course. The race ends when every duck is in, or after
2 minutes (the rest ranked by distance), with a results card: the winner, the
placings, and who does the chore (last place or the winner, a panel setting). The
follow camera (`F`, on by default) looks at the start line during the countdown,
then glides after the leading pack, then rests on the finish.

**Randomness** (`features.ts`, the panel's Race section; drawn for each course):
- **Rocks, swirls and power-ups** are set per 50 units of river (the panel shows the
  resulting count), so longer courses get more of each at the same density.
- **Rocks:** static solid boxes in the river, clear of the start, the waterfalls and
  the flip zones and the finish, always leaving a passage. Water flows around them; ducks
  bounce off (`solidConstraint`).
- **Moving obstacles** (`movers.ts`; the Race section's Moving dial, per 50 units,
  up to 8, with toggles for which kinds): kinematic bodies on the GPU, moved by a
  script, pushing the water (and the ducks, on the CPU) but never pushed back.
  - **Gates:** a wooden wall slides out of a bank 28–36% of the way across, holds,
    slides back in (out of sight under the bank) and rests, every 6–10 s.
  - **Pulsing rocks** grow and shrink (±35%) every 4–7 s; their boundary particles
    are rebuilt whenever the size has changed by a quarter spacing.
  - **Sliders:** a block wandering from bank to bank.
  - **Spinners:** a bar about half the width across turning about its middle.
  They share the GPU's body slots with the ducks (16 in all: 8 ducks, 8 movers).
- **Swirls** (up to 8 at a time, radius 4–6): whirlpools that turn the water's
  horizontal velocity toward a rotation about a point (fastest mid-radius) with a
  gentle inward pull, shown as glowing spiral discs. They come and go: each lasts
  5–10 s, spinning up and down over 1.2 s (the disc fades and grows with it), is gone
  for 1–4 s, then turns up somewhere else, so no duck stays stuck in one. Ducks also
  feel them directly, so a caught duck circles.
- **Power-ups:** floating gems; the first duck to touch one sets it off.
  ⬆ Flip reverses that duck's gravity (×−0.6) for 1.5 s, so it hops out and over;
  ⚡ Boost kicks it downstream; 🌌 Zero-G switches gravity off for 2 s for all the
  water and every duck, then everything splashes down (another one while it's on
  adds 2 s). An event feed says who grabbed what. Power-ups come back when the ducks
  line up again.
- **Lanes** are dealt at random each line-up: with a fixed order, two races in a row
  finished in identical order, because the course favours some lanes.

Known roughness: some splash where recycled water re-enters.

## Ripple Tank scene

A pool 24 m × 16 m and 1 m deep on a 300×200 grid of 8 cm cells, solved with the
shallow-water equations (`src/sim/shallow/shallow.ts`, plain TypeScript on the CPU,
2–3 ms a step, 2–3 steps a frame). The water is a height field over a bed, with
velocities on the faces between cells (a staggered grid). Each step:

1. carries momentum with the flow, in Stelling & Duinmeijer's (2003)
   momentum-conserving upwind form (it moves momentum into newly wetted cells, so a
   flood front runs at the right speed);
2. accelerates each face down the slope of the water surface (not the depth's, so a
   lake over a bumpy bed stays still); a face is dry when neither side's water reaches
   over the higher bed, which handles beaches and dam breaks;
3. moves water by upwind fluxes, scaled so no cell is emptied below zero;
4. relaxes sponge cells toward rest, so waves leave the tank instead of bouncing;
5. makes foam where the surface is steep or the flow converges, and carries it.

Presets: **Pool**; **Single slit** and **Double slit** (diffraction and
interference); **Underwater lens**: a round shoal where the water is 0.3 m deep. Waves
travel at √(g·h), so they slow over it and bend to a focus behind it. The wave maker is a
paddle on the left wall. It's absorbing: each row drives the water toward 2·η_in − η
(the shallow-water radiation condition), so waves reflected back from the slit wall are
swallowed instead of piling up into a standing wave. "Open water" puts sponges on the
other three walls.

Controls: click to drop a ripple; hold to keep bobbing (move while holding for a wake
and the Doppler effect); right-drag or Shift-drag to orbit; wheel to zoom; `W` wave
maker, `H` heights view (crests red, troughs blue: interference patterns stand out),
`R` reset, Space pause.

Rendering (`src/render/heightfieldWater.ts`, made to be reused by later height-field
scenes): the scene without the water renders to a target (colour and depth); then the
water's shader refracts that by the surface slope, absorbs by the distance through the
water, mixes in a sky reflection by Fresnel, adds the sun's glint and the simulation's
foam. The floor's shader adds caustics from the surface's curvature (a crest focuses
light below it).

`npm run test:shallow` checks:
- volume is kept (1e-8 relative);
- waves run at √(g·h) (3.120 vs 3.132 m/s);
- a lake at rest over bumps stays at rest;
- a dam break has no negative depths and matches Ritter's exact solution (6 mm RMS);
- sponges absorb ripples (10× less left after 20 s than in a closed pool);
- a step is fast enough for real time.

## Layout

```text
src/App.tsx    hash router: start screen at #/, each scene at its own route
src/scenes/    registry.ts (the scene list), Home.tsx
  sandbox2d/   the 2D sandbox: Sandbox2D.tsx, ControlPanel.tsx, loop.ts
  fluid3d/     the 3D sandbox: Fluid3D.tsx (with its panel), loop.ts, engines.ts (CPU/GPU)
  race/        the river race: course.ts (pieces → centreline, lookup, ground),
               riverScene.ts (water and boundary), features.ts (rocks, swirls,
               power-ups), loop.ts, RiverRace.tsx
  ripples/     the ripple tank: presets.ts (tank set-ups), loop.ts, RippleTank.tsx
               (scenes import shared modules, never each other; each is loaded
               only when opened)
src/engines/   sph.ts: the SPH solver as the scenes use it, CPU or GPU
src/sim/       solvers: plain TypeScript over typed arrays, no DOM, runs in Node
  shallow/       shallow.ts: shallow-water equations on a staggered height field
  params.ts      2D and 3D presets, dimOf, derived values (mass, CFL, brush caps)
  kernels.ts     Poly6 + Spiky; the only place with dimension-specific constants
  spatialHash.ts counting-sort cell grids (fluid, boundary), unique-pair list
  sph.ts         state arrays, boundary particles, the obstacle body, and step()
src/gpu/       WebGPU solver: sphShaders.ts (WGSL), sphGpu.ts (buffers, passes,
               readback), gpuChecks.ts (in-browser checks for npm run test:gpu)
src/render/    view3d.ts (renderer, camera, orbit controls, resizing: any 3D scene),
               renderer.ts (2D, orthographic), renderer3d.ts (the SPH scenes),
               colour modes, waterSurface.ts (2D metaballs),
               waterSurface3d.ts (3D screen-space fluid), foamField.ts (its foam),
               heightfieldWater.ts (height-field water: refraction, caustics)
src/ui/        shared React pieces: HUD, legend, panel controls
scripts/       determinism.ts, acceptance.ts, buoyancy.ts, acceptance3d.ts,
               shallow.ts, check-sim.ts (tuning)
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
