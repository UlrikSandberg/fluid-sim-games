# Next: boundary particles and a draggable rectangle obstacle

Plan agreed at the end of the Phase 1 session. **Status: all four steps done.**
Step 4 became: rotation, two-way coupling (pressure and XSPH reactions), corner
contacts, and `scripts/buoyancy.ts`. Read the
README's "Phase 1 notes" first.

## Goal

Replace the wall clamp's density deficiency (see README, "Known limitations") with
**boundary particles**: static particles that count toward fluid density and push
back through the normal pressure force. Then reuse the same mechanism for a
**rectangle obstacle** that you can place and drag with the mouse. Boundary
particles are **never rendered**; the obstacle is drawn as a rectangle outline.

This is the foundation for the duck: walls (fixed), dragged obstacle (moved by the
mouse) and duck (moved by fluid forces) differ only in who moves them.

## Steps

1. **Walls.** Generate 2–3 layers of boundary particles, about `h` thick, just
   outside each wall at the rest spacing.
   - Extend the spatial grid's domain by `h` on every side so they can be found.
   - Store them after the fluid particles (indices `n..n+nb`) or in separate arrays,
     with a per-particle flag. Either way the integrator, XSPH and boundary clamp only
     touch fluid particles.
   - Density: fluid particles sum boundary neighbours too (Akinci et al. 2012 weight
     each boundary particle by a volume ψ = ρ₀ / Σ_k W(r_bk) over its boundary
     neighbours, which handles uneven sampling).
   - Pressure force from boundary particle b on fluid i: use mirrored pressure
     (p_b = p_i), i.e. `−ψ_b · p_i/ρ_i² · ∇W`.
   - Keep the position clamp as a safety net.
   - **Verify:** the wall-layer probe (floor layer back to ~100 particles per row, no
     gap), `npm test` still passes, and the mouse spike count near walls drops.
2. **Static rectangle obstacle.** A rectangle `{centre, halfSize}` whose outline is
   filled with boundary particles the same way. Draw its outline in the renderer.
3. **Draggable.** Ctrl+click places one; dragging inside it moves it. Width and height
   sliders, right-click deletes.
   - Its boundary particles move with it and carry its velocity, so XSPH and pressure
     see a moving wall.
   - Cap drag speed with the same Mach 0.25 / CFL limit as the brush
     (`mouseSpeedCap`).
   - On placement, push any fluid particles trapped inside out to the nearest edge.
4. **Later: the duck.** Make the body dynamic: sum the fluid's forces on its boundary
   particles, integrate it as a rigid body, and buoyancy follows.

## Watch out for

- Boundary particles change the bulk density calibration slightly: re-check the
  ρ/ρ₀ ≈ 1 baseline and the acceptance thresholds.
- Neighbour counts per particle rise near walls. The neighbour list already grows
  as needed.
- Determinism: map obstacles are static, so race reproducibility is unaffected;
  dragging is a sandbox-only input.
- Performance: boundary particles add neighbours to wall-adjacent fluid. Measure
  ms/step with `npm run check`.
