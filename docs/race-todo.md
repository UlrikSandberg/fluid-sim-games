# River Race: randomness todo

The race works end to end (gate, finish, results, chore). This list adds the
randomness that makes every race different.

- [x] **1. New course each race.** Bends, bend size, waterfall height and length drawn
  from sensible ranges when a race starts (a panel toggle; off keeps your dials).
  The course is rebuilt and the race starts once it's ready.
- [x] **2. Rocks.** Random solid rocks in the river, kept clear of the start line, the
  waterfall lip and the finish, leaving a passage. Water flows around them (static
  boundary + push-out, as the ledge); ducks bounce off them (a CPU-side constraint,
  since the rigid bodies don't otherwise know about solids).
- [x] **3. Swirls.** Random whirlpools: a rotating current around a point that can trap
  a duck or fling it ahead. Shader-side (like the current), with a visible ring on
  the water.
- [x] **4. Power-ups.** Floating pickups along the course; the first duck to touch one
  sets it off:
  - **Flip:** that duck's gravity reverses for a moment, so it hops out of the water
    and over whatever is ahead, then drops back.
  - **Boost:** a kick downstream.
  - **Zero-G:** gravity switches off for 2 seconds for everything (water and ducks
    float up), then everything splashes down.
- [x] **Lanes shuffled each race.** Found while testing swirls: two races in a row
  finished in the same order, because each duck always started in the same lane.
- [x] **5. Event feed.** "Alex grabbed Zero-G!" messages on screen, so you can see why
  the race just changed.
- [x] **6. Set up, then start.** A new course is built with the ducks hanging at the
  gate while the water settles; Start counts down and opens it.
- [x] **7. Swirls come and go,** so a duck can't stay stuck in one.
- [x] **8. Composable course.** Straights, bends, U-turns, waterfalls (several) and
  flip zones (gravity reversed under a lid), in any order, or rolled at random.
