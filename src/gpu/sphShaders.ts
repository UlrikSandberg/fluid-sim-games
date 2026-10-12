// WGSL for the GPU SPH solver (3D). Same physics as src/sim/sph.ts: Poly6
// density, Spiky pressure gradient, linear/Tait EOS clamped at 0, mirrored
// pressure from boundary particles, XSPH with free-slip walls, the tank's
// rotating-frame accelerations, and two-way coupling with rigid bodies. Unlike
// the CPU version it is not bit-reproducible (atomics order the grid), which is
// fine for the game.
//
// One step is six dispatches over the particles:
//   binCount → scan → scatter (cell-sorted copy A → B) → density → accel → integrate (B → A)
// Every per-particle pass gathers from the 3×3×3 neighbouring cells; the x-row of
// three cells is one contiguous range of the sorted array.

export const WORKGROUP = 64
export const SCAN_THREADS = 256

// Field order must match GpuSph.writeParams.
export const MAX_BOXES = 8 // the ledge plus rocks
export const MAX_BODIES = 8
export const BODY_VEC4S = 10 // per Body struct
export const PARAMS_VEC4S = 18 + 2 * MAX_BOXES + BODY_VEC4S * MAX_BODIES

export const sphWgsl = /* wgsl */ `
// A rigid body (a box) the fluid pushes and is pushed by.
struct Body {
  center: vec4f,
  vel: vec4f,
  omega: vec4f,
  half: vec4f,       // half sizes
  rot0: vec4f,       // rotation matrix columns, body → tank
  rot1: vec4f,
  rot2: vec4f,
  boxMin: vec4f,     // bounding box grown by h
  boxMax: vec4f,
  range: vec4u,      // first boundary particle within the body section, count
}

struct Params {
  counts: vec4u,     // n, numCells, number of bodies, nWall
  cells: vec4u,      // cells per axis xyz
  gridMin: vec4f,    // grid origin xyz, cell size
  kernel: vec4f,     // h, h², poly6 coefficient, spiky gradient coefficient
  fluid: vec4f,      // mass, rest density, stiffness, useTait (0/1)
  stepping: vec4f,   // dt, XSPH ε, particle radius, restitution
  gravity: vec4f,    // tank coordinates
  omega: vec4f,      // tank angular velocity (tank coordinates)
  alpha: vec4f,      // tank angular acceleration
  pivot: vec4f,
  domainMin: vec4f,
  domainMax: vec4f,
  bodyMisc: vec4f,   // fixed-point scale for the force atomics
  current: vec4f,    // river current: target speed along +x, relaxation rate (0 = off)
  recycle: vec4f,    // past x = exitX, move to inletX + (x − exitX) and up by lift; enabled (0/1)
  boxCount: vec4u,   // static solid boxes in use
  channel: vec4f,    // winding channel: amplitude, wavenumber, half width, enabled (0/1)
  channel2: vec4f,   // centreline z at amplitude 0, 0, 0, 0
  boxes: array<vec4f, ${2 * MAX_BOXES}>, // min, max per box (static, solid)
  bodies: array<Body, ${MAX_BODIES}>,
}

struct Particle {
  pos: vec4f,
  vel: vec4f,
}

// Six storage buffers, the most WebGPU guarantees is 8 per stage (Chrome on
// Metal offers 10), so related arrays share a buffer at fixed offsets
// (n particles, C = numCells):
//   particles:   A [0, n)  (unsorted, the persistent state) · B [n, 2n) (cell-sorted)
//   fluid:       aux [0, n) (density, pressure, neighbours) · v* [n, 2n)
//   indices:     cellOf [0, n) · cellStart [n, n+C+1) · wallCellStart [n+C+1, n+2C+2)
//   cellAtomics: count [0, C+1) · cursor [C+1, 2C+1)
//   boundary:    walls [0, nWall) (xyz, ψ; cell-sorted) · bodies [nWall, …) (pos.xyz ψ, vel.xyz 0 per particle)
//   bodyForce:   per body: force xyz, torque xyz (fixed point)
@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(2) var<storage, read_write> fluid: array<vec4f>;
@group(0) @binding(3) var<storage, read_write> indices: array<u32>;
@group(0) @binding(4) var<storage, read_write> cellAtomics: array<atomic<u32>>;
@group(0) @binding(5) var<storage, read_write> boundary: array<vec4f>;
@group(0) @binding(6) var<storage, read_write> bodyForce: array<atomic<i32>>;

fn cellStartAt(c: u32) -> u32 { return indices[P.counts.x + c]; }
fn wallCellStartAt(c: u32) -> u32 { return indices[P.counts.x + P.counts.y + 1u + c]; }
fn bodyParticle(k: u32) -> vec4f { return boundary[P.counts.w + 2u * k]; }
fn bodyParticleVel(k: u32) -> vec3f { return boundary[P.counts.w + 2u * k + 1u].xyz; }

fn poly6(r2: f32) -> f32 {
  if (r2 >= P.kernel.y) { return 0.0; }
  let d = P.kernel.y - r2;
  return P.kernel.z * d * d * d;
}

// Magnitude of the Spiky gradient divided by r (multiply by the separation).
fn spikyOverR(r: f32) -> f32 {
  let d = P.kernel.x - r;
  return P.kernel.w * d * d / r;
}

fn equationOfState(rho: f32) -> f32 {
  var p: f32;
  if (P.fluid.w > 0.5) {
    let x = rho / P.fluid.y;
    let x2 = x * x;
    let x4 = x2 * x2;
    p = (P.fluid.y * P.fluid.z / 7.0) * (x4 * x2 * x - 1.0);
  } else {
    p = P.fluid.z * (rho - P.fluid.y);
  }
  return max(p, 0.0);
}

// Euler, centrifugal and Coriolis accelerations of the rotating tank frame.
fn frameAccel(x: vec3f, v: vec3f) -> vec3f {
  let r = x - P.pivot.xyz;
  let w = P.omega.xyz;
  return -cross(P.alpha.xyz, r) - cross(w, cross(w, r)) - 2.0 * cross(w, v);
}

// The winding channel's centreline z(x) = z₀ + A sin(k x) and its slope dz/dx.
fn channelZ(x: f32) -> f32 { return P.channel2.x + P.channel.x * sin(P.channel.y * x); }
fn channelSlope(x: f32) -> f32 { return P.channel.x * P.channel.y * cos(P.channel.y * x); }

// Downstream direction: +x, or along the channel where it winds.
fn downstream(x: vec3f) -> vec3f {
  if (P.channel.w < 0.5) { return vec3f(1.0, 0.0, 0.0); }
  return normalize(vec3f(1.0, 0.0, channelSlope(x.x)));
}

fn cellCoords(x: vec3f) -> vec3i {
  let c = vec3i(floor((x - P.gridMin.xyz) / P.gridMin.w));
  return clamp(c, vec3i(0), vec3i(P.cells.xyz) - vec3i(1));
}

fn cellIndex(c: vec3i) -> u32 {
  return u32(c.x) + u32(c.y) * P.cells.x + u32(c.z) * P.cells.x * P.cells.y;
}

fn nearBody(x: vec3f, b: u32) -> bool {
  return all(x >= P.bodies[b].boxMin.xyz) && all(x <= P.bodies[b].boxMax.xyz);
}

fn toBody(v: vec3f, b: u32) -> vec3f {
  return vec3f(dot(P.bodies[b].rot0.xyz, v), dot(P.bodies[b].rot1.xyz, v), dot(P.bodies[b].rot2.xyz, v));
}

fn fromBody(v: vec3f, b: u32) -> vec3f {
  return P.bodies[b].rot0.xyz * v.x + P.bodies[b].rot1.xyz * v.y + P.bodies[b].rot2.xyz * v.z;
}

// Adds force f acting at body particle position xb to body b's force/torque.
fn pushOnBody(b: u32, f: vec3f, xb: vec3f) {
  let s = P.bodyMisc.x;
  let t = cross(xb - P.bodies[b].center.xyz, f);
  let o = 6u * b;
  atomicAdd(&bodyForce[o], i32(round(f.x * s)));
  atomicAdd(&bodyForce[o + 1u], i32(round(f.y * s)));
  atomicAdd(&bodyForce[o + 2u], i32(round(f.z * s)));
  atomicAdd(&bodyForce[o + 3u], i32(round(t.x * s)));
  atomicAdd(&bodyForce[o + 4u], i32(round(t.y * s)));
  atomicAdd(&bodyForce[o + 5u], i32(round(t.z * s)));
}

// [start, end) of the sorted array for the x-row of three cells around c at
// offsets (dy, dz), or an empty range outside the grid.
fn rowRange(starts: ptr<function, vec2u>, c: vec3i, dy: i32, dz: i32, wall: bool) {
  let y = c.y + dy;
  let z = c.z + dz;
  if (y < 0 || y >= i32(P.cells.y) || z < 0 || z >= i32(P.cells.z)) {
    *starts = vec2u(0u, 0u);
    return;
  }
  let base = u32(y) * P.cells.x + u32(z) * P.cells.x * P.cells.y;
  let x0 = u32(max(c.x - 1, 0));
  let x1 = u32(min(c.x + 1, i32(P.cells.x) - 1));
  if (wall) {
    *starts = vec2u(wallCellStartAt(base + x0), wallCellStartAt(base + x1 + 1u));
  } else {
    // Ranges index B, which starts at n in the particles buffer.
    *starts = vec2u(cellStartAt(base + x0), cellStartAt(base + x1 + 1u)) + vec2u(P.counts.x);
  }
}

@compute @workgroup_size(${WORKGROUP})
fn binCount(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let c = cellIndex(cellCoords(particles[i].pos.xyz));
  indices[i] = c;
  atomicAdd(&cellAtomics[c], 1u);
}

var<workgroup> partial: array<u32, ${SCAN_THREADS}>;

// Exclusive prefix sum of the cell counts in one workgroup: each thread sums a
// contiguous block, the block sums are scanned, then each block is written out.
@compute @workgroup_size(${SCAN_THREADS})
fn scan(@builtin(local_invocation_id) lid: vec3u) {
  let n = P.counts.y;
  let per = (n + ${SCAN_THREADS}u - 1u) / ${SCAN_THREADS}u;
  let start = min(lid.x * per, n);
  let end = min(start + per, n);
  var sum = 0u;
  for (var i = start; i < end; i++) { sum += atomicLoad(&cellAtomics[i]); }
  partial[lid.x] = sum;
  workgroupBarrier();
  for (var offset = 1u; offset < ${SCAN_THREADS}u; offset *= 2u) {
    var add = 0u;
    if (lid.x >= offset) { add = partial[lid.x - offset]; }
    workgroupBarrier();
    partial[lid.x] += add;
    workgroupBarrier();
  }
  var running = 0u;
  if (lid.x > 0u) { running = partial[lid.x - 1u]; }
  for (var i = start; i < end; i++) {
    indices[P.counts.x + i] = running;
    running += atomicLoad(&cellAtomics[i]);
  }
  if (lid.x == ${SCAN_THREADS}u - 1u) { indices[P.counts.x + n] = partial[lid.x]; }
}

@compute @workgroup_size(${WORKGROUP})
fn scatter(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let c = indices[i];
  let slot = cellStartAt(c) + atomicAdd(&cellAtomics[P.counts.y + 1u + c], 1u);
  particles[P.counts.x + slot] = particles[i];
}

@compute @workgroup_size(${WORKGROUP})
fn density(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let xi = particles[P.counts.x + i].pos.xyz;
  let c = cellCoords(xi);
  let mass = P.fluid.x;
  var rho = mass * poly6(0.0);
  var neighbours = 0.0;
  var range: vec2u;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      rowRange(&range, c, dy, dz, false);
      for (var j = range.x; j < range.y; j++) {
        let d = xi - particles[j].pos.xyz;
        let r2 = dot(d, d);
        if (j != P.counts.x + i && r2 < P.kernel.y) {
          rho += mass * poly6(r2);
          neighbours += 1.0;
        }
      }
      rowRange(&range, c, dy, dz, true);
      for (var j = range.x; j < range.y; j++) {
        let b = boundary[j];
        let d = xi - b.xyz;
        let r2 = dot(d, d);
        if (r2 < P.kernel.y) {
          rho += b.w * poly6(r2);
          neighbours += 1.0;
        }
      }
    }
  }
  for (var b = 0u; b < P.counts.z; b++) {
    if (!nearBody(xi, b)) { continue; }
    let range = P.bodies[b].range;
    for (var k = range.x; k < range.x + range.y; k++) {
      let bp = bodyParticle(k);
      let d = xi - bp.xyz;
      let r2 = dot(d, d);
      if (r2 < P.kernel.y) {
        rho += bp.w * poly6(r2);
        neighbours += 1.0;
      }
    }
  }
  fluid[i] = vec4f(rho, equationOfState(rho), neighbours, 0.0);
}

// a_i = g + frame − Σ_j m (p_i + p_j)/(2 ρ_i ρ_j) ∇W − Σ_b ψ_b p_i/ρ_i² ∇W, then
// v* = v + a dt. Pushes from the body are reacted onto it.
@compute @workgroup_size(${WORKGROUP})
fn accel(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let xi = particles[P.counts.x + i].pos.xyz;
  let vi = particles[P.counts.x + i].vel.xyz;
  let rhoi = fluid[i].x;
  let pi = fluid[i].y;
  let c = cellCoords(xi);
  let mass = P.fluid.x;
  var a = P.gravity.xyz + frameAccel(xi, vi);
  // River current: relax the downstream velocity toward the target speed
  // (stands in for the slope plus bed friction of a real river).
  let t = downstream(xi);
  a += P.current.y * (P.current.x - dot(vi, t)) * t;
  var range: vec2u;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      rowRange(&range, c, dy, dz, false);
      for (var j = range.x; j < range.y; j++) {
        let d = xi - particles[j].pos.xyz;
        let r2 = dot(d, d);
        if (j != P.counts.x + i && r2 < P.kernel.y && r2 > 1e-12) {
          let other = fluid[j - P.counts.x];
          a += (mass * (pi + other.y) / (2.0 * rhoi * other.x)) * spikyOverR(sqrt(r2)) * d;
        }
      }
      rowRange(&range, c, dy, dz, true);
      for (var j = range.x; j < range.y; j++) {
        let b = boundary[j];
        let d = xi - b.xyz;
        let r2 = dot(d, d);
        if (r2 < P.kernel.y && r2 > 1e-12) {
          a += (b.w * pi / (rhoi * rhoi)) * spikyOverR(sqrt(r2)) * d;
        }
      }
    }
  }
  for (var b = 0u; b < P.counts.z; b++) {
    if (!nearBody(xi, b)) { continue; }
    let range = P.bodies[b].range;
    for (var k = range.x; k < range.x + range.y; k++) {
      let bp = bodyParticle(k);
      let d = xi - bp.xyz;
      let r2 = dot(d, d);
      if (r2 < P.kernel.y && r2 > 1e-12) {
        let s = (bp.w * pi / (rhoi * rhoi)) * spikyOverR(sqrt(r2));
        a += s * d;
        pushOnBody(b, -mass * s * d, bp.xyz);
      }
    }
  }
  fluid[P.counts.x + i] = vec4f(vi + a * P.stepping.x, 0.0);
}

// XSPH (fluid and body neighbours; walls are free-slip), position update, then
// the body push-out and the wall clamp, written back to A in sorted order.
@compute @workgroup_size(${WORKGROUP})
fn integrate(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= P.counts.x) { return; }
  let xi = particles[P.counts.x + i].pos.xyz;
  let vs = fluid[P.counts.x + i].xyz;
  let c = cellCoords(xi);
  let mass = P.fluid.x;
  let eps = P.stepping.y;
  let dt = P.stepping.x;
  var v = vs;
  var range: vec2u;
  for (var dz = -1; dz <= 1; dz++) {
    for (var dy = -1; dy <= 1; dy++) {
      rowRange(&range, c, dy, dz, false);
      for (var j = range.x; j < range.y; j++) {
        let d = xi - particles[j].pos.xyz;
        let r2 = dot(d, d);
        if (j != P.counts.x + i && r2 < P.kernel.y) {
          // j indexes B (offset n); aux sits at j − n and v* at j.
          v += (eps * mass * poly6(r2) / fluid[j - P.counts.x].x) * (fluid[j].xyz - vs);
        }
      }
    }
  }
  for (var b = 0u; b < P.counts.z; b++) {
    if (!nearBody(xi, b)) { continue; }
    let range = P.bodies[b].range;
    for (var k = range.x; k < range.x + range.y; k++) {
      let bp = bodyParticle(k);
      let d = xi - bp.xyz;
      let r2 = dot(d, d);
      if (r2 < P.kernel.y) {
        let dv = bodyParticleVel(k) - vs;
        let w = eps * bp.w * poly6(r2) / P.fluid.y;
        v += w * dv;
        pushOnBody(b, -mass * w * dv / dt, bp.xyz);
      }
    }
  }

  var x = xi + v * dt;
  let r = P.stepping.z;
  let e = P.stepping.w;

  // Body push-out: mirror across the nearest face, reflect the relative velocity.
  for (var b = 0u; b < P.counts.z; b++) {
    let body = P.bodies[b];
    var local = toBody(x - body.center.xyz, b);
    let depth = body.half.xyz + vec3f(r) - abs(local);
    if (all(depth > vec3f(0.0))) {
      var face = 0;
      if (depth.y < depth[face]) { face = 1; }
      if (depth.z < depth[face]) { face = 2; }
      let sgn = select(-1.0, 1.0, local[face] >= 0.0);
      local[face] += sgn * 2.0 * depth[face];
      let offset = fromBody(local, b);
      x = body.center.xyz + offset;
      var nLocal = vec3f(0.0);
      nLocal[face] = sgn;
      let n = fromBody(nLocal, b);
      let rel = v - (body.vel.xyz + cross(body.omega.xyz, offset));
      let vn = dot(rel, n);
      if (vn < 0.0) { v -= (1.0 + e) * vn * n; }
    }
  }

  // Static solid boxes: push out through the nearest exposed face (faces flush
  // with the domain's walls don't count), reflecting the normal velocity.
  for (var b = 0u; b < P.boxCount.x; b++) {
    let bmin = P.boxes[2u * b].xyz - vec3f(r);
    let bmax = P.boxes[2u * b + 1u].xyz + vec3f(r);
    if (any(x <= bmin) || any(x >= bmax)) { continue; }
    var best = 1e9;
    var axis = 0;
    var side = 0.0;
    for (var k = 0; k < 3; k++) {
      if (bmin[k] + r > P.domainMin[k] + 1e-3 && x[k] - bmin[k] < best) { best = x[k] - bmin[k]; axis = k; side = -1.0; }
      if (bmax[k] - r < P.domainMax[k] - 1e-3 && bmax[k] - x[k] < best) { best = bmax[k] - x[k]; axis = k; side = 1.0; }
    }
    if (side == 0.0) { continue; }
    x[axis] += side * 2.0 * best;
    if (v[axis] * side < 0.0) { v[axis] = -v[axis] * e; }
  }

  // Winding channel banks: keep the perpendicular distance from the centreline
  // within the half width (less the particle radius), reflecting off the bank.
  if (P.channel.w > 0.5) {
    let slope = channelSlope(x.x);
    let stretch = sqrt(1.0 + slope * slope);
    let across = (x.z - channelZ(x.x)) / stretch; // signed perpendicular distance
    let limit = P.channel.z - r;
    if (abs(across) > limit) {
      let n = sign(across) * vec3f(-slope, 0.0, 1.0) / stretch; // outward bank normal
      x -= (abs(across) - limit) * n;
      let vn = dot(v, n);
      if (vn > 0.0) { v -= (1.0 + e) * vn * n; }
    }
  }

  // Walls: clamp inset by the particle radius, reflect with restitution.
  let lo = P.domainMin.xyz + vec3f(r);
  let hi = P.domainMax.xyz - vec3f(r);
  for (var k = 0; k < 3; k++) {
    if (x[k] < lo[k]) {
      x[k] = lo[k];
      if (v[k] < 0.0) { v[k] = -v[k] * e; }
    } else if (x[k] > hi[k]) {
      x[k] = hi[k];
      if (v[k] > 0.0) { v[k] = -v[k] * e; }
    }
  }

  // Recycling: water reaching the end of the channel re-enters at the inlet,
  // lifted by the height difference (a pump, so the river never runs dry).
  if (P.recycle.w > 0.5 && x.x > P.recycle.x) {
    let exitedAt = x.x;
    x.x = x.x - P.recycle.x + P.recycle.y;
    x.y = min(x.y + P.recycle.z, hi.y);
    // Keep the particle's place across the channel.
    if (P.channel.w > 0.5) { x.z += channelZ(x.x) - channelZ(exitedAt); }
  }
  particles[i] = Particle(vec4f(x, 0.0), vec4f(v, 0.0));
}
`
