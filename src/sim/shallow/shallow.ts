// Shallow-water equations on a staggered grid: the water is a height field, not
// particles. Each cell holds a depth h over a bed elevation b; velocities live on
// the faces between cells (u across vertical faces, v across horizontal ones).
// A step:
//   1. (optional) carries momentum along with the flow (upwind, conservative),
//   2. accelerates each face down the slope of the water surface η = h + b,
//   3. moves water between cells by upwind fluxes, scaled so no cell goes below 0,
//   4. relaxes "sponge" cells toward rest, so waves leave instead of reflecting,
//   5. makes foam where the surface is steep or the flow converges, and carries it.
// Using the surface's slope (not the depth's) keeps a lake over an uneven bed
// still; a face is dry when the water on either side doesn't reach over the
// higher bed, which handles beaches and dam breaks. Plain TypeScript over typed
// arrays: no DOM, runs in Node (scripts/shallow.ts checks it).

export interface ShallowGrid {
  nx: number
  ny: number
  dx: number // metres per cell
  g: number
  bed: Float32Array // nx·ny, cell (i, j) at j·nx + i
  h: Float32Array // depth
  u: Float32Array // (nx+1)·ny: face i sits between cells i−1 and i
  v: Float32Array // nx·(ny+1): face j sits between cells j−1 and j
  solid: Uint8Array // walls: no water, no flow
  sponge: Float32Array // relaxation rate toward rest (1/s); 0 for most cells
  rest: Float32Array // the surface elevation the sponge relaxes toward
  foam: Float32Array // 0…1
  time: number
  // Scratch.
  fx: Float32Array
  fy: Float32Array
  scale: Float32Array
  tmpU: Float32Array
  tmpV: Float32Array
  tmpFoam: Float32Array
}

export interface StepOptions {
  advect?: boolean // carry momentum with the flow (matters for floods, not ripples)
  friction?: number // linear drag on the velocities (1/s)
  paddle?: number | ArrayLike<number> // a wave maker on the left wall: its velocity (m/s), one value or one per row
  foamRate?: number // how fast steep or converging water makes foam
  foamLife?: number // seconds for foam to fade to 1/e
}

const DRY = 1e-4 // metres: below this a cell or face counts as dry

export function createGrid(nx: number, ny: number, dx: number, g = 9.81): ShallowGrid {
  const cells = nx * ny
  return {
    nx,
    ny,
    dx,
    g,
    bed: new Float32Array(cells),
    h: new Float32Array(cells),
    u: new Float32Array((nx + 1) * ny),
    v: new Float32Array(nx * (ny + 1)),
    solid: new Uint8Array(cells),
    sponge: new Float32Array(cells),
    rest: new Float32Array(cells),
    foam: new Float32Array(cells),
    time: 0,
    fx: new Float32Array((nx + 1) * ny),
    fy: new Float32Array(nx * (ny + 1)),
    scale: new Float32Array(cells),
    tmpU: new Float32Array((nx + 1) * ny),
    tmpV: new Float32Array(nx * (ny + 1)),
    tmpFoam: new Float32Array(cells),
  }
}

// Fills every open cell up to the surface elevation `level` (its rest level too).
export function fillTo(s: ShallowGrid, level: number) {
  for (let k = 0; k < s.nx * s.ny; k++) {
    s.h[k] = s.solid[k] ? 0 : Math.max(0, level - s.bed[k])
    s.rest[k] = level
  }
  s.u.fill(0)
  s.v.fill(0)
  s.foam.fill(0)
  s.time = 0
}

// The largest stable step: waves (√(g·h)) plus the flow must not cross more than
// a fraction of a cell.
export function maxDt(s: ShallowGrid, cfl = 0.4): number {
  let hMax = 0
  for (let k = 0; k < s.h.length; k++) if (s.h[k] > hMax) hMax = s.h[k]
  let vMax = 0
  for (let k = 0; k < s.u.length; k++) vMax = Math.max(vMax, Math.abs(s.u[k]))
  for (let k = 0; k < s.v.length; k++) vMax = Math.max(vMax, Math.abs(s.v[k]))
  return (cfl * s.dx) / Math.max(1e-6, Math.sqrt(s.g * hMax) + vMax)
}

export const surfaceAt = (s: ShallowGrid, k: number) => s.h[k] + s.bed[k]

export function totalVolume(s: ShallowGrid): number {
  let v = 0
  for (let k = 0; k < s.h.length; k++) v += s.h[k]
  return v * s.dx * s.dx
}

// Bilinear sample of a face-centred or cell-centred field. (ox, oy) is where
// sample (0, 0) sits, in cells; w×hgt is the field's size.
function sample(f: Float32Array, w: number, hgt: number, x: number, y: number, ox: number, oy: number): number {
  const fx = Math.min(Math.max(x - ox, 0), w - 1.001)
  const fy = Math.min(Math.max(y - oy, 0), hgt - 1.001)
  const i = Math.floor(fx)
  const j = Math.floor(fy)
  const a = fx - i
  const b = fy - j
  const k = j * w + i
  return (f[k] * (1 - a) + f[k + 1] * a) * (1 - b) + (f[k + w] * (1 - a) + f[k + w + 1] * a) * b
}

// Velocity at a point given in cell units (cell (i, j)'s centre is at i+½, j+½).
function velocityAt(s: ShallowGrid, x: number, y: number): [number, number] {
  return [sample(s.u, s.nx + 1, s.ny, x, y, 0, 0.5), sample(s.v, s.nx, s.ny + 1, x, y, 0.5, 0)]
}

// Momentum carried with the flow, in the momentum-conserving upwind form of
// Stelling & Duinmeijer (2003): it moves momentum into newly wetted cells, which
// is what lets a flood front run at the right speed. Uses the current fluxes
// (fx, fy) and writes the advected velocities back into u and v.
function advectMomentum(s: ShallowGrid, dt: number) {
  const { nx, ny, dx, h, u, v, fx, fy, tmpU, tmpV } = s
  const U = (i: number, j: number) => u[Math.min(Math.max(j, 0), ny - 1) * (nx + 1) + Math.min(Math.max(i, 0), nx)]
  const V = (i: number, j: number) => v[Math.min(Math.max(j, 0), ny) * nx + Math.min(Math.max(i, 0), nx - 1)]
  const FX = (i: number, j: number) => fx[j * (nx + 1) + Math.min(Math.max(i, 0), nx)]
  const FY = (i: number, j: number) => fy[Math.min(Math.max(j, 0), ny) * nx + i]
  tmpU.set(u)
  for (let j = 0; j < ny; j++) {
    for (let i = 1; i < nx; i++) {
      const hf = (h[j * nx + i - 1] + h[j * nx + i]) / 2
      if (hf <= DRY) continue
      const ui = U(i, j)
      // Along x: the cell-centred fluxes either side of the face.
      const qL = (FX(i - 1, j) + FX(i, j)) / 2
      const qR = (FX(i, j) + FX(i + 1, j)) / 2
      const uL = qL > 0 ? U(i - 1, j) : ui
      const uR = qR > 0 ? ui : U(i + 1, j)
      let adv = qR * uR - qL * uL - ui * (qR - qL)
      // Along y: the fluxes at the face's two ends.
      const qB = (FY(i - 1, j) + FY(i, j)) / 2
      const qT = (FY(i - 1, j + 1) + FY(i, j + 1)) / 2
      const uB = qB > 0 ? U(i, j - 1) : ui
      const uT = qT > 0 ? ui : U(i, j + 1)
      adv += qT * uT - qB * uB - ui * (qT - qB)
      tmpU[j * (nx + 1) + i] = ui - (dt * adv) / (dx * hf)
    }
  }
  tmpV.set(v)
  for (let j = 1; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const hf = (h[(j - 1) * nx + i] + h[j * nx + i]) / 2
      if (hf <= DRY) continue
      const vi = V(i, j)
      const qD = (FY(i, j - 1) + FY(i, j)) / 2
      const qU = (FY(i, j) + FY(i, j + 1)) / 2
      const vD = qD > 0 ? V(i, j - 1) : vi
      const vU = qU > 0 ? vi : V(i, j + 1)
      let adv = qU * vU - qD * vD - vi * (qU - qD)
      const qLf = (FX(i, j - 1) + FX(i, j)) / 2
      const qRf = (FX(i + 1, j - 1) + FX(i + 1, j)) / 2
      const vL = qLf > 0 ? V(i - 1, j) : vi
      const vR = qRf > 0 ? vi : V(i + 1, j)
      adv += qRf * vR - qLf * vL - vi * (qRf - qLf)
      tmpV[j * nx + i] = vi - (dt * adv) / (dx * hf)
    }
  }
  u.set(tmpU)
  v.set(tmpV)
}

// Upwind fluxes q = u·h (from the cell the water comes from) on every face.
function computeFluxes(s: ShallowGrid) {
  const { nx, ny, h, u, v, fx, fy } = s
  for (let j = 0; j < ny; j++) {
    const row = j * (nx + 1)
    for (let i = 0; i <= nx; i++) {
      const vel = u[row + i]
      // At the walls the only flow is the paddle's, which pushes the edge cell's water.
      const up = vel > 0 ? (i > 0 ? i - 1 : 0) : i < nx ? i : nx - 1
      fx[row + i] = vel * h[j * nx + up]
    }
  }
  for (let j = 0; j <= ny; j++) {
    for (let i = 0; i < nx; i++) {
      const vel = v[j * nx + i]
      const up = vel > 0 ? Math.max(j - 1, 0) : Math.min(j, ny - 1)
      fy[j * nx + i] = vel * h[up * nx + i]
    }
  }
}

export function step(s: ShallowGrid, dt: number, opts: StepOptions = {}) {
  const { nx, ny, dx, g, h, bed, u, v, solid } = s
  if (opts.advect) {
    computeFluxes(s)
    advectMomentum(s, dt)
  }
  const drag = 1 / (1 + (opts.friction ?? 0) * dt)
  const paddle = opts.paddle ?? 0
  const paddleAt = typeof paddle === 'number' ? () => paddle : (j: number) => paddle[j]
  const vCap = (0.5 * dx) / dt // keeps a runaway face from breaking the CFL limit

  // 2. Faces accelerate down the slope of the water surface.
  const k = (g * dt) / dx
  for (let j = 0; j < ny; j++) {
    const row = j * (nx + 1)
    for (let i = 1; i < nx; i++) {
      const L = j * nx + i - 1
      const R = L + 1
      const f = row + i
      if (solid[L] || solid[R]) {
        u[f] = 0
        continue
      }
      const etaL = h[L] + bed[L]
      const etaR = h[R] + bed[R]
      if (Math.max(etaL, etaR) - Math.max(bed[L], bed[R]) <= DRY) {
        u[f] = 0
        continue
      }
      u[f] = Math.max(-vCap, Math.min(vCap, (u[f] - k * (etaR - etaL)) * drag))
    }
    // The walls, or the paddle on the left.
    u[row] = solid[j * nx] ? 0 : paddleAt(j)
    u[row + nx] = 0
  }
  for (let j = 1; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const D = (j - 1) * nx + i
      const U = D + nx
      const f = j * nx + i
      if (solid[D] || solid[U]) {
        v[f] = 0
        continue
      }
      const etaD = h[D] + bed[D]
      const etaU = h[U] + bed[U]
      if (Math.max(etaD, etaU) - Math.max(bed[D], bed[U]) <= DRY) {
        v[f] = 0
        continue
      }
      v[f] = Math.max(-vCap, Math.min(vCap, (v[f] - k * (etaU - etaD)) * drag))
    }
  }
  for (let i = 0; i < nx; i++) {
    v[i] = 0
    v[ny * nx + i] = 0
  }

  // 3. Upwind fluxes; a cell can't send out more water than it has.
  const { fx, fy, scale } = s
  computeFluxes(s)
  const c = dt / dx
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const cell = j * nx + i
      const fl = j * (nx + 1) + i
      const out = (Math.max(0, fx[fl + 1]) + Math.max(0, -fx[fl]) + Math.max(0, fy[cell + nx]) + Math.max(0, -fy[cell])) * c
      scale[cell] = out > h[cell] ? h[cell] / out : 1
    }
  }
  // Each face's flux is outgoing for its upwind cell: scale it by that cell's factor.
  for (let j = 0; j < ny; j++) {
    for (let i = 1; i < nx; i++) {
      const f = j * (nx + 1) + i
      fx[f] *= scale[j * nx + (fx[f] > 0 ? i - 1 : i)]
    }
    if (fx[j * (nx + 1)] < 0) fx[j * (nx + 1)] *= scale[j * nx] // the paddle pulling water out
  }
  for (let j = 1; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const f = j * nx + i
      fy[f] *= scale[(fy[f] > 0 ? j - 1 : j) * nx + i]
    }
  }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const cell = j * nx + i
      const fl = j * (nx + 1) + i
      h[cell] = Math.max(0, h[cell] - c * (fx[fl + 1] - fx[fl] + fy[cell + nx] - fy[cell]))
    }
  }

  // 4. Sponges pull the water back to rest, so waves die there instead of bouncing.
  const { sponge, rest } = s
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const cell = j * nx + i
      const r = sponge[cell]
      if (r <= 0 || solid[cell]) continue
      const a = 1 - Math.exp(-r * dt)
      h[cell] += (Math.max(0, rest[cell] - bed[cell]) - h[cell]) * a
      u[j * (nx + 1) + i] *= 1 - a
      u[j * (nx + 1) + i + 1] *= 1 - a
      v[cell] *= 1 - a
      v[cell + nx] *= 1 - a
    }
  }

  // 5. Foam.
  if (opts.foamRate) updateFoam(s, dt, opts.foamRate, opts.foamLife ?? 3)
  s.time += dt
}

function updateFoam(s: ShallowGrid, dt: number, rate: number, life: number) {
  const { nx, ny, dx, h, bed, u, v, foam, tmpFoam } = s
  const decay = Math.exp(-dt / life)
  const c = dt / dx
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const cell = j * nx + i
      if (h[cell] <= DRY) {
        tmpFoam[cell] = 0
        continue
      }
      // Carried by the flow (from where this water was a moment ago)…
      const [vx, vy] = velocityAt(s, i + 0.5, j + 0.5)
      let f = sample(foam, nx, ny, i + 0.5 - vx * c, j + 0.5 - vy * c, 0.5, 0.5) * decay
      // …and made where the surface is steep (waves about to break) or the flow
      // piles into itself.
      const e = (kk: number) => h[kk] + bed[kk]
      const sx = i > 0 && i < nx - 1 ? (e(cell + 1) - e(cell - 1)) / (2 * dx) : 0
      const sy = j > 0 && j < ny - 1 ? (e(cell + nx) - e(cell - nx)) / (2 * dx) : 0
      const slope = Math.hypot(sx, sy)
      const div = (u[j * (nx + 1) + i + 1] - u[j * (nx + 1) + i] + v[cell + nx] - v[cell]) / dx
      const speed = Math.hypot(vx, vy)
      f += dt * rate * (Math.max(0, slope - 0.12) * 6 + Math.max(0, -div - 0.4) * speed * 0.5)
      tmpFoam[cell] = Math.min(1, f)
    }
  }
  foam.set(tmpFoam)
}

// --- Things that disturb the water ---

// A drop: a raised centre with a trough around it (a "Mexican hat", which adds
// no water). `x`, `y` in metres; `radius` is the bump's size.
export function drop(s: ShallowGrid, x: number, y: number, radius: number, amplitude: number) {
  const { nx, ny, dx, h, solid } = s
  const reach = Math.ceil((radius * 3) / dx)
  const ci = Math.floor(x / dx)
  const cj = Math.floor(y / dx)
  for (let j = Math.max(0, cj - reach); j <= Math.min(ny - 1, cj + reach); j++) {
    for (let i = Math.max(0, ci - reach); i <= Math.min(nx - 1, ci + reach); i++) {
      const cell = j * nx + i
      if (solid[cell] || h[cell] <= DRY) continue
      const r2 = (((i + 0.5) * dx - x) ** 2 + ((j + 0.5) * dx - y) ** 2) / (radius * radius)
      h[cell] = Math.max(0, h[cell] + amplitude * (1 - r2) * Math.exp(-r2))
    }
  }
}

// An oscillating source (a finger bobbing in the water) for one step: the
// change in a drop of amplitude A·sin(ωt), so over a cycle it adds nothing.
export function oscillate(s: ShallowGrid, x: number, y: number, radius: number, amplitude: number, freq: number, dt: number) {
  const w = 2 * Math.PI * freq
  drop(s, x, y, radius, amplitude * w * Math.cos(w * s.time) * dt)
}
