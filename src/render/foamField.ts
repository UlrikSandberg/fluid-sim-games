import type { Params } from '../sim/params'
import type { SimState } from '../sim/sph'

// Foam and spray for the realistic water look (waterSurface3d.ts).
//
// Foam floats on the surface, so it lives in a 2D grid over the domain (x, z)
// rather than on the particles (the GPU solver reorders its particles every
// step, so a particle can't remember being foamy). Each frame:
//   - churning water makes foam in its column. That's a particle moving
//     differently from the water right around it (its 3D cell's mean): where a
//     waterfall plunges into the pool or the flow churns round a rock; or water
//     being slammed to a stop (the cell's mean flow decelerating hard): a wave
//     landing, a splash against a wall. Fast but smooth water (a river, the sheet
//     going over a lip) does neither, so it stays clear. Spray landing adds a
//     little;
//   - the foam drifts with the surface flow and fades over a couple of seconds,
//     so it lingers as patches and trails rather than flickering;
//   - particles at a free surface pick up the foam in their column. A free
//     surface is where a particle has fewer neighbours than in the bulk (walls,
//     beds and lids count as neighbours), which holds whichever way gravity
//     points: in a flip zone the free surface is the water's underside, not the
//     water pressed against the lid.
// The particles' colours then carry (foam, spray, air) for the surface shader:
// foam whitens the surface; air is fresh churn mixed into the water as bubbles,
// which rise out within a fraction of a second, briefly clouding it (seen
// through a tank's wall where it's hit); spray particles are left out of the
// surface and drawn as droplets.

const FOAM_LIFE = 1.8 // seconds to fade to 1/e
const EMPTY_LIFE = 0.25 // faster where no water is left under it
const FOAM_RATE = 1.5 // how fast turbulent water makes foam (per second)
const AIR_LIFE = 0.35 // churned-in bubbles rise out fast
const AIR_RATE = 3
const CHURN_START = 1.0 // velocity difference from the local flow where foam starts…
const CHURN_FULL = 3.5 // …and where it's made at full rate
const IMPACT_START = 30 // deceleration of the local flow (units/s²) where impact foam starts…
const IMPACT_FULL = 100 // …and full rate (frame-to-frame noise in steady flow stays below ~20)
const SPRAY_NEIGHBOURS = 0.35 // a particle with fewer than this share of the bulk's neighbours is spray
const SURFACE_FULL = 0.62 // at or below this share it's right at a free surface…
const SURFACE_NONE = 0.85 // …and above this it's inside the water
const MAX_CELLS_3D = 2_000_000

// The surface's flow, cell by cell, for the shader to move the foam's texture
// along (tank coordinates, units/s; stride 2: x, z).
export interface FoamFlow {
  data: Float32Array
  nx: number
  nz: number
  x0: number
  z0: number
  cell: number
}

export class FoamField {
  readonly flow: FoamFlow
  private readonly nx: number
  private readonly nz: number
  private readonly cell: number
  private readonly x0: number
  private readonly y0: number
  private readonly z0: number
  private foam: Float32Array
  private next: Float32Array
  private readonly air: Float32Array // bubbles mixed into the water, short-lived
  private readonly make: Float32Array
  private readonly velX: Float32Array
  private readonly velZ: Float32Array
  private readonly count: Float32Array
  private readonly wet: Uint8Array
  // The 3D grid of mean velocities, for turbulence.
  private readonly cell3: number
  private readonly ny3: number
  private readonly nx3: number
  private readonly nz3: number
  private readonly sum3: Float32Array // stride 4: vx, vy, vz, count
  private readonly prev3: Float32Array // stride 4: last frame's mean vx, vy, vz, and whether it had water
  private readonly impact3: Float32Array // how hard each cell's water is being stopped (deceleration)
  private readonly bulkNeighbours: number

  constructor(p: Params) {
    this.cell = 2 * p.spacing
    this.x0 = p.domain.min[0]
    this.y0 = p.domain.min[1]
    this.z0 = p.domain.min[2]
    const size = [0, 1, 2].map((d) => p.domain.max[d] - p.domain.min[d])
    this.nx = Math.max(2, Math.ceil(size[0] / this.cell) + 1)
    this.nz = Math.max(2, Math.ceil(size[2] / this.cell) + 1)
    const cells = this.nx * this.nz
    this.foam = new Float32Array(cells)
    this.next = new Float32Array(cells)
    this.air = new Float32Array(cells)
    this.make = new Float32Array(cells)
    this.velX = new Float32Array(cells)
    this.velZ = new Float32Array(cells)
    this.count = new Float32Array(cells)
    this.wet = new Uint8Array(cells)
    this.flow = { data: new Float32Array(cells * 2), nx: this.nx, nz: this.nz, x0: this.x0, z0: this.z0, cell: this.cell }
    // Turbulence cells about two particle spacings across (coarser on a huge domain).
    let c3 = this.cell
    while ((size[0] / c3 + 1) * (size[1] / c3 + 1) * (size[2] / c3 + 1) > MAX_CELLS_3D) c3 *= 1.25
    this.cell3 = c3
    this.nx3 = Math.ceil(size[0] / c3) + 1
    this.ny3 = Math.ceil(size[1] / c3) + 1
    this.nz3 = Math.ceil(size[2] / c3) + 1
    this.sum3 = new Float32Array(this.nx3 * this.ny3 * this.nz3 * 4)
    this.prev3 = new Float32Array(this.nx3 * this.ny3 * this.nz3 * 4)
    this.impact3 = new Float32Array(this.nx3 * this.ny3 * this.nz3)
    // Neighbours in the bulk: (h / spacing)³ · 4π/3, ~33 for the 3D preset.
    this.bulkNeighbours = ((4 * Math.PI) / 3) * (p.h / p.spacing) ** 3
  }

  private cellOf(x: number, z: number): number {
    const i = Math.min(this.nx - 1, Math.max(0, Math.round((x - this.x0) / this.cell)))
    const k = Math.min(this.nz - 1, Math.max(0, Math.round((z - this.z0) / this.cell)))
    return k * this.nx + i
  }

  private cell3Of(x: number, y: number, z: number): number {
    const i = Math.min(this.nx3 - 1, Math.max(0, Math.floor((x - this.x0) / this.cell3)))
    const j = Math.min(this.ny3 - 1, Math.max(0, Math.floor((y - this.y0) / this.cell3)))
    const k = Math.min(this.nz3 - 1, Math.max(0, Math.floor((z - this.z0) / this.cell3)))
    return ((k * this.ny3 + j) * this.nx3 + i) * 4
  }

  private sample(f: Float32Array, x: number, z: number): number {
    const fx = Math.min(Math.max((x - this.x0) / this.cell, 0), this.nx - 1.001)
    const fz = Math.min(Math.max((z - this.z0) / this.cell, 0), this.nz - 1.001)
    const i = Math.floor(fx)
    const k = Math.floor(fz)
    const a = fx - i
    const b = fz - k
    const c = k * this.nx + i
    return (f[c] * (1 - a) + f[c + 1] * a) * (1 - b) + (f[c + this.nx] * (1 - a) + f[c + this.nx + 1] * a) * b
  }

  // How much a particle with `nb` neighbours is at a free surface (0…1).
  private surface(nb: number): number {
    const share = nb / this.bulkNeighbours
    return Math.min(1, Math.max(0, (SURFACE_NONE - share) / (SURFACE_NONE - SURFACE_FULL)))
  }

  // Advances the foam by dt and writes each particle's (foam, spray, air) into
  // `out` (stride 3).
  update(s: SimState, dt: number, out: Float32Array) {
    const { pos, vel, neighborCount, n } = s
    const { make, velX, velZ, count, wet, sum3 } = this
    make.fill(0)
    velX.fill(0)
    velZ.fill(0)
    count.fill(0)
    wet.fill(0)
    sum3.fill(0)
    const sprayBelow = SPRAY_NEIGHBOURS * this.bulkNeighbours

    // The local mean flow, cell by cell in 3D.
    for (let i = 0; i < n; i++) {
      if (neighborCount[i] < sprayBelow) continue
      const c = this.cell3Of(pos[i * 3], pos[i * 3 + 1], pos[i * 3 + 2])
      sum3[c] += vel[i * 3]
      sum3[c + 1] += vel[i * 3 + 1]
      sum3[c + 2] += vel[i * 3 + 2]
      sum3[c + 3]++
    }

    // Impact: how fast each cell's mean flow is slowing. Water slammed to a
    // stop (a breaking wave landing, a splash against a wall) decelerates hard;
    // steady flow, however fast, doesn't change from frame to frame.
    const { prev3, impact3 } = this
    for (let c = 0, q = 0; q < impact3.length; q++, c += 4) {
      const m = sum3[c + 3]
      impact3[q] = 0
      if (m >= 2 && prev3[c + 3] > 0 && dt > 0) {
        const mx = sum3[c] / m
        const my = sum3[c + 1] / m
        const mz = sum3[c + 2] / m
        const before = Math.hypot(prev3[c], prev3[c + 1], prev3[c + 2])
        const now = Math.hypot(mx, my, mz)
        impact3[q] = Math.max(0, before - now) / dt
      }
      if (m >= 2) {
        prev3[c] = sum3[c] / m
        prev3[c + 1] = sum3[c + 1] / m
        prev3[c + 2] = sum3[c + 2] / m
        prev3[c + 3] = 1
      } else prev3[c + 3] = 0
    }

    // Where there's water, how its surface moves, and where it churns.
    for (let i = 0; i < n; i++) {
      const x = pos[i * 3]
      const y = pos[i * 3 + 1]
      const z = pos[i * 3 + 2]
      const c = this.cellOf(x, z)
      const nb = neighborCount[i]
      if (nb < sprayBelow) {
        make[c] = Math.max(make[c], 0.25) // spray landing stirs up a little foam
        continue
      }
      wet[c] = 1
      const vx = vel[i * 3]
      const vy = vel[i * 3 + 1]
      const vz = vel[i * 3 + 2]
      const at = this.surface(nb)
      if (at > 0) {
        velX[c] += vx * at
        velZ[c] += vz * at
        count[c] += at
      }
      const c3 = this.cell3Of(x, y, z)
      const m = sum3[c3 + 3]
      if (m < 2) continue
      const churn = Math.hypot(vx - sum3[c3] / m, vy - sum3[c3 + 1] / m, vz - sum3[c3 + 2] / m)
      const turbulent = (churn - CHURN_START) / (CHURN_FULL - CHURN_START)
      const slammed = (impact3[c3 / 4] - IMPACT_START) / (IMPACT_FULL - IMPACT_START)
      const f = Math.min(1, Math.max(0, turbulent, slammed))
      if (f > make[c]) make[c] = f
    }

    // Drift with the surface flow (semi-Lagrangian), fade, and add the new foam.
    const keep = Math.exp(-dt / FOAM_LIFE)
    const keepEmpty = Math.exp(-dt / EMPTY_LIFE)
    const keepAir = Math.exp(-dt / AIR_LIFE)
    for (let k = 0; k < this.nz; k++) {
      for (let i = 0; i < this.nx; i++) {
        const c = k * this.nx + i
        const m = count[c]
        const ux = m > 0 ? velX[c] / m : 0
        const uz = m > 0 ? velZ[c] / m : 0
        this.flow.data[c * 2] = ux
        this.flow.data[c * 2 + 1] = uz
        const x = this.x0 + i * this.cell - ux * dt
        const z = this.z0 + k * this.cell - uz * dt
        const f = this.sample(this.foam, x, z) * (wet[c] ? keep : keepEmpty)
        this.next[c] = Math.min(1, f + make[c] * FOAM_RATE * dt)
        this.air[c] = Math.min(1, this.air[c] * keepAir + make[c] * AIR_RATE * dt)
      }
    }
    ;[this.foam, this.next] = [this.next, this.foam]

    // Particles at a free surface show the foam in their column.
    for (let i = 0; i < n; i++) {
      const nb = neighborCount[i]
      const spray = nb < sprayBelow
      out[i * 3] = spray ? 1 : this.sample(this.foam, pos[i * 3], pos[i * 3 + 2]) * this.surface(nb)
      out[i * 3 + 1] = spray ? 1 : 0
      out[i * 3 + 2] = spray ? 0 : this.sample(this.air, pos[i * 3], pos[i * 3 + 2])
    }
  }
}
