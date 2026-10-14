// The planet's ground as a height map you can sculpt: heights above sea level
// on a cube wrapped onto the sphere (six square grids, one per face, the
// direction to a grid point being its spot on the cube, normalized). Any
// direction maps to one face and a point on its grid, so lookups interpolate
// between four samples, and a brush changes the samples near where it's held.
// Pure, so the checks can use it headless.

// Face f's grid point (u, v) in [−1, 1]² is the cube point AXES[f] applied to (u, v).
const FACES: ((u: number, v: number) => [number, number, number])[] = [
  (u, v) => [1, v, -u],
  (u, v) => [-1, v, u],
  (u, v) => [u, 1, -v],
  (u, v) => [u, -1, v],
  (u, v) => [u, v, 1],
  (u, v) => [-u, v, -1],
]

// Which face a direction falls on, and where on it (u, v in [−1, 1]).
function faceOf(x: number, y: number, z: number): [number, number, number] {
  const ax = Math.abs(x)
  const ay = Math.abs(y)
  const az = Math.abs(z)
  if (ax >= ay && ax >= az) return x > 0 ? [0, -z / ax, y / ax] : [1, z / ax, y / ax]
  if (ay >= az) return y > 0 ? [2, x / ay, -z / ay] : [3, x / ay, z / ay]
  return z > 0 ? [4, x / az, y / az] : [5, -x / az, y / az]
}

export class Terrain {
  readonly n: number // grid points per face edge
  readonly heights: Float32Array // 6 · n · n, metres above sea level
  readonly dirs: Float32Array // the unit direction of every grid point, stride 3
  min: number // the lowest and highest heights the brush may reach
  max: number

  constructor(n: number, min: number, max: number, init: (dir: number[]) => number) {
    this.n = n
    this.min = min
    this.max = max
    this.heights = new Float32Array(6 * n * n)
    this.dirs = new Float32Array(6 * n * n * 3)
    for (let f = 0; f < 6; f++) {
      for (let j = 0; j < n; j++) {
        for (let i = 0; i < n; i++) {
          const k = (f * n + j) * n + i
          const [x, y, z] = FACES[f]((2 * i) / (n - 1) - 1, (2 * j) / (n - 1) - 1)
          const l = Math.hypot(x, y, z)
          this.dirs.set([x / l, y / l, z / l], k * 3)
          this.heights[k] = Math.min(max, Math.max(min, init([x / l, y / l, z / l])))
        }
      }
    }
  }

  // Height above sea level in direction (x, y, z) (needn't be unit length).
  heightAt(x: number, y: number, z: number): number {
    const n = this.n
    const [f, u, v] = faceOf(x, y, z)
    const fi = Math.min(Math.max(((u + 1) / 2) * (n - 1), 0), n - 1.0001)
    const fj = Math.min(Math.max(((v + 1) / 2) * (n - 1), 0), n - 1.0001)
    const i = Math.floor(fi)
    const j = Math.floor(fj)
    const a = fi - i
    const b = fj - j
    const k = (f * n + j) * n + i
    const h = this.heights
    return (h[k] * (1 - a) + h[k + 1] * a) * (1 - b) + (h[k + n] * (1 - a) + h[k + n + 1] * a) * b
  }

  // Raises (amount > 0) or lowers the ground within `radius` radians of
  // direction `at` (unit), most at the centre, smoothly to nothing at the edge.
  // Returns the grid points it touched.
  brush(at: number[], radius: number, amount: number): number[] {
    const touched: number[] = []
    const cosR = Math.cos(radius)
    const { dirs, heights } = this
    for (let k = 0; k < heights.length; k++) {
      const c = dirs[k * 3] * at[0] + dirs[k * 3 + 1] * at[1] + dirs[k * 3 + 2] * at[2]
      if (c < cosR) continue
      const t = Math.acos(Math.min(1, c)) / radius
      const w = (1 - t * t) ** 2
      heights[k] = Math.min(this.max, Math.max(this.min, heights[k] + amount * w))
      touched.push(k)
    }
    return touched
  }

  // Where a ray first hits the ground (radius `sea` + height), or null. Marches
  // in small steps from where it enters the outermost sphere, then refines.
  raycast(origin: number[], dir: number[], sea: number): number[] | null {
    const outer = sea + this.max + 0.01
    // Entry into the outer sphere.
    const b = origin[0] * dir[0] + origin[1] * dir[1] + origin[2] * dir[2]
    const c = origin[0] ** 2 + origin[1] ** 2 + origin[2] ** 2 - outer * outer
    const disc = b * b - c
    if (disc < 0) return null
    let t = Math.max(0, -b - Math.sqrt(disc))
    const tEnd = -b + Math.sqrt(disc)
    const step = (this.max - this.min) / 40 + 0.02
    const inside = (s: number) => {
      const x = origin[0] + dir[0] * s
      const y = origin[1] + dir[1] * s
      const z = origin[2] + dir[2] * s
      return Math.hypot(x, y, z) <= sea + this.heightAt(x, y, z)
    }
    for (; t <= tEnd; t += step) {
      if (!inside(t)) continue
      let lo = t - step
      let hi = t
      for (let k = 0; k < 20; k++) {
        const mid = (lo + hi) / 2
        if (inside(mid)) hi = mid
        else lo = mid
      }
      return [origin[0] + dir[0] * hi, origin[1] + dir[1] * hi, origin[2] + dir[2] * hi]
    }
    return null
  }

  // Grid triangles (indices into the grid points), two per grid square.
  triangles(): Uint32Array {
    const n = this.n
    const out = new Uint32Array(6 * (n - 1) * (n - 1) * 6)
    let o = 0
    for (let f = 0; f < 6; f++) {
      for (let j = 0; j < n - 1; j++) {
        for (let i = 0; i < n - 1; i++) {
          const a = (f * n + j) * n + i
          out.set([a, a + 1, a + n, a + 1, a + n + 1, a + n], o)
          o += 6
        }
      }
    }
    return out
  }
}
