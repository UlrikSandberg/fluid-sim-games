import type { Domain } from './params.ts'

// Uniform grid over the (bounded) domain, padded by one cell on every side so
// boundary particles just outside the walls get their own cells, with cell
// size = h, built with a counting sort. Particles inside a cell are stored in ascending index order and
// cells are visited in a fixed order, so neighbour iteration order is fully
// deterministic.
//
// Axis 0 has stride 1, so the three cells of a "row" along axis 0 are adjacent
// in the sorted array. A neighbour query therefore scans 3^(dim−1) contiguous
// ranges instead of 3^dim separate cells.
//
// Fluid and boundary particles live in separate grids with the same cell layout:
// the fluid grid is rebuilt every step, the boundary grid only when boundary
// particles move. Fluid particles scan both; boundary particles never scan, so no
// time goes to boundary–boundary candidates (in 3D, walls outnumber the fluid).
//
// Neighbours are stored as a
// flat list of unique pairs (each pair once, not once per side), with the fluid
// particle first: pairI[k] < nFluid, pairJ[k] fluid or boundary, pairR2[k] the
// squared distance. Boundary–boundary pairs are never stored. The solver passes
// visit each pair once and apply its contribution to both ends, which halves
// their work compared to per-particle neighbour lists.
export interface SpatialGrid {
  dim: number
  cellSize: number
  min: number[]
  cellsPerAxis: Int32Array
  strides: Int32Array
  numCells: number
  cellStart: Int32Array // length numCells + 1
  sorted: Int32Array // particle indices grouped by cell
  cellCoord: Int32Array // per-particle cell coordinates, stride dim
  rowOffsets: Int32Array // the 3^(dim−1) offsets over axes 1..dim−1, stride dim−1
  pairI: Int32Array // grows as needed
  pairJ: Int32Array
  pairR2: Float64Array
  pairCount: number
}

export function createGrid(domain: Domain, cellSize: number, nFluid: number, nTotal: number): SpatialGrid {
  const dim = domain.min.length
  const cellsPerAxis = new Int32Array(dim)
  const strides = new Int32Array(dim)
  let numCells = 1
  for (let d = 0; d < dim; d++) {
    cellsPerAxis[d] = Math.max(1, Math.ceil((domain.max[d] - domain.min[d] + 2 * cellSize) / cellSize))
    strides[d] = numCells
    numCells *= cellsPerAxis[d]
  }

  const numRows = 3 ** (dim - 1)
  const rowOffsets = new Int32Array(numRows * (dim - 1))
  for (let o = 0; o < numRows; o++) {
    let rem = o
    for (let d = 0; d < dim - 1; d++) {
      rowOffsets[o * (dim - 1) + d] = (rem % 3) - 1
      rem = Math.floor(rem / 3)
    }
  }

  const capacity = nFluid * 16
  return {
    dim: dim,
    cellSize,
    min: domain.min.map((v) => v - cellSize),
    cellsPerAxis,
    strides,
    numCells,
    cellStart: new Int32Array(numCells + 1),
    sorted: new Int32Array(nTotal),
    cellCoord: new Int32Array(nTotal * dim),
    rowOffsets,
    pairI: new Int32Array(capacity),
    pairJ: new Int32Array(capacity),
    pairR2: new Float64Array(capacity),
    pairCount: 0,
  }
}

// Counting sort of particles into cells.
export function buildGrid(g: SpatialGrid, pos: Float32Array, n: number) {
  const dim = g.dim
  const { cellStart, sorted, cellCoord, cellsPerAxis, strides, numCells } = g
  const inv = 1 / g.cellSize
  cellStart.fill(0)

  // Count particles per cell. cellStart[c + 1] holds the count for cell c.
  for (let i = 0; i < n; i++) {
    let cell = 0
    for (let d = 0; d < dim; d++) {
      let c = Math.floor((pos[i * dim + d] - g.min[d]) * inv)
      if (c < 0) c = 0
      else if (c >= cellsPerAxis[d]) c = cellsPerAxis[d] - 1
      cellCoord[i * dim + d] = c
      cell += c * strides[d]
    }
    cellStart[cell + 1]++
  }

  // Prefix sum → start offset of each cell.
  for (let c = 0; c < numCells; c++) cellStart[c + 1] += cellStart[c]

  // Scatter in ascending particle order (stable). Reuse cellStart as a cursor,
  // then shift it back afterwards.
  for (let i = 0; i < n; i++) {
    let cell = 0
    for (let d = 0; d < dim; d++) cell += cellCoord[i * dim + d] * strides[d]
    sorted[cellStart[cell]++] = i
  }
  for (let c = numCells; c > 0; c--) cellStart[c] = cellStart[c - 1]
  cellStart[0] = 0
}


// For each of the m particles in `g`, Σ_j f(r²_ij) over every particle j of the
// same grid within its 3^dim cell stencil, itself included. Used to compute
// boundary weights ψ; it runs when boundaries are rebuilt, not every step.
export function stencilSums(g: SpatialGrid, pos: Float32Array, m: number, f: (r2: number) => number, out: Float64Array) {
  const dim = g.dim
  const { cellStart, sorted, cellCoord, cellsPerAxis, strides, rowOffsets } = g
  const nx = cellsPerAxis[0]
  const numRows = rowOffsets.length / (dim - 1) || 1
  for (let a = 0; a < m; a++) {
    const cx = cellCoord[a * dim]
    const x0 = cx > 0 ? cx - 1 : 0
    const x1 = cx < nx - 1 ? cx + 1 : nx - 1
    let sum = 0
    for (let o = 0; o < numRows; o++) {
      let base = 0
      let inside = true
      for (let d = 1; d < dim; d++) {
        const c = cellCoord[a * dim + d] + rowOffsets[o * (dim - 1) + d - 1]
        if (c < 0 || c >= cellsPerAxis[d]) {
          inside = false
          break
        }
        base += c * strides[d]
      }
      if (!inside) continue
      for (let q = cellStart[base + x0], to = cellStart[base + x1 + 1]; q < to; q++) {
        sum += f(dist2(pos, a, pos, sorted[q], dim))
      }
    }
    out[a] = sum
  }
}

// Squared distance between particle i of `a` and particle j of `b` (positions
// with stride dim). Unrolled per dimension: V8 does not unroll a `for (d < dim)`
// loop, and in the neighbour search (the hottest loop in the solver) that costs ~30%.
function dist2(a: Float32Array, i: number, b: Float32Array, j: number, dim: number): number {
  if (dim === 2) {
    const dx = a[i * 2] - b[j * 2]
    const dy = a[i * 2 + 1] - b[j * 2 + 1]
    return dx * dx + dy * dy
  }
  const dx = a[i * 3] - b[j * 3]
  const dy = a[i * 3 + 1] - b[j * 3 + 1]
  const dz = a[i * 3 + 2] - b[j * 3 + 2]
  return dx * dx + dy * dy + dz * dz
}

// Builds the unique pair list: every fluid–fluid and fluid–boundary pair with
// |x_i − x_j| < h, once. `g` holds the fluid particles (pos, indices 0..nFluid),
// `bg` the boundary particles (bpos, local indices 0..nb, stored as nFluid + local).
// If the buffers are too small, grow them and rebuild from scratch; that keeps
// the buffers fixed inside the hot loop, which V8 optimizes much better.
export function buildPairs(
  g: SpatialGrid, pos: Float32Array, nFluid: number, bg: SpatialGrid, bpos: Float32Array, h: number,
) {
  while (!tryBuildPairs(g, pos, nFluid, bg, bpos, h)) {
    const capacity = g.pairI.length * 2
    g.pairI = new Int32Array(capacity)
    g.pairJ = new Int32Array(capacity)
    g.pairR2 = new Float64Array(capacity)
  }
}

// Walks the fluid particles in sorted (cell) order.
//  - Fluid candidates: only those *later* in the sorted array, so every pair is
//    found exactly once. Rows are laid out in order, so the row below a particle
//    is skipped outright and its own row starts just after it: about half the
//    candidate checks of a full 3^dim stencil.
//  - Boundary candidates: the full stencil in the boundary grid (each
//    fluid–boundary pair is only ever seen from the fluid side).
// Every candidate is written unconditionally and `count` only advances for real
// pairs, avoiding a hard-to-predict branch. Returns false if the buffers overflow.
function tryBuildPairs(
  g: SpatialGrid, pos: Float32Array, nFluid: number, bg: SpatialGrid, bpos: Float32Array, h: number,
): boolean {
  const dim = g.dim
  const { cellStart, sorted, cellCoord, cellsPerAxis, strides, rowOffsets, pairI, pairJ, pairR2 } = g
  const bStart = bg.cellStart
  const bSorted = bg.sorted
  const capacity = pairI.length
  const h2 = h * h
  const nx = cellsPerAxis[0]
  const numRows = rowOffsets.length / (dim - 1) || 1
  let count = 0

  for (let k = 0; k < nFluid; k++) {
    const a = sorted[k]
    const cx = cellCoord[a * dim]
    const x0 = cx > 0 ? cx - 1 : 0
    const x1 = cx < nx - 1 ? cx + 1 : nx - 1

    for (let o = 0; o < numRows; o++) {
      let base = 0
      let inside = true
      for (let d = 1; d < dim; d++) {
        const c = cellCoord[a * dim + d] + rowOffsets[o * (dim - 1) + d - 1]
        if (c < 0 || c >= cellsPerAxis[d]) {
          inside = false
          break
        }
        base += c * strides[d]
      }
      if (!inside) continue

      const to = cellStart[base + x1 + 1]
      if (to > k + 1) {
        const from = Math.max(cellStart[base + x0], k + 1)
        if (count + (to - from) > capacity) return false
        for (let q = from; q < to; q++) {
          const b = sorted[q]
          const r2 = dist2(pos, a, pos, b, dim)
          pairI[count] = a
          pairJ[count] = b
          pairR2[count] = r2
          count += r2 < h2 ? 1 : 0
        }
      }

      const bFrom = bStart[base + x0]
      const bTo = bStart[base + x1 + 1]
      if (count + (bTo - bFrom) > capacity) return false
      for (let q = bFrom; q < bTo; q++) {
        const b = bSorted[q]
        const r2 = dist2(pos, a, bpos, b, dim)
        pairI[count] = a
        pairJ[count] = nFluid + b
        pairR2[count] = r2
        count += r2 < h2 ? 1 : 0
      }
    }
  }
  g.pairCount = count
  return true
}
