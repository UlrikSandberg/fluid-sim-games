import { DIM, type Domain } from './params.ts'

// Uniform grid over the (bounded) domain, padded by one cell on every side so
// boundary particles just outside the walls get their own cells, with cell
// size = h, built with a counting sort. Particles inside a cell are stored in ascending index order and
// cells are visited in a fixed order, so neighbour iteration order is fully
// deterministic.
//
// Axis 0 has stride 1, so the three cells of a "row" along axis 0 are adjacent
// in the sorted array. A neighbour query therefore scans 3^(DIM−1) contiguous
// ranges instead of 3^DIM separate cells.
//
// The grid holds all particles (fluid and boundary), but neighbour lists are only
// built for the fluid ones, indices 0..nFluid. The neighbours of i (excluding i
// itself) are neighbors[neighborStart[i] .. neighborEnd[i]), with the matching
// squared distances in neighborDist2.
export interface SpatialGrid {
  cellSize: number
  min: number[]
  cellsPerAxis: Int32Array
  strides: Int32Array
  numCells: number
  cellStart: Int32Array // length numCells + 1
  sorted: Int32Array // particle indices grouped by cell
  cellCoord: Int32Array // per-particle cell coordinates, stride DIM
  rowOffsets: Int32Array // the 3^(DIM−1) offsets over axes 1..DIM−1, stride DIM−1
  neighborStart: Int32Array
  neighborEnd: Int32Array
  neighbors: Int32Array // grows as needed
  neighborDist2: Float64Array // grows with neighbors
}

export function createGrid(domain: Domain, cellSize: number, nFluid: number, nTotal: number): SpatialGrid {
  const cellsPerAxis = new Int32Array(DIM)
  const strides = new Int32Array(DIM)
  let numCells = 1
  for (let d = 0; d < DIM; d++) {
    cellsPerAxis[d] = Math.max(1, Math.ceil((domain.max[d] - domain.min[d] + 2 * cellSize) / cellSize))
    strides[d] = numCells
    numCells *= cellsPerAxis[d]
  }

  const numRows = 3 ** (DIM - 1)
  const rowOffsets = new Int32Array(numRows * (DIM - 1))
  for (let o = 0; o < numRows; o++) {
    let rem = o
    for (let d = 0; d < DIM - 1; d++) {
      rowOffsets[o * (DIM - 1) + d] = (rem % 3) - 1
      rem = Math.floor(rem / 3)
    }
  }

  const capacity = nFluid * 32
  return {
    cellSize,
    min: domain.min.map((v) => v - cellSize),
    cellsPerAxis,
    strides,
    numCells,
    cellStart: new Int32Array(numCells + 1),
    sorted: new Int32Array(nTotal),
    cellCoord: new Int32Array(nTotal * DIM),
    rowOffsets,
    neighborStart: new Int32Array(nFluid),
    neighborEnd: new Int32Array(nFluid),
    neighbors: new Int32Array(capacity),
    neighborDist2: new Float64Array(capacity),
  }
}

// Counting sort of particles into cells.
export function buildGrid(g: SpatialGrid, pos: Float32Array, n: number) {
  const { cellStart, sorted, cellCoord, cellsPerAxis, strides, numCells } = g
  const inv = 1 / g.cellSize
  cellStart.fill(0)

  // Count particles per cell. cellStart[c + 1] holds the count for cell c.
  for (let i = 0; i < n; i++) {
    let cell = 0
    for (let d = 0; d < DIM; d++) {
      let c = Math.floor((pos[i * DIM + d] - g.min[d]) * inv)
      if (c < 0) c = 0
      else if (c >= cellsPerAxis[d]) c = cellsPerAxis[d] - 1
      cellCoord[i * DIM + d] = c
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
    for (let d = 0; d < DIM; d++) cell += cellCoord[i * DIM + d] * strides[d]
    sorted[cellStart[cell]++] = i
  }
  for (let c = numCells; c > 0; c--) cellStart[c] = cellStart[c - 1]
  cellStart[0] = 0
}

// Squared distance between particles i and j. Unrolled per dimension: V8 does
// not unroll a `for (d < DIM)` loop, and in the neighbour search (the hottest
// loop in the solver) that costs ~30%.
function dist2(pos: Float32Array, i: number, j: number): number {
  if (DIM === 2) {
    const a = i * 2
    const b = j * 2
    const dx = pos[a] - pos[b]
    const dy = pos[a + 1] - pos[b + 1]
    return dx * dx + dy * dy
  }
  const a = i * 3
  const b = j * 3
  const dx = pos[a] - pos[b]
  const dy = pos[a + 1] - pos[b + 1]
  const dz = pos[a + 2] - pos[b + 2]
  return dx * dx + dy * dy + dz * dz
}

// Builds neighbour lists for the fluid particles i < n (all j ≠ i, fluid or
// boundary, with |x_i − x_j| < h).
// If the list buffers are too small, grow them and rebuild from scratch; that
// keeps the buffers fixed inside the hot loop, which V8 optimizes much better.
export function buildNeighbors(g: SpatialGrid, pos: Float32Array, n: number, h: number) {
  while (!tryBuildNeighbors(g, pos, n, h)) {
    const capacity = g.neighbors.length * 2
    g.neighbors = new Int32Array(capacity)
    g.neighborDist2 = new Float64Array(capacity)
  }
}

// Particles are visited in cell order for memory locality. Every candidate is
// written unconditionally and `count` only advances for real neighbours, which
// avoids a hard-to-predict branch. Returns false if the buffers overflow.
function tryBuildNeighbors(g: SpatialGrid, pos: Float32Array, n: number, h: number): boolean {
  const { cellStart, sorted, cellCoord, cellsPerAxis, strides, rowOffsets, neighborStart, neighborEnd } = g
  const total = cellStart[g.numCells]
  const list = g.neighbors
  const dist = g.neighborDist2
  const capacity = list.length
  const h2 = h * h
  const nx = cellsPerAxis[0]
  const numRows = rowOffsets.length / (DIM - 1) || 1
  let count = 0

  for (let k = 0; k < total; k++) {
    const i = sorted[k]
    if (i >= n) continue
    neighborStart[i] = count
    const cx = cellCoord[i * DIM]
    const x0 = cx > 0 ? cx - 1 : 0
    const x1 = cx < nx - 1 ? cx + 1 : nx - 1

    for (let o = 0; o < numRows; o++) {
      let base = 0
      let inside = true
      for (let d = 1; d < DIM; d++) {
        const c = cellCoord[i * DIM + d] + rowOffsets[o * (DIM - 1) + d - 1]
        if (c < 0 || c >= cellsPerAxis[d]) {
          inside = false
          break
        }
        base += c * strides[d]
      }
      if (!inside) continue

      const from = cellStart[base + x0]
      const to = cellStart[base + x1 + 1]
      if (count + (to - from) > capacity) return false
      for (let q = from; q < to; q++) {
        const j = sorted[q]
        const r2 = dist2(pos, i, j)
        list[count] = j
        dist[count] = r2
        count += r2 < h2 && j !== i ? 1 : 0
      }
    }
    neighborEnd[i] = count
  }
  return true
}
