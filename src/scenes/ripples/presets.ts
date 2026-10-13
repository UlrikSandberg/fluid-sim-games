import { createGrid, fillTo, type ShallowGrid } from '../../sim/shallow/shallow.ts'

// The ripple tank's set-ups: a pool 24 m × 16 m, 1 m deep, on a grid of 8 cm
// cells. A wave maker can push plane waves in from the left wall; walls with
// slits show diffraction and interference; a shallow bump on the floor slows
// the waves over it and bends them to a focus, like a lens.

export type PresetId = 'pool' | 'single' | 'double' | 'lens'

export interface Preset {
  id: PresetId
  name: string
  blurb: string
  waveMaker: boolean // starts with the paddle on
}

export const PRESETS: Preset[] = [
  { id: 'pool', name: 'Pool', blurb: 'Open water. Click or drag to make ripples; hold still to keep bobbing.', waveMaker: false },
  { id: 'single', name: 'Single slit', blurb: 'Plane waves squeeze through a gap and spread out the other side: diffraction.', waveMaker: true },
  {
    id: 'double',
    name: 'Double slit',
    blurb: 'Two gaps make two sources. Where their crests meet the waves add up; where a crest meets a trough they cancel.',
    waveMaker: true,
  },
  {
    id: 'lens',
    name: 'Underwater lens',
    blurb: 'Waves are slower in shallow water (c = √(g·h)). A round shoal holds back the middle of each wave and bends it to a focus.',
    waveMaker: true,
  },
]

export const TANK = { width: 24, length: 16, depth: 1, cell: 0.08 }
const SPONGE_WIDTH = 1.2 // metres of absorbing margin
const SPONGE_RATE = 3 // 1/s at the very edge
const WALL_X = 8 // the slit wall's position, metres from the wave maker
const WALL_THICK = 0.24

export interface TankSetup {
  grid: ShallowGrid
  level: number // the rest surface elevation
}

// `absorb`: sponges on the three walls away from the wave maker, so the tank
// behaves like open water; off, the waves bounce around a closed pool.
export function buildTank(id: PresetId, absorb: boolean): TankSetup {
  const { width, length, depth, cell } = TANK
  const nx = Math.round(width / cell)
  const ny = Math.round(length / cell)
  const s = createGrid(nx, ny, cell)
  const at = (i: number, j: number) => j * nx + i
  const x = (i: number) => (i + 0.5) * cell
  const y = (j: number) => (j + 0.5) * cell

  if (id === 'single' || id === 'double') {
    const gaps = id === 'single' ? [[length / 2, 1.6]] : [[length / 2 - 2, 1], [length / 2 + 2, 1]]
    for (let j = 0; j < ny; j++) {
      const open = gaps.some(([c, w]) => Math.abs(y(j) - c) < w / 2)
      if (open) continue
      for (let i = 0; i < nx; i++) if (Math.abs(x(i) - WALL_X) < WALL_THICK / 2) s.solid[at(i, j)] = 1
    }
  }
  if (id === 'lens') {
    // A round shoal, 0.7 m high at its centre, so the water over it is 0.3 m deep.
    const cx = WALL_X
    const cy = length / 2
    const R = 3.2
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const r = Math.hypot(x(i) - cx, y(j) - cy) / R
        if (r < 1) s.bed[at(i, j)] = 0.7 * Math.cos((r * Math.PI) / 2) ** 2
      }
    }
  }
  if (absorb) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const d = Math.min(width - x(i), y(j), length - y(j))
        if (d < SPONGE_WIDTH) s.sponge[at(i, j)] = SPONGE_RATE * (1 - d / SPONGE_WIDTH) ** 2
      }
    }
  }
  fillTo(s, depth)
  return { grid: s, level: depth }
}
