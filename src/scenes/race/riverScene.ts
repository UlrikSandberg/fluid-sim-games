import { defaultParams3D, particleRadius, type Params } from '../../sim/params'
import { staticBoundary, type SolidBox } from '../../sim/sph'
import type { GpuScene } from '../../gpu/sphGpu'

// The race course: a straight channel along +x with a waterfall. A solid ledge
// fills the upstream third, so water runs along the upper reach, pours over the
// ledge's edge and continues along the lower reach; at the end it is recycled to
// the inlet on top of the ledge. Sized for modest GPUs (~16k particles).
export const RIVER = {
  length: 60,
  height: 14,
  width: 12,
  ledgeLength: 20, // the waterfall's lip is at x = ledgeLength
  ledgeHeight: 6,
  upperDepth: 2.5,
  lowerDepth: 3,
  exitMargin: 3, // water within this of the far end is recycled
  current: { speed: 3, rate: 1.5 },
}

export function riverParams(): Params {
  return {
    ...structuredClone(defaultParams3D),
    domain: { min: [0, 0, 0], max: [RIVER.length, RIVER.height, RIVER.width] },
  }
}

export function riverBoxes(): SolidBox[] {
  return [{ min: [0, 0, 0], max: [RIVER.ledgeLength, RIVER.ledgeHeight, RIVER.width] }]
}

// Rest-spacing lattice filling [min, max], inset by the particle radius.
function fill(p: Params, min: number[], max: number[], out: number[]) {
  const r = particleRadius(p)
  for (let x = min[0] + r; x <= max[0] - r + 1e-6; x += p.spacing) {
    for (let y = min[1] + r; y <= max[1] - r + 1e-6; y += p.spacing) {
      for (let z = min[2] + r; z <= max[2] - r + 1e-6; z += p.spacing) out.push(x, y, z)
    }
  }
}

export function riverScene(p: Params): GpuScene {
  const { length, width, ledgeLength, ledgeHeight, upperDepth, lowerDepth, exitMargin } = RIVER
  const fluid: number[] = []
  fill(p, [0, ledgeHeight, 0], [ledgeLength, ledgeHeight + upperDepth, width], fluid)
  fill(p, [ledgeLength, 0, 0], [length, lowerDepth, width], fluid)
  const boxes = riverBoxes()
  return {
    fluid: Float32Array.from(fluid),
    boundary: staticBoundary(p, boxes),
    boxes,
    current: { ...RIVER.current },
    recycle: { exitX: length - exitMargin, inletX: particleRadius(p) * 2, lift: ledgeHeight },
  }
}
