// Unit quaternions [w, x, y, z] for orientations (the tank, rigid bodies).
// Vectors are 3-component; 2D uses rotations about z only.

export type Quat = [number, number, number, number]

export function quatMultiply(a: Quat, b: Quat): Quat {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ]
}

export function quatFromAxisAngle(axis: number[], angle: number): Quat {
  const len = Math.hypot(axis[0], axis[1], axis[2])
  if (len < 1e-12 || angle === 0) return [1, 0, 0, 0]
  const s = Math.sin(angle / 2) / len
  return [Math.cos(angle / 2), axis[0] * s, axis[1] * s, axis[2] * s]
}

export function normalize(q: Quat): Quat {
  const len = Math.hypot(q[0], q[1], q[2], q[3])
  return [q[0] / len, q[1] / len, q[2] / len, q[3] / len]
}

// Rotates v by q (or by q⁻¹ with inverse = true).
export function rotate(q: Quat, v: number[], inverse = false): number[] {
  const [w, x, y, z] = q
  const s = inverse ? -1 : 1
  // v' = v + 2 u × (u × v + w v), with u the vector part (negated for q⁻¹)
  const ux = s * x
  const uy = s * y
  const uz = s * z
  const tx = uy * v[2] - uz * v[1] + w * v[0]
  const ty = uz * v[0] - ux * v[2] + w * v[1]
  const tz = ux * v[1] - uy * v[0] + w * v[2]
  return [v[0] + 2 * (uy * tz - uz * ty), v[1] + 2 * (uz * tx - ux * tz), v[2] + 2 * (ux * ty - uy * tx)]
}

// q⁻¹ for a unit quaternion.
export function conjugate(q: Quat): Quat {
  return [q[0], -q[1], -q[2], -q[3]]
}

export function cross(a: number[], b: number[]): number[] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
