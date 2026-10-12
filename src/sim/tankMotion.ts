import type { FrameMotion } from './sph.ts'

// Rotating the tank. The user drags a target orientation around; the tank
// follows it with capped angular speed and acceleration, and every frame this
// produces what the solver needs to simulate in tank coordinates: the frame's
// angular velocity and acceleration (for the rotating-frame forces) and gravity
// rotated into the tank. The caps are what make wild swooshing safe: the walls
// move through the water at |ω × r| ≤ MAX_SPIN · r (~20 at the tank's corners,
// about Mach 0.2), and they never start or stop that motion abruptly.
//
// Quaternions are [w, x, y, z]; angular vectors are in world coordinates.

export const MAX_SPIN = 2 // rad/s
export const MAX_SPIN_ACCEL = 8 // rad/s²
const SPIN_GAIN = 6 // 1/s, proportional pull toward the target for small errors

export type Quat = [number, number, number, number]

export interface TankMotion {
  q: Quat // current orientation (tank → world)
  target: Quat
  omega: number[] // current angular velocity
}

export function createTankMotion(): TankMotion {
  return { q: [1, 0, 0, 0], target: [1, 0, 0, 0], omega: [0, 0, 0] }
}

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

function normalize(q: Quat): Quat {
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

// Applies a user rotation (world axis, angle) to the target orientation.
export function nudgeTarget(t: TankMotion, axis: number[], angle: number) {
  t.target = normalize(quatMultiply(quatFromAxisAngle(axis, angle), t.target))
}

export function levelTarget(t: TankMotion) {
  t.target = [1, 0, 0, 0]
}

// Advances the tank by `seconds` of simulated time (the substeps about to run)
// and returns the frame motion and gravity for those substeps, in tank
// coordinates. `pivot` is the tank's centre in tank coordinates.
export function advanceTank(
  t: TankMotion, seconds: number, pivot: number[], gravity: number,
): { frame: FrameMotion; gravity: number[] } {
  let alpha = [0, 0, 0]
  if (seconds > 0) {
    // Error rotation target · q⁻¹ as axis-angle, taking the short way round.
    let e = quatMultiply(t.target, [t.q[0], -t.q[1], -t.q[2], -t.q[3]])
    if (e[0] < 0) e = [-e[0], -e[1], -e[2], -e[3]]
    const angle = 2 * Math.acos(Math.min(1, e[0]))
    const sinHalf = Math.sqrt(Math.max(0, 1 - e[0] * e[0]))
    const axis = sinHalf > 1e-9 ? [e[1] / sinHalf, e[2] / sinHalf, e[3] / sinHalf] : [0, 0, 0]
    // Desired spin: proportional near the target, otherwise the fastest speed
    // from which the acceleration cap can still stop on it, capped overall.
    const speed = Math.min(MAX_SPIN, SPIN_GAIN * angle, Math.sqrt(2 * MAX_SPIN_ACCEL * angle))
    const delta = [0, 1, 2].map((d) => axis[d] * speed - t.omega[d])
    const deltaLen = Math.hypot(delta[0], delta[1], delta[2])
    const maxDelta = MAX_SPIN_ACCEL * seconds
    const f = deltaLen > maxDelta ? maxDelta / deltaLen : 1
    alpha = delta.map((v) => (v * f) / seconds)
    t.omega = t.omega.map((v, d) => v + delta[d] * f)
    const spin = Math.hypot(t.omega[0], t.omega[1], t.omega[2])
    t.q = normalize(quatMultiply(quatFromAxisAngle(t.omega, spin * seconds), t.q))
  }
  const dim = pivot.length
  const g = rotate(t.q, [0, -gravity, 0], true)
  return {
    frame: { omega: rotate(t.q, t.omega, true), alpha: rotate(t.q, alpha, true), pivot },
    gravity: g.slice(0, dim),
  }
}
