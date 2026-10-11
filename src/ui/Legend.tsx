import type { ColorMode } from '../render/colorModes'

const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : v.toFixed(2))

export function Legend({ mode }: { mode: ColorMode }) {
  const [lo, hi] = mode.range
  return (
    <div className="legend">
      <div className="legend-label">{mode.label}</div>
      <div className="legend-bar" style={{ background: `linear-gradient(to right, ${mode.stops.join(', ')})` }} />
      <div className="legend-ticks">
        <span>≤{fmt(lo)}</span>
        <span>{fmt((lo + hi) / 2)}</span>
        <span>≥{fmt(hi)}</span>
      </div>
    </div>
  )
}
