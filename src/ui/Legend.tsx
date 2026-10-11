import { colorModes, modeRange, type ColorMode } from '../render/colorModes'

// Decimals follow the range's span, so a narrow range like 0.9–1.1 keeps its precision.
const formatter = (span: number) => (v: number) =>
  Math.abs(v) >= 1000 ? `${(v / 1000).toFixed(0)}k` : v.toFixed(span < 10 ? 2 : 0)

export function Legend({ mode, dim }: { mode: ColorMode; dim: number }) {
  const [lo, hi] = modeRange(mode, dim)
  const fmt = formatter(hi - lo)
  return (
    <div className="legend">
      <div className="legend-label">
        {mode.label} <span className="legend-key">[{colorModes.indexOf(mode) + 1}]</span>
      </div>
      <div className="legend-bar" style={{ background: `linear-gradient(to right, ${mode.stops.join(', ')})` }} />
      <div className="legend-ticks">
        <span>≤{fmt(lo)}</span>
        <span>{fmt((lo + hi) / 2)}</span>
        <span>≥{fmt(hi)}</span>
      </div>
    </div>
  )
}
