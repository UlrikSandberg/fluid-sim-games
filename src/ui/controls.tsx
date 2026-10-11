import type { ReactNode } from 'react'

// Building blocks for the scenes' control panels.

export function Section({ title, children }: { title: string; children: ReactNode }) {
  return (
    <fieldset className="section">
      <legend>{title}</legend>
      {children}
    </fieldset>
  )
}

interface SliderProps {
  label: string
  value: number
  min: number
  max: number
  step?: number
  log?: boolean // logarithmic track, for values spanning orders of magnitude
  format: (v: number) => string
  onChange: (v: number) => void
}

const LOG_STEPS = 1000

export function Slider({ label, value, min, max, step, log, format, onChange }: SliderProps) {
  const toTrack = (v: number) => (log ? (Math.log(v / min) / Math.log(max / min)) * LOG_STEPS : v)
  const fromTrack = (t: number) => (log ? min * (max / min) ** (t / LOG_STEPS) : t)
  return (
    <label className="row">
      <span>{label}</span>
      <input
        type="range"
        min={log ? 0 : min}
        max={log ? LOG_STEPS : max}
        step={log ? 1 : step}
        value={toTrack(value)}
        onChange={(e) => onChange(fromTrack(Number(e.target.value)))}
      />
      <output>{format(value)}</output>
    </label>
  )
}
