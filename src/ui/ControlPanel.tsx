import { useReducer, useState, type ReactNode } from 'react'
import { cflDt, defaultParams, mouseSpeedCap, params, type Params } from '../sim/params'
import { blockCapacity } from '../sim/sph'
import { colorModes, type ColorModeId } from '../render/colorModes'
import { BRUSH_MODES, type BrushMode, type SimController } from '../loop'

// Fields only read when the particles are created; editing them marks a reset as pending.
const INIT_KEYS = ['particleCount', 'spacing', 'seed'] as const
type InitSnapshot = Pick<Params, (typeof INIT_KEYS)[number]>
const snapshotInit = (): InitSnapshot => ({ particleCount: params.particleCount, spacing: params.spacing, seed: params.seed })

interface Props {
  controller: () => SimController | null
  colorMode: ColorModeId
  onColorMode: (id: ColorModeId) => void
  paused: boolean
  onTogglePause: () => void
  brushMode: BrushMode
  onBrushMode: (mode: BrushMode) => void
}

// Writes straight into the shared mutable `params`; the loop picks changes up on
// its next step. React re-renders only on input events, never per frame.
export function ControlPanel({ controller, colorMode, onColorMode, paused, onTogglePause, brushMode, onBrushMode }: Props) {
  const [, rerender] = useReducer((x: number) => x + 1, 0)
  const [applied, setApplied] = useState(snapshotInit)

  const update = (fn: () => void) => {
    fn()
    rerender()
  }
  const resetPending = INIT_KEYS.some((k) => params[k] !== applied[k])
  const reset = () => {
    controller()?.reset()
    setApplied(snapshotInit())
  }

  const dtLimit = cflDt(params)
  const substepsNeeded = 1 / (60 * params.dt)

  return (
    <details className="panel" open>
      <summary>Controls</summary>

      <div className="buttons">
        <button onClick={onTogglePause} title="Space">
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button onClick={() => controller()?.stepOnce()} disabled={!paused}>
          Step
        </button>
        <button onClick={reset} className={resetPending ? 'pending' : undefined}>
          {resetPending ? 'Reset to apply' : 'Reset'}
        </button>
      </div>

      <Section title="View">
        <label className="row">
          <span>Colour</span>
          <select value={colorMode} onChange={(e) => onColorMode(e.target.value as ColorModeId)}>
            {colorModes.map((m, i) => (
              <option key={m.id} value={m.id}>
                {i + 1}. {m.label}
              </option>
            ))}
          </select>
        </label>
      </Section>

      <Section title="Solver (live)">
        <Slider label="h" value={params.h} min={0.6} max={1.6} step={0.01} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (params.h = v))} />
        <Slider label="Stiffness k" value={params.stiffness} min={500} max={50000} log format={(v) => v.toFixed(0)}
          onChange={(v) => update(() => (params.stiffness = Math.round(v)))} />
        <label className="row">
          <span>EOS</span>
          <select value={params.useTait ? 'tait' : 'linear'}
            onChange={(e) => update(() => (params.useTait = e.target.value === 'tait'))}>
            <option value="linear">Linear</option>
            <option value="tait">Tait (γ=7)</option>
          </select>
        </label>
        <Slider label="XSPH ε" value={params.viscosityEps} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (params.viscosityEps = v))} />
        <Slider label="Gravity" value={-params.gravity[1]} min={0} max={20} step={0.1} format={(v) => v.toFixed(1)}
          onChange={(v) => update(() => (params.gravity[1] = -v))} />
        <Slider label="dt" value={params.dt} min={0.0005} max={0.008} step={0.0005} format={(v) => v.toFixed(4)}
          onChange={(v) => update(() => (params.dt = v))} />
        <p className={params.dt > dtLimit ? 'note warn' : 'note'}>
          CFL limit {dtLimit.toFixed(4)}
          {params.dt > dtLimit && ' — dt exceeds it, expect blow-up'}
        </p>
        <Slider label="Max substeps" value={params.maxSubsteps} min={1} max={16} step={1} format={(v) => v.toFixed(0)}
          onChange={(v) => update(() => (params.maxSubsteps = v))} />
        <p className={substepsNeeded > params.maxSubsteps ? 'note warn' : 'note'}>
          {substepsNeeded.toFixed(1)} substeps/frame needed at 60 fps
          {substepsNeeded > params.maxSubsteps && ' — sim will run slow-motion'}
        </p>
        <Slider label="Restitution" value={params.restitution} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (params.restitution = v))} />
      </Section>

      <Section title="Mouse">
        <div className="row">
          <span>Drag</span>
          <div className="toggle" role="radiogroup" title="B cycles">
            {BRUSH_MODES.map((m) => (
              <button key={m} role="radio" aria-checked={brushMode === m} className={brushMode === m ? `on ${m}` : undefined}
                onClick={() => onBrushMode(m)}>
                {m[0].toUpperCase() + m.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <p className="note">
          {brushMode === 'current'
            ? 'press to anchor, drag to draw the flow · right/shift-drag reverses'
            : 'right/shift-drag does the opposite'}{' '}
          · B cycles
        </p>
        <Slider label="Strength" value={params.mouseStrength} min={0} max={1} step={0.05}
          format={(v) => `${(v * 100).toFixed(0)}%`} onChange={(v) => update(() => (params.mouseStrength = v))} />
        <Slider label="Max speed" value={params.mouseMaxSpeed} min={1} max={40} step={1} format={(v) => v.toFixed(0)}
          onChange={(v) => update(() => (params.mouseMaxSpeed = v))} />
        {mouseSpeedCap(params) < params.mouseMaxSpeed && (
          <p className="note">capped to {mouseSpeedCap(params).toFixed(1)} by CFL at this dt</p>
        )}
      </Section>

      <Section title="Initial state (on reset)">
        <Slider label="Particles" value={params.particleCount} min={500} max={8000} step={100}
          format={(v) => v.toFixed(0)} onChange={(v) => update(() => (params.particleCount = v))} />
        {params.particleCount > blockCapacity(params) && (
          <p className="note warn">Only {blockCapacity(params)} fit at this spacing</p>
        )}
        <Slider label="Spacing" value={params.spacing} min={0.3} max={0.6} step={0.01} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (params.spacing = v))} />
        <p className="note">spacing/h = {(params.spacing / params.h).toFixed(2)} (≈0.4 → ~20 neighbours)</p>
        <label className="row">
          <span>Seed</span>
          <input type="number" value={params.seed} step={1}
            onChange={(e) => update(() => (params.seed = Math.trunc(Number(e.target.value)) || 0))} />
        </label>
      </Section>

      <button className="wide" onClick={() => update(() => Object.assign(params, structuredClone(defaultParams)))}>
        Restore defaults
      </button>
    </details>
  )
}

function Section({ title, children }: { title: string; children: ReactNode }) {
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

function Slider({ label, value, min, max, step, log, format, onChange }: SliderProps) {
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
