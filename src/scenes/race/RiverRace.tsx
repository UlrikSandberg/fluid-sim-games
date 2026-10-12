import { useEffect, useReducer, useRef, useState } from 'react'
import { cflDt } from '../../sim/params'
import { colorModes, getColorMode, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { Hud } from '../../ui/Hud'
import { Legend } from '../../ui/Legend'
import { Section, Slider } from '../../ui/controls'
import { raceParams, raceView, startRaceLoop, type RaceController } from './loop'
import { defaultRiverLayout, estimateParticles, riverLayout, type RiverLayout } from './riverScene'

// The game scene (work in progress): a winding river with a waterfall on the GPU
// solver. Ducks, randomness and race rules come next.
export function RiverRace() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<RaceController | null>(null)
  const [paused, setPaused] = useState(false)
  const [noWebGpu, setNoWebGpu] = useState(false)
  const [colorMode, setColorMode] = useState<ColorModeId>(raceView.colorMode)
  const [style, setStyle] = useState<RenderStyle>(raceView.style)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    const controller = startRaceLoop(host, (s) => {
      if (s.status === 'no-webgpu') setNoWebGpu(true)
      if (hudRef.current) {
        hudRef.current.textContent =
          `GPU · ${s.fps.toFixed(0)} fps · ${s.particles.toLocaleString()} particles · ` +
          (s.status === 'running' ? `${s.substeps.toFixed(1)} substeps/frame` : s.status) +
          `\ndrag: orbit · right-drag: pan · wheel: zoom · D: drop a duck · Space: pause`
      }
    })
    controllerRef.current = controller
    return () => {
      controller.dispose()
      controllerRef.current = null
    }
  }, [])

  const togglePause = () => {
    controllerRef.current?.setPaused(!paused)
    setPaused(!paused)
  }
  const changeColorMode = (id: ColorModeId) => {
    raceView.colorMode = id
    setColorMode(id)
  }
  const changeStyle = (s: RenderStyle) => {
    raceView.style = s
    setStyle(s)
  }

  // D drops a duck, Space pauses, number keys pick the colour mode, V toggles
  // water/particles (all ignored while a control has focus).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.closest('input, select, textarea, button') || e.metaKey || e.ctrlKey || e.altKey) return
      const mode = colorModes[Number(e.key) - 1]
      if (mode) changeColorMode(mode.id)
      else if (e.code === 'KeyD') controllerRef.current?.dropDuck()
      else if (e.code === 'KeyV') changeStyle(style === 'water' ? 'particles' : 'water')
      else if (e.code === 'Space') {
        e.preventDefault()
        togglePause()
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  })

  return (
    <div className="app">
      <div ref={canvasHostRef} className="canvas-host canvas-3d" />
      <Hud ref={hudRef} />
      {noWebGpu && (
        <div className="notice">
          The river needs WebGPU (recent Chrome, Edge or Safari). The 3D Sandbox runs on the CPU.
        </div>
      )}
      <div className="side">
        {style === 'particles' && <Legend mode={getColorMode(colorMode)} dim={3} />}
        <RacePanel
          controller={() => controllerRef.current}
          paused={paused}
          onTogglePause={togglePause}
          colorMode={colorMode}
          onColorMode={changeColorMode}
          style={style}
          onStyle={changeStyle}
        />
      </div>
    </div>
  )
}

interface PanelProps {
  controller: () => RaceController | null
  paused: boolean
  onTogglePause: () => void
  colorMode: ColorModeId
  onColorMode: (id: ColorModeId) => void
  style: RenderStyle
  onStyle: (s: RenderStyle) => void
}

const DUCK_BASE = [1.6, 1, 1.2]

// Course dials edit riverLayout and apply on Reset; water and duck dials are live.
function RacePanel({ controller, paused, onTogglePause, colorMode, onColorMode, style, onStyle }: PanelProps) {
  const [, rerender] = useReducer((x: number) => x + 1, 0)
  const [applied, setApplied] = useState(() => JSON.stringify(riverLayout))
  const [duckScale, setDuckScale] = useState(1)
  const update = (fn: () => void) => {
    fn()
    rerender()
  }
  const reset = () => {
    controller()?.reset()
    setApplied(JSON.stringify(riverLayout))
  }
  const resetPending = JSON.stringify(riverLayout) !== applied
  const course = (label: string, key: keyof RiverLayout, min: number, max: number, step: number, digits = 1) => (
    <Slider key={key} label={label} value={riverLayout[key]} min={min} max={max} step={step}
      format={(v) => v.toFixed(digits)} onChange={(v) => update(() => (riverLayout[key] = v))} />
  )
  const dtLimit = cflDt(raceParams)

  return (
    <details className="panel" open>
      <summary>River</summary>
      <div className="buttons">
        <button onClick={onTogglePause} title="Space">
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button onClick={reset} className={resetPending ? 'pending' : undefined}>
          {resetPending ? 'Reset to apply' : 'Reset'}
        </button>
        <button onClick={() => controller()?.dropDuck()} title="D">
          Drop duck
        </button>
      </div>

      <Section title="Course (on reset)">
        {course('Particle size', 'spacing', 0.3, 0.6, 0.05, 2)}
        <p className="note">≈{estimateParticles(riverLayout).toLocaleString()} particles · finer means more substeps too</p>
        {course('Length', 'length', 30, 120, 5, 0)}
        {course('Width', 'width', 6, 16, 1, 0)}
        {course('Bends', 'bends', 0, 3, 0.5)}
        {course('Bend size', 'bendSize', 0, 8, 0.5)}
        {course('Waterfall', 'ledgeHeight', 0, 10, 0.5)}
        {course('Upper depth', 'upperDepth', 1, 5, 0.5)}
        {course('Lower depth', 'lowerDepth', 1, 6, 0.5)}
        <button className="wide" onClick={() => update(() => Object.assign(riverLayout, structuredClone(defaultRiverLayout)))}>
          Default course
        </button>
      </Section>

      <Section title="Water (live)">
        <Slider label="Current" value={raceView.current.speed} min={0} max={8} step={0.25} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (raceView.current.speed = v))} />
        <Slider label="Pull" value={raceView.current.rate} min={0} max={5} step={0.1} format={(v) => v.toFixed(1)}
          onChange={(v) => update(() => (raceView.current.rate = v))} />
        <Slider label="Gravity" value={raceView.gravity} min={0} max={20} step={0.1} format={(v) => v.toFixed(1)}
          onChange={(v) => update(() => (raceView.gravity = v))} />
        <Slider label="Stiffness k" value={raceParams.stiffness} min={500} max={50000} log format={(v) => v.toFixed(0)}
          onChange={(v) => update(() => (raceParams.stiffness = Math.round(v)))} />
        <Slider label="XSPH ε" value={raceParams.viscosityEps} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (raceParams.viscosityEps = v))} />
        <Slider label="dt" value={raceParams.dt} min={0.0005} max={0.008} step={0.0005} format={(v) => v.toFixed(4)}
          onChange={(v) => update(() => (raceParams.dt = v))} />
        <p className={raceParams.dt > dtLimit ? 'note warn' : 'note'}>
          CFL limit {dtLimit.toFixed(4)}
          {raceParams.dt > dtLimit && ' — dt exceeds it, expect blow-up'}
        </p>
      </Section>

      <Section title="Duck">
        <Slider label="Density" value={raceParams.obstacleDensity} min={0.1} max={2} step={0.05}
          format={(v) => `${v.toFixed(2)}ρ₀`} onChange={(v) => update(() => (raceParams.obstacleDensity = v))} />
        <Slider label="Size" value={duckScale} min={0.5} max={2.5} step={0.1} format={(v) => `${v.toFixed(1)}×`}
          onChange={(v) => {
            setDuckScale(v)
            raceView.duckSize = DUCK_BASE.map((s) => s * v)
          }} />
        <p className="note">size applies on the next drop</p>
      </Section>

      <Section title="View">
        <div className="row">
          <span>Draw</span>
          <div className="toggle" role="radiogroup" title="V toggles">
            {(['particles', 'water'] as const).map((s) => (
              <button key={s} role="radio" aria-checked={style === s}
                className={style === s ? 'on attract' : undefined} onClick={() => onStyle(s)}>
                {s === 'water' ? 'Water' : 'Particles'}
              </button>
            ))}
          </div>
        </div>
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
    </details>
  )
}
