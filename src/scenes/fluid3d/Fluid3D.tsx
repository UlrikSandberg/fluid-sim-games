import { useEffect, useReducer, useRef, useState } from 'react'
import { cflDt, defaultParams3D } from '../../sim/params'
import { blockCapacity } from '../../sim/sph'
import { colorModes, getColorMode, type ColorModeId } from '../../render/colorModes'
import { Hud } from '../../ui/Hud'
import { Legend } from '../../ui/Legend'
import { Section, Slider } from '../../ui/controls'
import { params3d, startLoop3D, tankGravity, view3d, type DragMode, type SimController3D } from './loop'

// The 3D fluid scene: the shared solver with a 3D domain, an orbit camera and
// lit spheres. No mouse forces yet (the mouse drives the camera); that needs ray
// picking into the tank.
export function Fluid3D() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<SimController3D | null>(null)
  const [colorMode, setColorMode] = useState<ColorModeId>(view3d.colorMode)
  const [paused, setPaused] = useState(false)
  const [dragMode, setDragMode] = useState<DragMode>(view3d.drag)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    const controller = startLoop3D(host, (s) => {
      if (hudRef.current) {
        hudRef.current.textContent =
          `${s.fps.toFixed(0)} fps · ${s.particles} particles · ` +
          (s.paused ? 'paused' : `${s.substeps.toFixed(1)} substeps/frame · ${s.stepMs.toFixed(2)} ms/substep`) +
          `\nρ/ρ₀ avg ${s.densityAvg.toFixed(3)} · neighbours avg ${s.neighborsAvg.toFixed(1)}` +
          `\n${view3d.drag === 'tank' ? 'drag: rotate tank' : 'drag: orbit · shift-drag: rotate tank'} · right-drag: pan · wheel: zoom` +
          `\nT: toggle drag mode · L: level tank`
      }
    })
    controllerRef.current = controller
    return () => {
      controller.dispose()
      controllerRef.current = null
    }
  }, [])

  const changeColorMode = (id: ColorModeId) => {
    view3d.colorMode = id
    setColorMode(id)
  }
  const changeDragMode = (mode: DragMode) => {
    view3d.drag = mode
    setDragMode(mode)
  }
  const togglePause = () => {
    controllerRef.current?.setPaused(!paused)
    setPaused(!paused)
  }

  // Number keys pick the colour mode, T toggles the drag mode, L levels the tank,
  // Space pauses (all ignored while a control has focus).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.closest('input, select, textarea, button') || e.metaKey || e.ctrlKey || e.altKey) return
      const mode = colorModes[Number(e.key) - 1]
      if (mode) changeColorMode(mode.id)
      else if (e.code === 'KeyT') changeDragMode(dragMode === 'tank' ? 'orbit' : 'tank')
      else if (e.code === 'KeyL') controllerRef.current?.levelTank()
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
      <div ref={canvasHostRef} className={`canvas-host canvas-3d ${dragMode === 'tank' ? 'drag-tank' : ''}`} />
      <Hud ref={hudRef} />
      <div className="side">
        <Legend mode={getColorMode(colorMode)} dim={3} />
        <Panel3D
          controller={() => controllerRef.current}
          colorMode={colorMode}
          onColorMode={changeColorMode}
          paused={paused}
          onTogglePause={togglePause}
          dragMode={dragMode}
          onDragMode={changeDragMode}
        />
      </div>
    </div>
  )
}

interface PanelProps {
  controller: () => SimController3D | null
  colorMode: ColorModeId
  onColorMode: (id: ColorModeId) => void
  paused: boolean
  onTogglePause: () => void
  dragMode: DragMode
  onDragMode: (mode: DragMode) => void
}

// Writes straight into params3d; the loop picks changes up on its next step.
function Panel3D({ controller, colorMode, onColorMode, paused, onTogglePause, dragMode, onDragMode }: PanelProps) {
  const [, rerender] = useReducer((x: number) => x + 1, 0)
  const [appliedCount, setAppliedCount] = useState(params3d.particleCount)
  const update = (fn: () => void) => {
    fn()
    rerender()
  }
  const reset = () => {
    controller()?.reset()
    setAppliedCount(params3d.particleCount)
  }
  const resetPending = params3d.particleCount !== appliedCount
  const dtLimit = cflDt(params3d)
  const capacity = blockCapacity(params3d)

  return (
    <details className="panel" open>
      <summary>Controls</summary>
      <div className="buttons">
        <button onClick={onTogglePause} title="Space">
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button onClick={reset} className={resetPending ? 'pending' : undefined}>
          {resetPending ? 'Reset to apply' : 'Reset'}
        </button>
      </div>

      <Section title="Tank">
        <div className="row">
          <span>Drag</span>
          <div className="toggle" role="radiogroup" title="T toggles">
            {(['orbit', 'tank'] as const).map((m) => (
              <button key={m} role="radio" aria-checked={dragMode === m}
                className={dragMode === m ? 'on current' : undefined} onClick={() => onDragMode(m)}>
                {m === 'orbit' ? 'Orbit' : 'Rotate tank'}
              </button>
            ))}
          </div>
        </div>
        <p className="note">shift-drag rotates the tank in either mode</p>
        <button className="wide" onClick={() => controller()?.levelTank()} title="L">
          Level tank
        </button>
      </Section>

      <Section title="Floating box">
        <div className="buttons">
          <button onClick={() => update(() => controller()?.dropBody())}>
            {controller()?.hasBody() ? 'Drop again' : 'Drop box'}
          </button>
          <button onClick={() => update(() => controller()?.removeBody())} disabled={!controller()?.hasBody()}>
            Remove
          </button>
        </div>
        <Slider label="Density" value={params3d.obstacleDensity} min={0.1} max={2} step={0.05}
          format={(v) => `${v.toFixed(2)}ρ₀`} onChange={(v) => update(() => (params3d.obstacleDensity = v))} />
        <p className="note">
          {params3d.obstacleDensity < 1
            ? `floats ${(params3d.obstacleDensity * 100).toFixed(0)}% submerged`
            : 'denser than water: sinks'} · live
        </p>
        {(['Width', 'Height', 'Depth'] as const).map((label, d) => (
          <Slider key={label} label={label} value={params3d.obstacleSize[d]} min={1} max={8} step={0.5}
            format={(v) => v.toFixed(1)} onChange={(v) => update(() => (params3d.obstacleSize[d] = v))} />
        ))}
        <p className="note">size applies on the next drop</p>
      </Section>

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
        <Slider label="Stiffness k" value={params3d.stiffness} min={500} max={50000} log format={(v) => v.toFixed(0)}
          onChange={(v) => update(() => (params3d.stiffness = Math.round(v)))} />
        <Slider label="XSPH ε" value={params3d.viscosityEps} min={0} max={0.5} step={0.01} format={(v) => v.toFixed(2)}
          onChange={(v) => update(() => (params3d.viscosityEps = v))} />
        <Slider label="Gravity" value={tankGravity.value} min={0} max={20} step={0.1} format={(v) => v.toFixed(1)}
          onChange={(v) => update(() => (tankGravity.value = v))} />
        <Slider label="dt" value={params3d.dt} min={0.0005} max={0.008} step={0.0005} format={(v) => v.toFixed(4)}
          onChange={(v) => update(() => (params3d.dt = v))} />
        <p className={params3d.dt > dtLimit ? 'note warn' : 'note'}>
          CFL limit {dtLimit.toFixed(4)}
          {params3d.dt > dtLimit && ' — dt exceeds it, expect blow-up'}
        </p>
      </Section>

      <Section title="Initial state (on reset)">
        <Slider label="Particles" value={params3d.particleCount} min={500} max={16000} step={500}
          format={(v) => v.toFixed(0)} onChange={(v) => update(() => (params3d.particleCount = v))} />
        <p className={params3d.particleCount > capacity ? 'note warn' : 'note'}>
          {params3d.particleCount > capacity
            ? `Only ${capacity} fit in the tank`
            : '~0.9 µs per particle per substep; 60 fps fits ~4000'}
        </p>
      </Section>

      <button className="wide"
        onClick={() => update(() => {
          Object.assign(params3d, structuredClone(defaultParams3D))
          tankGravity.value = -defaultParams3D.gravity[1]
        })}>
        Restore defaults
      </button>
    </details>
  )
}
