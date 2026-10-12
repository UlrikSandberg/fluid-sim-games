import { useEffect, useRef, useState } from 'react'
import { colorModes, getColorMode, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { Hud } from '../../ui/Hud'
import { Legend } from '../../ui/Legend'
import { Section, Slider } from '../../ui/controls'
import { raceView, startRaceLoop, type RaceController } from './loop'

// The game scene (work in progress): a river with a waterfall on the GPU solver.
// Ducks, randomness and race rules come next.
export function RiverRace() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<RaceController | null>(null)
  const [paused, setPaused] = useState(false)
  const [noWebGpu, setNoWebGpu] = useState(false)
  const [current, setCurrent] = useState(raceView.current)
  const [colorMode, setColorMode] = useState<ColorModeId>(raceView.colorMode)
  const [style, setStyle] = useState<RenderStyle>(raceView.style)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    const controller = startRaceLoop(host, (s) => {
      if (s.status === 'no-webgpu') setNoWebGpu(true)
      if (hudRef.current) {
        hudRef.current.textContent =
          `GPU · ${s.fps.toFixed(0)} fps · ${s.particles} particles · ` +
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
  const changeCurrent = (v: number) => {
    controllerRef.current?.setCurrent(v)
    setCurrent(v)
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
        <details className="panel" open>
          <summary>River</summary>
          <div className="buttons">
            <button onClick={togglePause} title="Space">
              {paused ? 'Resume' : 'Pause'}
            </button>
            <button onClick={() => controllerRef.current?.reset()}>Reset</button>
            <button onClick={() => controllerRef.current?.dropDuck()} title="D">
              Drop duck
            </button>
          </div>
          <Section title="Water">
            <Slider label="Current" value={current} min={0} max={8} step={0.25} format={(v) => v.toFixed(2)}
              onChange={changeCurrent} />
            <div className="row">
              <span>Draw</span>
              <div className="toggle" role="radiogroup" title="V toggles">
                {(['particles', 'water'] as const).map((s) => (
                  <button key={s} role="radio" aria-checked={style === s}
                    className={style === s ? 'on attract' : undefined} onClick={() => changeStyle(s)}>
                    {s === 'water' ? 'Water' : 'Particles'}
                  </button>
                ))}
              </div>
            </div>
            <label className="row">
              <span>Colour</span>
              <select value={colorMode} onChange={(e) => changeColorMode(e.target.value as ColorModeId)}>
                {colorModes.map((m, i) => (
                  <option key={m.id} value={m.id}>
                    {i + 1}. {m.label}
                  </option>
                ))}
              </select>
            </label>
          </Section>
        </details>
      </div>
    </div>
  )
}
