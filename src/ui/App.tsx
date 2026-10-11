import { useEffect, useRef, useState } from 'react'
import { brush, startLoop, type BrushMode, type SimController } from '../loop'
import { colorModes, getColorMode, view, type ColorModeId } from '../render/colorModes'
import { ControlPanel } from './ControlPanel'
import { Hud } from './Hud'
import { Legend } from './Legend'

export function App() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<SimController | null>(null)
  const [colorMode, setColorMode] = useState<ColorModeId>(view.colorMode)
  const [paused, setPaused] = useState(false)
  const [brushMode, setBrushMode] = useState<BrushMode>(brush.mode)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    const controller = startLoop(host, (s) => {
      if (hudRef.current) {
        hudRef.current.textContent =
          `${s.fps.toFixed(0)} fps · ${s.particles} particles · ` +
          (s.paused ? 'paused' : `${s.substeps.toFixed(1)} substeps/frame`) +
          `\nρ/ρ₀ avg ${s.densityAvg.toFixed(3)} · max ${s.densityMax.toFixed(3)} · neighbours avg ${s.neighborsAvg.toFixed(1)}` +
          `\nbrush ${s.brushMode} [B] · R ${s.brushRadius.toFixed(1)} · accel ${s.brushAccel.toFixed(0)}` +
          ` · right/shift-drag: opposite · wheel: resize`
      }
    })
    controllerRef.current = controller
    return () => {
      controller.dispose()
      controllerRef.current = null
    }
  }, [])

  const changeColorMode = (id: ColorModeId) => {
    view.colorMode = id
    setColorMode(id)
  }
  const changeBrushMode = (mode: BrushMode) => {
    brush.mode = mode
    setBrushMode(mode)
  }
  const togglePause = () => {
    controllerRef.current?.setPaused(!paused)
    setPaused(!paused)
  }

  // Keys 1–4 pick the colour mode, B toggles the brush mode, Space pauses. Ignored
  // while a panel control has focus, so typing and native button/Space keep working.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.closest('input, select, textarea, button') || e.metaKey || e.ctrlKey || e.altKey) return
      const mode = colorModes[Number(e.key) - 1]
      if (mode) changeColorMode(mode.id)
      else if (e.code === 'KeyB') changeBrushMode(brushMode === 'repel' ? 'attract' : 'repel')
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
      <div ref={canvasHostRef} className="canvas-host" />
      <Hud ref={hudRef} />
      <div className="side">
        <Legend mode={getColorMode(colorMode)} />
        <ControlPanel
          controller={() => controllerRef.current}
          colorMode={colorMode}
          onColorMode={changeColorMode}
          paused={paused}
          onTogglePause={togglePause}
          brushMode={brushMode}
          onBrushMode={changeBrushMode}
        />
      </div>
    </div>
  )
}
