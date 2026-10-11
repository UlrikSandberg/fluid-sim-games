import { useEffect, useRef, useState } from 'react'
import { startLoop, type SimController } from '../loop'
import { getColorMode, view, type ColorModeId } from '../render/colorModes'
import { ControlPanel } from './ControlPanel'
import { Hud } from './Hud'
import { Legend } from './Legend'

export function App() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<SimController | null>(null)
  const [colorMode, setColorMode] = useState<ColorModeId>(view.colorMode)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    const controller = startLoop(host, (s) => {
      if (hudRef.current) {
        hudRef.current.textContent =
          `${s.fps.toFixed(0)} fps · ${s.particles} particles · ` +
          (s.paused ? 'paused' : `${s.substeps.toFixed(1)} substeps/frame`) +
          `\nρ/ρ₀ avg ${s.densityAvg.toFixed(3)} · max ${s.densityMax.toFixed(3)} · neighbours avg ${s.neighborsAvg.toFixed(1)}`
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

  return (
    <div className="app">
      <div ref={canvasHostRef} className="canvas-host" />
      <Hud ref={hudRef} />
      <div className="side">
        <Legend mode={getColorMode(colorMode)} />
        <ControlPanel controller={() => controllerRef.current} colorMode={colorMode} onColorMode={changeColorMode} />
      </div>
    </div>
  )
}
