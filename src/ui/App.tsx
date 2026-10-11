import { useEffect, useRef } from 'react'
import { startLoop } from '../loop'
import { getColorMode, view } from '../render/colorModes'
import { Hud } from './Hud'
import { Legend } from './Legend'

export function App() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    return startLoop(host, (s) => {
      if (hudRef.current) {
        hudRef.current.textContent =
          `${s.fps.toFixed(0)} fps · ${s.particles} particles · ${s.substeps.toFixed(1)} substeps/frame\n` +
          `ρ/ρ₀ avg ${s.densityAvg.toFixed(3)} · max ${s.densityMax.toFixed(3)} · neighbours avg ${s.neighborsAvg.toFixed(1)}`
      }
    })
  }, [])

  return (
    <div className="app">
      <div ref={canvasHostRef} className="canvas-host" />
      <Hud ref={hudRef} />
      <Legend mode={getColorMode(view.colorMode)} />
    </div>
  )
}
