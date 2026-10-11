import { useEffect, useRef } from 'react'
import { startLoop } from '../loop'
import { Hud } from './Hud'

export function App() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    return startLoop(host, (s) => {
      if (hudRef.current) {
        hudRef.current.textContent =
          `${s.fps.toFixed(0)} fps · ${s.particles} particles · ${s.substeps.toFixed(1)} substeps/frame`
      }
    })
  }, [])

  return (
    <div className="app">
      <div ref={canvasHostRef} className="canvas-host" />
      <Hud ref={hudRef} />
    </div>
  )
}
