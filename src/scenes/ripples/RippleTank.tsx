import { useEffect, useReducer, useRef } from 'react'
import { Hud } from '../../ui/Hud'
import { Section, Slider } from '../../ui/controls'
import { PRESETS, type PresetId } from './presets'
import { rippleView, startRippleLoop, type RippleController } from './loop'

// The ripple tank: a pool of shallow water (src/sim/shallow) with a wave maker,
// walls with slits, and an underwater lens. Poke it, watch waves diffract and
// interfere, and see the floor's caustics through the water.
export function RippleTank() {
  const hostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<RippleController | null>(null)
  const [, rerender] = useReducer((x: number) => x + 1, 0)

  useEffect(() => {
    const controller = startRippleLoop(hostRef.current!, (s) => {
      if (!hudRef.current) return
      hudRef.current.textContent =
        `CPU · ${s.fps.toFixed(0)} fps · ${s.grid} cells · ` +
        (rippleView.paused ? 'paused' : `${s.steps} steps/frame · ${s.stepMs.toFixed(2)} ms/step`) +
        `\nwave speed ${s.speed.toFixed(2)} m/s · wavelength ${s.wavelength.toFixed(2)} m at ${rippleView.freq.toFixed(2)} Hz` +
        `\nclick/drag: poke · right-drag or shift-drag: orbit · wheel: zoom` +
        `\nW: wave maker · H: heights/real · R: reset · Space: pause`
    })
    controllerRef.current = controller
    return () => {
      controller.dispose()
      controllerRef.current = null
    }
  }, [])

  const update = (fn: () => void) => {
    fn()
    rerender()
  }
  const choosePreset = (id: PresetId) =>
    update(() => {
      rippleView.preset = id
      rippleView.waveMaker = PRESETS.find((p) => p.id === id)!.waveMaker
      controllerRef.current?.reset()
    })

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof Element && e.target.closest('input, select, textarea, button')) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      if (e.code === 'KeyW') update(() => (rippleView.waveMaker = !rippleView.waveMaker))
      else if (e.code === 'KeyH') update(() => (rippleView.look = rippleView.look === 'real' ? 'heights' : 'real'))
      else if (e.code === 'KeyR') controllerRef.current?.reset()
      else if (e.code === 'Space') {
        e.preventDefault()
        update(() => (rippleView.paused = !rippleView.paused))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const preset = PRESETS.find((p) => p.id === rippleView.preset)!
  return (
    <div className="app ripple">
      <div ref={hostRef} className="canvas-host canvas-3d" />
      <Hud ref={hudRef} />
      <div className="side">
        <div className="panel">
          <Section title="Tank">
            <div className="toggle preset-toggle">
              {PRESETS.map((p) => (
                <button key={p.id} className={p.id === rippleView.preset ? 'on' : ''} onClick={() => choosePreset(p.id)}>
                  {p.name}
                </button>
              ))}
            </div>
            <p className="blurb">{preset.blurb}</p>
            <label className="check-row">
              <input
                type="checkbox"
                checked={rippleView.absorb}
                onChange={(e) =>
                  update(() => {
                    rippleView.absorb = e.target.checked
                    controllerRef.current?.reset()
                  })
                }
              />
              Open water (far walls absorb waves)
            </label>
          </Section>
          <Section title="Waves">
            <label className="check-row">
              <input type="checkbox" checked={rippleView.waveMaker} onChange={(e) => update(() => (rippleView.waveMaker = e.target.checked))} />
              Wave maker (left wall)
            </label>
            <Slider label="Frequency" value={rippleView.freq} min={0.3} max={2} step={0.05} format={(v) => `${v.toFixed(2)} Hz`} onChange={(v) => update(() => (rippleView.freq = v))} />
            <Slider label="Strength" value={rippleView.amplitude} min={0.1} max={1} step={0.05} format={(v) => `${Math.round(v * 100)}%`} onChange={(v) => update(() => (rippleView.amplitude = v))} />
          </Section>
          <Section title="View">
            <div className="toggle">
              <button className={rippleView.look === 'real' ? 'on' : ''} onClick={() => update(() => (rippleView.look = 'real'))}>
                Water
              </button>
              <button className={rippleView.look === 'heights' ? 'on' : ''} onClick={() => update(() => (rippleView.look = 'heights'))}>
                Heights
              </button>
            </div>
            <div className="buttons">
              <button onClick={() => update(() => (rippleView.paused = !rippleView.paused))}>{rippleView.paused ? 'Resume' : 'Pause'}</button>
              <button onClick={() => controllerRef.current?.reset()}>Reset</button>
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}
