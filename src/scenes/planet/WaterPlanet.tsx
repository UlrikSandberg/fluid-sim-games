import { useEffect, useReducer, useRef, useState } from 'react'
import { Hud } from '../../ui/Hud'
import { Legend } from '../../ui/Legend'
import { Section, Slider } from '../../ui/controls'
import { colorModes, getColorMode, type ColorModeId } from '../../render/colorModes'
import { planetView, startPlanetLoop, type PlanetController, type Show, type Tool } from './loop'
import { estimateParticles, seaRadius, type PlanetLayout } from './planet'

// The water planet: an ocean held to a rocky ball by gravity toward its centre,
// which weakens with distance (G·M/r²), on the GPU solver. Sculpt the ground,
// pour water, watch it run downhill.

const TOOLS: { id: Tool; label: string; key: string }[] = [
  { id: 'orbit', label: 'Orbit', key: 'O' },
  { id: 'raise', label: 'Raise', key: 'R' },
  { id: 'lower', label: 'Lower', key: 'L' },
  { id: 'pour', label: 'Pour', key: 'P' },
]
const SHOWS: { id: Show; label: string }[] = [
  { id: 'water', label: 'Water' },
  { id: 'particles', label: 'Particles' },
  { id: 'none', label: 'Hidden' },
]
// Above this many particles (rock, water and room to pour) a planet won't build:
// past it the GPU falls behind real time.
const PARTICLE_LIMIT = 450_000
const POUR_ROOM = 80_000

export function WaterPlanet() {
  const hostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const controllerRef = useRef<PlanetController | null>(null)
  const [, rerender] = useReducer((x: number) => x + 1, 0)
  const [noWebGpu, setNoWebGpu] = useState(false)
  const [water, setWater] = useState({ n: 0, capacity: 0 })
  // The planet the panel describes; it's built when you press Build.
  const [draft, setDraft] = useState<PlanetLayout>(() => structuredClone(planetView.layout))

  useEffect(() => {
    const controller = startPlanetLoop(hostRef.current!, (s) => {
      if (s.status === 'no-webgpu') setNoWebGpu(true)
      setWater((w) => (w.n === s.water && w.capacity === s.capacity ? w : { n: s.water, capacity: s.capacity }))
      if (!hudRef.current) return
      const l = planetView.layout
      hudRef.current.textContent =
        `GPU · ${s.fps.toFixed(0)} fps · ${s.water.toLocaleString()} water + ${s.rock.toLocaleString()} rock particles · ` +
        (planetView.paused ? 'paused' : s.status === 'running' ? `${s.substeps.toFixed(1)} substeps/frame` : s.status) +
        (s.rebuildMs > 0 ? ` · last rebuild ${s.rebuildMs.toFixed(0)} ms` : '') +
        `\ngravity ${planetView.gravity.toFixed(1)} at sea level (r = ${seaRadius(l)}), weakening as 1/r²` +
        `\n${planetView.tool === 'orbit' ? 'drag: orbit' : 'left-drag: ' + planetView.tool + ' · right-drag or shift-drag: orbit'} · wheel: zoom` +
        `\nO/R/L/P: orbit, raise, lower, pour · V: water / particles / hidden · N: new planet · Space: pause`
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
  const build = (layout: PlanetLayout) =>
    update(() => {
      planetView.layout = structuredClone(layout)
      setDraft(structuredClone(layout))
      controllerRef.current?.rebuild()
    })
  const setTool = (t: Tool) => update(() => (planetView.tool = t))

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.target instanceof Element && e.target.closest('input, select, textarea, button')) return
      if (e.metaKey || e.ctrlKey || e.altKey) return
      const tool = TOOLS.find((t) => `Key${t.key}` === e.code)
      if (tool) setTool(tool.id)
      else if (e.code === 'KeyN') build({ ...planetView.layout, seed: planetView.layout.seed + 1, blank: false })
      else if (e.code === 'KeyV') {
        const i = SHOWS.findIndex((s) => s.id === planetView.show)
        update(() => (planetView.show = SHOWS[(i + 1) % SHOWS.length].id))
      } else if (e.code === 'Space') {
        e.preventDefault()
        update(() => (planetView.paused = !planetView.paused))
      }
    }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [])

  const est = estimateParticles(draft)
  const total = est.rock + est.water + POUR_ROOM
  const tooMany = total > PARTICLE_LIMIT
  const pending = JSON.stringify(draft) !== JSON.stringify(planetView.layout)
  const set = (patch: Partial<PlanetLayout>) => setDraft((d) => ({ ...d, ...patch }))
  const tool = planetView.tool

  return (
    <div className="app">
      <div ref={hostRef} className={`canvas-host canvas-3d ${tool !== 'orbit' ? 'tool-active' : ''}`} />
      <Hud ref={hudRef} />
      {noWebGpu && <div className="notice">The water planet runs on the GPU solver, and this browser has no WebGPU.</div>}
      <div className="side">
        {planetView.show !== 'none' && planetView.colorMode !== 'water' && <Legend mode={getColorMode(planetView.colorMode)} dim={3} />}
        <div className="panel">
          <Section title="Tools">
            <div className="toggle">
              {TOOLS.map((t) => (
                <button key={t.id} className={tool === t.id ? 'on' : ''} onClick={() => setTool(t.id)}>
                  {t.label}
                </button>
              ))}
            </div>
            <Slider label="Brush" value={planetView.brushSize} min={0.5} max={8} step={0.25} format={(v) => v.toFixed(2)} onChange={(v) => update(() => (planetView.brushSize = v))} />
            <Slider label="Strength" value={planetView.brushStrength} min={0.25} max={6} step={0.25} format={(v) => v.toFixed(2)} onChange={(v) => update(() => (planetView.brushStrength = v))} />
            <p className="blurb">
              {tool === 'orbit' && 'Pick a tool, then drag on the planet. Right-drag (or shift-drag) still orbits.'}
              {(tool === 'raise' || tool === 'lower') &&
                'Hold and drag to sculpt. The water catches up a few times a second while you hold (a brief hitch each time the rock is rebuilt).'}
              {tool === 'pour' && 'Hold to pour from above the spot you point at; the brush size sets the stream’s width (up to 4).'}
            </p>
          </Section>
          <Section title="Water">
            <div className="row-text">
              {water.n.toLocaleString()} particles, room for {Math.max(0, water.capacity - water.n).toLocaleString()} more
            </div>
            <div className="buttons">
              <button onClick={() => controllerRef.current?.clearWater()}>Remove all</button>
              <button onClick={() => controllerRef.current?.refill()}>Refill the ocean</button>
            </div>
          </Section>
          <Section title="View">
            <div className="toggle">
              {SHOWS.map((s) => (
                <button key={s.id} className={planetView.show === s.id ? 'on' : ''} onClick={() => update(() => (planetView.show = s.id))}>
                  {s.label}
                </button>
              ))}
            </div>
            <label className="row">
              <span>Colour</span>
              <select value={planetView.colorMode} onChange={(e) => update(() => (planetView.colorMode = e.target.value as ColorModeId))}>
                {colorModes.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="buttons">
              <button onClick={() => update(() => (planetView.paused = !planetView.paused))}>{planetView.paused ? 'Resume' : 'Pause'}</button>
            </div>
          </Section>
          <Section title="Gravity (live)">
            <Slider label="At sea level" value={planetView.gravity} min={1} max={25} step={0.5} format={(v) => v.toFixed(1)} onChange={(v) => update(() => (planetView.gravity = v))} />
          </Section>
          <Section title="Planet (applies on Build)">
            <Slider label="Size" value={draft.radius} min={5} max={25} step={1} format={(v) => `r ${v}`} onChange={(v) => set({ radius: v })} />
            <Slider label="Particles" value={draft.spacing} min={0.25} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => set({ spacing: v })} />
            <Slider label="Ocean depth" value={draft.ocean} min={0.5} max={4} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ ocean: v })} />
            <Slider label="Mountains" value={draft.relief} min={0.3} max={3} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ relief: v })} />
            <label className="check-row">
              <input type="checkbox" checked={draft.blank} onChange={(e) => set({ blank: e.target.checked })} />
              Blank ball (a smooth dry planet to sculpt)
            </label>
            <p className={`note estimate ${tooMany ? 'warn' : ''}`}>
              ≈ {est.rock.toLocaleString()} rock + {est.water.toLocaleString()} water particles
              {tooMany ? ` — too many (limit ${PARTICLE_LIMIT.toLocaleString()} with room to pour); smaller planet or bigger particles` : ''}
            </p>
            <div className="buttons">
              <button className={pending ? 'pending' : ''} disabled={tooMany} onClick={() => build(draft)}>
                Build
              </button>
              <button disabled={tooMany} onClick={() => build({ ...draft, seed: draft.seed + 1, blank: false })}>
                New planet
              </button>
            </div>
          </Section>
        </div>
      </div>
    </div>
  )
}
