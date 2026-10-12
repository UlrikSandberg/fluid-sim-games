import { useEffect, useReducer, useRef, useState } from 'react'
import { cflDt } from '../../sim/params'
import { colorModes, getColorMode, type ColorModeId, type RenderStyle } from '../../render/colorModes'
import { Hud } from '../../ui/Hud'
import { Legend } from '../../ui/Legend'
import { Section, Slider } from '../../ui/controls'
import { RIDE_DISTANCE, courseState, raceParams, raceView, startRaceLoop, type RaceController, type Standing } from './loop'
import { RANDOM_RANGES, countFor, raceSettings, type RandomDial } from './features'
import { duckColor, duckName, racers, renderEvents, renderStandings, saveRacers } from './racers'
import { defaultRiverLayout, estimateParticles, riverLayout } from './riverScene'
import { MOVER_KINDS, type MoverKind } from './movers'
import { courseLength, minRadius, piecesOverlap, type Piece, type PieceCounts, type PieceKind } from './course'

// The game scene: named ducks race down a river built from pieces (straights,
// bends, U-turns, waterfalls, flip zones) on the GPU solver. The ducks hang at the
// gate while the water settles; Start counts down and opens it, a finish line
// ends it, and the results say who does the chore.
export function RiverRace() {
  const canvasHostRef = useRef<HTMLDivElement>(null)
  const hudRef = useRef<HTMLDivElement>(null)
  const countdownRef = useRef<HTMLDivElement>(null)
  const standingsRef = useRef<HTMLDivElement>(null)
  const eventsRef = useRef<HTMLDivElement>(null)
  const phaseRef = useRef('idle')
  const courseRef = useRef(0)
  const [, setCourse] = useState(0)
  const [results, setResults] = useState<Standing[] | null>(null)
  const [phase, setPhase] = useState('idle') // re-renders the page when a race changes phase
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
      // 3, 2, 1 while the ducks hang at the gate, then GO! for a second.
      if (countdownRef.current) {
        const zeroG = s.zeroG > 0
        const text = s.phase === 'countdown'
          ? String(Math.max(1, Math.ceil(s.countdown)))
          : zeroG ? 'ZERO-G!' : s.phase === 'racing' && s.sinceStart < 1 ? 'GO!' : ''
        countdownRef.current.textContent = text
        countdownRef.current.style.display = text ? 'block' : 'none'
        countdownRef.current.style.color = zeroG ? '#54d6ff' : ''
      }
      if (eventsRef.current) renderEvents(eventsRef.current, s.events, 4)
      if (standingsRef.current) {
        const show = s.phase === 'racing' || s.phase === 'countdown'
        standingsRef.current.style.display = show ? 'block' : 'none'
        if (show) renderStandings(standingsRef.current, s.standings)
      }
      // React only hears about the race when its phase changes, or the course does.
      if (s.course !== courseRef.current) {
        courseRef.current = s.course
        setCourse(s.course)
      }
      if (s.phase !== phaseRef.current) {
        phaseRef.current = s.phase
        setResults(s.phase === 'finished' ? s.standings : null)
        setPhase(s.phase)
      }
      if (hudRef.current) {
        hudRef.current.textContent =
          `GPU · ${s.fps.toFixed(0)} fps · ${s.particles.toLocaleString()} particles · ${s.ducks} ducks · ` +
          (s.status === 'running' ? `${s.substeps.toFixed(1)} substeps/frame` : s.status) +
          `\nS: start · N: new course · C: duck cam (1st/3rd), ←/→ steer, ↑ paddle · D: add a duck · F: follow · Space: pause`
      }
    })
    controllerRef.current = controller
    return () => {
      controller.dispose()
      controllerRef.current = null
    }
  }, [])

  // Cycles the camera: orbit → first person → third person → orbit, riding the
  // last ridden duck (the first racer to begin with).
  const lastRide = useRef(0)
  const [, setRideTick] = useState(0)
  const toggleRide = () => {
    if (raceView.ride < 0) {
      raceView.ride = Math.min(lastRide.current, raceView.duckCount - 1)
      raceView.rideView = 'first'
    } else if (raceView.rideView === 'first') {
      raceView.rideView = 'third'
    } else {
      lastRide.current = raceView.ride
      raceView.ride = -1
    }
    setRideTick((t) => t + 1)
  }

  // The mouse wheel moves the third-person camera nearer or farther (the orbit
  // controls, which zoom otherwise, are off while riding).
  useEffect(() => {
    const host = canvasHostRef.current
    if (!host) return
    const onWheel = (e: WheelEvent) => {
      if (raceView.ride < 0 || raceView.rideView !== 'third') return
      const [lo, hi] = RIDE_DISTANCE
      raceView.rideDistance = Math.min(hi, Math.max(lo, raceView.rideDistance * Math.exp(e.deltaY * 0.001)))
    }
    host.addEventListener('wheel', onWheel, { passive: true })
    return () => host.removeEventListener('wheel', onWheel)
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

  // C rides a duck, the arrow keys steer it (held: they set raceView.steer;
  // ignored while a slider or text field has focus, which use them).
  useEffect(() => {
    const held = new Set<string>()
    const apply = () => {
      raceView.steer.turn = (held.has('ArrowLeft') ? 1 : 0) - (held.has('ArrowRight') ? 1 : 0)
      raceView.steer.paddle = held.has('ArrowUp') ? 1 : 0
    }
    const arrows = ['ArrowLeft', 'ArrowRight', 'ArrowUp']
    const down = (e: KeyboardEvent) => {
      if (!arrows.includes(e.code) || (e.target as HTMLElement).closest('input, select, textarea')) return
      e.preventDefault()
      held.add(e.code)
      apply()
    }
    const up = (e: KeyboardEvent) => {
      held.delete(e.code)
      apply()
    }
    const clear = () => {
      held.clear()
      apply()
    }
    window.addEventListener('keydown', down)
    window.addEventListener('keyup', up)
    window.addEventListener('blur', clear)
    return () => {
      window.removeEventListener('keydown', down)
      window.removeEventListener('keyup', up)
      window.removeEventListener('blur', clear)
    }
  }, [])

  // S starts a race, D adds a duck, Space pauses, number keys pick the colour mode, V toggles
  // water/particles, C rides a duck (all ignored while a control has focus).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement
      if (target.closest('input, select, textarea, button') || e.metaKey || e.ctrlKey || e.altKey) return
      const mode = colorModes[Number(e.key) - 1]
      if (mode) changeColorMode(mode.id)
      else if (e.code === 'KeyD') controllerRef.current?.dropDuck()
      else if (e.code === 'KeyS') controllerRef.current?.startRace()
      else if (e.code === 'KeyN') controllerRef.current?.newCourse()
      else if (e.code === 'KeyF') raceView.follow = !raceView.follow
      else if (e.code === 'KeyC') toggleRide()
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
      <div ref={countdownRef} className="countdown" style={{ display: 'none' }} />
      <div ref={standingsRef} className="standings" style={{ display: 'none' }} />
      <div ref={eventsRef} className="events" />
      {results && (
        <Results standings={results} onAgain={() => controllerRef.current?.startRace()}
          onNewCourse={() => controllerRef.current?.newCourse()} />
      )}
      {phase === 'setup' && (
        <div className="setup">
          <button className="go" onClick={() => controllerRef.current?.startRace()}>▶ Start race (S)</button>
          <button onClick={() => controllerRef.current?.newCourse()}>🎲 New course (N)</button>
        </div>
      )}
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

// The end-of-race card: the winner, the placings, and who does the chore.
function Results({ standings, onAgain, onNewCourse }: { standings: Standing[]; onAgain: () => void; onNewCourse: () => void }) {
  const winner = standings[0]
  const chore = racers.chore === 'last' ? standings[standings.length - 1] : winner
  return (
    <div className="results">
      <h2>
        🏆 <span style={{ color: duckColor(winner.index) }}>{duckName(winner.index)}</span> wins!
      </h2>
      <ol>
        {standings.map((s) => (
          <li key={s.index}>
            <span className="dot" style={{ background: duckColor(s.index) }} />
            {duckName(s.index)}
            <span className="status">
              {s.finishTime != null ? `${s.finishTime.toFixed(1)} s` : `${Math.round(s.progress * 100)}% (out of time)`}
            </span>
          </li>
        ))}
      </ol>
      <p className="chore">
        🧽 <strong style={{ color: duckColor(chore.index) }}>{duckName(chore.index)}</strong> does the chore
        {standings.length > 1 && (racers.chore === 'last' ? ' (last place)' : ' (the winner)')}.
      </p>
      <div className="buttons">
        <button onClick={onAgain}>Race again here (S)</button>
        <button onClick={onNewCourse}>New course (N)</button>
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

// Course dials edit riverLayout and apply to the next course; water and duck
// dials are live.
function RacePanel({ controller, paused, onTogglePause, colorMode, onColorMode, style, onStyle }: PanelProps) {
  const [, rerender] = useReducer((x: number) => x + 1, 0)
  const [duckScale, setDuckScale] = useState(1)
  const update = (fn: () => void) => {
    fn()
    rerender()
  }
  const pending = JSON.stringify(riverLayout) !== courseState.built
  const length = courseLength(riverLayout.pieces, riverLayout.width / 2)
  const overlap = piecesOverlap(riverLayout.pieces, riverLayout.width)
  const dice = (key: RandomDial) => (
    <button className={raceSettings.random[key] ? 'dice on' : 'dice'} aria-pressed={raceSettings.random[key]}
      title={raceSettings.random[key] ? 'random for each new course' : 'fixed at this value'}
      onClick={() => update(() => (raceSettings.random[key] = !raceSettings.random[key]))}>
      🎲
    </button>
  )
  const course = (label: string, key: 'spacing' | 'width' | 'depth', min: number, max: number, step: number, digits = 1) => (
    <Slider key={key} label={label} value={riverLayout[key]} min={min} max={max} step={step}
      format={(v) => v.toFixed(digits)} onChange={(v) => update(() => (riverLayout[key] = v))} />
  )
  // A random course's ingredient, with a 🎲 toggle: on, it's drawn from its range.
  const count = (label: string, key: keyof PieceCounts, max: number, step = 1) => (
    <div key={key} className="dial">
      <Slider label={label} value={raceSettings.counts[key]} min={key === 'length' ? 40 : 0} max={max} step={step}
        format={(v) => (raceSettings.random[key] ? `${RANDOM_RANGES[key].join('–')}` : v.toFixed(0))}
        onChange={(v) => update(() => {
          raceSettings.counts[key] = v
          raceSettings.random[key] = false
        })} />
      {dice(key)}
    </div>
  )
  const perFifty = (label: string, key: 'rocks' | 'movers' | 'swirls' | 'pickups') => (
    <Slider key={key} label={label} value={raceSettings[key]} min={0} max={5} step={0.5}
      format={(v) => `${v.toFixed(1)} → ${countFor(v, length, key)}`} onChange={(v) => update(() => (raceSettings[key] = v))} />
  )
  // Editing the pieces keeps them: the next course uses them as they are.
  const editPieces = (fn: (pieces: Piece[]) => void) => update(() => {
    fn(riverLayout.pieces)
    raceSettings.random.pieces = false
  })
  const dtLimit = cflDt(raceParams)

  return (
    <details className="panel" open>
      <summary>River</summary>
      <div className="buttons">
        <button onClick={onTogglePause} title="Space">
          {paused ? 'Resume' : 'Pause'}
        </button>
        <button onClick={() => controller()?.newCourse()} title="N" className={pending ? 'pending' : undefined}
          disabled={overlap && !raceSettings.random.pieces}>
          {pending ? 'New course to apply' : 'New course'}
        </button>
        <button onClick={() => controller()?.startRace()} title="S">
          Start
        </button>
      </div>

      <Section title="Race (each course)">
        <p className="note">per 50 units of river, so longer courses get more</p>
        {perFifty('Rocks', 'rocks')}
        {perFifty('Moving', 'movers')}
        <div className="row">
          <span />
          <div className="toggle kinds">
            {MOVER_KINDS.map((k) => (
              <button key={k} aria-pressed={raceSettings.moverKinds[k]} className={raceSettings.moverKinds[k] ? 'on current' : undefined}
                title="include in new courses" onClick={() => update(() => (raceSettings.moverKinds[k] = !raceSettings.moverKinds[k]))}>
                {MOVER_LABELS[k]}
              </button>
            ))}
          </div>
        </div>
        {perFifty('Swirls', 'swirls')}
        {perFifty('Power-ups', 'pickups')}
        <p className="note">swirls come and go · ⬆ Flip · ⚡ Boost · 🌌 Zero-G</p>
      </Section>

      <Section title="Course (each new course)">
        {course('Particle size', 'spacing', 0.3, 0.6, 0.05, 2)}
        <p className="note">≈{estimateParticles(riverLayout).toLocaleString()} particles · finer means more substeps too</p>
        <p className="note">🎲 = drawn at random for each new course; off = your value</p>
        <div className="dial">
          {course('Width', 'width', 8, 20, 1, 0)}
          {dice('width')}
        </div>
        {course('Depth', 'depth', 1.5, 4, 0.5)}
        <div className="dial">
          <span className="note">{raceSettings.random.pieces ? 'Random pieces, made of:' : 'Your pieces (🎲 for random)'}</span>
          {dice('pieces')}
        </div>
        {raceSettings.random.pieces && (
          <>
            {count('Length', 'length', 160, 5)}
            {count('Waterfalls', 'waterfalls', 4)}
            {count('U-turns', 'uTurns', 2)}
            {count('Bends', 'bends', 4)}
            {count('Flip zones', 'flipZones', 3)}
          </>
        )}
        <PieceList pieces={riverLayout.pieces} halfWidth={riverLayout.width / 2} onEdit={editPieces} />
        <p className="note">
          {Math.round(length)} long · start and finish straights included
        </p>
        {overlap && !raceSettings.random.pieces && (
          <p className="note warn">The river runs into itself: change a bend or add a straight.</p>
        )}
        <button className="wide" onClick={() => update(() => {
          Object.assign(riverLayout, structuredClone(defaultRiverLayout))
          raceSettings.random.pieces = false
          raceSettings.random.width = false
        })}>
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

      <Section title="Racers">
        <Slider label="Racers" value={raceView.duckCount} min={1} max={8} step={1} format={(v) => v.toFixed(0)}
          onChange={(v) => update(() => {
            raceView.duckCount = v
            controller()?.lineUp(v) // re-hangs the gate before a race
          })} />
        {racers.names.slice(0, raceView.duckCount).map((name, i) => (
          <label key={i} className="row racer">
            <span className="dot" style={{ background: duckColor(i) }} />
            <input type="text" value={name} maxLength={24} aria-label={`Duck ${i + 1} name`}
              onChange={(e) => update(() => {
                racers.names[i] = e.target.value
                saveRacers()
              })} />
          </label>
        ))}
        <div className="row">
          <span>Chore</span>
          <div className="toggle" role="radiogroup">
            {(['last', 'winner'] as const).map((who) => (
              <button key={who} role="radio" aria-checked={racers.chore === who}
                className={racers.chore === who ? 'on current' : undefined}
                onClick={() => update(() => {
                  racers.chore = who
                  saveRacers()
                })}>
                {who === 'last' ? 'Last place' : 'Winner'}
              </button>
            ))}
          </div>
        </div>
        <label className="row">
          <span>Follow</span>
          <input type="checkbox" checked={raceView.follow} onChange={(e) => update(() => (raceView.follow = e.target.checked))} />
        </label>
        <label className="row">
          <span>Ride</span>
          <select value={raceView.ride} title="C toggles; ←/→ steer, ↑ paddles while racing"
            onChange={(e) => update(() => (raceView.ride = Number(e.target.value)))}>
            <option value={-1}>Off (orbit camera)</option>
            {racers.names.slice(0, raceView.duckCount).map((_, i) => (
              <option key={i} value={i}>{duckName(i)}</option>
            ))}
          </select>
        </label>
        <div className="row">
          <span>View</span>
          <div className="toggle" role="radiogroup">
            {(['first', 'third'] as const).map((v) => (
              <button key={v} role="radio" aria-checked={raceView.rideView === v}
                className={raceView.rideView === v ? 'on current' : undefined}
                onClick={() => update(() => (raceView.rideView = v))}>
                {v === 'first' ? 'First person' : 'Third person'}
              </button>
            ))}
          </div>
        </div>
        <p className="note">the ridden duck steers with ←/→ and paddles with ↑ (a little faster than the current)</p>
      </Section>

      <Section title="Ducks">
        <div className="buttons">
          <button onClick={() => controller()?.dropDuck()} title="D">
            Add one
          </button>
          <button onClick={() => controller()?.clearDucks()}>Clear</button>
        </div>
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

const MOVER_LABELS: Record<MoverKind, string> = { gate: 'Gates', pulse: 'Pulsing', slider: 'Sliders', spinner: 'Spinners' }

const PIECE_INFO: Record<PieceKind, { icon: string; label: string }> = {
  straight: { icon: '━', label: 'Straight' },
  bend: { icon: '↷', label: 'Bend' },
  waterfall: { icon: '⤵', label: 'Waterfall' },
  flip: { icon: '⇅', label: 'Flip zone' },
}

// The course's pieces, in order from the start straight to the finish straight:
// each with its dial, move up/down and remove, plus buttons to add more.
function PieceList({ pieces, halfWidth, onEdit }: { pieces: Piece[]; halfWidth: number; onEdit: (fn: (pieces: Piece[]) => void) => void }) {
  const tightest = minRadius(halfWidth)
  const add: [string, () => Piece][] = [
    ['Straight', () => ({ kind: 'straight', length: 8 })],
    ['Bend', () => ({ kind: 'bend', angle: 90, radius: tightest + 6 })],
    ['U-turn', () => ({ kind: 'bend', angle: 180, radius: tightest })],
    ['Waterfall', () => ({ kind: 'waterfall', drop: 4 })],
    ['Flip zone', () => ({ kind: 'flip', length: 12 })],
  ]
  return (
    <div className="pieces">
      {pieces.map((p, i) => (
        <div key={i} className="piece">
          <div className="piece-head">
            <span>{PIECE_INFO[p.kind].icon} {p.kind === 'bend' && Math.abs(p.angle) === 180 ? 'U-turn' : PIECE_INFO[p.kind].label}</span>
            <button title="move up" disabled={i === 0} onClick={() => onEdit((ps) => ps.splice(i - 1, 2, ps[i], ps[i - 1]))}>↑</button>
            <button title="move down" disabled={i === pieces.length - 1} onClick={() => onEdit((ps) => ps.splice(i, 2, ps[i + 1], ps[i]))}>↓</button>
            <button title="remove" onClick={() => onEdit((ps) => ps.splice(i, 1))}>✕</button>
          </div>
          {(p.kind === 'straight' || p.kind === 'flip') && (
            <Slider label="Length" value={p.length} min={p.kind === 'flip' ? 6 : 2} max={40} step={1} format={(v) => v.toFixed(0)}
              onChange={(v) => onEdit(() => (p.length = v))} />
          )}
          {p.kind === 'waterfall' && (
            <Slider label="Drop" value={p.drop} min={1} max={8} step={0.5} format={(v) => v.toFixed(1)}
              onChange={(v) => onEdit(() => (p.drop = v))} />
          )}
          {p.kind === 'bend' && (
            <>
              <Slider label="Turn" value={p.angle} min={-180} max={180} step={15}
                format={(v) => `${v < 0 ? '↶' : '↷'} ${Math.abs(v)}°`} onChange={(v) => onEdit(() => (p.angle = v))} />
              <Slider label="Radius" value={Math.max(p.radius, tightest)} min={tightest} max={40} step={1}
                format={(v) => v.toFixed(0)} onChange={(v) => onEdit(() => (p.radius = v))} />
            </>
          )}
        </div>
      ))}
      <div className="buttons add">
        {add.map(([label, make]) => (
          <button key={label} onClick={() => onEdit((ps) => ps.push(make()))}>+ {label}</button>
        ))}
      </div>
    </div>
  )
}
