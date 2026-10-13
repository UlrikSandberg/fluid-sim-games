import { useEffect, useId, useRef, useState } from 'react'
import type { Standing } from './loop'
import { duckColor, duckName, racers } from './racers'
import { createFinishScene, type FinishRacer } from './finishScene'

interface RaceResultsProps {
  standings: Standing[]
  onAgain: () => void
  onNewCourse: () => void
}

export function RaceResults({ standings, onAgain, onNewCourse }: RaceResultsProps) {
  const [snapshot] = useState(() => ({
    standings: standings.map((standing): FinishRacer => ({
      ...standing,
      halfSize: [...standing.halfSize],
      name: duckName(standing.index),
      color: duckColor(standing.index),
    })),
    chore: racers.chore,
  }))
  const [unavailable, setUnavailable] = useState(false)
  const sectionRef = useRef<HTMLElement>(null)
  const previousFocusRef = useRef<HTMLElement | null>(null)
  const hostRef = useRef<HTMLDivElement>(null)
  const podiumRef = useRef<HTMLDivElement>(null)
  const headingId = useId()
  const scoresId = useId()
  const winner = snapshot.standings[0]
  const chore = snapshot.chore === 'last' ? snapshot.standings.at(-1) : winner

  useEffect(() => {
    const section = sectionRef.current
    const active = document.activeElement instanceof HTMLElement ? document.activeElement : null
    if (!section?.contains(active)) previousFocusRef.current = active
    // Focus the section, not a button: the existing S/N/Space shortcuts still work.
    section?.focus({ preventScroll: true })
    return () => {
      const previous = previousFocusRef.current
      if (previous?.isConnected && (document.activeElement === document.body || section?.contains(document.activeElement))) {
        previous.focus({ preventScroll: true })
      }
    }
  }, [])

  useEffect(() => {
    if (!hostRef.current || !podiumRef.current) return
    const dispose = createFinishScene(hostRef.current, podiumRef.current, snapshot.standings)
    setUnavailable(!dispose)
    return dispose ?? undefined
  }, [snapshot])

  return (
    <section className="race-results" ref={sectionRef} tabIndex={-1} aria-labelledby={headingId}>
      <div className="race-results-canvas" ref={hostRef} aria-hidden="true" />
      <div className="race-results-stage">
        <div className="race-results-podium-region" ref={podiumRef} aria-hidden="true" />
        <div className="race-results-podium-labels" role="group" aria-label="Podium" tabIndex={0}>
          {[1, 0, 2].map((place) => {
            const racer = snapshot.standings[place]
            return (
              <div key={place} className={`race-results-podium-label place-${place + 1}`}>
                {racer && <>
                  <strong>{['1st · Gold', '2nd · Silver', '3rd · Bronze'][place]}</strong>
                  <span style={{ color: racer.color }}>{racer.name}</span>
                </>}
              </div>
            )
          })}
        </div>
        {unavailable && <p className="race-results-fallback">3D celebration unavailable — results below.</p>}
      </div>
      <div className="race-results-scores">
        <h2 id={headingId}>
          {winner ? <>🏆 <span style={{ color: winner.color }}>{winner.name}</span> wins!</> : 'Race complete'}
        </h2>
        <h3 id={scoresId}>Final standings</h3>
        <ol className="race-results-ranking" aria-labelledby={scoresId} tabIndex={0}>
          {snapshot.standings.map((racer, place) => (
            <li key={racer.index}>
              <span className="race-results-place" aria-label={`Place ${place + 1}`}>{place + 1}.</span>
              <span className="dot" style={{ background: racer.color }} aria-hidden="true" />
              <span className="race-results-name">{racer.name}</span>
              <span className="race-results-time">
                {racer.finishTime != null ? `${racer.finishTime.toFixed(1)} s` : `${Math.round(racer.progress * 100)}% (out of time)`}
              </span>
            </li>
          ))}
        </ol>
        {!winner && <p>No racers finished this race.</p>}
        {chore && <p className="race-results-chore">
          🧽 <strong style={{ color: chore.color }}>{chore.name}</strong> does the chore
          {snapshot.standings.length > 1 && (snapshot.chore === 'last' ? ' (last place)' : ' (the winner)')}.
        </p>}
        <div className="race-results-actions">
          <button type="button" onClick={onAgain}>Race again here (S)</button>
          <button type="button" onClick={onNewCourse}>New course (N)</button>
        </div>
      </div>
    </section>
  )
}
