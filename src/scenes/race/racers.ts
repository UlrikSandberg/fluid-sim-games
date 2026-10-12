import { DUCK_COLORS } from '../../render/renderer3d'
import type { RaceEvent, Standing } from './loop'
import { PICKUP_INFO } from './features'

// Who is racing: a name per duck (duck i has DUCK_COLORS[i]) and whether the chore
// goes to the last duck or the winner. Remembered in this browser between visits
// (a convenience; storage may be unavailable, so every access is guarded).

export type ChoreGoesTo = 'last' | 'winner'

const DEFAULT_NAMES = ['Quackers', 'Waddles', 'Puddles', 'Dabble', 'Splash', 'Bubbles', 'Nugget', 'Ducky']
const STORAGE_KEY = 'duck-race:racers'

export const racers = {
  names: DEFAULT_NAMES.slice(),
  chore: 'last' as ChoreGoesTo,
}

try {
  const saved = JSON.parse(localStorage.getItem(STORAGE_KEY) ?? 'null')
  if (saved && Array.isArray(saved.names)) {
    saved.names.slice(0, DEFAULT_NAMES.length).forEach((n: unknown, i: number) => {
      if (typeof n === 'string') racers.names[i] = n
    })
  }
  if (saved?.chore === 'last' || saved?.chore === 'winner') racers.chore = saved.chore
} catch {
  // No storage (private window, blocked): defaults it is.
}

export function saveRacers() {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(racers))
  } catch {
    // Not remembered this time.
  }
}

export function duckName(index: number): string {
  return racers.names[index]?.trim() || `Duck ${index + 1}`
}

export function duckColor(index: number): string {
  return `#${DUCK_COLORS[index % DUCK_COLORS.length].toString(16).padStart(6, '0')}`
}

// Fills `el` with the live standings: place, colour dot, name, and time or progress.
// Built with textContent, since names are user input.
export function renderStandings(el: HTMLElement, standings: Standing[]) {
  el.replaceChildren(
    ...standings.map((s, place) => {
      const row = document.createElement('div')
      row.className = 'standing'
      const dot = document.createElement('span')
      dot.className = 'dot'
      dot.style.background = duckColor(s.index)
      const name = document.createElement('span')
      name.className = 'name'
      name.textContent = `${place + 1}. ${duckName(s.index)}`
      const status = document.createElement('span')
      status.className = 'status'
      status.textContent = s.finishTime != null ? `${s.finishTime.toFixed(1)} s` : `${Math.round(s.progress * 100)}%`
      row.append(dot, name, status)
      return row
    }),
  )
}

// Fills `el` with recent power-up events ("Alex grabbed ⚡ Boost"), fading with age.
export function renderEvents(el: HTMLElement, events: RaceEvent[], fadeAfter: number) {
  el.replaceChildren(
    ...events.map((e) => {
      const row = document.createElement('div')
      row.className = 'event'
      row.style.opacity = String(Math.max(0.2, 1 - e.age / fadeAfter))
      const dot = document.createElement('span')
      dot.className = 'dot'
      dot.style.background = duckColor(e.duck)
      const text = document.createElement('span')
      text.textContent = `${duckName(e.duck)} grabbed ${PICKUP_INFO[e.kind].label}`
      row.append(dot, text)
      return row
    }),
  )
}
