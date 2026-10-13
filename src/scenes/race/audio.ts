import type { Obstacle } from '../../sim/sph'

interface DuckVoice {
  next: number
  source: AudioBufferSourceNode | null
}

export function createRaceAudio() {
  let context: AudioContext | null = null
  let gain: GainNode | null = null
  let buffer: AudioBuffer | null = null
  let disposed = false
  let failed = false
  let resuming = false
  let warnedResume = false
  const loading = new AbortController()
  const ducks = new Map<Obstacle, DuckVoice>()
  const voices = new Set<AudioBufferSourceNode>()

  const stopVoice = (duck: DuckVoice) => {
    const source = duck.source
    if (!source) return
    duck.source = null
    voices.delete(source)
    source.onended = null
    source.stop()
    source.disconnect()
  }
  const silence = () => {
    for (const duck of ducks.values()) stopVoice(duck)
  }
  const reset = () => {
    silence()
    ducks.clear()
  }
  const fail = (message: string, error: unknown) => {
    failed = true
    silence()
    console.warn(message, error)
  }

  const load = async (audio: AudioContext) => {
    try {
      const response = await fetch(new URL('./quack1.mp3', import.meta.url), { signal: loading.signal })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      const bytes = await response.arrayBuffer()
      if (disposed) return
      const decoded = await audio.decodeAudioData(bytes)
      if (!disposed) buffer = decoded
    } catch (error) {
      if (!disposed) fail('Race quack audio could not be loaded', error)
    }
  }

  const unlock = (event: Event) => {
    if (!event.isTrusted || disposed || failed || document.hidden) return
    if (!context) {
      try {
        context = new AudioContext()
        gain = context.createGain()
        gain.gain.value = 0.18
        gain.connect(context.destination)
        context.onstatechange = () => {
          if (context?.state !== 'running') silence()
        }
        void load(context)
      } catch (error) {
        fail('Race quack audio is unavailable', error)
        return
      }
    }
    if (context.state === 'running' || resuming) return
    // Resume synchronously in the trusted gesture, never with a dummy sound.
    resuming = true
    try {
      void context.resume().catch((error: unknown) => {
        if (!disposed && !warnedResume) {
          warnedResume = true
          console.warn('Race quack audio could not be unlocked; retry on the next interaction', error)
        }
      }).finally(() => { resuming = false })
    } catch (error) {
      resuming = false
      console.warn('Race quack audio could not be unlocked', error)
    }
  }
  const visibility = () => {
    if (document.hidden) silence()
  }
  document.addEventListener('pointerdown', unlock, true)
  document.addEventListener('keydown', unlock, true)
  document.addEventListener('click', unlock, true)
  document.addEventListener('visibilitychange', visibility)

  // Audio randomness must not consume the physics/course's Math.random sequence.
  const interval = () => 3 + 6 * crypto.getRandomValues(new Uint32Array(1))[0] / 0x100000000

  return {
    silence,
    reset,
    update(time: number, bodies: Obstacle[], finishTimes: (number | null)[], active: boolean) {
      if (disposed || failed) return
      if (!active || document.hidden) {
        silence()
        return
      }
      const eligible = new Set(bodies.filter((body, i) => !body.shape && finishTimes[i] == null))
      for (const [body, duck] of ducks) {
        if (!eligible.has(body)) {
          stopVoice(duck)
          ducks.delete(body)
        }
      }
      try {
        for (const body of eligible) {
          let duck = ducks.get(body)
          if (!duck) {
            duck = { next: time + interval(), source: null }
            ducks.set(body, duck)
          }
          if (time < duck.next) continue
          duck.next = time + interval()
          // Missed slots are skipped, not queued into a burst on resume/load.
          if (!context || context.state !== 'running' || !buffer || !gain || duck.source || voices.size >= 3) continue
          const source = context.createBufferSource()
          source.buffer = buffer
          source.connect(gain)
          duck.source = source
          voices.add(source)
          source.onended = () => {
            if (duck.source === source) duck.source = null
            voices.delete(source)
            source.disconnect()
          }
          try {
            source.start()
          } catch (error) {
            duck.source = null
            voices.delete(source)
            source.onended = null
            source.disconnect()
            throw error
          }
        }
      } catch (error) {
        fail('Race quack audio playback failed', error)
      }
    },
    dispose() {
      if (disposed) return
      disposed = true
      document.removeEventListener('pointerdown', unlock, true)
      document.removeEventListener('keydown', unlock, true)
      document.removeEventListener('click', unlock, true)
      document.removeEventListener('visibilitychange', visibility)
      loading.abort()
      reset()
      buffer = null
      gain?.disconnect()
      if (context) {
        context.onstatechange = null
        void context.close().catch((error: unknown) => {
          console.warn('Race quack audio could not be closed', error)
        })
      }
    },
  }
}
