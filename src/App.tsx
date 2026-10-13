import { Suspense, useEffect, useState } from 'react'
import { Home } from './scenes/Home'
import { scenes } from './scenes/registry'

// Hash routing: bookmarkable, back button works, and no server config is needed
// on a static host.
function useHash(): string {
  const [hash, setHash] = useState(window.location.hash)
  useEffect(() => {
    const onChange = () => setHash(window.location.hash)
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return hash
}

export function App() {
  const hash = useHash()
  const scene = scenes.find((s) => s.path === hash)
  if (!scene) return <Home scenes={scenes} />
  const Scene = scene.component
  return (
    <>
      <Suspense fallback={<div className="loading">Loading…</div>}>
        <Scene />
      </Suspense>
      <a className="home-link" href="#/">
        ← Scenes
      </a>
    </>
  )
}
