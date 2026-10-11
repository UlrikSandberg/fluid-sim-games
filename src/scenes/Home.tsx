import type { SceneInfo } from './registry'

export function Home({ scenes }: { scenes: SceneInfo[] }) {
  return (
    <main className="home">
      <h1>Duck Game</h1>
      <p className="home-subtitle">Fluid simulation playground. Pick a scene.</p>
      <div className="scene-cards">
        {scenes.map((s) => (
          <a key={s.path} className={`scene-card ${s.status}`} href={s.path}>
            <span className="scene-status">{s.status === 'ready' ? 'Ready' : 'Planned'}</span>
            <h2>{s.title}</h2>
            <p>{s.description}</p>
          </a>
        ))}
      </div>
    </main>
  )
}
