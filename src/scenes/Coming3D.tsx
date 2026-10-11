// Placeholder until the 3D scene exists. First step there: make the simulation's
// dimension a per-simulation runtime value instead of the module constant DIM, so
// the 2D and 3D scenes can share src/sim in one app.
export function Coming3D() {
  return (
    <main className="home">
      <h1>3D Fluid</h1>
      <p className="home-subtitle">Not built yet.</p>
      <ol className="plan">
        <li>Make the simulation dimension a runtime property, so 2D and 3D share the solver.</li>
        <li>Perspective camera, orbit controls, ray-cast mouse picking.</li>
        <li>Performance work for 3D particle counts (PBF, Web Workers or WebGPU).</li>
      </ol>
    </main>
  )
}
