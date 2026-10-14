import { lazy, type ComponentType } from 'react'

export interface SceneInfo {
  path: string // hash route, e.g. '#/2d'
  title: string
  description: string
  status: 'ready' | 'planned'
  component: ComponentType
}

// The scenes the start screen offers. Each one mounts its own renderer and loop
// and tears them down on unmount, so switching never leaks a WebGL context.
// Each is loaded only when opened, so adding a scene doesn't slow the others.
export const scenes: SceneInfo[] = [
  {
    path: '#/2d',
    title: '2D Sandbox',
    description: 'Side-view SPH tank: dam break, mouse push/pull/current, floating bodies, water rendering.',
    status: 'ready',
    component: lazy(() => import('./sandbox2d/Sandbox2D').then((m) => ({ default: m.Sandbox2D }))),
  },
  {
    path: '#/3d',
    title: '3D Sandbox',
    description: 'The same solver in a 3D tank: dam break, orbit camera, lit particles, colour modes.',
    status: 'ready',
    component: lazy(() => import('./fluid3d/Fluid3D').then((m) => ({ default: m.Fluid3D }))),
  },
  {
    path: '#/race',
    title: 'River Race',
    description: 'The game: named ducks race down a river you build from bends, U-turns, waterfalls and flip zones (or roll at random). GPU solver.',
    status: 'ready',
    component: lazy(() => import('./race/RiverRace').then((m) => ({ default: m.RiverRace }))),
  },
  {
    path: '#/ripples',
    title: 'Ripple Tank',
    description: 'Shallow-water waves on a height field: poke the pool, send plane waves through one or two slits, focus them with an underwater lens. See-through water with caustics.',
    status: 'ready',
    component: lazy(() => import('./ripples/RippleTank').then((m) => ({ default: m.RippleTank }))),
  },
  {
    path: '#/planet',
    title: 'Water Planet',
    description: 'An ocean held to a rocky planet by gravity toward its centre, weakening as 1/r². GPU solver.',
    status: 'ready',
    component: lazy(() => import('./planet/WaterPlanet').then((m) => ({ default: m.WaterPlanet }))),
  },
]
