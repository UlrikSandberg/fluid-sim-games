import type { ComponentType } from 'react'
import { Coming3D } from './Coming3D'
import { Sandbox2D } from './sandbox2d/Sandbox2D'

export interface SceneInfo {
  path: string // hash route, e.g. '#/2d'
  title: string
  description: string
  status: 'ready' | 'planned'
  component: ComponentType
}

// The scenes the start screen offers. Each one mounts its own renderer and loop
// and tears them down on unmount, so switching never leaks a WebGL context.
export const scenes: SceneInfo[] = [
  {
    path: '#/2d',
    title: '2D Sandbox',
    description: 'Side-view SPH tank: dam break, mouse push/pull/current, floating bodies, water rendering.',
    status: 'ready',
    component: Sandbox2D,
  },
  {
    path: '#/3d',
    title: '3D Fluid',
    description: 'The same solver with DIM = 3, a perspective camera and orbit controls.',
    status: 'planned',
    component: Coming3D,
  },
]
