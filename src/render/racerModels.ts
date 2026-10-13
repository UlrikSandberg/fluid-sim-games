import * as THREE from 'three'

export const DUCK_COLORS = [0xf2c230, 0xff8a3d, 0xff5d8f, 0x7ad151, 0x3fb7e8, 0xb07cff, 0xf5f5f5, 0xff4d4d]

// Shared visual hulls face +x; simulation geometry remains independent.
export function makeDuck(half: number[], color: number): THREE.Group {
  const [hx, hy, hz] = half
  const duck = new THREE.Group()
  const skin = new THREE.MeshLambertMaterial({ color })
  const orange = new THREE.MeshLambertMaterial({ color: 0xff8c1a })
  const black = new THREE.MeshLambertMaterial({ color: 0x111111 })
  const sphere = new THREE.SphereGeometry(1, 20, 14)
  const body = new THREE.Mesh(sphere, skin)
  body.scale.set(hx * 1.05, hy * 0.85, hz * 1.0)
  body.position.y = -0.1 * hy
  const r = 0.6 * hz
  const head = new THREE.Mesh(sphere, skin)
  head.scale.setScalar(r)
  head.position.set(0.55 * hx, 0.75 * hy + 0.3 * r, 0)
  const beak = new THREE.Mesh(new THREE.ConeGeometry(0.35 * r, 0.8 * r, 12), orange)
  beak.rotation.z = -Math.PI / 2
  beak.position.set(head.position.x + 1.1 * r, head.position.y - 0.15 * r, 0)
  const tail = new THREE.Mesh(new THREE.ConeGeometry(0.35 * hy, 0.7 * hy, 10), skin)
  tail.rotation.z = Math.PI / 2.6
  tail.position.set(-0.95 * hx, 0.3 * hy, 0)
  duck.add(body, head, beak, tail)
  for (const side of [-1, 1]) {
    const eye = new THREE.Mesh(sphere, black)
    eye.scale.setScalar(0.13 * r)
    eye.position.set(head.position.x + 0.55 * r, head.position.y + 0.3 * r, side * 0.65 * r)
    duck.add(eye)
  }
  duck.userData.half = half.join(',')
  return duck
}

export function makeBoat(half: number[], color: number, bow: number): THREE.Group {
  const [hx, hy, hz] = half
  const start = bow * hx
  const outline = new THREE.Shape()
  outline.moveTo(-hx, -hz)
  const steps = 16
  for (let i = 0; i <= steps; i++) {
    const t = i / steps
    outline.lineTo(start + t * (hx - start), -hz * (1 - t * t))
  }
  for (let i = steps - 1; i >= 0; i--) {
    const t = i / steps
    outline.lineTo(start + t * (hx - start), hz * (1 - t * t))
  }
  outline.lineTo(-hx, hz)
  outline.closePath()
  const hull = new THREE.ExtrudeGeometry(outline, { depth: 2 * hy, bevelEnabled: false })
  hull.rotateX(-Math.PI / 2)
  hull.translate(0, -hy, 0)
  const boat = new THREE.Group()
  const paint = new THREE.MeshLambertMaterial({ color })
  const wood = new THREE.MeshLambertMaterial({ color: 0xb98a55 })
  const white = new THREE.MeshLambertMaterial({ color: 0xf2f2f2 })
  boat.add(new THREE.Mesh(hull, paint))
  const deck = new THREE.Mesh(new THREE.ShapeGeometry(outline), wood)
  deck.rotation.x = -Math.PI / 2
  deck.scale.set(0.92, 0.85, 1)
  deck.position.y = hy + 0.01
  const cabin = new THREE.Mesh(new THREE.BoxGeometry(0.7 * hx, 0.9 * hy, 1.3 * hz), white)
  cabin.position.set(-0.4 * hx, hy + 0.45 * hy, 0)
  const mast = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 3 * hy, 8), wood)
  mast.position.set(0.15 * hx, hy + 1.5 * hy, 0)
  const pennant = new THREE.Shape()
  pennant.moveTo(0, 0)
  pennant.lineTo(-0.7 * hz, -0.2 * hy)
  pennant.lineTo(0, -0.4 * hy)
  const flag = new THREE.Mesh(new THREE.ShapeGeometry(pennant), new THREE.MeshLambertMaterial({ color, side: THREE.DoubleSide }))
  flag.position.set(0.15 * hx, hy + 3 * hy, 0)
  boat.add(deck, cabin, mast, flag)
  boat.userData.half = `${half.join(',')}:hull`
  return boat
}

// Models share geometry and materials internally; release each resource once.
export function disposeObject(root: THREE.Object3D) {
  const geometries = new Set<THREE.BufferGeometry>()
  const materials = new Set<THREE.Material>()
  const textures = new Set<THREE.Texture>()
  root.traverse((child) => {
    if (child instanceof THREE.Mesh || child instanceof THREE.Line || child instanceof THREE.Points) {
      geometries.add(child.geometry)
      for (const material of Array.isArray(child.material) ? child.material : [child.material]) {
        materials.add(material)
        for (const value of Object.values(material)) {
          if (value instanceof THREE.Texture) textures.add(value)
        }
      }
    }
  })
  for (const texture of textures) texture.dispose()
  for (const geometry of geometries) geometry.dispose()
  for (const material of materials) material.dispose()
}
