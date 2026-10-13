import * as THREE from 'three'
import type { Standing } from './loop'
import { disposeObject, makeBoat, makeDuck } from '../../render/racerModels'

export interface FinishRacer extends Standing {
  name: string
  color: string
}

interface DisplayRacer {
  root: THREE.Group
  size: THREE.Vector3
  baseY: number
}

// The DOM podium region is also the coordinate system for the canvas, keeping the
// models aligned with their HTML labels across stacked and landscape layouts.
export function createFinishScene(
  host: HTMLElement,
  podiumRegion: HTMLElement,
  racers: readonly FinishRacer[],
): (() => void) | null {
  let renderer: THREE.WebGLRenderer
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true })
  } catch {
    return null
  }
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2))
  renderer.outputColorSpace = THREE.SRGBColorSpace
  host.appendChild(renderer.domElement)
  const scene = new THREE.Scene()
  const camera = new THREE.OrthographicCamera(0, 1, 1, 0, 0.1, 2000)
  camera.position.z = 1000
  scene.add(new THREE.HemisphereLight(0xe8f6ff, 0x283349, 2.5))
  const light = new THREE.DirectionalLight(0xffffff, 3)
  light.position.set(-200, 400, 600)
  scene.add(light)

  const models: DisplayRacer[] = racers.slice(0, 3).map((racer) => {
    const color = new THREE.Color(racer.color).getHex()
    const model = racer.vessel === 'boat'
      ? makeBoat(racer.halfSize, color, racer.bow ?? 0.2)
      : makeDuck(racer.halfSize, color)
    model.rotation.set(0.16, -0.55, 0)
    const bounds = new THREE.Box3().setFromObject(model)
    const size = bounds.getSize(new THREE.Vector3())
    const center = bounds.getCenter(new THREE.Vector3())
    model.position.set(-center.x, -bounds.min.y, -center.z)
    const root = new THREE.Group()
    root.add(model)
    scene.add(root)
    return { root, size, baseY: 0 }
  })
  const medals = [0xe5b94f, 0xaabccc, 0xbd8050]
  const blocks = racers.slice(0, 3).map((_, place) => {
    const block = new THREE.Mesh(
      new THREE.BoxGeometry(1, 1, 1),
      new THREE.MeshStandardMaterial({ color: medals[place], metalness: 0.55, roughness: 0.35 }),
    )
    block.rotation.x = 0.12
    scene.add(block)
    return block
  })
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)')
  let reduced = motion.matches
  let disposed = false
  let frame = 0
  let width = 1
  let height = 1
  let winnerAmplitude = 0
  const started = performance.now()
  let confetti: THREE.Group | null = null
  const particles: { mesh: THREE.Mesh; x: number; y: number; spin: number }[] = []
  if (!reduced && racers.length > 0) {
    confetti = new THREE.Group()
    confetti.visible = false
    const geometry = new THREE.PlaneGeometry(1, 1)
    const materials = [0xf4cb65, 0x7ecbff, 0xff8caa, 0x9ce8b3].map(
      (color) => new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide }),
    )
    for (let i = 0; i < 64; i++) {
      const mesh = new THREE.Mesh(geometry, materials[i % materials.length])
      confetti.add(mesh)
      particles.push({ mesh, x: Math.random(), y: Math.random(), spin: Math.random() * Math.PI * 2 })
    }
    scene.add(confetti)
  }

  const clearConfetti = () => {
    if (!confetti) return
    scene.remove(confetti)
    disposeObject(confetti)
    confetti = null
    particles.length = 0
  }
  const render = () => renderer.render(scene, camera)
  const resize = () => {
    if (disposed) return
    const hostRect = host.getBoundingClientRect()
    const podium = podiumRegion.getBoundingClientRect()
    width = Math.max(1, hostRect.width)
    height = Math.max(1, hostRect.height)
    renderer.setSize(width, height, false)
    camera.right = width
    camera.top = height
    camera.updateProjectionMatrix()
    const slot = podium.width / 3
    const base = height - (podium.bottom - hostRect.top)
    const podiumHeights = [0.39, 0.25, 0.17]
    const slots = [1, 0, 2] // left to right: silver, gold, bronze
    blocks.forEach((block, place) => {
      const blockHeight = podium.height * podiumHeights[place]
      const x = podium.left - hostRect.left + slot * (slots[place] + 0.5)
      block.scale.set(slot * 0.84, blockHeight, Math.min(24, slot * 0.18))
      block.position.set(x, base + blockHeight / 2, -25)
      const display = models[place]
      const scale = Math.min(
        slot * 0.77 / Math.max(display.size.x, 0.001),
        podium.height * 0.42 / Math.max(display.size.y, 0.001),
      )
      display.root.scale.setScalar(scale)
      display.baseY = base + blockHeight + 2
      display.root.position.set(x, display.baseY, 0)
    })
    winnerAmplitude = Math.min(32, podium.height * 0.09)
    render()
  }
  const tick = (now: number) => {
    frame = 0
    if (disposed || reduced) return
    const elapsed = (now - started) / 1000
    const winner = models[0]
    if (winner) {
      // One full turn during each jump, followed by a short landing pause.
      const jump = Math.min(1, (elapsed % 2.4) / 1.2)
      const spin = jump * jump * (3 - 2 * jump)
      winner.root.position.y = winner.baseY + 4 * jump * (1 - jump) * winnerAmplitude
      winner.root.rotation.y = spin * Math.PI * 2
      winner.root.rotation.z = Math.sin(jump * Math.PI * 2) * 0.08
    }
    if (confetti) {
      if (elapsed >= 2.4) {
        clearConfetti()
      } else {
        confetti.visible = true
        particles.forEach(({ mesh, x, y, spin }) => {
          mesh.position.set(
            width * (0.08 + x * 0.84) + Math.sin(elapsed * 2 + spin) * 15,
            height * (1.08 + y * 0.3 - elapsed * 0.68),
            60,
          )
          mesh.rotation.set(elapsed * 3 + spin, elapsed * 2, spin + elapsed)
          mesh.scale.set(5, 9, 1)
        })
      }
    }
    render()
    if (winner || confetti) frame = requestAnimationFrame(tick)
  }
  const onMotion = () => {
    reduced = motion.matches
    cancelAnimationFrame(frame)
    frame = 0
    // Once omitted or interrupted, the one-shot burst is never restarted.
    clearConfetti()
    if (models[0]) {
      models[0].root.position.y = models[0].baseY
      models[0].root.rotation.y = 0
      models[0].root.rotation.z = 0
    }
    render()
    if (!reduced && models.length) frame = requestAnimationFrame(tick)
  }
  motion.addEventListener('change', onMotion)
  window.addEventListener('resize', resize)
  const observer = new ResizeObserver(resize)
  observer.observe(host)
  observer.observe(podiumRegion)
  resize()
  if (!reduced && models.length) frame = requestAnimationFrame(tick)

  return () => {
    disposed = true
    cancelAnimationFrame(frame)
    observer.disconnect()
    window.removeEventListener('resize', resize)
    motion.removeEventListener('change', onMotion)
    clearConfetti()
    disposeObject(scene)
    renderer.dispose()
    renderer.forceContextLoss()
    renderer.domElement.remove()
  }
}
