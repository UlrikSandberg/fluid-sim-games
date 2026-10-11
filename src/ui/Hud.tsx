import type { Ref } from 'react'

// Static shell; the loop writes text into it directly via the ref.
export function Hud({ ref }: { ref: Ref<HTMLDivElement> }) {
  return <div ref={ref} className="hud" />
}
