import { useMemo } from 'react'
import { Line } from '@react-three/drei'
import { Vector3 } from 'three'
import { sensorWorldPos } from './utils'
import { SCENE_COLORS as C } from './colors'
import type { RoomDevice, V3 } from './types'

/**
 * Wi-Fi CSI links — dashed arcs from every `csi_tx` sensor to every
 * `csi_rx` sensor (CONTRACT §1.2). Positions are device-pose composed.
 */
export function CsiLinks({ devices = [] }: { devices?: RoomDevice[] }) {
  const links = useMemo(() => {
    const tx: V3[] = []
    const rx: V3[] = []
    for (const d of devices) {
      for (const s of d.sensors ?? []) {
        if (s.type === 'csi_tx') tx.push(sensorWorldPos(d, s))
        if (s.type === 'csi_rx') rx.push(sensorWorldPos(d, s))
      }
    }
    return tx.flatMap((a, i) => rx.map((b, j) => [`${i}-${j}`, a, b] as const))
  }, [devices])

  return (
    <group>
      {links.map(([key, a, b]) => {
        const va = new Vector3(...a)
        const vb = new Vector3(...b)
        const mid = va.clone().lerp(vb, 0.5).add(new Vector3(0, 0.4, 0))
        return (
          <Line key={key} points={[va, mid, vb]} color={C.overlay}
                lineWidth={1} dashed dashSize={0.09} gapSize={0.07}
                transparent opacity={0.45} />
        )
      })}
    </group>
  )
}
