import { useMemo } from 'react'
import { SCENE_COLORS as C } from './colors'
import type { RoomDims, RoomWall, V3 } from './types'

/**
 * Open-roof room shell: floor plate + perimeter walls + optional
 * explicit wall segments from the room doc.
 *
 * Per CONTRACT §1.2, auto perimeter walls render at `dims.h * 0.35` so
 * the interior is always visible. Explicit `walls[]` segments carry their
 * own height and are rendered as given.
 */
export function RoomShell({ dims, walls = [], wallScale = 0.35, grid = true,
                            floor = true }: {
  dims: RoomDims
  walls?: RoomWall[]
  /** Perimeter wall height as a fraction of room height. */
  wallScale?: number
  grid?: boolean
  /** Render the floor plate + ground (disable when the host does it). */
  floor?: boolean
}) {
  const { w: W, d: D } = dims
  const h = Math.max(0.4, dims.h * wallScale)
  const t = 0.1
  const hw = W / 2
  const hd = D / 2
  const plate = 0.12

  const perimeter: { p: V3; s: V3 }[] = useMemo(() => [
    { p: [0, h / 2, -hd - t / 2], s: [W + 2 * t, h, t] },
    { p: [0, h / 2, hd + t / 2], s: [W + 2 * t, h, t] },
    { p: [-hw - t / 2, h / 2, 0], s: [t, h, D] },
    { p: [hw + t / 2, h / 2, 0], s: [t, h, D] },
  ], [W, D, h, hw, hd, t])

  return (
    <group>
      {floor && (<>
        {/* floor plate */}
        <mesh position={[0, -plate / 2, 0]} receiveShadow>
          <boxGeometry args={[W + 0.6, plate, D + 0.6]} />
          <meshStandardMaterial color={C.floorPlate} roughness={1} />
        </mesh>
        <mesh position={[0, 0.001, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[W, D]} />
          <meshStandardMaterial color={C.floor} roughness={0.95} />
        </mesh>
        {/* oversized ground plane for soft shadows */}
        <mesh position={[0, -plate - 0.002, 0]} rotation={[-Math.PI / 2, 0, 0]} receiveShadow>
          <planeGeometry args={[80, 80]} />
          <meshStandardMaterial color={C.ground} roughness={1} />
        </mesh>
      </>)}
      {grid && (
        <gridHelper
          args={[Math.max(W, D), Math.round(Math.max(W, D) / 0.5), C.grid, C.grid]}
          position={[0, 0.006, 0]}
        />
      )}

      {/* perimeter (open roof) */}
      {perimeter.map((w, i) => (
        <mesh key={`p${i}`} position={w.p} castShadow receiveShadow>
          <boxGeometry args={w.s} />
          <meshStandardMaterial color={C.wall} roughness={0.95} />
        </mesh>
      ))}
      {/* explicit interior/exterior segments from the doc */}
      {walls.map((w, i) => (
        <mesh key={`w${i}`} position={w.p} castShadow receiveShadow>
          <boxGeometry args={w.s} />
          <meshStandardMaterial color={C.wall} roughness={0.95} />
        </mesh>
      ))}
    </group>
  )
}
