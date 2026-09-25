import type { ThreeEvent } from '@react-three/fiber'
import { Box } from './parts'
import { FURNITURE_COLORS, SCENE_COLORS as C } from './colors'
import type { RoomFurniture } from './types'

/**
 * One furniture item from the room doc — an honest primitive at scale.
 * `onPick` fires on click; `hoverTint` marks the item as hovered/selected.
 */
export function FurnitureItem({ item, selected = false, onPick }: {
  item: RoomFurniture
  selected?: boolean
  onPick?: (f: RoomFurniture) => void
}) {
  const color = FURNITURE_COLORS[item.type] ?? C.fabric
  const [w, h, d] = item.dims
  const handle = (e: ThreeEvent<MouseEvent>) => {
    if (!onPick) return
    e.stopPropagation()
    onPick(item)
  }
  return (
    <group position={item.pos} rotation={[0, item.rot_y ?? 0, 0]} onClick={handle}>
      {item.type === 'bed' ? (
        <>
          <Box p={[0, h * 0.3, 0]} s={[w, h * 0.6, d]} c={C.bed} />
          <Box p={[0, h * 0.68, 0]} s={[w * 0.94, h * 0.2, d * 0.94]} c={C.blanket} />
          <Box p={[0, h * 0.7, -d * 0.44]} s={[w, h * 1.1, 0.08]} c={C.deskDark} />
        </>
      ) : item.type === 'table' || item.type === 'desk' ? (
        <>
          <Box p={[0, h - 0.035, 0]} s={[w, 0.07, d]} c={color} />
          {[[-1, -1], [1, -1], [-1, 1], [1, 1]].map(([sx, sz], i) => (
            <Box key={i}
                 p={[sx * (w / 2 - 0.05), (h - 0.07) / 2, sz * (d / 2 - 0.05)]}
                 s={[0.06, h - 0.07, 0.06]} c={C.deskDark} />
          ))}
        </>
      ) : item.type === 'sofa' ? (
        <>
          <Box p={[0, h * 0.25, 0]} s={[w, h * 0.5, d]} c={color} />
          <Box p={[0, h * 0.62, -d * 0.38]} s={[w, h * 0.75, d * 0.24]} c={C.fabricDark} />
          <Box p={[-w * 0.42, h * 0.6, 0]} s={[w * 0.16, h * 0.7, d]} c={C.fabricDark} />
          <Box p={[w * 0.42, h * 0.6, 0]} s={[w * 0.16, h * 0.7, d]} c={C.fabricDark} />
        </>
      ) : item.type === 'shelf' ? (
        <>
          <Box p={[0, h / 2, 0]} s={[w, h, d]} c={color} />
          {[0.25, 0.5, 0.75].map((f, i) => (
            <Box key={i} p={[0, h * f, d * 0.02]} s={[w * 0.92, 0.02, d * 0.92]} c={C.deskDark} />
          ))}
        </>
      ) : (
        <Box p={[0, h / 2, 0]} s={[w, h, d]} c={color} />
      )}
      {selected && (
        <mesh position={[0, 0.015, 0]} rotation={[-Math.PI / 2, 0, 0]}>
          <ringGeometry args={[Math.max(w, d) * 0.62, Math.max(w, d) * 0.62 + 0.04, 32]} />
          <meshBasicMaterial color={C.selected} transparent opacity={0.9} side={2} depthWrite={false} />
        </mesh>
      )}
    </group>
  )
}

/** Renders `furniture[]` from a room doc. */
export function Furniture({ items = [], selectedId, onPick }: {
  items?: RoomFurniture[]
  selectedId?: string | null
  onPick?: (f: RoomFurniture) => void
}) {
  return (
    <group>
      {items.map((f, i) => (
        <FurnitureItem
          key={f.id ?? `${f.type}-${i}`}
          item={f}
          selected={f.id != null && f.id === selectedId}
          onPick={onPick}
        />
      ))}
    </group>
  )
}
