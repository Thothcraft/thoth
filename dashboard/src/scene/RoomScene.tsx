import { RoomCanvas } from './RoomCanvas'
import { RoomShell } from './RoomShell'
import { Furniture } from './Furniture'
import { DeviceModel } from './DeviceModel'
import { CsiLinks } from './CsiLinks'
import { EMPTY_ROOM } from './types'
import type { RoomDevice, RoomDoc, RoomFurniture } from './types'

export interface RoomScenePick {
  kind: 'device' | 'furniture'
  item: RoomDevice | RoomFurniture
}

/**
 * The shared open-roof room visualization (CONTRACT §5): floor, walls,
 * furniture, devices with per-sensor FOV wedges, and CSI links.
 *
 * Both dashboards embed this: the node dashboard hits the local API for
 * the room doc; the portal hits the Brain relay. Same prop surface.
 *
 * `overlays` renders caller-supplied decoration (live markers, labels);
 * `onPick` reports clicked devices/furniture to the host page.
 */
export function RoomScene({ room, overlays, onPick, selectedId, controls,
                            controlsEnabled, onBackgroundClick, className }: {
  room?: RoomDoc | null
  overlays?: React.ReactNode
  onPick?: (pick: RoomScenePick) => void
  selectedId?: string | null
  controls?: boolean
  controlsEnabled?: boolean
  onBackgroundClick?: () => void
  className?: string
}) {
  const doc = room ?? EMPTY_ROOM
  return (
    <div className={className} style={{ position: 'relative', width: '100%', height: '100%' }}>
      <RoomCanvas dims={doc.dims} controls={controls}
                  controlsEnabled={controlsEnabled}
                  onBackgroundClick={onBackgroundClick}>
        <RoomShell dims={doc.dims} walls={doc.walls} />
        <Furniture
          items={doc.furniture}
          selectedId={selectedId}
          onPick={onPick ? (f) => onPick({ kind: 'furniture', item: f }) : undefined}
        />
        {(doc.devices ?? []).map((d, i) => (
          <DeviceModel
            key={d.device_id || `dev-${i}`}
            device={d}
            selected={d.device_id === selectedId}
            onPick={onPick ? (dev) => onPick({ kind: 'device', item: dev }) : undefined}
          />
        ))}
        <CsiLinks devices={doc.devices} />
        {overlays}
      </RoomCanvas>
    </div>
  )
}
