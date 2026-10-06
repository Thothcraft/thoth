/**
 * room/v1 — the synced room document (plans/CONTRACT.md §1.2).
 *
 * Units are metres. The coordinate frame is centered on the room floor:
 * x across the width, z across the depth, y up. A device/sensor `pos` is
 * its center; `rot_y` is yaw in radians (0 = facing +z).
 */

export type V3 = [number, number, number]

export interface RoomDims {
  w: number
  d: number
  h: number
}

export interface RoomWall {
  /** center of the wall segment */
  p: V3
  /** size [w, h, d] */
  s: V3
}

export type FurnitureType =
  | 'sofa' | 'table' | 'bed' | 'desk' | 'shelf' | 'wall' | (string & {})

export interface RoomFurniture {
  id?: string
  type: FurnitureType
  pos: V3
  rot_y?: number
  /** [w, h, d] in metres */
  dims: V3
}

export type SensorType =
  | 'radar' | 'camera' | 'csi_rx' | 'csi_tx' | 'mic' | (string & {})

export interface RoomSensorSpec {
  type: SensorType
  /** position relative to the device's own frame */
  pos: V3
  /** yaw offset added to the device yaw (radians) */
  rot_y?: number
  /** vertical tilt, radians (positive = looking down) */
  tilt?: number
  fov_deg?: number
  range_m?: number
}

export type Mount = 'wall' | 'table' | 'floor' | 'ceiling' | (string & {})

export interface RoomSpatial {
  surveyed: boolean
  origin_enu_m: V3 | null
  heading_deg: number | null
  floor: number | null
}
export interface BuildingAnchor {
  id: string
  name: string
  anchor: { latitude: number | null; longitude: number | null; altitude_m: number | null }
}
export interface RoomDevice {
  position_uncertainty_m?: number | null
  device_id: string
  pos: V3
  rot_y?: number
  mount?: Mount
  /** which named room this device lives in; '' = the primary room */
  room_id?: string
  sensors?: RoomSensorSpec[]
}

/** A secondary named room: own geometry + furniture, devices join it via
 * `RoomDevice.room_id`. */
export interface RoomSpec {
  spatial?: RoomSpatial
  room_id: string
  name?: string
  dims: RoomDims
  walls?: RoomWall[]
  furniture?: RoomFurniture[]
}

export interface RoomDoc {
  spatial?: RoomSpatial
  building?: BuildingAnchor
  format: 'room/v1' | string
  room_id?: string
  name?: string
  dims: RoomDims
  walls?: RoomWall[]
  furniture?: RoomFurniture[]
  devices?: RoomDevice[]
  /** extra rooms beyond the primary (top-level) one */
  rooms?: RoomSpec[]
  updated_at?: number
}

/** All selectable rooms in a doc: primary first, then `rooms[]` entries. */
export function roomOptions(doc: RoomDoc):
    Array<{ room_id: string; name: string }> {
  const out = [{ room_id: doc.room_id || '',
                 name: doc.name || 'main room' }]
  for (const r of doc.rooms ?? [])
    out.push({ room_id: r.room_id, name: r.name || r.room_id })
  return out
}

/** Renderable RoomDoc for one room: that room's geometry/furniture plus
 * only the devices assigned to it (unassigned = primary room). */
export function roomView(doc: RoomDoc, roomId: string): RoomDoc {
  const primary = doc.room_id || ''
  if (roomId === primary) {
    return { ...doc,
      devices: (doc.devices ?? []).filter(
        (d) => !d.room_id || d.room_id === primary) }
  }
  const r = (doc.rooms ?? []).find((x) => x.room_id === roomId)
  if (!r) return { ...doc, devices: [] }
  return {
    ...doc,
    room_id: r.room_id,
    name: r.name || r.room_id,
    dims: r.dims,
    spatial: r.spatial,
    walls: r.walls ?? [],
    furniture: r.furniture ?? [],
    devices: (doc.devices ?? []).filter((d) => d.room_id === roomId),
  }
}

export const EMPTY_ROOM: RoomDoc = {
  format: 'room/v1',
  room_id: '',
  name: '',
  dims: { w: 6, d: 4, h: 2.6 },
  walls: [],
  furniture: [],
  devices: [],
  updated_at: 0,
}
