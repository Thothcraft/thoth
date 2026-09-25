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

export interface RoomDevice {
  device_id: string
  pos: V3
  rot_y?: number
  mount?: Mount
  sensors?: RoomSensorSpec[]
}

export interface RoomDoc {
  format: 'room/v1' | string
  room_id?: string
  name?: string
  dims: RoomDims
  walls?: RoomWall[]
  furniture?: RoomFurniture[]
  devices?: RoomDevice[]
  updated_at?: number
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
