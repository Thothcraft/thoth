import type { RoomDevice, RoomSensorSpec, V3 } from './types'

/** World-space position of a sensor (device pose ∘ sensor pos). */
export function sensorWorldPos(device: RoomDevice, s: RoomSensorSpec): V3 {
  const yaw = device.rot_y ?? 0
  const cos = Math.cos(yaw)
  const sin = Math.sin(yaw)
  const [x, y, z] = s.pos
  return [
    device.pos[0] + x * cos + z * sin,
    device.pos[1] + y,
    device.pos[2] - x * sin + z * cos,
  ]
}
