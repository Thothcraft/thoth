import { Html } from '@react-three/drei'
import type { ThreeEvent } from '@react-three/fiber'
import { DoubleSide } from 'three'
import { Box } from './parts'
import { SCENE_COLORS as C } from './colors'
import type { RoomDevice, RoomSensorSpec } from './types'

const FOV_COLORS: Record<string, string> = {
  radar: '#a3502e',
  camera: '#5c7a99',
  mic: '#7a8a5a',
  csi_rx: '#7a6f5c',
  csi_tx: '#7a6f5c',
}

/**
 * Sensor FOV wedge — a flat sector anchored at the sensor position,
 * yawed by `rot_y` (+ device yaw) and optionally pitched by `tilt`.
 * Contract §1.2: wedge for camera/radar; csi_* get link lines in
 * RoomScene instead of wedges.
 */
export function FovWedge({ sensor, yaw = 0, color }: {
  sensor: RoomSensorSpec
  yaw?: number
  color?: string
}) {
  const fov = ((sensor.fov_deg ?? 60) * Math.PI) / 180
  const range = sensor.range_m ?? 6
  const tilt = sensor.tilt ?? 0
  const c = color ?? FOV_COLORS[sensor.type] ?? C.overlay
  return (
    <group
      position={sensor.pos}
      rotation={[0, yaw + (sensor.rot_y ?? 0), 0]}
    >
      <group rotation={[tilt, 0, 0]}>
        {/* sector: geometry -Y maps to world +Z after the -π/2 X rotation,
            so the arc is centered on -π/2 = forward */}
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.004, 0]}>
          <circleGeometry args={[range, 40, -Math.PI / 2 - fov / 2, fov]} />
          <meshBasicMaterial color={c} transparent opacity={0.12}
                             side={DoubleSide} depthWrite={false} />
        </mesh>
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.006, 0]}>
          <ringGeometry args={[range - 0.03, range, 40, 1,
                               -Math.PI / 2 - fov / 2, fov]} />
          <meshBasicMaterial color={c} transparent opacity={0.5}
                             side={DoubleSide} depthWrite={false} />
        </mesh>
      </group>
    </group>
  )
}

/**
 * A Thoth node device — wall/table/floor enclosure plus one FOV wedge
 * per radar/camera sensor. Clickable; `selected` lifts the LED.
 */
export function DeviceModel({ device, selected = false, showFov = true,
                              showLabel = true, onPick }: {
  device: RoomDevice
  selected?: boolean
  showFov?: boolean
  showLabel?: boolean
  onPick?: (d: RoomDevice) => void
}) {
  const yaw = device.rot_y ?? 0
  const handle = (e: ThreeEvent<MouseEvent>) => {
    if (!onPick) return
    e.stopPropagation()
    onPick(device)
  }
  const tableMount = device.mount === 'table' || device.mount === 'floor'

  return (
    <group position={device.pos} rotation={[0, yaw, 0]}>
      <group onClick={handle}>
        {tableMount ? (
          // desk puck form for commodity devices
          <>
            <Box p={[0, 0.02, 0]} s={[0.16, 0.04, 0.16]} c={C.node} />
            <mesh position={[0, 0.05, 0.06]}>
              <sphereGeometry args={[0.008, 10, 8]} />
              <meshStandardMaterial color={C.nodeLed}
                emissive={C.nodeLed}
                emissiveIntensity={selected ? 3 : 1.6} />
            </mesh>
          </>
        ) : (
          // wall/ceiling enclosure with radar aperture face
          <>
            <Box p={[0, 0, -0.015]} s={[0.16, 0.2, 0.02]} c={C.metal} />
            <Box p={[0, 0, 0.03]} s={[0.13, 0.17, 0.05]} c={C.node} />
            <Box p={[0, 0.02, 0.058]} s={[0.09, 0.09, 0.008]} c="#3d3b34" />
            <mesh position={[0, -0.06, 0.058]}>
              <sphereGeometry args={[0.008, 10, 8]} />
              <meshStandardMaterial color={C.nodeLed}
                emissive={C.nodeLed}
                emissiveIntensity={selected ? 3 : 1.6} />
            </mesh>
          </>
        )}
      </group>

      {showFov && (device.sensors ?? [])
        .filter((s) => s.type !== 'csi_rx' && s.type !== 'csi_tx')
        .map((s, i) => <FovWedge key={i} sensor={s} yaw={0} />)}

      {showLabel && (
        <Html position={[0, 0.28, 0]} center zIndexRange={[30, 0]}
              wrapperClass="scene-label-wrap">
          <span className="scene-label">{device.device_id || 'node'}</span>
        </Html>
      )}
    </group>
  )
}
