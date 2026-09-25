import type { ThreeElements } from '@react-three/fiber'
import type { V3 } from './types'

interface BoxProps extends Omit<ThreeElements['mesh'], 'position' | 'rotation'> {
  p?: V3
  s: V3
  c: string
  r?: V3
  rough?: number
  metal?: number
}

/** Rounded-corner-free box with standard material defaults. */
export function Box({ p = [0, 0, 0], s, c, r = [0, 0, 0], rough = 0.92, metal = 0, ...rest }: BoxProps) {
  return (
    <mesh position={p} rotation={r} castShadow receiveShadow {...rest}>
      <boxGeometry args={s} />
      <meshStandardMaterial color={c} roughness={rough} metalness={metal} />
    </mesh>
  )
}

interface CylProps extends Omit<ThreeElements['mesh'], 'position' | 'rotation'> {
  p?: V3
  dims: [number, number, number, number?]
  c: string
  r?: V3
  rough?: number
}

export function Cyl({ p = [0, 0, 0], dims, c, r = [0, 0, 0], rough = 0.9, ...rest }: CylProps) {
  return (
    <mesh position={p} rotation={r} castShadow receiveShadow {...rest}>
      <cylinderGeometry args={[dims[0], dims[1], dims[2], dims[3] ?? 24]} />
      <meshStandardMaterial color={c} roughness={rough} />
    </mesh>
  )
}
