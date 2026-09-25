import { Suspense } from 'react'
import { Canvas } from '@react-three/fiber'
import { OrbitControls, PerspectiveCamera } from '@react-three/drei'
import { SCENE_COLORS as C } from './colors'
import type { RoomDims } from './types'

/**
 * Canvas + camera rig + lights for the room scene. The camera distance
 * scales with the room footprint; OrbitControls is on unless
 * `controls={false}` (embedding) or while an editor drag is active.
 */
export function RoomCanvas({ dims, controls = true, controlsEnabled = true,
                             children, onBackgroundClick }: {
  dims: RoomDims
  controls?: boolean
  controlsEnabled?: boolean
  children?: React.ReactNode
  onBackgroundClick?: () => void
}) {
  const span = Math.max(dims.w, dims.d)
  const dist = span * 1.35 + 3
  return (
    <Canvas
      dpr={[1, 1.5]}
      shadows
      gl={{ antialias: true, powerPreference: 'high-performance' }}
      style={{ position: 'absolute', inset: 0 }}
      onPointerMissed={onBackgroundClick}
    >
      <PerspectiveCamera makeDefault position={[dist * 0.72, dist * 0.7, dist * 0.8]} fov={34} />
      {controls && (
        <OrbitControls
          makeDefault
          enabled={controlsEnabled}
          target={[0, 0, 0]}
          maxPolarAngle={Math.PI / 2.05}
          minDistance={span * 0.5}
          maxDistance={span * 4}
        />
      )}

      <color attach="background" args={[C.floorPlate]} />
      <fog attach="fog" args={[C.floorPlate, dist * 2.4, dist * 4.2]} />
      <hemisphereLight args={['#fff8ec', '#8a8272', 0.85]} />
      <directionalLight
        position={[6, 10, 4]}
        intensity={2.1}
        color="#fff4e0"
        castShadow
        shadow-mapSize={[1024, 1024]}
        shadow-camera-left={-8}
        shadow-camera-right={8}
        shadow-camera-top={8}
        shadow-camera-bottom={-8}
        shadow-bias={-0.0004}
      />

      <Suspense fallback={null}>
        {children}
      </Suspense>
    </Canvas>
  )
}
