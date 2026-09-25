/**
 * Shared room visualization (plans/CONTRACT.md §5) — prop-driven,
 * dependency-free except react-three. Mirrored into
 * `thoth/dashboard/src/scene/` for the node dashboard; keep imports
 * relative and self-contained so the copy stays a straight file drop.
 */
export * from './types'
export { SCENE_COLORS, FURNITURE_COLORS } from './colors'
export { Box, Cyl } from './parts'
export { RoomShell } from './RoomShell'
export { Furniture, FurnitureItem } from './Furniture'
export { DeviceModel, FovWedge } from './DeviceModel'
export { sensorWorldPos } from './utils'
export { CsiLinks } from './CsiLinks'
export { RoomCanvas } from './RoomCanvas'
export { RoomScene } from './RoomScene'
export type { RoomScenePick } from './RoomScene'
