/** Scene palette — warm architectural maquette, shared by both
 * dashboards and the marketing site. Mirrors src/lib/scene-config.ts. */
export const SCENE_COLORS = {
  floor: '#d9cfbc',
  floorPlate: '#c9bda6',
  ground: '#e7e1d2',
  wall: '#efe9dc',
  wallCap: '#e2dac8',
  grid: '#bfb49d',
  desk: '#b99a72',
  deskDark: '#8a6f52',
  fabric: '#b6b0a1',
  fabricDark: '#9a937f',
  bed: '#ddd6c6',
  blanket: '#a8b09a',
  metal: '#4a463e',
  node: '#2b2a26',
  nodeLed: '#c96f3f',
  overlay: '#7a6f5c',
  overlayStrong: '#a3502e',
  selected: '#a3502e',
} as const

export const FURNITURE_COLORS: Record<string, string> = {
  sofa: '#b6b0a1',
  table: '#b99a72',
  bed: '#ddd6c6',
  desk: '#8a6f52',
  shelf: '#a89878',
  wall: '#efe9dc',
}
