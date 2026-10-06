import type { BuildingAnchor, RoomDoc, RoomSpatial } from './types'

export function SpatialSettings({ room, view, onBuilding, onSpatial }: {
  room: RoomDoc; view: RoomDoc
  onBuilding: (b: BuildingAnchor) => void
  onSpatial: (s: RoomSpatial) => void
}) {
  const b = room.building ?? { id: '', name: '', anchor: { latitude: null, longitude: null, altitude_m: null } }
  const s = view.spatial ?? { surveyed: false, origin_enu_m: null, heading_deg: null, floor: null }
  const field = (label: string, value: number | null, update: (v: number | null) => void, step = 'any') =>
    <label className="field" key={label}><span>{label}</span><input type="number" step={step}
      value={value ?? ''} placeholder="Not set" onChange={(e) => update(e.target.value === '' ? null : Number(e.target.value))} /></label>
  return <section aria-label="House and room anchor">
    <h3>House location</h3>
    <p className="muted">Use the same house ID and origin on every node. Coordinates remain unknown until configured.</p>
    <label className="field"><span>House ID</span><input value={b.id} onChange={(e) => onBuilding({ ...b, id: e.target.value })} /></label>
    <label className="field"><span>House name</span><input value={b.name} onChange={(e) => onBuilding({ ...b, name: e.target.value })} /></label>
    {field('Latitude', b.anchor.latitude, (v) => onBuilding({ ...b, anchor: { ...b.anchor, latitude: v } }))}
    {field('Longitude', b.anchor.longitude, (v) => onBuilding({ ...b, anchor: { ...b.anchor, longitude: v } }))}
    {field('Altitude (m, optional)', b.anchor.altitude_m, (v) => onBuilding({ ...b, anchor: { ...b.anchor, altitude_m: v } }))}
    <h3>Room anchor</h3>
    <p className="muted">Room origin is its floor centre. East, north and height are offsets from the house origin. Heading is clockwise from north toward the roomâ€™s âˆ’Z direction.</p>
    {field('Floor (ground = 0)', s.floor, (v) => onSpatial({ ...s, floor: v }), '1')}
    {field('Heading (degrees)', s.heading_deg, (v) => onSpatial({ ...s, heading_deg: v }))}
    {['East (m)', 'North (m)', 'Height (m)'].map((label, i) => field(label, s.origin_enu_m?.[i] ?? null, (v) => {
      if (v === null) { onSpatial({ ...s, origin_enu_m: null }); return }
      const origin: [number, number, number] = [...(s.origin_enu_m ?? [0, 0, 0])]
      origin[i] = v; onSpatial({ ...s, origin_enu_m: origin })
    }))}
    <label className="field"><input type="checkbox" checked={s.surveyed} onChange={(e) => onSpatial({ ...s, surveyed: e.target.checked })} />
      Measured dimensions and room anchor confirmed</label>
    <p className="muted">Save with the room below. Radio strength does not confirm floors or wall boundaries.</p>
  </section>
}
