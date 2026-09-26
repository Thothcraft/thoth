import { useCallback, useEffect, useRef, useState } from 'react'
import type { ThreeEvent } from '@react-three/fiber'
import { get, put } from '../api'
import { RoomScene, roomOptions, roomView } from '../scene'
import { EMPTY_ROOM } from '../scene/types'
import type { RoomDevice, RoomDoc, RoomFurniture, RoomSpec, V3 } from '../scene'

const ts = (t?: number) => t ? new Date(t * 1000).toLocaleString() : '–'

type Sel = { kind: 'device' | 'furniture'; id: string } | null

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v))

/**
 * Invisible drag handles over each device/furniture item + a floor-sized
 * pointer plane that delivers world-space drag points. Dragging updates
 * the local room doc live; pointer-up commits via PUT /api/v1/room.
 */
function DragLayer({ room, setSel, onMove, onBegin, onCommit }: {
  room: RoomDoc
  setSel: (s: Sel) => void
  onMove: (kind: 'device' | 'furniture', id: string, pos: V3) => void
  onBegin: () => void
  onCommit: () => void
}) {
  const drag = useRef<{ kind: 'device' | 'furniture'; id: string;
                       off: V3; y: number } | null>(null)
  const W = room.dims.w / 2
  const D = room.dims.d / 2

  useEffect(() => {
    const up = () => {
      if (drag.current) { drag.current = null; onCommit() }
    }
    window.addEventListener('pointerup', up)
    return () => window.removeEventListener('pointerup', up)
  }, [onCommit])

  const begin = (e: ThreeEvent<PointerEvent>, kind: 'device' | 'furniture',
                 id: string, pos: V3) => {
    e.stopPropagation()
    // NOTE: no setPointerCapture — that would redirect pointermove to
    // this mesh and starve the floor drag plane.
    setSel({ kind, id })
    onBegin()
    drag.current = {
      kind, id,
      off: [pos[0] - e.point.x, 0, pos[2] - e.point.z],
      y: pos[1],
    }
  }

  const move = (e: ThreeEvent<PointerEvent>) => {
    const d = drag.current
    if (!d) return
    e.stopPropagation()
    onMove(d.kind, d.id, [
      clamp(e.point.x + d.off[0], -W + 0.1, W - 0.1),
      d.y,
      clamp(e.point.z + d.off[2], -D + 0.1, D - 0.1),
    ])
  }

  return (
    <group>
      {/* floor drag plane (invisible hit target) */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.001, 0]}
            onPointerMove={move}>
        <planeGeometry args={[500, 500]} />
        <meshBasicMaterial visible={false} />
      </mesh>

      {(room.devices ?? []).map((d, i) => {
        const id = d.device_id || `dev-${i}`
        return (
          <mesh key={`dh-${id}`} position={d.pos}
                onPointerDown={(e) => begin(e, 'device', id, d.pos)}>
            <cylinderGeometry args={[0.22, 0.22, 0.34, 16]} />
            <meshBasicMaterial visible={false} />
          </mesh>
        )
      })}
      {(room.furniture ?? []).map((f, i) => {
        const id = f.id ?? `furniture-${i}`
        const [w, h, dd] = f.dims
        return (
          <mesh key={`fh-${id}`}
                position={[f.pos[0], f.pos[1] + h / 2, f.pos[2]]}
                rotation={[0, f.rot_y ?? 0, 0]}
                onPointerDown={(e) => begin(e, 'furniture', id, f.pos)}>
            <boxGeometry args={[Math.max(w, 0.2), Math.max(h, 0.2), Math.max(dd, 0.2)]} />
            <meshBasicMaterial visible={false}
              opacity={0} transparent depthWrite={false} />
          </mesh>
        )
      })}
    </group>
  )
}

function Num({ v, onChange, step = 0.1 }: {
  v: number
  onChange: (n: number) => void
  step?: number
}) {
  return (
    <input className="num" type="number" step={step} value={v}
           onChange={(e) => onChange(parseFloat(e.target.value) || 0)} />
  )
}

export default function MetadataPage() {
  const [meta, setMeta] = useState<any>(null)
  const [room, setRoom] = useState<RoomDoc | null>(null)
  const [manual, setManual] = useState({ room_name: '', friendly_name: '', room_id: '' })
  const [sel, setSel] = useState<Sel>(null)
  const [dragging, setDragging] = useState(false)
  const [saved, setSaved] = useState('')
  const [addType, setAddType] = useState('table')
  const [newDevice, setNewDevice] = useState('')
  const [selRoom, setSelRoom] = useState('')

  const refresh = useCallback(async () => {
    const m = await get<any>('/api/v1/metadata')
    if (m.body) {
      setMeta(m.body)
      setManual(m.body.manual ?? {})
    }
    const r = await get<RoomDoc>('/api/v1/room')
    if (r.body) {
      setRoom(r.body)
      setSelRoom((prev) => prev || (r.body!.room_id ?? ''))
    }
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(async () => {
      const m = await get<any>('/api/v1/metadata')
      if (m.body) setMeta(m.body)
    }, 10000)
    return () => clearInterval(t)
  }, [refresh])

  const saveManual = async () => {
    const r = await put<any>('/api/v1/metadata', manual)
    if (r.status === 200) setSaved(`metadata saved ${ts(Date.now() / 1000)}`)
  }

  const putRoom = useCallback(async (doc: RoomDoc) => {
    const r = await put<RoomDoc>('/api/v1/room', doc)
    if (r.status === 200 && r.body) {
      setRoom(r.body)
      setSaved(`room saved ${ts(Date.now() / 1000)}`)
    }
  }, [])

  // -- room doc mutation helpers ---------------------------------------------
  const mutate = (fn: (d: RoomDoc) => void) => {
    setRoom((prev) => {
      if (!prev) return prev
      const next: RoomDoc = JSON.parse(JSON.stringify(prev))
      fn(next)
      return next
    })
  }

  /** Mutate the ACTIVE room's sections — top-level when editing the
   * primary room, rooms[] entry otherwise. Devices always mutate at the
   * top level (they carry room_id). */
  const mutateRoom = (fn: (d: RoomDoc | RoomSpec) => void) => {
    mutate((doc) => {
      const primary = doc.room_id || ''
      if (selRoom === primary) return fn(doc)
      const r = (doc.rooms ?? []).find((x) => x.room_id === selRoom)
      if (r) fn(r)
    })
  }

  // The doc as rendered for the active room.
  const view = room ? roomView(room, selRoom || (room.room_id ?? ''))
                    : null

  const selId = sel?.id ?? null

  const findItem = (kind: 'device' | 'furniture', id: string) =>
    kind === 'device'
      ? room?.devices?.find((d, i) => (d.device_id || `dev-${i}`) === id)
      : (view?.furniture ?? []).find((f, i) => (f.id ?? `furniture-${i}`) === id)

  const onMove = (kind: 'device' | 'furniture', id: string, pos: V3) => {
    setDragging(true)
    if (kind === 'device') {
      mutate((doc) => {
        const d = doc.devices?.find((x, i) => (x.device_id || `dev-${i}`) === id)
        if (d) d.pos = pos
      })
    } else {
      mutateRoom((r) => {
        const f = r.furniture?.find((x, i) => (x.id ?? `furniture-${i}`) === id)
        if (f) f.pos = pos
      })
    }
  }
  const onCommit = () => {
    setDragging(false)
    if (room) void putRoom(room)
  }

  const selDevice = sel?.kind === 'device'
    ? (findItem('device', sel.id) as RoomDevice | undefined) : undefined
  const selFurn = sel?.kind === 'furniture'
    ? (findItem('furniture', sel.id) as RoomFurniture | undefined) : undefined

  const inf = meta?.inferred ?? {}

  return (
    <div className="meta-grid">
      <div>
        <div className="card">
          <h3>Manual metadata</h3>
          <div className="field"><span>friendly name</span>
            <input value={manual.friendly_name}
                   onChange={(e) => setManual({ ...manual, friendly_name: e.target.value })} /></div>
          <div className="field"><span>room name</span>
            <input value={manual.room_name}
                   onChange={(e) => setManual({ ...manual, room_name: e.target.value })} /></div>
          <div className="field"><span>room id</span>
            <input value={manual.room_id}
                   onChange={(e) => setManual({ ...manual, room_id: e.target.value })} /></div>
          <button className="go" onClick={saveManual}>save metadata</button>
        </div>

        <div className="card">
          <h3>Inferred <small className="muted">(auto-refreshed)</small></h3>
          <table>
            <tbody>
              <tr><td className="muted">location</td>
                <td>{inf.location?.city || '–'} {inf.location?.postal_code || ''}<br />
                  <small className="muted">
                    {inf.location?.lat?.toFixed?.(3) ?? '–'},{' '}
                    {inf.location?.lon?.toFixed?.(3) ?? '–'} · {ts(inf.location?.updated_at)}
                  </small></td></tr>
              <tr><td className="muted">activity</td>
                <td><span className="pill">{inf.activity?.kind ?? 'idle'}</span>{' '}
                  {(inf.activity?.confidence ?? 0).toFixed(2)}<br />
                  <small className="muted">{ts(inf.activity?.updated_at)}</small></td></tr>
              <tr><td className="muted">battery</td>
                <td>{inf.battery?.percent == null ? '–'
                  : `${inf.battery.percent}%${inf.battery.charging ? ' (charging)' : ''}`}</td></tr>
              <tr><td className="muted">application</td>
                <td>{inf.application?.foreground || '–'}<br />
                  <small className="muted">{inf.application?.platform}</small></td></tr>
            </tbody>
          </table>
        </div>

        <div className="card">
          <h3>Room — {roomOptions(room ?? EMPTY_ROOM).find((r) => r.room_id === selRoom)?.name ?? '…'}</h3>
          <div className="field"><span>name</span>
            <input value={view?.name ?? ''}
                   onChange={(e) => mutateRoom((r) => { r.name = e.target.value })} /></div>
          <div className="field"><span>dims (w × d × h)</span>
            <div className="row">
              <Num v={view?.dims.w ?? 0} onChange={(n) => mutateRoom((d) => { d.dims.w = n })} />
              <Num v={view?.dims.d ?? 0} onChange={(n) => mutateRoom((d) => { d.dims.d = n })} />
              <Num v={view?.dims.h ?? 0} onChange={(n) => mutateRoom((d) => { d.dims.h = n })} />
            </div></div>
          <div className="row">
            <select value={addType} onChange={(e) => setAddType(e.target.value)}>
              {['sofa', 'table', 'bed', 'desk', 'shelf', 'wall'].map((t) =>
                <option key={t}>{t}</option>)}
            </select>
            <button onClick={() => mutateRoom((r) => {
              r.furniture = r.furniture ?? []
              r.furniture.push({
                id: `f-${Date.now().toString(36)}`, type: addType,
                pos: [0, 0, 0], rot_y: 0,
                dims: addType === 'bed' ? [2, 0.5, 1.6]
                  : addType === 'sofa' ? [1.8, 0.8, 0.8]
                  : addType === 'shelf' ? [1.2, 1.8, 0.35] : [1.0, 0.75, 0.8],
              })
            })}>+ furniture</button>
          </div>
          <div className="row" style={{ marginTop: 6 }}>
            <input placeholder="device id" size={12} value={newDevice}
                   onChange={(e) => setNewDevice(e.target.value)} />
            <button onClick={() => {
              if (!newDevice.trim()) return
              mutate((d) => {
                d.devices = d.devices ?? []
                d.devices.push({
                  device_id: newDevice.trim(), pos: [0, 1.2, 0],
                  rot_y: 0, mount: 'wall', room_id: selRoom,
                  sensors: [{ type: 'radar', pos: [0, 0, 0.05],
                              rot_y: 0, tilt: 0, fov_deg: 60, range_m: 6 }],
                })
              })
              setNewDevice('')
            }}>+ device</button>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="go" onClick={() => room && putRoom(room)}>
              save room</button>
            {saved && <span className="muted">{saved}</span>}
          </div>
        </div>
      </div>

      <div className="editor-pane" style={{ position: 'relative' }}>
        {room && (
          <div className="row" style={{ marginBottom: 8, flexWrap: 'wrap' }}>
            {roomOptions(room).map((r) => (
              <button key={r.room_id || 'primary'}
                      className={r.room_id === selRoom ? 'go' : ''}
                      onClick={() => { setSelRoom(r.room_id); setSel(null) }}>
                {r.name}
              </button>
            ))}
            <button className="mini" onClick={() => {
              const id = `room-${Date.now().toString(36)}`
              mutate((d) => {
                d.rooms = d.rooms ?? []
                d.rooms.push({ room_id: id, name: 'new room',
                               dims: { ...d.dims }, walls: [], furniture: [] })
              })
              setSelRoom(id)
            }}>+ room</button>
          </div>
        )}
        {room && (
          <RoomScene
            room={view ?? room}
            selectedId={selId}
            controlsEnabled={!dragging}
            onBackgroundClick={() => setSel(null)}
            onPick={(p) => setSel({
              kind: p.kind,
              id: p.kind === 'device'
                ? (p.item as RoomDevice).device_id
                : ((p.item as RoomFurniture).id ?? ''),
            })}
            overlays={view && (
              <DragLayer room={view} setSel={setSel}
                         onMove={onMove} onCommit={onCommit}
                         onBegin={() => setDragging(true)} />
            )}
          />
        )}

        {sel && (selDevice || selFurn) && (
          <div className="card" style={{
            position: 'absolute', right: 10, top: 46, width: 250,
            background: 'rgba(23,28,38,0.94)' }}>
            <h3>{sel.kind === 'device' ? 'Device' : 'Furniture'} — {sel.id}</h3>
            {selDevice && (
              <>
                <div className="field"><span>device id</span>
                  <input value={selDevice.device_id}
                         onChange={(e) => mutate((d) => {
                           const t = d.devices?.find(
                             (x, i) => (x.device_id || `dev-${i}`) === sel.id)
                           if (t) t.device_id = e.target.value
                         })} /></div>
                <div className="field"><span>room</span>
                  <select value={selDevice.room_id ?? ''}
                          onChange={(e) => mutate((d) => {
                            const t = d.devices?.find(
                              (x, i) => (x.device_id || `dev-${i}`) === sel.id)
                            if (t) t.room_id = e.target.value
                            if (t) {
                              // keep a sane pos inside the new room's walls
                              const rid = e.target.value
                              const rr = rid === (d.room_id || '')
                                ? d.dims
                                : (d.rooms ?? []).find(
                                    (x) => x.room_id === rid)?.dims
                              if (rr && t) t.pos = [0, Math.min(t.pos[1], rr.h), 0]
                            }
                          })}>
                    {roomOptions(room!).map((r) => (
                      <option key={r.room_id || 'primary'}
                              value={r.room_id}>{r.name}</option>
                    ))}
                  </select></div>
                <div className="field"><span>mount</span>
                  <select value={selDevice.mount ?? 'wall'}
                          onChange={(e) => mutate((d) => {
                            const t = d.devices?.find(
                              (x, i) => (x.device_id || `dev-${i}`) === sel.id)
                            if (t) t.mount = e.target.value
                          })}>
                    {['wall', 'table', 'floor', 'ceiling'].map((m) =>
                      <option key={m}>{m}</option>)}
                  </select></div>
                <div className="field"><span>pos (x y z)</span>
                  <div className="row">
                    {selDevice.pos.map((v, i) => (
                      <Num key={i} v={v} onChange={(n) => mutate((d) => {
                        const t = d.devices?.find(
                          (x, j) => (x.device_id || `dev-${j}`) === sel.id)
                        if (t) t.pos[i] = n
                      })} />
                    ))}
                  </div></div>
                <div className="field"><span>yaw (deg)</span>
                  <Num v={((selDevice.rot_y ?? 0) * 180) / Math.PI}
                       step={5}
                       onChange={(n) => mutate((d) => {
                         const t = d.devices?.find(
                           (x, i) => (x.device_id || `dev-${i}`) === sel.id)
                         if (t) t.rot_y = (n * Math.PI) / 180
                       })} /></div>
                <div className="field"><span>sensors (JSON)</span>
                  <textarea
                    style={{ minHeight: 80 }}
                    defaultValue={JSON.stringify(selDevice.sensors ?? [], null, 1)}
                    onBlur={(e) => {
                      try {
                        const arr = JSON.parse(e.target.value)
                        mutate((d) => {
                          const t = d.devices?.find(
                            (x, i) => (x.device_id || `dev-${i}`) === sel.id)
                          if (t) t.sensors = arr
                        })
                      } catch { /* leave invalid JSON unsaved */ }
                    }} /></div>
              </>
            )}
            {selFurn && (
              <>
                <div className="field"><span>type</span>
                  <select value={selFurn.type}
                          onChange={(e) => mutateRoom((r) => {
                            const t = r.furniture?.find(
                              (x, i) => (x.id ?? `furniture-${i}`) === sel.id)
                            if (t) t.type = e.target.value
                          })}>
                    {['sofa', 'table', 'bed', 'desk', 'shelf', 'wall'].map((t) =>
                      <option key={t}>{t}</option>)}
                  </select></div>
                <div className="field"><span>pos (x y z)</span>
                  <div className="row">
                    {selFurn.pos.map((v, i) => (
                      <Num key={i} v={v} onChange={(n) => mutateRoom((r) => {
                        const t = r.furniture?.find(
                          (x, j) => (x.id ?? `furniture-${j}`) === sel.id)
                        if (t) t.pos[i] = n
                      })} />
                    ))}
                  </div></div>
                <div className="field"><span>dims (w h d)</span>
                  <div className="row">
                    {selFurn.dims.map((v, i) => (
                      <Num key={i} v={v} onChange={(n) => mutateRoom((r) => {
                        const t = r.furniture?.find(
                          (x, j) => (x.id ?? `furniture-${j}`) === sel.id)
                        if (t) t.dims[i] = n
                      })} />
                    ))}
                  </div></div>
                <div className="field"><span>yaw (deg)</span>
                  <Num v={((selFurn.rot_y ?? 0) * 180) / Math.PI}
                       step={5}
                       onChange={(n) => mutateRoom((r) => {
                         const t = r.furniture?.find(
                           (x, i) => (x.id ?? `furniture-${i}`) === sel.id)
                         if (t) t.rot_y = (n * Math.PI) / 180
                       })} /></div>
              </>
            )}
            <div className="row">
              <button className="danger mini" onClick={() => {
                if (sel.kind === 'device') {
                  mutate((d) => {
                    d.devices = (d.devices ?? []).filter(
                      (x, i) => (x.device_id || `dev-${i}`) !== sel.id)
                  })
                } else {
                  mutateRoom((r) => {
                    r.furniture = (r.furniture ?? []).filter(
                      (x, i) => (x.id ?? `furniture-${i}`) !== sel.id)
                  })
                }
                setSel(null)
              }}>remove</button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
