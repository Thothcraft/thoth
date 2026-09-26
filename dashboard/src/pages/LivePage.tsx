import { useCallback, useEffect, useRef, useState } from 'react'
import { get } from '../api'
import { RoomScene, roomOptions, roomView } from '../scene'
import type { RoomDevice, RoomDoc } from '../scene'
import { StreamView, type Sample } from '../components/StreamView'

const fmtTs = (t?: number) =>
  t ? new Date(t * 1000).toLocaleTimeString() : '–'

interface SensorInfo {
  id: string
  type?: string
  capabilities?: string[]
  sample_rate?: number
  metadata?: { name?: string }
}

const sensorLabel = (s: SensorInfo) =>
  s.metadata?.name || s.id

const MAX_KEEP = 200

/**
 * Live: sensor list · open-roof RoomScene · sample tail stream.
 * Sensors in the room doc get FOV wedges; selecting a sensor in the
 * sidebar selects its device in the scene and starts the tail.
 */
export default function LivePage() {
  const [sensors, setSensors] = useState<SensorInfo[]>([])
  const [room, setRoom] = useState<RoomDoc | null>(null)
  const [selRoom, setSelRoom] = useState('')
  const [selSensor, setSelSensor] = useState<string>('')
  const [streaming, setStreaming] = useState(false)
  const [count, setCount] = useState(0)
  const [rate, setRate] = useState(0)
  const [lastTs, setLastTs] = useState<number>(0)
  const [samples, setSamples] = useState<Sample[]>([])
  const outRef = useRef<HTMLPreElement>(null)
  const cursorRef = useRef(0)
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const rateRef = useRef<{ at: number; n: number }>({ at: 0, n: 0 })

  useEffect(() => {
    get<{ sensors: SensorInfo[] }>('/api/sensors').then((r) => {
      setSensors(r.body?.sensors ?? [])
    })
    get<RoomDoc>('/api/v1/room').then((r) => {
      if (r.body?.format === 'room/v1') {
        setRoom(r.body)
        setSelRoom((prev) => prev || (r.body!.room_id ?? ''))
      }
    })
  }, [])

  const stop = useCallback(() => {
    if (timerRef.current) clearInterval(timerRef.current)
    timerRef.current = null
    setStreaming(false)
  }, [])

  const start = useCallback((sid: string) => {
    stop()
    cursorRef.current = 0
    setCount(0)
    setRate(0)
    setSamples([])
    if (outRef.current) outRef.current.textContent = ''
    if (!sid) return
    setStreaming(true)
    timerRef.current = setInterval(async () => {
      const r = await get<{ cursor?: number; samples?: any[] }>(
        `/api/sensors/${encodeURIComponent(sid)}/tail?cursor=${cursorRef.current}`)
      if (r.status !== 200 || !r.body) return
      const list = r.body.samples ?? []
      cursorRef.current = r.body.cursor ?? cursorRef.current
      setCount((c) => c + list.length)
      const now = Date.now() / 1000
      const rr = rateRef.current
      if (list.length) {
        if (now - rr.at > 1.5) { rr.at = now; rr.n = list.length }
        else rr.n += list.length
        setRate(rr.n / Math.max(0.001, now - rr.at))
        setLastTs(list[list.length - 1].timestamp ?? 0)
      }
      const el = outRef.current
      if (list.length)
        setSamples((prev) => [...prev, ...list].slice(-MAX_KEEP))
      if (el) {
        let text = el.textContent ?? ''
        for (const s of list) {
          text +=
            `${fmtTs(s.timestamp)} ${JSON.stringify(s.payload).slice(0, 300)}\n`
        }
        if (text.length > 20000) text = text.slice(-10000)
        el.textContent = text
        el.scrollTop = el.scrollHeight
      }
    }, 700)
  }, [stop])

  useEffect(() => stop, [stop])

  const selectSensor = (sid: string) => {
    setSelSensor(sid)
    start(sid)
  }

  // Find which room device claims the selected sensor (type match —
  // room/v1 sensors are typed, not id'd).
  const selDeviceId = (() => {
    const sel = sensors.find((s) => s.id === selSensor)
    if (!sel || !room?.devices) return null
    const typeHint = (sel.type ?? '').toLowerCase()
    for (const d of room.devices) {
      if (d.device_id === selSensor) return d.device_id
      for (const s of d.sensors ?? []) {
        if (s.type === typeHint) return d.device_id
      }
    }
    return null
  })()

  return (
    <div className="live-grid">
      <div className="pane sensor-list">
        <div className="card" style={{ flex: 1, overflow: 'auto' }}>
          <h3>Sensors</h3>
          {sensors.map((s) => (
            <button key={s.id}
                    className={s.id === selSensor ? 'on' : ''}
                    onClick={() => selectSensor(s.id)}>
              {sensorLabel(s)}
              <div className="muted"><small>{s.type} · {s.id}</small></div>
            </button>
          ))}
          {!sensors.length && <span className="muted">no sensors</span>}
        </div>
      </div>

      <div className="room-pane">
        {room && roomOptions(room).length > 1 && (
          <div style={{ position: 'absolute', top: 8, left: 8, zIndex: 5,
                        display: 'flex', gap: 4, flexWrap: 'wrap' }}>
            {roomOptions(room).map((r) => (
              <button key={r.room_id || 'primary'} className="mini"
                      style={{
                        opacity: r.room_id === selRoom ? 1 : 0.55,
                        borderColor: r.room_id === selRoom
                          ? 'var(--acc)' : 'var(--line)',
                      }}
                      onClick={() => setSelRoom(r.room_id)}>
                {r.name}
              </button>
            ))}
          </div>
        )}
        <RoomScene
          room={room ? roomView(room, selRoom || (room.room_id ?? '')) : room}
          selectedId={selDeviceId}
          onPick={(p) => {
            if (p.kind === 'device') {
              // pick the first sensor of that device in the sidebar list
              const picked = p.item as RoomDevice
              const dev = room?.devices?.find(
                (d) => d.device_id === picked.device_id)
              const st = dev?.sensors?.[0]?.type
              const match = sensors.find((s) =>
                s.id === picked.device_id ||
                (st && (s.type ?? '').toLowerCase() === st))
              if (match) selectSensor(match.id)
            }
          }}
        />
      </div>

      <div className="pane">
        <div className="card" style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          <h3>Stream — {sensors.find((s) => s.id === selSensor)
            ? sensorLabel(sensors.find((s) => s.id === selSensor)!)
            : 'select a sensor'}</h3>
          <div className="row" style={{ marginBottom: 8 }}>
            <button className="go" disabled={!selSensor || streaming}
                    onClick={() => start(selSensor)}>stream</button>
            <button disabled={!streaming} onClick={stop}>stop</button>
            <span className="muted">
              {count} samples · {rate.toFixed(1)}/s · last {fmtTs(lastTs)}
            </span>
          </div>
          <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
            <StreamView
              type={sensors.find((s) => s.id === selSensor)?.type ?? ''}
              samples={samples} />
            <details style={{ marginTop: 10 }}>
              <summary className="muted" style={{ cursor: 'pointer', fontSize: 12 }}>
                raw payload tail
              </summary>
              <pre id="liveOut" ref={outRef} style={{ flex: 1, maxHeight: 200 }} />
            </details>
          </div>
        </div>
      </div>
    </div>
  )
}
