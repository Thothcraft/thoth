import { useCallback, useEffect, useRef, useState } from 'react'
import { get } from '../api'
import { RoomScene, roomOptions, roomView } from '../scene'
import type { RoomDevice, RoomDoc } from '../scene'
import { StreamView, type Sample } from '../components/StreamView'

/** WebGL availability probe — the scene degrades to a 2D plan list. */
function webglAvailable(): boolean {
  try {
    const c = document.createElement('canvas')
    return !!(c.getContext('webgl2') || c.getContext('webgl'))
  } catch {
    return false
  }
}

interface Prediction {
  model_id?: string
  label?: string
  confidence?: number
  timestamp?: number
  [key: string]: unknown
}

/** /api/v1/context — node-local derived states/events. The field shape is
 * the node's own; we read defensively and degrade to "no context" when
 * the endpoint isn't implemented yet. */
interface NodeContext {
  states?: Array<Record<string, unknown>>
  events?: Array<Record<string, unknown>>
  [key: string]: unknown
}

const fmtTs = (t?: number) =>
  t ? new Date(t * 1000).toLocaleTimeString() : '–'

interface SensorInfo {
  id: string
  type?: string
  capabilities?: string[]
  sample_rate?: number
  metadata?: { name?: string }
}

const sensorLabel = (s: SensorInfo) => s.metadata?.name || s.id

// Camera frames are huge JPEG payloads and only the newest frame is drawn;
// chart sensors benefit from a longer retained tail.
const keepFor = (type: string) => (/cam|video|image|jpeg/i.test(type) ? 6 : 80)

const FETCH_LIMIT = 100   // bounded delta per sensor per tick
const POLL_MS = 700

interface StreamState {
  cursor: number
  samples: Sample[]
  count: number
  lastTs: number
  ema: number
  lastTick: number
  err: string
  keep: number
}

const newStream = (type: string): StreamState => ({
  cursor: 0, samples: [], count: 0, lastTs: 0,
  ema: 0, lastTick: 0, err: '', keep: keepFor(type),
})

/** Renders a lazily-expanded raw tail — JSON is only stringified while open. */
function RawTail({ samples }: { samples: Sample[] }) {
  const [open, setOpen] = useState(false)
  return (
    <details style={{ marginTop: 10 }}
             onToggle={(e) => setOpen((e.target as HTMLDetailsElement).open)}>
      <summary className="muted" style={{ cursor: 'pointer', fontSize: 12 }}>
        raw payload tail
      </summary>
      {open && (
        <pre style={{ maxHeight: 200 }}>
          {JSON.stringify(samples.slice(-8), null, 1)}
        </pre>
      )}
    </details>
  )
}

/**
 * Live: sensor list · open-roof RoomScene · per-sensor stream cards.
 * Clicking a sensor toggles its stream — several sensors can stream in
 * parallel. One timer fans out bounded /tail requests; stream state lives
 * in a ref and the page re-renders once per round, so a 134 Hz sensor
 * cannot flood React with renders.
 */
export default function LivePage() {
  const [sensors, setSensors] = useState<SensorInfo[]>([])
  const [room, setRoom] = useState<RoomDoc | null>(null)
  const [selRoom, setSelRoom] = useState('')
  const [, bump] = useState(0)
  const streamsRef = useRef(new Map<string, StreamState>())
  const sensorsRef = useRef<SensorInfo[]>([])
  const busyRef = useRef(false)
  const [predictions, setPredictions] = useState<Prediction[]>([])
  const [context, setContext] = useState<NodeContext | null>(null)
  const [ctxOnline, setCtxOnline] = useState<boolean | null>(null)
  const [webgl] = useState(webglAvailable)

  useEffect(() => {
    get<{ sensors: SensorInfo[] }>('/api/sensors').then((r) => {
      const list = r.body?.sensors ?? []
      sensorsRef.current = list
      setSensors(list)
    })
    get<RoomDoc>('/api/v1/room').then((r) => {
      if (r.body?.format === 'room/v1') {
        setRoom(r.body)
        setSelRoom((prev) => prev || (r.body!.room_id ?? ''))
      }
    })
    // Prediction + context layers poll at a slower cadence — they're
    // derived, not sample-rate data.
    const ctxPoll = async () => {
      const p = await get<{ predictions?: Prediction[] }>(
        '/api/v1/predictions?limit=12')
      setPredictions(p.body?.predictions ?? [])
      const c = await get<NodeContext>('/api/v1/context')
      setCtxOnline(c.status === 200)
      setContext(c.status === 200 ? c.body : null)
    }
    void ctxPoll()
    const t = setInterval(() => { void ctxPoll() }, 3000)
    return () => clearInterval(t)
  }, [])

  const tick = useCallback(async () => {
    if (busyRef.current) return            // never overlap fetch rounds
    busyRef.current = true
    try {
      const streams = streamsRef.current
      const jobs = [...streams].map(async ([id, st]) => {
        if (st.cursor === 0) {
          // Prime at the live edge — jump past the ring backlog.
          try {
            const p = await get<any>(
              `/api/sensors/${encodeURIComponent(id)}/tail?limit=0`)
            if (p.status === 404) {
              st.err = 'sensor gone'
              streams.delete(id)
              return
            }
            st.cursor = p.body?.cursor ?? 0
          } catch {
            st.err = 'node unreachable'
          }
          return
        }
        try {
          const r = await get<any>(
            `/api/sensors/${encodeURIComponent(id)}/tail` +
            `?cursor=${st.cursor}&limit=${FETCH_LIMIT}`)
          if (r.status === 404) {
            st.err = 'sensor gone'
            streams.delete(id)
            return
          }
          const body = r.body ?? {}
          const now = performance.now()
          const dt = st.lastTick ? (now - st.lastTick) / 1000 : 0
          const arrived = (body.samples?.length ?? 0) + (body.skipped ?? 0)
          if (dt > 0) {
            const inst = arrived / dt
            st.ema = st.ema ? st.ema * 0.6 + inst * 0.4 : inst
          }
          st.lastTick = now
          st.cursor = body.cursor ?? st.cursor
          if (body.samples?.length) {
            st.samples = [...st.samples, ...body.samples].slice(-st.keep)
            st.count += body.samples.length
            st.lastTs = body.samples.at(-1)?.timestamp ?? st.lastTs
          }
          st.err = ''
        } catch {
          st.err = 'fetch failed'
        }
      })
      await Promise.all(jobs)
    } finally {
      busyRef.current = false
    }
    bump((n) => n + 1)
  }, [])

  useEffect(() => {
    const t = setInterval(() => { void tick() }, POLL_MS)
    return () => clearInterval(t)
  }, [tick])

  const toggleStream = useCallback((sid: string) => {
    const streams = streamsRef.current
    if (streams.has(sid)) {
      streams.delete(sid)
    } else {
      const type = sensorsRef.current.find((s) => s.id === sid)?.type ?? ''
      streams.set(sid, newStream(type))
      void tick()                        // prime immediately, don't wait a tick
    }
    bump((n) => n + 1)
  }, [tick])

  // Scene highlight: the most recently toggled stream's sensor.
  const activeIds = [...streamsRef.current.keys()]
  const selSensor = activeIds.at(-1) ?? ''
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

  const streams = streamsRef.current

  // Pipeline strip liveness: observations = open streams with samples,
  // predictions = recent inference rows, context = node context endpoint.
  const obsActive = [...streams.values()].filter((s) => s.count > 0).length
  const predAge = predictions.length
    ? Date.now() / 1000 - (predictions[predictions.length - 1].timestamp ?? 0)
    : Infinity
  const ctxStates = context?.states ?? []
  const ctxEvents = context?.events ?? []

  return (
    <>
    {/* PHYSICAL → OBSERVATIONS → PREDICTIONS → CONTEXT pipeline strip —
        each stage shows whether data is flowing *right now*. */}
    <div className="pipe-strip">
      <span className="pipe-stage ok">physical world</span>
      <span className="pipe-arrow">→</span>
      <span className={`pipe-stage ${obsActive ? 'ok' : ''}`}>
        observations {obsActive ? `· ${obsActive} streaming` : '· idle'}
      </span>
      <span className="pipe-arrow">→</span>
      <span className={`pipe-stage ${predAge < 30 ? 'ok' : ''}`}>
        predictions {predAge < 30 ? `· ${predictions.length} recent` : '· quiet'}
      </span>
      <span className="pipe-arrow">→</span>
      <span className={`pipe-stage ${ctxOnline ? 'ok' : ctxOnline === false ? 'err' : ''}`}>
        context {ctxOnline === false ? '· unavailable' : ctxStates.length ? `· ${ctxStates.length} live` : '· none'}
      </span>
    </div>
    <div className="live-grid">
      <div className="pane sensor-list">
        <div className="card" style={{ flex: 1, overflow: 'auto' }}>
          <h3>Sensors — click to stream</h3>
          {sensors.map((s) => {
            const st = streams.get(s.id)
            return (
              <button key={s.id}
                      className={st ? 'on' : ''}
                      onClick={() => toggleStream(s.id)}>
                {sensorLabel(s)}
                {st && !st.err &&
                  <span className="muted"> · {st.ema.toFixed(1)}/s</span>}
                <div className="muted"><small>{s.type} · {s.id}</small></div>
              </button>
            )
          })}
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
        {webgl ? (
          <RoomScene
            room={room ? roomView(room, selRoom || (room.room_id ?? '')) : room}
            selectedId={selDeviceId}
            onPick={(p) => {
              if (p.kind === 'device') {
                const picked = p.item as RoomDevice
                const dev = room?.devices?.find(
                  (d) => d.device_id === picked.device_id)
                const st = dev?.sensors?.[0]?.type
                const match = sensors.find((s) =>
                  s.id === picked.device_id ||
                  (st && (s.type ?? '').toLowerCase() === st))
                if (match) toggleStream(match.id)
              }
            }}
          />
        ) : (
          /* Non-WebGL fallback (Part 9): 2D plan — same content, no 3D. */
          <div className="card" style={{ margin: 8, padding: 12 }}>
            <h3>Space — 2D view (WebGL unavailable)</h3>
            {(room?.devices ?? []).map((d) => (
              <div key={d.device_id} className="row" style={{ padding: '4px 0' }}>
                <span>{d.device_id}</span>
                <span className="muted">
                  [{d.pos.map((n) => n.toFixed(1)).join(', ')}]
                  {d.sensors?.length ? ` · ${d.sensors.length} sensors` : ''}
                </span>
              </div>
            ))}
            {!room?.devices?.length && (
              <span className="muted">no devices placed in the room doc</span>)}
          </div>
        )}
      </div>

      <div className="pane">
        {/* ── prediction layer (evidence, not truth) ─────────────────── */}
        <div className="card" style={{ maxHeight: 180, overflow: 'auto' }}>
          <h3>Predictions <span className="muted">evidence</span></h3>
          {predictions.length === 0 && (
            <span className="muted">no recent inference output</span>)}
          {predictions.map((p, i) => (
            <div key={i} className="row" style={{ padding: '2px 0', fontSize: 12 }}>
              <span className="muted" style={{ fontFamily: 'monospace' }}>
                {p.model_id ?? 'model'}
              </span>
              <span>{String(p.label ?? (p.payload ? 'payload' : '—'))}</span>
              {p.confidence != null && (
                <span className="pill">{Math.round(p.confidence * 100)}%</span>)}
            </div>
          ))}
        </div>

        {/* ── context layer (derived state + transitions) ────────────── */}
        <div className="card" style={{ maxHeight: 200, overflow: 'auto' }}>
          <h3>Context <span className="muted">derived · {ctxOnline === false ? 'endpoint unavailable' : 'live'}</span></h3>
          {ctxStates.length === 0 && ctxEvents.length === 0 && (
            <span className="muted">
              {ctxOnline === false
                ? 'the node does not expose /api/v1/context yet'
                : 'no active context states'}
            </span>)}
          {ctxStates.map((s, i) => (
            <div key={`s${i}`} className="row" style={{ padding: '2px 0', fontSize: 12 }}>
              <span style={{ fontFamily: 'monospace' }}>{String(s.key ?? 'state')}</span>
              <span>{String(s.value ?? s.entity_id ?? '')}</span>
              <span className="pill live">
                {s.confidence != null ? `${Math.round(Number(s.confidence) * 100)}%` : 'on'}
              </span>
            </div>
          ))}
          {ctxEvents.slice(-6).map((e, i) => (
            <div key={`e${i}`} className="row" style={{ padding: '2px 0', fontSize: 12 }}>
              <span className="muted" style={{ fontFamily: 'monospace' }}>
                {String(e.key ?? 'event')}
              </span>
              <span className="muted">
                {String(e.type ?? '')} {e.value != null ? String(e.value) : ''}
              </span>
            </div>
          ))}
        </div>

        {/* ── BLE diagnostics ─────────────────────────────────────────── */}
        {sensors.some((s) => /ble|rssi|prox/i.test(`${s.id} ${s.type ?? ''}`)) && (
          <div className="card" style={{ maxHeight: 140, overflow: 'auto' }}>
            <h3>BLE diagnostics</h3>
            {sensors
              .filter((s) => /ble|rssi|prox/i.test(`${s.id} ${s.type ?? ''}`))
              .map((s) => {
                const st = streams.get(s.id)
                return (
                  <div key={s.id} className="row" style={{ padding: '2px 0', fontSize: 12 }}>
                    <span>{sensorLabel(s)}</span>
                    <span className="muted">
                      {st && st.samples.length
                        ? `rssi ${JSON.stringify(st.samples.at(-1)?.payload ?? {})}`
                        : 'idle — click to stream'}
                    </span>
                    {st && <span className="pill live">{st.ema.toFixed(1)}/s</span>}
                  </div>
                )
              })}
          </div>
        )}

        <div className="card" style={{ flex: 1, display: 'flex',
                                      flexDirection: 'column', minHeight: 0 }}>
          <h3>Observations {streams.size > 0 && `· ${streams.size} live`}</h3>
          <div style={{ flex: 1, overflow: 'auto', minHeight: 0 }}>
            {streams.size === 0 && (
              <span className="muted">
                select sensors on the left — click again to stop
              </span>
            )}
            {[...streams].map(([id, st]) => {
              const sensor = sensors.find((s) => s.id === id)
              return (
                <div key={id} className="stream-card">
                  <div className="row" style={{ marginBottom: 6 }}>
                    <strong style={{ fontSize: 13 }}>
                      {sensor ? sensorLabel(sensor) : id}
                    </strong>
                    {st.err
                      ? <span className="pill err">{st.err}</span>
                      : <span className="pill live">● live · {st.ema.toFixed(1)}/s</span>}
                    <span className="muted" style={{ marginLeft: 'auto' }}>
                      {st.count} · last {fmtTs(st.lastTs)}
                    </span>
                    <button className="mini danger"
                            onClick={() => toggleStream(id)}>✕</button>
                  </div>
                  <StreamView type={sensor?.type ?? ''} samples={st.samples} />
                  <RawTail samples={st.samples} />
                </div>
              )
            })}
          </div>
        </div>
      </div>
    </div>
    </>
  )
}
