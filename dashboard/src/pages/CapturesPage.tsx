import { useCallback, useEffect, useRef, useState } from 'react'
import { del, downloadUrl, get, isPortalViewer, post } from '../api'

const ts = (t?: number) =>
  t ? new Date(t * 1000).toLocaleString() : '–'

const dur = (c: any, now: number) => {
  const end = c.stopped_at ?? now
  const s = Math.max(0, end - (c.started_at ?? end))
  if (s < 60) return `${s.toFixed(0)}s`
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.floor(s % 60)}s`
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`
}

const sensorLabel = (s: any) => s.metadata?.name || s.id

/** Per-second coverage strip over [started_at, stopped_at|now] — shows
 * which seconds of the capture actually hold aligned data, plus a
 * predictions overlay lane when samples carry them. */
function CoverageStrip({ cap, now, height = 12 }: {
  cap: any; now: number; height?: number
}) {
  const start = cap.started_at ?? now
  const end = cap.stopped_at ?? now
  const span = Math.max(1, end - start)
  const buckets = Math.min(60, Math.ceil(span))
  const secs = cap.seconds ?? {}
  const data = new Set<number>()
  let sensors = new Set<string>()
  for (const k of Object.keys(secs)) {
    const b = Math.floor(((Number(k) - start) / span) * buckets)
    if (b >= 0 && b < buckets) {
      data.add(b)
      for (const sid of Object.keys(secs[k] ?? {})) sensors.add(sid)
    }
  }
  sensors = new Set(sensors)   // keep ordering stable for the tooltip
  const pct = Math.round((data.size / buckets) * 100)
  return (
    <div title={`${data.size}/${buckets} buckets with data · ` +
               `${[...sensors].join(', ') || 'no samples'}`}
         style={{ display: 'flex', gap: 1, height, marginTop: 4 }}>
      {Array.from({ length: buckets }, (_, i) => (
        <div key={i} style={{
          flex: 1, borderRadius: 1,
          background: data.has(i) ? 'var(--ok)' : '#2a3348',
        }} />
      ))}
      <span className="muted" style={{ fontSize: 10, marginLeft: 6,
                                       lineHeight: `${height}px` }}>
        {pct}%
      </span>
    </div>
  )
}

export default function CapturesPage() {
  const [sensors, setSensors] = useState<any[]>([])
  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [captures, setCaptures] = useState<any[]>([])
  const [capId, setCapId] = useState('')
  const [label, setLabel] = useState('')
  const [msg, setMsg] = useState('')
  const [busy, setBusy] = useState(false)
  const tickRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const [, force] = useState(0)   // drives the live duration column

  const refresh = useCallback(async () => {
    const sn = await get<any>('/api/sensors')
    setSensors(sn.body?.sensors ?? [])
    const r = await get<any>('/api/captures')
    setCaptures(r.body?.captures ?? [])
  }, [])

  useEffect(() => { void refresh() }, [refresh])

  // While a capture is active, poll every 3s for counts + ticking duration.
  const hasActive = captures.some((c) => c.state === 'active')
  useEffect(() => {
    if (tickRef.current) clearInterval(tickRef.current)
    tickRef.current = hasActive
      ? setInterval(() => { void refresh(); force((n) => n + 1) }, 3000)
      : null
    return () => { if (tickRef.current) clearInterval(tickRef.current) }
  }, [hasActive, refresh])

  const startCapture = async () => {
    setBusy(true)
    const ids = [...checked]
    const r = await post<any>('/api/captures/start',
      { sensors: ids.length ? ids : null })
    setBusy(false)
    if (r.status === 200 && r.body?.id) setCapId(r.body.id)
    else setMsg(r.body?.error ?? 'start failed')
    refresh()
  }
  const stopCapture = async (id: string) => {
    await post('/api/captures/stop', { capture_id: id })
    refresh()
  }
  const deleteCapture = async (id: string) => {
    if (!confirm(`delete capture ${id}?`)) return
    const r = await del<any>(`/api/captures/${id}`)
    if (r.status !== 200) setMsg(r.body?.error ?? 'delete failed')
    else if (capId === id) setCapId('')
    refresh()
  }
  const addLabel = async () => {
    if (!capId || !label) return
    await post(`/api/captures/${capId}/label`, { label })
    setLabel('')
    refresh()
  }
  const autoLabel = async () => {
    if (!capId) return
    await post(`/api/captures/${capId}/autolabel`, {})
    refresh()
  }
  const clearLabels = async () => {
    if (!capId) return
    await post(`/api/captures/${capId}/clear-labels`, {})
    refresh()
  }

  const sel = captures.find((c) => c.id === capId)
  const now = Date.now() / 1000
  const nameOf = (sid: string) =>
    sensors.find((s) => s.id === sid)?.metadata?.name || sid

  return (
    <>
      <div className="card">
        <h3>New capture</h3>
        <div className="chips">
          {sensors.map((s) => (
            <button key={s.id}
                    className={`chip ${checked.has(s.id) ? 'on' : ''}`}
                    onClick={() => {
                      const next = new Set(checked)
                      if (next.has(s.id)) next.delete(s.id)
                      else next.add(s.id)
                      setChecked(next)
                    }}>
              {sensorLabel(s)}{' '}
              <span className="muted">{s.type}</span>
            </button>
          ))}
          {!sensors.length && <span className="muted">no sensors</span>}
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="go" disabled={busy} onClick={startCapture}>
            ● record
          </button>
          <span className="muted">
            {checked.size
              ? `${checked.size} sensor${checked.size > 1 ? 's' : ''} selected`
              : 'no selection = all sensors'}
          </span>
        </div>
      </div>

      <div className="card">
        <h3>Captures {hasActive && <span className="pill live">● recording</span>}</h3>
        {!captures.length && (
          <span className="muted">no captures yet — pick sensors above and record.</span>
        )}
        {captures.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>id</th><th>state</th><th>started</th><th>duration</th>
              <th>per-sensor samples</th><th>labels</th><th />
            </tr>
          </thead>
          <tbody>
            {[...captures].reverse().map((c) => {
              const counts = Object.entries(c.sample_counts ?? {})
                .map(([sid, n]) => `${nameOf(sid)}: ${n}`)
                .join(' · ') || '0'
              const labels = (c.labels ?? []).map((l: any) =>
                `${l.label}/${l.source}`).join(', ') || '–'
              return (
                <tr key={c.id} className={c.id === capId ? 'sel' : ''}>
                  <td><a href="#" onClick={(e) => {
                    e.preventDefault(); setCapId(c.id) }}>{c.id}</a></td>
                  <td><span className={`pill ${c.state === 'active' ? 'live' : ''}`}>
                    {c.state}</span></td>
                  <td>{ts(c.started_at)}</td>
                  <td>{dur(c, now)}</td>
                  <td><small>{counts}</small>
                    <CoverageStrip cap={c} now={now} height={8} /></td>
                  <td><small>{labels}</small></td>
                  <td className="row-cells">
                    {c.state === 'active' && (
                      <button className="mini rec"
                              onClick={() => stopCapture(c.id)}>■ stop</button>
                    )}
                    {/* zip download is local-only — relay mode returns JSON, not bytes */}
                    {!isPortalViewer && (
                      <button className="mini"
                              onClick={() => window.open(downloadUrl(`/api/captures/${c.id}/download`))}>
                        zip
                      </button>
                    )}
                    <button className="mini danger"
                            onClick={() => deleteCapture(c.id)}>delete</button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        )}
      </div>

      {sel && (
        <div className="card">
          <h3>Labels — {sel.id}</h3>
          <div style={{ marginBottom: 10 }}>
            <span className="stat">state <b>{sel.state}</b></span>
            <span className="stat">duration <b>{dur(sel, now)}</b></span>
            <span className="stat">sensors <b>
              {(sel.sensors ?? []).map(nameOf).join(', ') || '–'}</b></span>
            <span className="stat">aligned seconds <b>
              {Object.keys(sel.seconds ?? {}).length}</b></span>
          </div>
          <CoverageStrip cap={sel} now={now} height={14} />
          <div className="row">
            <input placeholder="label text" size={22} value={label}
                   onChange={(e) => setLabel(e.target.value)}
                   onKeyDown={(e) => e.key === 'Enter' && addLabel()} />
            <button onClick={addLabel} disabled={!label}>label</button>
            <button onClick={autoLabel}>auto-label from predictions</button>
            <button className="danger" onClick={clearLabels}>clear labels</button>
          </div>
          {(sel.labels ?? []).length > 0 && (
            <pre style={{ maxHeight: 140, marginTop: 10 }}>
              {sel.labels.map((l: any) =>
                `${l.label}  [${l.source}${l.confidence != null
                  ? ` ${(l.confidence * 100).toFixed(0)}%` : ''}]` +
                `  ${ts(l.start)}–${ts(l.end)}\n`).join('')}
            </pre>
          )}
        </div>
      )}
      {msg && <div className="muted" style={{ marginTop: 6 }}>{msg}</div>}
    </>
  )
}
