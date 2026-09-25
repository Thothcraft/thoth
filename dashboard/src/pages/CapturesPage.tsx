import { useCallback, useEffect, useState } from 'react'
import { del, downloadUrl, get, isPortalViewer, post } from '../api'

const ts = (t?: number) =>
  t ? new Date(t * 1000).toLocaleTimeString() : '–'

export default function CapturesPage() {
  const [sensors, setSensors] = useState<any[]>([])
  const [checked, setChecked] = useState<Set<string>>(new Set())
  const [captures, setCaptures] = useState<any[]>([])
  const [capId, setCapId] = useState('')
  const [label, setLabel] = useState('')
  const [msg, setMsg] = useState('')

  const refresh = useCallback(async () => {
    const sn = await get<any>('/api/sensors')
    setSensors(sn.body?.sensors ?? [])
    const r = await get<any>('/api/captures')
    setCaptures(r.body?.captures ?? [])
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const startCapture = async () => {
    const ids = [...checked]
    await post('/api/captures/start', { sensors: ids.length ? ids : null })
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

  return (
    <>
      <div className="card">
        <h3>New capture</h3>
        <div className="row">
          {sensors.map((s) => (
            <label key={s.id}>
              <input
                type="checkbox" value={s.id}
                checked={checked.has(s.id)}
                onChange={(e) => {
                  const next = new Set(checked)
                  if (e.target.checked) next.add(s.id)
                  else next.delete(s.id)
                  setChecked(next)
                }}
              />{' '}{s.id}
            </label>
          ))}
        </div>
        <div className="row" style={{ marginTop: 8 }}>
          <button className="go" onClick={startCapture}>start</button>
          <span className="muted">no selection = all sensors</span>
        </div>
      </div>

      <div className="card">
        <h3>Captures</h3>
        <table>
          <thead>
            <tr><th>id</th><th>state</th><th>started</th><th>samples</th><th>labels</th><th /></tr>
          </thead>
          <tbody>
            {[...captures].reverse().map((c) => {
              const samples = Object.values(c.sample_counts ?? {})
                .reduce((a: number, b) => a + (b as number), 0)
              const labels = (c.labels ?? []).map((l: any) =>
                `${l.label}/${l.source}`).join(', ') || '–'
              return (
                <tr key={c.id} className={c.id === capId ? 'sel' : ''}>
                  <td><a href="#" onClick={(e) => { e.preventDefault(); setCapId(c.id) }}>{c.id}</a></td>
                  <td><span className={`pill ${c.state === 'active' ? 'ok' : ''}`}>{c.state}</span></td>
                  <td>{ts(c.started_at)}</td>
                  <td>{samples}</td>
                  <td>{labels}</td>
                  <td className="row-cells">
                    {c.state === 'active' && (
                      <button className="mini" onClick={() => stopCapture(c.id)}>stop</button>
                    )}
                    {/* zip download is local-only — relay mode returns JSON, not bytes */}
                    {!isPortalViewer && (
                      <button className="mini"
                              onClick={() => window.open(downloadUrl(`/api/captures/${c.id}/download`))}>
                        zip
                      </button>
                    )}
                    <button className="mini danger" onClick={() => deleteCapture(c.id)}>delete</button>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
        <div className="row" style={{ marginTop: 8 }}>
          <input placeholder="capture id" size={14} value={capId}
                 onChange={(e) => setCapId(e.target.value)} />
          <input placeholder="label text" size={18} value={label}
                 onChange={(e) => setLabel(e.target.value)}
                 onKeyDown={(e) => e.key === 'Enter' && addLabel()} />
          <button onClick={addLabel}>label</button>
          <button onClick={autoLabel}>auto-label from predictions</button>
          <button className="danger" onClick={clearLabels}>clear labels</button>
        </div>
        {msg && <div className="muted" style={{ marginTop: 6 }}>{msg}</div>}
      </div>
    </>
  )
}
