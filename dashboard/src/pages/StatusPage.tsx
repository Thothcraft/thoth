import { useCallback, useEffect, useState } from 'react'
import { get } from '../api'

const ts = (t?: number) =>
  t ? new Date(t * 1000).toLocaleTimeString() : '–'

interface Pred { timestamp?: number; label?: string; confidence?: number; runtime_model_id?: string }

export default function StatusPage() {
  const [status, setStatus] = useState<any>({})
  const [device, setDevice] = useState<any>({})
  const [meta, setMeta] = useState<any>({})
  const [sensors, setSensors] = useState<any[]>([])
  const [actuators, setActuators] = useState<any[]>([])
  const [preds, setPreds] = useState<Pred[]>([])

  const refresh = useCallback(async () => {
    const [s, inf, mt, sn, ac, pr] = await Promise.all([
      get<any>('/api/status'),
      get<any>('/api/v1/device'),
      get<any>('/api/v1/metadata'),
      get<any>('/api/sensors'),
      get<any>('/api/actuators'),
      get<any>('/api/predictions?limit=12'),
    ])
    if (s.body) setStatus(s.body)
    if (inf.body) setDevice(inf.body)
    if (mt.body) setMeta(mt.body)
    setSensors(sn.body?.sensors ?? [])
    setActuators(ac.body?.actuators ?? [])
    setPreds(pr.body?.predictions ?? [])
  }, [])

  useEffect(() => {
    refresh()
    const t = setInterval(refresh, 5000)
    return () => clearInterval(t)
  }, [refresh])

  const inf = meta.inferred ?? {}
  return (
    <>
      <div className="card">
        <h3>Device</h3>
        <b>{device.name || device.device_name || status.device_name || '?'}</b>{' '}
        <span className="pill">{status.device_id || device.id || '–'}</span>
        <div className="muted" style={{ marginTop: 6 }}>
          uptime {Math.round(status.uptime_s || 0)}s ·
          {' '}{(status.sensors?.length ?? sensors.length)} sensors ·
          {' '}{status.active_models ?? '–'} active models ·
          {' '}{status.captures ?? '–'} captures
        </div>
        <div className="muted">
          {inf.location?.city || ''}
          {inf.location?.postal_code ? ` ${inf.location.postal_code}` : ''}
          {inf.battery?.percent != null
            ? ` · battery ${inf.battery.percent}%${inf.battery.charging ? ' ⚡' : ''}`
            : ''}
          {inf.application?.foreground
            ? ` · fg: ${inf.application.foreground}`
            : ''}
        </div>
      </div>

      <div className="card">
        <h3>Sensors</h3>
        <table>
          <thead><tr><th>id</th><th>type</th><th>capabilities</th></tr></thead>
          <tbody>
            {sensors.map((s) => (
              <tr key={s.id}>
                <td>{s.id}</td><td>{s.type}</td>
                <td className="muted">{(s.capabilities ?? []).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Actuators</h3>
        <table>
          <thead><tr><th>id</th><th>kind</th><th>operations</th></tr></thead>
          <tbody>
            {actuators.map((a) => (
              <tr key={a.id}>
                <td>{a.id}</td><td>{a.kind}</td>
                <td className="muted">{(a.operations ?? []).join(', ')}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Recent predictions</h3>
        <table>
          <thead><tr><th>time</th><th>label</th><th>conf</th><th>model</th></tr></thead>
          <tbody>
            {preds.slice(-12).reverse().map((p, i) => (
              <tr key={i}>
                <td>{ts(p.timestamp)}</td><td>{p.label}</td>
                <td>{(p.confidence ?? 0).toFixed(2)}</td>
                <td className="muted">{p.runtime_model_id}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
