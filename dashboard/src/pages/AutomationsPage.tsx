import { useCallback, useEffect, useState } from 'react'
import { del, get, post } from '../api'

/**
 * Automations — guided creation (no raw JSON by default) + an advanced
 * JSON editor for power users. Trigger and action are built from plain
 * form fields and serialized on save.
 */

type TrigKind = 'condition' | 'event' | 'time'
type ActKind = 'home_assistant' | 'lan' | 'webhook' | 'notification'

interface TrigState {
  kind: TrigKind
  when: string
  labels: string
  for_s: string
  on: string
  label: string
  interval_s: string
  at: string
}

interface ActState {
  kind: ActKind
  ha_url: string
  ha_token: string
  entity_id: string
  service: string
  color: string
  host: string
  port: string
  token: string
  actuator: string
  operation: string
  params: string
  url: string
  body: string
  message: string
}

const T0: TrigState = {
  kind: 'condition', when: 'snr_mean > 4', labels: 'occupied', for_s: '10',
  on: 'label', label: '', interval_s: '3600', at: '08:00',
}
const A0: ActState = {
  kind: 'notification', ha_url: 'http://10.0.0.22:8123', ha_token: '',
  entity_id: 'light.gad_room_hue', service: 'turn_on', color: '0,255,0',
  host: '', port: '5001', token: '', actuator: '', operation: '',
  params: '{}', url: '', body: '{"text":"fired"}', message: 'automation fired',
}

function buildTrigger(t: TrigState): Record<string, any> {
  if (t.kind === 'time') {
    const out: Record<string, any> = { type: 'time' }
    const iv = parseFloat(t.interval_s)
    if (Number.isFinite(iv) && iv > 0) out.interval_s = iv
    if (t.at.trim()) out.at = [t.at.trim()]
    return out
  }
  if (t.kind === 'event') {
    return { type: 'event', on: t.on || 'label',
             label: t.label || undefined }
  }
  return {
    type: 'condition',
    when: t.when || 'snr_mean > 4',
    labels: t.labels.split(',').map((s) => s.trim()).filter(Boolean),
    for_s: parseFloat(t.for_s) || 0,
  }
}

function buildAction(a: ActState): Record<string, any> {
  if (a.kind === 'webhook') {
    let body: any = {}
    try { body = JSON.parse(a.body || '{}') } catch { /* keep empty */ }
    return { type: 'webhook', config: { url: a.url, body } }
  }
  if (a.kind === 'lan') {
    let params: any = {}
    try { params = JSON.parse(a.params || '{}') } catch { /* keep */ }
    return {
      type: 'lan',
      config: {
        host: a.host, port: parseInt(a.port) || 5001, token: a.token,
        actuator: a.actuator, operation: a.operation, params,
      },
      cooldown_seconds: 30,
    }
  }
  if (a.kind === 'home_assistant') {
    const rgb = a.color.split(',').map((s) => parseInt(s.trim()) || 0)
    return {
      type: 'home_assistant',
      config: {
        ha_url: a.ha_url, ha_token: a.ha_token,
        entity_id: a.entity_id, service: a.service || 'turn_on',
        data: { rgb_color: rgb },
      },
      cooldown_seconds: 30,
    }
  }
  return { type: 'notification', config: { message: a.message } }
}

const F = ({ label, children }: { label: string; children: React.ReactNode }) => (
  <div className="field"><span>{label}</span>{children}</div>
)

function TriggerEditor({ t, setT }: {
  t: TrigState; setT: (v: TrigState) => void
}) {
  return (
    <div className="card" style={{ flex: 1 }}>
      <h4>When — trigger</h4>
      <F label="type">
        <select value={t.kind}
                onChange={(e) => setT({ ...t, kind: e.target.value as TrigKind })}>
          <option value="condition">condition (rule over sensor data)</option>
          <option value="event">event (prediction label)</option>
          <option value="time">time (interval / daily)</option>
        </select>
      </F>
      {t.kind === 'condition' && (
        <>
          <F label="rule"><input value={t.when}
            onChange={(e) => setT({ ...t, when: e.target.value })} /></F>
          <F label="labels (comma)"><input value={t.labels}
            placeholder="occupied,present"
            onChange={(e) => setT({ ...t, labels: e.target.value })} /></F>
          <F label="hold seconds"><input type="number" value={t.for_s}
            onChange={(e) => setT({ ...t, for_s: e.target.value })} /></F>
        </>
      )}
      {t.kind === 'event' && (
        <>
          <F label="on"><select value={t.on}
            onChange={(e) => setT({ ...t, on: e.target.value })}>
            <option value="label">label appears</option>
            <option value="prediction">any prediction</option>
          </select></F>
          <F label="label"><input value={t.label} placeholder="e.g. gad"
            onChange={(e) => setT({ ...t, label: e.target.value })} /></F>
        </>
      )}
      {t.kind === 'time' && (
        <>
          <F label="every (seconds)"><input type="number" value={t.interval_s}
            onChange={(e) => setT({ ...t, interval_s: e.target.value })} /></F>
          <F label="or daily at (HH:MM)"><input value={t.at} placeholder="08:00"
            onChange={(e) => setT({ ...t, at: e.target.value })} /></F>
        </>
      )}
    </div>
  )
}

function ActionEditor({ a, setA }: {
  a: ActState; setA: (v: ActState) => void
}) {
  return (
    <div className="card" style={{ flex: 1 }}>
      <h4>Then — action</h4>
      <F label="type">
        <select value={a.kind}
                onChange={(e) => setA({ ...a, kind: e.target.value as ActKind })}>
          <option value="notification">notification</option>
          <option value="home_assistant">Home Assistant light</option>
          <option value="lan">actuator on another node</option>
          <option value="webhook">webhook</option>
        </select>
      </F>
      {a.kind === 'notification' && (
        <F label="message"><input value={a.message}
          onChange={(e) => setA({ ...a, message: e.target.value })} /></F>
      )}
      {a.kind === 'home_assistant' && (
        <>
          <F label="ha url"><input value={a.ha_url}
            onChange={(e) => setA({ ...a, ha_url: e.target.value })} /></F>
          <F label="token"><input value={a.ha_token} type="password"
            onChange={(e) => setA({ ...a, ha_token: e.target.value })} /></F>
          <F label="entity"><input value={a.entity_id} placeholder="light.xxx"
            onChange={(e) => setA({ ...a, entity_id: e.target.value })} /></F>
          <F label="service"><input value={a.service}
            onChange={(e) => setA({ ...a, service: e.target.value })} /></F>
          <F label="rgb"><input value={a.color} placeholder="0,255,0"
            onChange={(e) => setA({ ...a, color: e.target.value })} /></F>
        </>
      )}
      {a.kind === 'lan' && (
        <>
          <F label="host"><input value={a.host} placeholder="10.0.0.88"
            onChange={(e) => setA({ ...a, host: e.target.value })} /></F>
          <F label="port"><input type="number" value={a.port}
            onChange={(e) => setA({ ...a, port: e.target.value })} /></F>
          <F label="token"><input value={a.token} type="password"
            onChange={(e) => setA({ ...a, token: e.target.value })} /></F>
          <F label="actuator"><input value={a.actuator} placeholder="matrix"
            onChange={(e) => setA({ ...a, actuator: e.target.value })} /></F>
          <F label="operation"><input value={a.operation} placeholder="scroll"
            onChange={(e) => setA({ ...a, operation: e.target.value })} /></F>
          <F label="params (JSON)"><input value={a.params}
            onChange={(e) => setA({ ...a, params: e.target.value })} /></F>
        </>
      )}
      {a.kind === 'webhook' && (
        <>
          <F label="url"><input value={a.url} placeholder="https://…"
            onChange={(e) => setA({ ...a, url: e.target.value })} /></F>
          <F label="body (JSON)"><input value={a.body}
            onChange={(e) => setA({ ...a, body: e.target.value })} /></F>
        </>
      )}
    </div>
  )
}

export default function AutomationsPage() {
  const [autos, setAutos] = useState<any[]>([])
  const [name, setName] = useState('')
  const [trig, setTrig] = useState<TrigState>(T0)
  const [act, setAct] = useState<ActState>(A0)
  const [rawTrig, setRawTrig] = useState('')
  const [rawAct, setRawAct] = useState('')
  const [advanced, setAdvanced] = useState(false)
  const [out, setOut] = useState('')

  const refresh = useCallback(async () => {
    const r = await get<any>('/api/automations')
    setAutos(r.body?.automations ?? [])
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const create = async () => {
    let t: unknown
    let a: unknown
    if (advanced) {
      try {
        t = JSON.parse(rawTrig || '{}')
        a = JSON.parse(rawAct || '{}')
      } catch { alert('bad JSON'); return }
    } else {
      t = buildTrigger(trig)
      a = buildAction(act)
    }
    const r = await post<any>('/api/automations', {
      name: name || 'automation', trigger: t, action: a,
    })
    setOut(r.status >= 300 ? `error ${r.status}: ${JSON.stringify(r.body)}`
                           : 'created')
    setName('')
    refresh()
  }

  const toggle = async (id: string, enabled: boolean) => {
    await post(`/api/automations/${id}`, { enabled })
    refresh()
  }
  const remove = async (id: string) => {
    await del(`/api/automations/${id}`)
    refresh()
  }

  return (
    <>
      <div className="card">
        <h3>Automations</h3>
        <table>
          <thead><tr><th>name</th><th>trigger</th><th>action</th><th>enabled</th><th>fires</th><th /></tr></thead>
          <tbody>
            {autos.map((a) => (
              <tr key={a.id}>
                <td>{a.name}</td>
                <td className="muted">{a.trigger?.type}</td>
                <td className="muted">{a.action?.type}</td>
                <td><span className={`pill ${a.enabled ? 'ok' : ''}`}>{a.enabled ? 'on' : 'off'}</span></td>
                <td>{a.state?.fires ?? 0}</td>
                <td className="row-cells">
                  <button className="mini" onClick={() => toggle(a.id, !a.enabled)}>
                    {a.enabled ? 'disable' : 'enable'}</button>
                  <button className="mini danger" onClick={() => remove(a.id)}>delete</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <details style={{ marginTop: 8 }}>
          <summary className="muted" style={{ cursor: 'pointer', fontSize: 12 }}>
            raw automations JSON
          </summary>
          <pre>{JSON.stringify(autos, null, 2)}</pre>
        </details>
      </div>

      <div className="card">
        <h3>Create automation</h3>
        <div className="field"><span>name</span>
          <input placeholder="my automation" value={name}
                 onChange={(e) => setName(e.target.value)} /></div>
        <div className="row" style={{ alignItems: 'stretch', gap: 12 }}>
          <TriggerEditor t={trig} setT={setTrig} />
          <ActionEditor a={act} setA={setAct} />
        </div>
        <div className="row" style={{ marginTop: 10 }}>
          <button className="go" onClick={create}>create</button>
          <label className="muted" style={{ fontSize: 12, cursor: 'pointer' }}>
            <input type="checkbox" checked={advanced}
                   onChange={(e) => {
                     setAdvanced(e.target.checked)
                     if (e.target.checked) {
                       setRawTrig(JSON.stringify(buildTrigger(trig), null, 1))
                       setRawAct(JSON.stringify(buildAction(act), null, 1))
                     }
                   }} />{' '}
            advanced JSON
          </label>
          {out && <span className="muted">{out}</span>}
        </div>
        {advanced && (
          <div className="row" style={{ marginTop: 8, alignItems: 'stretch' }}>
            <div style={{ flex: 1 }}>
              <span className="muted">trigger (JSON)</span>
              <textarea value={rawTrig} onChange={(e) => setRawTrig(e.target.value)} />
            </div>
            <div style={{ flex: 1 }}>
              <span className="muted">action (JSON)</span>
              <textarea value={rawAct} onChange={(e) => setRawAct(e.target.value)} />
            </div>
          </div>
        )}
      </div>
    </>
  )
}
