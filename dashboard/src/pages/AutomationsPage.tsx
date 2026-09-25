import { useCallback, useEffect, useState } from 'react'
import { del, get, post } from '../api'

const TEMPLATES: Record<string, { trig: string; act: string }> = {
  condition: {
    trig: '{"type":"condition","when":"snr_mean > 4","labels":["occupied"],"for_s":10}',
    act: '{"type":"home_assistant","config":{"ha_url":"http://10.0.0.22:8123","ha_token":"…","entity_id":"light.gad_room_hue","service":"turn_on","data":{"rgb_color":[0,255,0]}},"cooldown_seconds":30}',
  },
  event: {
    trig: '{"type":"event","on":"label","label":"gad"}',
    act: '{"type":"lan","config":{"host":"10.0.0.88","port":5001,"token":"…","actuator":"matrix","operation":"scroll","params":{"text":"{label}"}},"cooldown_seconds":60}',
  },
  time: {
    trig: '{"type":"time","interval_s":3600}',
    act: '{"type":"webhook","config":{"url":"https://example.com/hook","body":{"text":"tick from {device_id}"}}}',
  },
}

export default function AutomationsPage() {
  const [autos, setAutos] = useState<any[]>([])
  const [name, setName] = useState('')
  const [tpl, setTpl] = useState('')
  const [trig, setTrig] = useState('')
  const [act, setAct] = useState('')
  const [out, setOut] = useState('')

  const refresh = useCallback(async () => {
    const r = await get<any>('/api/automations')
    setAutos(r.body?.automations ?? [])
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const applyTemplate = (key: string) => {
    setTpl(key)
    const t = TEMPLATES[key]
    if (t) { setTrig(t.trig); setAct(t.act) }
  }

  const create = async () => {
    let t: unknown, a: unknown
    try {
      t = JSON.parse(trig); a = JSON.parse(act)
    } catch { alert('bad JSON'); return }
    const r = await post<any>('/api/automations', {
      name: name || 'automation', trigger: t, action: a,
    })
    setOut(JSON.stringify(r.body, null, 2))
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
      </div>

      <div className="card">
        <h3>Create</h3>
        <div className="row">
          <input placeholder="name" size={22} value={name}
                 onChange={(e) => setName(e.target.value)} />
          <select value={tpl} onChange={(e) => applyTemplate(e.target.value)}>
            <option value="">— template —</option>
            <option value="condition">condition → HA light</option>
            <option value="event">event (label) → lan actuator</option>
            <option value="time">time interval → webhook</option>
          </select>
        </div>
        <div className="row" style={{ marginTop: 8, alignItems: 'stretch' }}>
          <div style={{ flex: 1 }}>
            <span className="muted">trigger (JSON)</span>
            <textarea value={trig} onChange={(e) => setTrig(e.target.value)} />
          </div>
          <div style={{ flex: 1 }}>
            <span className="muted">action (JSON)</span>
            <textarea value={act} onChange={(e) => setAct(e.target.value)} />
          </div>
        </div>
        <div style={{ marginTop: 8 }}>
          <button className="go" onClick={create}>create</button>
        </div>
        <pre>{out}</pre>
      </div>
    </>
  )
}
