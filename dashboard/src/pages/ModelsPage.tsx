import { useCallback, useEffect, useState } from 'react'
import { get, post } from '../api'

export default function ModelsPage() {
  const [catalog, setCatalog] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [name, setName] = useState('')
  const [proc, setProc] = useState('rule')
  const [cfg, setCfg] = useState(
    '{"rules":[{"when":"snr_mean > 4","label":"occupied"}],"else":"empty","actions":[]}')
  const [manifest, setManifest] = useState('')
  const [out, setOut] = useState('')

  const refresh = useCallback(async () => {
    const c = await get<any>('/api/v1/model-catalog')
    setCatalog(c.body?.models ?? [])
    const r = await get<any>('/api/models')
    setModels(r.body?.models ?? [])
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const install = async () => {
    let config: unknown
    try { config = JSON.parse(cfg || '{}') } catch {
      alert('bad config JSON'); return
    }
    const r = await post<any>('/api/models/install', {
      name: name || 'model', processor: proc || 'rule', config,
    })
    setOut(JSON.stringify(r.body, null, 2))
    if (r.body?.runtime_model_id) {
      await post('/api/models/activate', {
        runtime_model_id: r.body.runtime_model_id, active: true,
      })
    }
    refresh()
  }

  const activate = async (id: string, on: boolean) => {
    await post('/api/models/activate', { runtime_model_id: id, active: on })
    refresh()
  }

  const deploy = async () => {
    let body: unknown
    try { body = JSON.parse(manifest) } catch {
      alert('bad manifest JSON'); return
    }
    const r = await post<any>('/api/deployments',
      { deployment_id: `dash-${Date.now()}`, ...(body as object) })
    setOut(JSON.stringify(r.body, null, 2))
    refresh()
  }

  return (
    <>
      <div className="card">
        <h3>Catalog</h3>
        <table>
          <thead><tr><th>name</th><th>kind</th><th>source</th><th>available</th></tr></thead>
          <tbody>
            {catalog.map((m, i) => (
              <tr key={m.model_id ?? m.name ?? i}>
                <td>{m.name || m.model_id}</td>
                <td>{m.kind || m.processor || ''}</td>
                <td className="muted">
                  {m.builtin ? 'builtin' : (m.package || m.source || 'cloud')}
                </td>
                <td><span className={`pill ${m.available === false ? 'err' : 'ok'}`}>
                  {m.available === false ? 'no' : 'yes'}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Installed</h3>
        <table>
          <thead><tr><th>runtime id</th><th>name</th><th>processor</th><th>state</th><th /></tr></thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.runtime_model_id}>
                <td className="muted">{m.runtime_model_id}</td>
                <td>{m.name}</td><td>{m.processor}</td>
                <td><span className={`pill ${m.active ? 'ok' : ''}`}>
                  {m.active ? 'active' : 'idle'}</span></td>
                <td>
                  <button className="mini"
                          onClick={() => activate(m.runtime_model_id, !m.active)}>
                    {m.active ? 'deactivate' : 'activate'}
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Install / upload</h3>
        <div className="row">
          <input placeholder="name" size={16} value={name}
                 onChange={(e) => setName(e.target.value)} />
          <input placeholder="processor (e.g. rule, pca-face-recognizer)"
                 size={34} value={proc}
                 onChange={(e) => setProc(e.target.value)} />
          <button onClick={install}>install+activate</button>
        </div>
        <div style={{ marginTop: 8 }}>
          <span className="muted">config (JSON)</span>
          <textarea value={cfg} onChange={(e) => setCfg(e.target.value)} />
        </div>
        <div style={{ marginTop: 8 }}>
          <span className="muted">or upload a whispy-model/v1 manifest (JSON)</span>
          <textarea
            placeholder='{"manifest":{"model_id":"…","version":"…","processor":{"type":"rule","config":{}},"inputs":[],"bindings":[]}}'
            value={manifest} onChange={(e) => setManifest(e.target.value)} />
          <button onClick={deploy}>deploy</button>
        </div>
        <pre>{out}</pre>
      </div>
    </>
  )
}
