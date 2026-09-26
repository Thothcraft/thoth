import { useCallback, useEffect, useState } from 'react'
import { get, post } from '../api'

/**
 * Models — one-click presets for the common pipelines plus an advanced
 * JSON installer. Each preset installs one or more runtime models in
 * dependency order (detectors first so their detections land in the
 * window for downstream recognizers).
 */

interface Preset {
  id: string
  label: string
  blurb: string
  steps: Array<{ name: string; processor: string; config: any }>
}

const PRESETS: Preset[] = [
  {
    id: 'face-pipeline',
    label: 'Face detection + recognition',
    blurb: 'Haar detector → PCA recognizer on the camera stream',
    steps: [
      { name: 'face-detect', processor: 'opencv-haar-face', config: {} },
      { name: 'face-recognize', processor: 'pca-face-recognizer', config: {} },
    ],
  },
  {
    id: 'person-occupancy',
    label: 'Person → occupancy rules',
    blurb: 'Haar person detector, then a rule that marks the room occupied',
    steps: [
      { name: 'person-detect', processor: 'opencv-haar-person', config: {} },
      { name: 'occupancy', processor: 'rule', config: {
        rules: [{ when: 'person_detect == True', label: 'occupied' }],
        else: 'empty', actions: [] } },
    ],
  },
  {
    id: 'speech-stt',
    label: 'Speech → text',
    blurb: 'Whisper tiny.en on the microphone stream',
    steps: [
      { name: 'stt', processor: 'whisper-stt', config: {
        model_size: 'tiny.en', normalize_gain: true, target_peak: 0.5 } },
    ],
  },
]

export default function ModelsPage() {
  const [catalog, setCatalog] = useState<any[]>([])
  const [models, setModels] = useState<any[]>([])
  const [advanced, setAdvanced] = useState(false)
  const [name, setName] = useState('')
  const [proc, setProc] = useState('rule')
  const [cfg, setCfg] = useState('{}')
  const [manifest, setManifest] = useState('')
  const [out, setOut] = useState('')
  const [busy, setBusy] = useState('')

  const refresh = useCallback(async () => {
    const c = await get<any>('/api/v1/model-catalog')
    setCatalog(c.body?.models ?? [])
    const r = await get<any>('/api/models')
    setModels(r.body?.models ?? [])
  }, [])

  useEffect(() => { refresh() }, [refresh])

  const installOne = async (step: { name: string; processor: string; config: any }) => {
    const r = await post<any>('/api/models/install', {
      name: step.name, processor: step.processor, config: step.config,
    })
    if (r.status >= 300)
      throw new Error(`${step.name}: ${JSON.stringify(r.body)}`)
    if (r.body?.runtime_model_id)
      await post('/api/models/activate', {
        runtime_model_id: r.body.runtime_model_id, active: true,
      })
    return r.body
  }

  const runPreset = async (p: Preset) => {
    setBusy(p.id)
    try {
      for (const step of p.steps) await installOne(step)
      setOut(`${p.label}: installed + active (${p.steps.length} model(s))`)
    } catch (e) {
      setOut(`${p.label} failed: ${e}`)
    }
    setBusy('')
    refresh()
  }

  const activate = async (id: string, on: boolean) => {
    await post('/api/models/activate', { runtime_model_id: id, active: on })
    refresh()
  }

  const installAdvanced = async () => {
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
        <h3>Pipelines</h3>
        <div className="row" style={{ alignItems: 'stretch' }}>
          {PRESETS.map((p) => (
            <button key={p.id} className="go" disabled={busy === p.id}
                    onClick={() => void runPreset(p)}
                    style={{ flex: 1, textAlign: 'left', padding: '10px 14px' }}>
              <div style={{ fontWeight: 600 }}>{p.label}</div>
              <div className="muted" style={{ fontSize: 11, marginTop: 2 }}>
                {p.blurb}
              </div>
              {busy === p.id && <div className="muted">installing…</div>}
            </button>
          ))}
        </div>
        {out && <pre style={{ marginTop: 8 }}>{out}</pre>}
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
        <details style={{ marginTop: 8 }}>
          <summary className="muted" style={{ cursor: 'pointer', fontSize: 12 }}>
            raw model JSON
          </summary>
          <pre>{JSON.stringify(models, null, 2)}</pre>
        </details>
      </div>

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
        <h3>
          <label className="muted" style={{ fontSize: 13, cursor: 'pointer' }}
                 onClick={() => setAdvanced(!advanced)}>
            {advanced ? '▾' : '▸'} advanced — raw install / manifest upload
          </label>
        </h3>
        {advanced && (
          <>
            <div className="row">
              <input placeholder="name" size={16} value={name}
                     onChange={(e) => setName(e.target.value)} />
              <input placeholder="processor (e.g. rule, pca-face-recognizer)"
                     size={34} value={proc}
                     onChange={(e) => setProc(e.target.value)} />
              <button onClick={installAdvanced}>install+activate</button>
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
          </>
        )}
      </div>
    </>
  )
}
