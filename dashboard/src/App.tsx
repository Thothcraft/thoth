import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { clearToken, getToken, isPortalViewer, saveToken } from './api'
import StatusPage from './pages/StatusPage'
import LivePage from './pages/LivePage'
import CapturesPage from './pages/CapturesPage'
import ModelsPage from './pages/ModelsPage'
import AutomationsPage from './pages/AutomationsPage'
import MetadataPage from './pages/MetadataPage'

const TABS = [
  { to: '/status', label: 'Status' },
  { to: '/live', label: 'Live' },
  { to: '/captures', label: 'Captures' },
  { to: '/models', label: 'Models' },
  { to: '/automations', label: 'Automations' },
  { to: '/metadata', label: 'Metadata' },
]

type Phase = 'checking' | 'locked' | 'open'

/* Verify a candidate/stored token against the node API. */
async function tokenOk(t: string): Promise<'ok' | 'bad' | 'error'> {
  try {
    const r = await fetch('/api/status', {
      headers: { Authorization: `Bearer ${t}` },
    })
    if (r.status === 401 || r.status === 403) return 'bad'
    return r.ok ? 'ok' : 'error'
  } catch {
    return 'error'
  }
}

function SignIn({ onUnlock }: { onUnlock: (t: string) => Promise<void> }) {
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)
  async function submit(v: string) {
    const t = v.trim()
    if (!t || busy) return
    setBusy(true)
    try {
      await onUnlock(t)
      setErr('invalid token (check `thoth token` on the node)')
    } catch {
      setErr('node unreachable — is the daemon running?')
    }
    setBusy(false)
  }
  return (
    <div className="gate">
      <div className="gatecard">
        <h2>
          <span className="brand"><b>◈</b></span> thoth node
        </h2>
        <div className="sub">{location.hostname} — sign in</div>
        <input
          autoFocus type="password" placeholder="node token"
          autoComplete="current-password"
          onKeyDown={(e) => {
            if (e.key === 'Enter') submit((e.target as HTMLInputElement).value)
          }}
        />
        <button className="go" disabled={busy}
                onClick={() => {
                  const el = document.querySelector<HTMLInputElement>(
                    '.gatecard input')
                  void submit(el?.value ?? '')
                }}>
          {busy ? 'checking…' : 'sign in'}
        </button>
        <div className="err">{err}</div>
        <small className="muted">
          token = the node's <code>local_token</code>
          (<code>thoth token</code> on the device)
        </small>
      </div>
    </div>
  )
}

export default function App() {
  const [tok, setTok] = useState(getToken())
  const [phase, setPhase] = useState<Phase>(
    isPortalViewer || getToken() ? 'checking' : 'locked')

  useEffect(() => {
    // Strip ?token= from the address bar once captured.
    if (new URLSearchParams(location.search).get('token')) {
      history.replaceState(null, '', location.pathname + location.hash)
    }
    // Portal embeds authenticate through the relay, not the gate.
    if (isPortalViewer) { setPhase('open'); return }
    const t = getToken()
    if (!t) { setPhase('locked'); return }
    void tokenOk(t).then((r) => {
      if (r === 'ok') setPhase('open')
      else if (r === 'bad') { clearToken(); setPhase('locked') }
      else setPhase('open') // node unreachable briefly — let pages show errors
    })
  }, [])

  async function unlock(t: string) {
    const r = await tokenOk(t)
    if (r !== 'ok') {
      if (r === 'error') throw new Error('node unreachable')
      return
    }
    saveToken(t)
    setTok(t)
    setPhase('open')
  }

  if (phase === 'checking') return null
  if (phase === 'locked') return <SignIn onUnlock={unlock} />

  return (
    <>
      <header>
        <span className="brand"><b>◈</b> thoth node</span>
        <nav>
          {TABS.map((t) => (
            <NavLink key={t.to} to={t.to}
                     className={({ isActive }) => (isActive ? 'on' : '')}>
              {t.label}
            </NavLink>
          ))}
        </nav>
        {/* viewer=portal hides the raw local token field (CONTRACT §5) */}
        {!isPortalViewer && (
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            <input
              placeholder="local token" size={22} defaultValue={tok}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  saveToken((e.target as HTMLInputElement).value)
                  setTok(getToken())
                }
              }}
            />
            <button onClick={() => {
              const el = document.querySelector<HTMLInputElement>(
                'header input')
              saveToken(el?.value ?? '')
              setTok(getToken())
            }}>unlock</button>
          </div>
        )}
        {isPortalViewer && (
          <span className="muted" style={{ marginLeft: 'auto' }}>
            portal viewer
          </span>
        )}
      </header>
      <main>
        <Routes>
          <Route path="/" element={<Navigate to="/status" replace />} />
          <Route path="/status" element={<StatusPage />} />
          <Route path="/live" element={<LivePage />} />
          <Route path="/captures" element={<CapturesPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/automations" element={<AutomationsPage />} />
          <Route path="/metadata" element={<MetadataPage />} />
          <Route path="*" element={<Navigate to="/status" replace />} />
        </Routes>
      </main>
    </>
  )
}
