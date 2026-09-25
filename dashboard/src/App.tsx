import { useEffect, useState } from 'react'
import { NavLink, Navigate, Route, Routes } from 'react-router-dom'
import { getToken, isPortalViewer, saveToken } from './api'
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

export default function App() {
  const [tok, setTok] = useState(getToken())

  useEffect(() => {
    // Strip ?token= from the address bar once captured.
    if (new URLSearchParams(location.search).get('token')) {
      history.replaceState(null, '', location.pathname + location.hash)
    }
  }, [])

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
