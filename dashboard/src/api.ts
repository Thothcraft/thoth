/**
 * Node API client (CONTRACT §5).
 *
 * Local mode  — hits the node's own /api/* on this origin with
 *               `Authorization: Bearer <local_token>` (?token= once, then
 *               localStorage + the thoth_dash cookie the server sets).
 * Relay mode  — `?relay=<brain-base>/<device-id>` rewrites every call to
 *               `POST {relay}/api {method,path,body}` so the same UI runs
 *               embedded in the portal against the Brain WS tunnel
 *               (Agent B serves the page; this flag needs no code here).
 */
const qs = new URLSearchParams(location.search)

export const viewer: 'local' | 'portal' =
  qs.get('viewer') === 'portal' ? 'portal' : 'local'
export const relay: string | null = qs.get('relay')
export const isPortalViewer = viewer === 'portal' || relay != null

let token =
  qs.get('token') ?? localStorage.getItem('thoth_tok') ?? ''

export function getToken(): string {
  return token
}

export function saveToken(t: string) {
  token = t.trim()
  localStorage.setItem('thoth_tok', token)
}

export function clearToken() {
  token = ''
  localStorage.removeItem('thoth_tok')
}

export interface ApiResult<T = unknown> {
  status: number
  body: T | null
}

async function request<T>(method: string, path: string,
                          body?: unknown): Promise<ApiResult<T>> {
  if (relay) {
    // Brain relay: REST→WS tunnel returns the api_response body directly.
    try {
      const res = await fetch(`${relay.replace(/\/$/, '')}/api`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          ...(token ? { Authorization: `Bearer ${token}` } : {}),
        },
        body: JSON.stringify({ method, path, body: body ?? null }),
      })
      const payload = await res.json().catch(() => null)
      return { status: res.status, body: payload as T }
    } catch {
      return { status: 0, body: null }
    }
  }
  try {
    const res = await fetch(path, {
      method,
      headers: {
        ...(body !== undefined ? { 'Content-Type': 'application/json' } : {}),
        ...(token ? { Authorization: `Bearer ${token}` } : {}),
      },
      body: body !== undefined ? JSON.stringify(body) : undefined,
    })
    const payload = await res.json().catch(() => null)
    return { status: res.status, body: payload as T }
  } catch {
    return { status: 0, body: null }
  }
}

export const get = <T>(p: string) => request<T>('GET', p)
export const post = <T>(p: string, b?: unknown) => request<T>('POST', p, b)
export const put = <T>(p: string, b?: unknown) => request<T>('PUT', p, b)
export const del = <T>(p: string) => request<T>('DELETE', p)

/** Direct download URL — only valid in local mode (browser navigates
 *  with ?token=; the relay returns JSON, not zip bytes). */
export function downloadUrl(path: string): string {
  const sep = path.includes('?') ? '&' : '?'
  return `${path}${sep}token=${encodeURIComponent(token)}`
}
