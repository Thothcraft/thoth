import { useEffect, useMemo, useRef } from 'react'

/** Renderers for live sensor payloads — images for cameras, signals for
 * audio/CSI, and multi-view for radar. Pure presentation: parent feeds
 * samples as they arrive. */

const b64ToBytes = (b64: string): Uint8Array => {
  const bin = atob(b64)
  const out = new Uint8Array(bin.length)
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i)
  return out
}

/** PCM s16le base64 → Float32Array in [-1,1]. */
const pcmToFloat = (b64: string): Float32Array => {
  const bytes = b64ToBytes(b64)
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  const out = new Float32Array(bytes.length / 2)
  for (let i = 0; i < out.length; i++) out[i] = view.getInt16(i * 2, true) / 32768
  return out
}

/** Draw a waveform/trace into a canvas (dark theme, accent line). */
function Trace({
  values, label, height = 90, max,
}: {
  values: number[]; label?: string; height?: number; max?: number
}) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !values.length) return
    const w = (c.width = c.clientWidth * devicePixelRatio)
    const h = (c.height = height * devicePixelRatio)
    const ctx = c.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#10141c'
    ctx.fillRect(0, 0, w, h)
    ctx.strokeStyle = '#243049'
    ctx.lineWidth = 1
    ctx.beginPath()
    ctx.moveTo(0, h / 2)
    ctx.lineTo(w, h / 2)
    ctx.stroke()
    const lim = max ?? Math.max(...values.map(Math.abs), 1e-6)
    ctx.strokeStyle = '#58a6ff'
    ctx.lineWidth = 1.5 * devicePixelRatio
    ctx.beginPath()
    values.forEach((v, i) => {
      const x = (i / Math.max(1, values.length - 1)) * w
      const y = h / 2 - (v / lim) * (h / 2 - 4)
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    })
    ctx.stroke()
  }, [values, height, max])
  return (
    <div>
      {label && <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>{label}</div>}
      <canvas ref={ref} style={{ width: '100%', height, borderRadius: 6 }} />
    </div>
  )
}

/** Bar/line chart for positive series (range profile, variance, bytes). */
function Bars({
  values, label, height = 90,
}: { values: number[]; label?: string; height?: number }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !values.length) return
    const w = (c.width = c.clientWidth * devicePixelRatio)
    const h = (c.height = height * devicePixelRatio)
    const ctx = c.getContext('2d')
    if (!ctx) return
    ctx.fillStyle = '#10141c'
    ctx.fillRect(0, 0, w, h)
    const lim = Math.max(...values, 1e-6)
    const bw = w / values.length
    ctx.fillStyle = '#3fb950'
    values.forEach((v, i) => {
      const bh = (v / lim) * (h - 6)
      ctx.fillRect(i * bw + 0.5, h - bh, Math.max(1, bw - 1), bh)
    })
  }, [values, height])
  return (
    <div>
      {label && <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>{label}</div>}
      <canvas ref={ref} style={{ width: '100%', height, borderRadius: 6 }} />
    </div>
  )
}

/** Heatmap for 2-D grids (radar xy_map). */
function Heatmap({ grid, label }: { grid: number[][]; label?: string }) {
  const ref = useRef<HTMLCanvasElement>(null)
  useEffect(() => {
    const c = ref.current
    if (!c || !grid.length) return
    const size = 220
    const w = (c.width = size * devicePixelRatio)
    const h = (c.height = size * devicePixelRatio)
    const ctx = c.getContext('2d')
    if (!ctx) return
    const rows = grid.length
    const cols = Math.max(...grid.map((r) => r.length), 1)
    const lim = Math.max(...grid.flat().map(Math.abs), 1e-6)
    const cw = w / cols
    const ch = h / rows
    for (let y = 0; y < rows; y++) {
      for (let x = 0; x < (grid[y]?.length ?? 0); x++) {
        const v = Math.abs(grid[y][x]) / lim
        // dark-blue → cyan → yellow ramp
        const rC = Math.round(20 + v * v * 235)
        const gC = Math.round(30 + v * 190)
        const bC = Math.round(60 + v * 130)
        ctx.fillStyle = `rgb(${rC},${gC},${bC})`
        ctx.fillRect(x * cw, y * ch, cw + 1, ch + 1)
      }
    }
  }, [grid])
  return (
    <div>
      {label && <div className="muted" style={{ fontSize: 11, marginBottom: 4 }}>{label}</div>}
      <canvas ref={ref} style={{ width: '100%', borderRadius: 6, display: 'block' }} />
    </div>
  )
}

export interface Sample {
  timestamp?: number
  payload?: Record<string, any>
  [k: string]: any
}

export function StreamView({
  type, samples,
}: { type: string; samples: Sample[] }) {
  const last = samples[samples.length - 1]
  const t = (type || '').toLowerCase()

  // Derived series — hooks must run unconditionally (sensor type can change).
  const wave = useMemo(() => {
    const b64 = last?.payload?.data
    if (!b64 || last?.payload?.encoding !== 'pcm_s16le') return [] as number[]
    const f = pcmToFloat(b64)
    const stride = Math.max(1, Math.floor(f.length / 600))
    const out: number[] = []
    for (let i = 0; i < f.length; i += stride) out.push(f[i])
    return out
  }, [last])

  const rmsSeries = useMemo(() => samples.slice(-60).map((s) => {
    const b64 = s.payload?.data
    if (!b64 || s.payload?.encoding !== 'pcm_s16le') return 0
    const f = pcmToFloat(b64)
    let acc = 0
    for (let i = 0; i < f.length; i++) acc += f[i] * f[i]
    return f.length ? Math.sqrt(acc / f.length) : 0
  }), [samples])

  const csiSeries = useMemo(() => samples.slice(-40).map((s) => {
    const b64 = s.payload?.data
    if (!b64 || s.payload?.encoding !== 'csi_raw')
      return { amp: 0, variance: 0, bytes: 0 }
    const bytes = b64ToBytes(b64)
    let acc = 0
    for (let i = 0; i < bytes.length; i++) acc += bytes[i]
    const mean = bytes.length ? acc / bytes.length : 0
    let varAcc = 0
    for (let i = 0; i < bytes.length; i++) {
      const d = bytes[i] - mean
      varAcc += d * d
    }
    return {
      amp: mean,
      variance: bytes.length ? varAcc / bytes.length : 0,
      bytes: bytes.length,
    }
  }), [samples])

  // ---- camera: latest jpeg frame -------------------------------------------
  if (t.includes('camera')) {
    const data = last?.payload?.data
    const w = last?.payload?.width
    const h = last?.payload?.height
    return (
      <div>
        {data ? (
          <img
            src={`data:image/jpeg;base64,${data}`}
            alt="camera frame"
            style={{ width: '100%', borderRadius: 8, display: 'block' }}
          />
        ) : (
          <div className="muted">waiting for frames…</div>
        )}
        <div className="muted" style={{ fontSize: 11, marginTop: 4 }}>
          {w ?? '?'}×{h ?? '?'} jpeg · {samples.length} frames received
        </div>
      </div>
    )
  }

  // ---- microphone: waveform + rolling RMS -----------------------------------
  if (t.includes('microphone') || t.includes('audio')) {
    return (
      <div>
        <Trace values={wave} label="waveform (100 ms)" height={110} />
        <div style={{ height: 8 }} />
        <Bars values={rmsSeries} label="rolling RMS (last 60 chunks)" height={70} />
      </div>
    )
  }

  // ---- radar: 3 views --------------------------------------------------------
  if (t.includes('radar')) {
    const xy = (last?.payload?.xy_map ?? []) as number[][]
    const prof = (last?.payload?.range_profile ?? []) as number[]
    const snr = last?.payload?.snr_db
    return (
      <div>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {xy.length > 0 && <Heatmap grid={xy} label="range × azimuth energy" />}
          <Bars values={prof} label="range profile" height={220} />
        </div>
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          SNR {typeof snr === 'number' ? `${snr.toFixed(1)} dB` : '–'} ·{' '}
          energy {last?.payload?.energy?.toFixed?.(3) ?? '–'} ·{' '}
          {samples.length} frames
        </div>
        <div style={{ height: 8 }} />
        <Bars
          values={samples.slice(-50).map((s) => s.payload?.snr_db ?? 0)}
          label="SNR history (last 50 frames)"
          height={60}
        />
      </div>
    )
  }

  // ---- wifi_csi: amplitude / rolling variance --------------------------------
  if (t.includes('csi')) {
    return (
      <div>
        <Bars
          values={csiSeries.map((d) => d.amp)}
          label="mean amplitude (per frame)"
          height={80}
        />
        <div style={{ height: 8 }} />
        <Bars
          values={csiSeries.map((d) => d.variance)}
          label="rolling variance"
          height={80}
        />
        <div className="muted" style={{ fontSize: 12, marginTop: 8 }}>
          {last?.payload?.bytes ?? csiSeries[csiSeries.length - 1]?.bytes ?? 0} bytes/frame
          · {samples.length} frames
        </div>
      </div>
    )
  }

  // ---- fallback: raw JSON tail ----------------------------------------------
  return (
    <pre id="liveOut" style={{ flex: 1, minHeight: 200 }}>
      {samples.map((s) =>
        `${s.timestamp ? new Date(s.timestamp * 1000).toLocaleTimeString() : '–'} ` +
        `${JSON.stringify(s.payload ?? s).slice(0, 300)}\n`).join('')}
    </pre>
  )
}
