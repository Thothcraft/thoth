import numpy as np, sys, os
sys.path.insert(0, 'src')
from backend.model_runtime import _e2_read_uint12, E2_FRAME_PAYLOAD_BYTES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    frames = [bytes(raw[off[i]:off[i+1]]) for i in order]
    ts = d['radar_sample_unix_ns'].astype(np.float64)[order] / 1e9
    return frames, ts

def frame_adc(pkt):
    payload = pkt[12:] if len(pkt) >= 12 else pkt
    if len(payload) != E2_FRAME_PAYLOAD_BYTES:
        return None
    return _e2_read_uint12(payload).reshape(64, 128, 3)  # chirps x samples x rx

def doppler_energy(frames):
    """Mean |FFT| energy in non-zero Doppler bins, per frame."""
    out = []
    for f in frames:
        adc = frame_adc(f)
        if adc is None:
            continue
        rfft = np.fft.fft(adc, axis=1)                       # range
        dfft = np.fft.fftshift(np.fft.fft(rfft, axis=0), axes=0)  # doppler
        mag = np.abs(dfft)
        nz = mag[1:, :, :]  # skip DC doppler bin
        out.append(nz.mean())
    return np.array(out)

tmp = os.environ['TEMP'].replace('\\', '/')
for tag, truth in [('live_2153', 'empty?'), ('live_2047', 'empty')]:
    frames, ts = load_frames(tmp + '/' + tag + '.npz')
    n = len(frames)
    nW = n // 50
    print('===', tag, truth, 'frames', n)
    for k in range(nW):
        wf = frames[k*50:(k+1)*50]
        wt = ts[k*50:(k+1)*50]
        e = doppler_energy(wf)
        gaps = np.diff(wt)
        print('  win %d: doppler_E mean %.1f max %.1f | maxgap %.2fs | span %.1fs' % (
            k, e.mean(), e.max(), gaps.max() if len(gaps) else 0, wt[-1]-wt[0]))

# compare to val occupied + empty
for p, truth in [
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz', 'val-occupied'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz', 'val-empty'),
]:
    frames, ts = load_frames(p)
    n = len(frames); nW = n // 50
    print('===', truth, 'frames', n)
    for k in range(min(nW, 11)):
        wf = frames[k*50:(k+1)*50]
        e = doppler_energy(wf)
        print('  win %d: doppler_E mean %.1f max %.1f' % (k, e.mean(), e.max()))
