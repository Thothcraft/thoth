import numpy as np, sys, os
sys.path.insert(0, 'src')
from backend.model_runtime import _e2_read_uint12, E2_FRAME_PAYLOAD_BYTES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    return [bytes(raw[off[i]:off[i+1]]) for i in order]

def rx_stats(frames, tag):
    adcs = []
    for pkt in frames[:200]:
        pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES:
            adcs.append(_e2_read_uint12(pl).reshape(64, 128, 3))
    a = np.stack(adcs)  # (N,64,128,3)
    print('=== %s ===' % tag)
    for rx in range(3):
        ch = a[:, :, :, rx]
        print('  rx%d: mean %.1f std %.1f min %.0f max %.0f' % (rx, ch.mean(), ch.std(), ch.min(), ch.max()))
    # cross-rx correlation of range profiles
    rfft = np.abs(np.fft.fft(a, axis=2)).mean(axis=(0, 1))  # (128,3)
    for rx in range(3):
        print('  rx%d range profile: %s' % (rx, np.round(rfft[:12, rx], 0)))

tmp = os.environ['TEMP'].replace('\\', '/')
rx_stats(load_frames(tmp + '/live_2153.npz'), 'live-empty')
rx_stats(load_frames(r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz'), 'val-empty')
