import numpy as np, sys, os
sys.path.insert(0, 'src')
from backend.model_runtime import _e2_read_uint12, E2_FRAME_PAYLOAD_BYTES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    return [bytes(raw[off[i]:off[i+1]]) for i in order]

def rx_profiles(frames, tag):
    adcs = []
    for pkt in frames[:150]:
        pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES:
            adcs.append(_e2_read_uint12(pl).reshape(64, 128, 3))
    a = np.stack(adcs)
    rfft = np.abs(np.fft.fft(a, axis=2)).mean(axis=(0, 1))  # (128,3)
    print('=== %s ===' % tag)
    for rx in range(3):
        print('  rx%d: %s' % (rx, np.round(rfft[:8, rx], 0)))

tmp = os.environ['TEMP'].replace('\\', '/')
rx_profiles(load_frames(tmp + '/live_2047.npz'), 'live-2047')
rx_profiles(load_frames(tmp + '/live_2046.npz'), 'live-2046')
rx_profiles(load_frames(tmp + '/live_2042.npz'), 'live-2042')
rx_profiles(load_frames(r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1806\capture.npz'), 'val-1806')
rx_profiles(load_frames(r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz'), 'val-1910-occ')
