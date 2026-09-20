import numpy as np, sys, os
sys.path.insert(0, 'src')
from backend.model_runtime import _e2_frames_to_maps, E2_FRAME_PAYLOAD_BYTES, E2_WINDOW_FRAMES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    return [bytes(raw[off[i]:off[i+1]]) for i in order]

def window_map(frames, k):
    pay = [bytes(f[12:]) for f in frames[k*50:(k+1)*50] if len(f[12:]) == E2_FRAME_PAYLOAD_BYTES]
    if len(pay) < 50:
        return None
    return _e2_frames_to_maps(pay)  # (50,2,24,24)

def summarize(m, tag, k):
    rd = m[:, 0]  # range-doppler (50,24,24)
    ra = m[:, 1]  # range-azimuth
    # doppler profile: mean over range, per doppler bin
    dop = rd.mean(axis=(0, 1))  # (24,) over doppler axis
    rng = rd.mean(axis=(0, 2))  # (24,) over range axis
    print('%s win%d | doppler profile: %s' % (tag, k, np.round(dop, 1)))
    print('%s win%d | range profile:   %s' % (tag, k, np.round(rng, 1)))
    # peak location
    avg = rd.mean(axis=0)
    pk = np.unravel_index(np.argmax(avg), avg.shape)
    print('%s win%d | peak at range-bin %d dop-bin %d val %.1f | mean %.1f' % (tag, k, pk[0], pk[1], avg[pk], avg.mean()))

tmp = os.environ['TEMP'].replace('\\', '/')
# live empty - a high-prob clean window (win 9 of 2153) and low-prob (win 5)
f = load_frames(tmp + '/live_2153.npz')
for k in [5, 9]:
    m = window_map(f, k)
    if m is not None:
        summarize(m, 'live2153', k)
# val empty + occupied
f = load_frames(r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz')
summarize(window_map(f, 3), 'val-empty', 3)
f = load_frames(r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz')
summarize(window_map(f, 3), 'val-occ', 3)
