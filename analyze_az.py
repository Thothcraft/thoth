import numpy as np, sys, os
sys.path.insert(0, 'src')
from backend.model_runtime import _e2_frames_to_maps, E2_FRAME_PAYLOAD_BYTES

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
    return _e2_frames_to_maps(pay)

tmp = os.environ['TEMP'].replace('\\', '/')
for tag, path, k in [
    ('live2153', tmp + '/live_2153.npz', 9),
    ('val-empty', r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz', 3),
    ('val-occ', r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz', 3),
]:
    m = window_map(load_frames(path), k)
    ra = m[:, 1].mean(axis=0)  # (24,24) range-azimuth avg
    print('=== %s win%d azimuth map (range rows x az cols):' % (tag, k))
    for row in ra[::3]:
        print('  ' + ' '.join('%5.1f' % v for v in row[::2]))
    print('  azimuth profile (mean over range):', np.round(ra.mean(axis=0), 2))
