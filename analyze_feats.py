import numpy as np, sys, os
sys.path.insert(0, 'src')
import torch
from backend.model_runtime import _e2_frames_to_maps, E2_FRAME_PAYLOAD_BYTES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    frames = [bytes(raw[off[i]:off[i+1]]) for i in order]
    ts = d['radar_sample_unix_ns'].astype(np.float64)[order] / 1e9
    return frames, ts

def window_feats(frames, ts):
    pay = [bytes(f[12:]) for f in frames if len(f[12:]) == E2_FRAME_PAYLOAD_BYTES]
    nW = len(pay) // 50
    m = _e2_frames_to_maps(pay[:nW*50])  # (nW*50, 2, 24, 24)
    w = m.reshape(nW, 50, 2, 24, 24)
    sd = w.std(axis=1)  # (nW, 2, 24, 24) temporal std
    feats = np.stack([sd.mean(axis=(2,3)), sd.std(axis=(2,3)), sd.max(axis=(2,3))], axis=1)  # (nW,3,2)
    gaps = np.diff(ts).reshape(-1)[:nW*50].reshape(nW, 50)[:, 1:].max(axis=1) if len(ts) >= nW*50 else np.zeros(nW)
    return feats, gaps

tmp = os.environ['TEMP'].replace('\\', '/')
for tag, truth in [
    (tmp + '/live_2153.npz', 'live-empty'),
    (tmp + '/live_2047.npz', 'live-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz', 'val-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz', 'val-occ'),
]:
    frames, ts = load_frames(tag)
    feats, gaps = window_feats(frames, ts)
    print('=== %s ===' % truth)
    print('  ch0(rd): mean/std/max per win')
    for k in range(feats.shape[0]):
        f = feats[k]
        print('   w%d gap%.1f | rd mean %.3f std %.3f max %.3f | az mean %.3f std %.3f max %.3f' % (
            k, gaps[k], f[0,0], f[1,0], f[2,0], f[0,1], f[1,1], f[2,1]))
