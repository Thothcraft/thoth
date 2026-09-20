import numpy as np, sys, os, json, zipfile
sys.path.insert(0, 'src')
import torch
from backend.model_runtime import _e2_frames_to_maps, E2_FRAME_PAYLOAD_BYTES, E2_WINDOW_FRAMES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    return [bytes(raw[off[i]:off[i+1]]) for i in order]

def windows_of(frames, mean, std):
    payloads = []
    for pkt in frames:
        pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES:
            payloads.append(bytes(pl))
    nW = len(payloads) // E2_WINDOW_FRAMES
    maps = _e2_frames_to_maps(payloads[:nW * E2_WINDOW_FRAMES])
    w = maps.reshape(nW, E2_WINDOW_FRAMES, 2, 24, 24)
    mean = np.asarray(mean, dtype=np.float32).reshape(1, 1, 2, 1, 1)
    std = np.asarray(std, dtype=np.float32).reshape(1, 1, 2, 1, 1)
    return ((w - mean) / std).astype(np.float32)

model = torch.jit.load('models/examples/radar_occupancy_e2.pt', map_location='cpu').eval()

def probs(w):
    with torch.inference_mode():
        out = model(torch.from_numpy(w))
        v = out.detach().cpu().float().reshape(w.shape[0], -1)
        return torch.sigmoid(v[:, 0]).numpy() if v.shape[1] == 1 else torch.softmax(v, 1)[:, 1].numpy()

tmp = os.environ['TEMP'].replace('\\', '/')
live_empty_frames = load_frames(tmp + '/live_2047.npz') + load_frames(tmp + '/live_2153.npz')

# live stats from empty data
pay = [bytes(f[12:]) for f in live_empty_frames if len(f[12:]) == E2_FRAME_PAYLOAD_BYTES]
live_maps = _e2_frames_to_maps(pay)
live_mean = live_maps.mean(axis=(0, 2, 3))  # per-channel
live_std = live_maps.std(axis=(0, 2, 3))
print('live stats: mean', live_mean, 'std', live_std)

train_mean = [3.4973549023206707, 2.981877020132232]
train_std = [1.3730000988391573, 1.9651408268153345]

for tag, truth in [
    (tmp + '/live_2047.npz', 'live-empty'),
    (tmp + '/live_2153.npz', 'live-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz', 'val-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz', 'val-occupied'),
]:
    frames = load_frames(tag)
    for label, m, s in [('train-norm', train_mean, train_std), ('live-norm', live_mean, live_std)]:
        w = windows_of(frames, m, s)
        p = probs(w)
        top2 = np.sort(p)[-2:].mean() if len(p) >= 2 else p.mean()
        print('%s %s: top2 %.3f -> %s | probs %s' % (
            truth, label, top2, 'OCC' if top2 >= 0.695 else 'empty', np.round(p, 2)))
