import numpy as np, sys, os
sys.path.insert(0, 'src')
import torch
from scipy import fft as sfft
from scipy.ndimage import zoom
from backend.model_runtime import _e2_read_uint12, E2_FRAME_PAYLOAD_BYTES, E2_WINDOW_FRAMES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    return [bytes(raw[off[i]:off[i+1]]) for i in order]

def maps_of(frames, perm=None):
    pay = [bytes(f[12:]) for f in frames if len(f[12:]) == E2_FRAME_PAYLOAD_BYTES]
    n = len(pay)
    adc = _e2_read_uint12(b''.join(pay)).reshape(n, 64, 128, 3)
    if perm is not None:
        adc = adc[:, :, :, perm]
    adc *= np.hanning(128)[None, None, :, None].astype(np.float32) * np.hanning(64)[None, :, None, None].astype(np.float32)
    R = sfft.fft(adc, axis=2, workers=-1)
    RD = sfft.fftshift(sfft.fft(R, axis=1, workers=-1), axes=1)
    rd = np.abs(RD[:, :, :64, :]).mean(axis=3)
    RA = sfft.fft(R[:, :, :64, :], n=16, axis=3, workers=-1)
    ra = np.abs(RA).mean(axis=1).transpose(0, 2, 1)
    rd = zoom(np.log1p(rd), (1, 24/64, 24/64), order=1)
    ra = zoom(np.log1p(ra), (1, 24/16, 24/64), order=1)
    return np.stack([rd, ra], axis=1).astype(np.float32)

model = torch.jit.load('models/examples/radar_occupancy_e2.pt', map_location='cpu').eval()
MEAN = np.array([3.4973549023206707, 2.981877020132232], dtype=np.float32).reshape(1,1,2,1,1)
STD = np.array([1.3730000988391573, 1.9651408268153345], dtype=np.float32).reshape(1,1,2,1,1)

def probs(frames, perm):
    m = maps_of(frames, perm)
    nW = m.shape[0] // 50
    w = m[:nW*50].reshape(nW, 50, 2, 24, 24)
    w = (w - MEAN) / STD
    with torch.inference_mode():
        v = model(torch.from_numpy(w)).detach().cpu().float().reshape(nW, -1)
        return torch.sigmoid(v[:, 0]).numpy() if v.shape[1] == 1 else torch.softmax(v, 1)[:, 1].numpy()

tmp = os.environ['TEMP'].replace('\\', '/')
for tag, truth in [
    (tmp + '/live_2047.npz', 'live-empty'),
    (tmp + '/live_2153.npz', 'live-empty'),
    (tmp + '/live_2046.npz', 'live-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz', 'val-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz', 'val-occ'),
]:
    frames = load_frames(tag)
    for perm_name, perm in [('identity', None), ('perm[2,0,1]', [2,0,1])]:
        p = probs(frames, perm)
        top2 = np.sort(p)[-2:].mean()
        print('%s %s: top2 %.3f -> %s | %s' % (truth, perm_name, top2, 'OCC' if top2 >= 0.695 else 'empty', np.round(p, 2)))
