import numpy as np, sys, os, json, zipfile
sys.path.insert(0, 'src')
from backend.model_runtime import _e2_frames_to_maps, E2_FRAME_PAYLOAD_BYTES

def load_frames(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    frames = [bytes(raw[off[i]:off[i+1]]) for i in order]
    return frames

def maps_of(frames):
    payloads = []
    for pkt in frames:
        pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES:
            payloads.append(bytes(pl))
    return _e2_frames_to_maps(payloads)

# embedded norm stats from the model
pt = 'models/examples/radar_occupancy_e2.pt'
with zipfile.ZipFile(pt) as z:
    name = [i.filename for i in z.infolist() if i.filename.endswith('meta.json')][0]
    meta = json.loads(z.read(name))
norm = meta.get('normalization') or {}
print('embedded radar_mean:', norm.get('radar_mean'), 'radar_std:', norm.get('radar_std'))

tmp = os.environ['TEMP'].replace('\\', '/')
for tag, truth in [
    (tmp + '/live_2153.npz', 'live-empty'),
    (tmp + '/live_2047.npz', 'live-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1804\capture.npz', 'val-empty'),
    (r'C:\Users\ggad\Desktop\radar\val_minutes\20260906_1910\capture.npz', 'val-occupied'),
]:
    frames = load_frames(tag)
    m = maps_of(frames[:550])
    print('%s: maps mean %.3f std %.3f | ch0 mean %.3f std %.3f | ch1 mean %.3f std %.3f' % (
        truth, m.mean(), m.std(),
        m[:, 0].mean(), m[:, 0].std(), m[:, 1].mean(), m[:, 1].std()))
