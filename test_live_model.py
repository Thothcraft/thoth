import numpy as np, sys, os, shutil
sys.path.insert(0, 'src')
from pathlib import Path
from backend.model_runtime import ModelRegistry

# register the live-finetuned model into a temp registry
tmp_reg = Path(os.environ['TEMP']) / 'live_reg'
(tmp_reg / 'artifacts').mkdir(parents=True, exist_ok=True)
shutil.copy(r'C:\Users\ggad\Desktop\radar\deploy\radar_occupancy_live.pt', tmp_reg / 'artifacts' / 'live.pt')
reg = ModelRegistry(tmp_reg)
reg.add(tmp_reg / 'artifacts' / 'live.pt', {
    'schema': 'thoth-model/v1',
    'name': 'Radar Occupancy Live', 'version': 'live1',
    'inputs': [{'sensor': 'radar', 'representation': 'e2_maps', 'frames': 50, 'shape': [1, 50, 2, 24, 24], 'fit': 'left_pad_latest', 'normalization': {'kind': 'none'}}],
    'output': {'kind': 'logits'},
    'class_names': ['empty', 'occupied'],
    'execution': 'minute', 'aggregation': {'kind': 'top2'},
}, source='local')
for m in reg.list():
    reg.set_enabled(m['id'], True)

def load_npz(p):
    d = np.load(p, allow_pickle=False)
    off = d['radar_sample_offsets']; raw = d['radar_sample_bytes']
    n = len(off) - 1
    seq = d['radar_sample_sequence'].astype(np.int64); order = np.argsort(seq, kind='stable')
    frames = [bytes(raw[off[i]:off[i+1]]) for i in order]
    ts = d['radar_sample_unix_ns'].astype(np.float64)[order] / 1e9
    return frames, ts.tolist()

tmp = os.environ['TEMP'].replace('\\', '/')
for tag, truth in [
    ('live_empty/2153.npz', 'empty'), ('live_empty/2210.npz', 'empty'),
    ('live_empty/2212.npz', 'empty'), ('live_empty/2214.npz', 'empty'),
    ('live_empty/2216.npz', 'empty'),
    ('live_2042.npz', 'occ?'), ('live_2043.npz', 'occ?'),
]:
    frames, times = load_npz(tmp + '/' + tag)
    res = reg.run_minute(frames, times, [], 't')
    r = res[0]
    print('%s truth=%s -> %s conf %.3f thr %.2f excl %s | probs %s' % (
        tag, truth, r.get('class'), r.get('confidence') or 0,
        r.get('threshold') or 0, r.get('windows_excluded'), r.get('window_probabilities')))
