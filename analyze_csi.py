"""Compare live CSI statistics vs cached empty/present to see if a still
person leaves a detectable CSI signature on the live device."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path.home() / "Desktop/radar/E2"))
import common as C

def csi_feats(grid):
    """grid (nW,128,52) normalized -> per-window feature summary."""
    # temporal std per subcarrier, then stats — breathing shows as low-freq power
    sd = grid.std(axis=1)                    # (nW,52)
    return sd.mean(1), sd.std(1), sd.max(1)  # (nW,) each

def summarize(name, grids):
    m, s, mx = csi_feats(grids)
    print(f"{name:22} n={len(grids):3}  sd.mean {m.mean():.3f}±{m.std():.3f}  "
          f"sd.max {mx.mean():.3f}±{mx.std():.3f}")

# cached: split by label
for split in ("val_minutes", "test_minutes"):
    emp, occ = [], []
    for f in sorted((C.CACHE_DIR / split).glob("*.npz")):
        d = np.load(f); meta = json.loads(str(d["meta"]))
        lab = int(meta.get("label", -1))
        csi = d["csi"].astype(np.float32)
        valid = d["csi_valid"].astype(bool)
        if lab == 0: emp.append(csi[valid])
        elif lab == 1: occ.append(csi[valid])
    if emp: summarize(f"{split} EMPTY", np.concatenate(emp))
    if occ: summarize(f"{split} OCC", np.concatenate(occ))

# live: rebuild csi windows from a capture
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import e2_csi_windows, E2_FRAME_PAYLOAD_BYTES, _e2_frames_to_maps
import torch
extra = {"meta.json": ""}
torch.jit.load(str(Path.home()/"Desktop/radar/deploy/fusion_occupancy.pt"),
               map_location="cpu", _extra_files=extra)
norm = json.loads(extra["meta.json"])["norm"]

for tag in sys.argv[1:]:
    d = np.load(tag, allow_pickle=False)
    off = d["radar_sample_offsets"]; raw = d["radar_sample_bytes"]
    seq = d["radar_sample_sequence"].astype(np.int64)
    order = np.argsort(seq, kind="stable")
    rts = d["radar_sample_unix_ns"].astype(np.float64)[order]/1e9
    pay = []
    for i in order:
        pkt = bytes(raw[off[i]:off[i+1]]); pl = pkt[12:] if len(pkt)>=12 else pkt
        if len(pl)==E2_FRAME_PAYLOAD_BYTES: pay.append(bytes(pl))
    nW = len(pay)//50
    times = np.asarray([float(rts[i]) for i in range(nW*50)]).reshape(nW,50)
    cs = []
    if "csi_sample_offsets" in d.files:
        coff=d["csi_sample_offsets"]; craw=d["csi_sample_bytes"]
        crx=d["csi_sample_receiver_index"]; cts=d["csi_sample_unix_ns"].astype(np.float64)/1e9
        for i in range(len(coff)-1):
            cs.append((int(crx[i]),float(cts[i]),bytes(craw[coff[i]:coff[i+1]]).decode("utf-8","replace")))
    grid,valid = e2_csi_windows(cs, times[:,0], times[:,-1], norm)
    summarize(f"LIVE {Path(tag).stem}", grid[valid])
