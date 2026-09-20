"""Verify quantized TorchScript models predict correctly on live captures."""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import _e2_frames_to_maps, e2_csi_windows, E2_FRAME_PAYLOAD_BYTES

def load_meta(path):
    e = {"meta.json": ""}
    m = torch.jit.load(path, map_location="cpu", _extra_files=e)
    return m, json.loads(e["meta.json"])

def run(model_path, npz):
    model, meta = load_meta(model_path)
    norm = meta["norm"]
    rmean = np.asarray(norm["radar_mean"], np.float32); rstd = np.asarray(norm["radar_std"], np.float32)
    d = np.load(npz, allow_pickle=False)
    off = d["radar_sample_offsets"]; raw = d["radar_sample_bytes"]
    seq = d["radar_sample_sequence"].astype(np.int64); order = np.argsort(seq, kind="stable")
    rts = d["radar_sample_unix_ns"].astype(np.float64)[order] / 1e9
    pay = []
    for i in order:
        pkt = bytes(raw[off[i]:off[i+1]]); pl = pkt[12:] if len(pkt) >= 12 else pkt
        if len(pl) == E2_FRAME_PAYLOAD_BYTES: pay.append(bytes(pl))
    nW = len(pay) // 50
    m = _e2_frames_to_maps(pay[:nW*50]).reshape(nW, 50, 2, 24, 24)
    r = ((m - rmean.reshape(1,1,2,1,1)) / rstd.reshape(1,1,2,1,1)).astype(np.float32)
    times = np.asarray([float(rts[i]) for i in range(nW*50)]).reshape(nW, 50)
    cs = []
    if "csi_sample_offsets" in d.files:
        coff = d["csi_sample_offsets"]; craw = d["csi_sample_bytes"]
        crx = d["csi_sample_receiver_index"]; cts = d["csi_sample_unix_ns"].astype(np.float64)/1e9
        for i in range(len(coff)-1):
            cs.append((int(crx[i]), float(cts[i]), bytes(craw[coff[i]:coff[i+1]]).decode("utf-8","replace")))
    c, _ = e2_csi_windows(cs, times[:,0], times[:,-1], norm)
    with torch.inference_mode():
        if meta["modality"] == "fusion":
            p = torch.sigmoid(model(torch.from_numpy(r), torch.from_numpy(c))).numpy()
        else:
            p = torch.sigmoid(model(torch.from_numpy(r))).numpy()
    top2 = float(np.sort(p)[-2:].mean()) if len(p) >= 2 else float(p.mean())
    return top2

import glob, os
for tag, npz in [("OCC", os.path.join(os.environ["TEMP"], "live_now.npz")),
                 ("EMPTY", os.path.join(os.environ["TEMP"], "live_empty", "2214.npz"))]:
    for name, mp in [("radar_q", r"C:\Users\ggad\Desktop\radar\deploy\radar_occupancy_q.pt"),
                     ("fusion_q", r"C:\Users\ggad\Desktop\radar\deploy\fusion_occupancy_q.pt")]:
        try:
            print(f"{tag} {name}: top2={run(mp, npz):.3f}")
        except Exception as e:
            print(f"{tag} {name}: ERR {e}")
