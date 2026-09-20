"""Run the fine-tuned fusion model on a live capture to check occupancy."""
import sys, json
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path.home() / "Desktop/radar/E2"))
import common as C
from models import build_model
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import _e2_frames_to_maps, e2_csi_windows, E2_FRAME_PAYLOAD_BYTES

ckpt = torch.load(C.OUTPUT_DIR/"model_fusion_live.pt", map_location="cpu", weights_only=False)
norm = ckpt["norm"]; cfg = ckpt["cfg"]
rmean = np.asarray(norm["radar_mean"], np.float32); rstd = np.asarray(norm["radar_std"], np.float32)
model = build_model("fusion", cfg["embed_dim"], cfg["dropout"]); model.load_state_dict(ckpt["state_dict"]); model.eval()

for tag in sys.argv[1:]:
    d = np.load(tag, allow_pickle=False)
    off=d["radar_sample_offsets"]; raw=d["radar_sample_bytes"]
    seq=d["radar_sample_sequence"].astype(np.int64); order=np.argsort(seq,kind="stable")
    rts=d["radar_sample_unix_ns"].astype(np.float64)[order]/1e9
    pay=[]
    for i in order:
        pkt=bytes(raw[off[i]:off[i+1]]); pl=pkt[12:] if len(pkt)>=12 else pkt
        if len(pl)==E2_FRAME_PAYLOAD_BYTES: pay.append(bytes(pl))
    nW=len(pay)//50
    m=_e2_frames_to_maps(pay[:nW*50]).reshape(nW,50,2,24,24)
    r=((m-rmean.reshape(1,1,2,1,1))/rstd.reshape(1,1,2,1,1)).astype(np.float32)
    times=np.asarray([float(rts[i]) for i in range(nW*50)]).reshape(nW,50)
    cs=[]
    coff=d["csi_sample_offsets"]; craw=d["csi_sample_bytes"]
    crx=d["csi_sample_receiver_index"]; cts=d["csi_sample_unix_ns"].astype(np.float64)/1e9
    for i in range(len(coff)-1):
        cs.append((int(crx[i]),float(cts[i]),bytes(craw[coff[i]:coff[i+1]]).decode("utf-8","replace")))
    c,_=e2_csi_windows(cs,times[:,0],times[:,-1],norm)
    with torch.inference_mode():
        p=torch.sigmoid(model(torch.from_numpy(r),torch.from_numpy(c))).numpy()
    top2=float(np.sort(p)[-2:].mean())
    print(f"{Path(tag).stem}: top2={top2:.3f} -> {'OCCUPIED' if top2>=0.5 else 'empty'}  winp={np.round(p,2)}")
