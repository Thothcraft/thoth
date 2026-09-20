"""Per-window RAW CSI temporal std for live captures — is the breathing
signal present inside a 5s window, or only across the whole minute?"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import _e2_parse_csi_amplitude, E2_FRAME_PAYLOAD_BYTES

for tag in sys.argv[1:]:
    d = np.load(tag, allow_pickle=False)
    off = d["radar_sample_offsets"]; raw = d["radar_sample_bytes"]
    seq = d["radar_sample_sequence"].astype(np.int64)
    order = np.argsort(seq, kind="stable")
    rts = d["radar_sample_unix_ns"].astype(np.float64)[order]/1e9
    npay = 0
    for i in order:
        pkt = bytes(raw[off[i]:off[i+1]]); pl = pkt[12:] if len(pkt)>=12 else pkt
        if len(pl)==E2_FRAME_PAYLOAD_BYTES: npay += 1
    nW = npay//50
    times = np.asarray([float(rts[i]) for i in range(nW*50)]).reshape(nW,50)
    # csi samples
    coff=d["csi_sample_offsets"]; craw=d["csi_sample_bytes"]
    cts=d["csi_sample_unix_ns"].astype(np.float64)/1e9
    amps=[]; tt=[]
    for i in range(len(coff)-1):
        v=_e2_parse_csi_amplitude(bytes(craw[coff[i]:coff[i+1]]).decode("utf-8","replace"))
        if v is not None: amps.append(v); tt.append(float(cts[i]))
    A=np.stack(amps); tt=np.asarray(tt)
    print(f"{Path(tag).stem}: {nW} windows, {len(A)} csi samples")
    for k in range(nW):
        m=(tt>=times[k,0])&(tt<=times[k,-1])
        if m.sum()<3: continue
        w=A[m]
        print(f"  win{k}: n={m.sum():3} span={times[k,-1]-times[k,0]:.1f}s "
              f"rawstd={w.std(0).mean():.3f} max={w.std(0).max():.3f}")
