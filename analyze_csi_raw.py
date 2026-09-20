"""Inspect RAW live CSI amplitudes (pre-normalization) for variation."""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import _e2_parse_csi_amplitude

for tag in sys.argv[1:]:
    d = np.load(tag, allow_pickle=False)
    if "csi_sample_offsets" not in d.files:
        print(tag, "no csi"); continue
    coff = d["csi_sample_offsets"]; craw = d["csi_sample_bytes"]
    crx = d["csi_sample_receiver_index"]
    amps = []
    for i in range(len(coff) - 1):
        v = _e2_parse_csi_amplitude(bytes(craw[coff[i]:coff[i+1]]).decode("utf-8","replace"))
        if v is not None: amps.append(v)
    if not amps:
        print(tag, "no parsed csi"); continue
    A = np.stack(amps)                      # (nS, 52) raw amplitudes
    print(f"{Path(tag).stem}: nS={len(A)} receivers={np.unique(crx)}")
    print(f"  raw amp: mean {A.mean():.1f}  std {A.std():.2f}  range [{A.min():.0f},{A.max():.0f}]")
    print(f"  per-subcarrier temporal std: mean {A.std(0).mean():.3f}  max {A.std(0).max():.3f}")
    print(f"  sample[0][:8]={np.round(A[0,:8],1)}")
    print(f"  sample[-1][:8]={np.round(A[-1,:8],1)}")
