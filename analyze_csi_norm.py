"""Compute live CSI per-subcarrier mean/std of log1p(amplitude) — the norm
stats the deployed model should embed so live CSI isn't crushed by the
training device's ~30x larger amplitude scale."""
import sys, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "Desktop/thothcraft/thoth/src"))
from backend.model_runtime import _e2_parse_csi_amplitude

amps = []
for tag in sys.argv[1:]:
    d = np.load(tag, allow_pickle=False)
    if "csi_sample_offsets" not in d.files: continue
    coff = d["csi_sample_offsets"]; craw = d["csi_sample_bytes"]
    for i in range(len(coff) - 1):
        v = _e2_parse_csi_amplitude(bytes(craw[coff[i]:coff[i+1]]).decode("utf-8","replace"))
        if v is not None: amps.append(v)
A = np.log1p(np.stack(amps))            # (nS,52)
mean = A.mean(0); std = A.std(0) + 1e-6
print("live csi_mean[:8]:", np.round(mean[:8],3))
print("live csi_std [:8]:", np.round(std[:8],3))
print("std range:", round(float(std.min()),3), "-", round(float(std.max()),3))
out = {"csi_mean": mean.tolist(), "csi_std": std.tolist()}
Path(sys.argv[1]).with_suffix(".csinorm.json").write_text(json.dumps(out))
print("wrote", Path(sys.argv[1]).with_suffix(".csinorm.json"))
