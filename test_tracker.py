"""Run the example_2 XY tracker on a live capture's radar frames and report
location/score/detected/snr per frame — does it localize the still person?"""
import sys, json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[0]
sys.path.insert(0, str(ROOT / "src"))
MMW = ROOT / "WS" / "MMW-HAT" / "MMW-HAT-Release"
sys.path.insert(0, str(MMW))
sys.path.insert(0, str(MMW / "example_2_track"))
from utility.helper import parse_radar_cfg, read_uint12, split_samples
from backend.radar_analysis import create_example2_processor

setting = json.loads(next((MMW/"radar_config"/"config_3rx_3m").glob("BGT60TR13C_settings_*.json")).read_text())
proc = create_example2_processor(parse_radar_cfg(setting))
proc.detection_threshold_db = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0

d = np.load(sys.argv[1], allow_pickle=False)
off = d["radar_sample_offsets"]; raw = d["radar_sample_bytes"]
seq = d["radar_sample_sequence"].astype(np.int64); order = np.argsort(seq, kind="stable")
rp = proc.radar_config
det = 0; n = 0
for i in order[:120]:
    pkt = bytes(raw[off[i]:off[i+1]]); pl = pkt[12:] if len(pkt) >= 12 else pkt
    try:
        adc = read_uint12(pl)
        split = split_samples(adc, 1, rp['num_chirps_per_frame'], rp['num_samples_per_chirp'], rp['num_antennas'])
        frame = np.transpose(split[0], (2, 0, 1))
        location, score, gui = proc.update(frame)
        ld = proc.last_detection
        n += 1
        if ld.get('detected'):
            det += 1
        if n % 10 == 0 or ld.get('detected'):
            print(f"f{n}: det={ld.get('detected')} loc={location} score={score:.2f} "
                  f"snr={ld.get('snr_db')} thr={ld.get('threshold_db')} peak={ld.get('peak_power_db')}")
    except Exception as e:
        print("err", e); break
print(f"detected {det}/{n} frames")
