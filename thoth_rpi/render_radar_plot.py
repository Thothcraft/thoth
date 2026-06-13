#!/usr/bin/env python3
import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _resolve_radar_root() -> Path:
    root = os.environ.get("THOTH_MMW_ROOT", "/home/pi/Desktop/thoth/WS/MMW-HAT/MMW-HAT-Release")
    return Path(root)


def _load_processor(cfg_dir: Path):
    radar_root = _resolve_radar_root()
    sys.path.insert(0, str(radar_root))

    from utility.helper import find_setting_in_directory
    from utility.mmw_cube_proc_v0 import CubeProcessor

    setting_fn = find_setting_in_directory(str(cfg_dir))
    with open(setting_fn, "r", encoding="utf-8") as handle:
        setting = json.load(handle)
    return CubeProcessor(setting, num_azimuth_bin=16, num_elevation_bin=16)


def _plot_axes(kind: str):
    mapping = {
        "range-doppler": ("Range", "Doppler"),
        "azimuth-range": ("Azimuth", "Range"),
        "azimuth-doppler": ("Azimuth", "Doppler"),
    }
    if kind not in mapping:
        raise ValueError(f"Unsupported radar plot kind: {kind}")
    return mapping[kind]


def main():
    if len(sys.argv) != 4:
        print("Usage: render_radar_plot.py <input.bin> <plot-kind> <output.png>", file=sys.stderr)
        return 1

    input_path = Path(sys.argv[1])
    plot_kind = sys.argv[2]
    output_path = Path(sys.argv[3])

    cfg_dir = _resolve_radar_root() / "radar_config" / "config_3rx_3m"
    processor = _load_processor(cfg_dir)

    with input_path.open("rb") as handle:
        frame_count = 0
        while True:
            version_bytes = handle.read(4)
            if not version_bytes:
                break

            version = int.from_bytes(version_bytes, byteorder="little", signed=False)
            if version != 0:
                break

            handle.read(4)
            data_len = int.from_bytes(handle.read(4), byteorder="little", signed=False)
            raw_frame = handle.read(data_len)
            if not raw_frame:
                break

            processor.process_raw_data(raw_frame)
            frame_count += 1

    if frame_count == 0:
        print("No radar frames found", file=sys.stderr)
        return 2

    axis_0, axis_1 = _plot_axes(plot_kind)
    image = np.log10(np.maximum(processor.vis_2d(axis_0, axis_1), 1e-9))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6), dpi=160)
    plt.imshow(image, cmap="viridis", aspect="auto")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
