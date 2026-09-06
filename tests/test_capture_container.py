import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from backend.capture_container import (
    CONTAINER_FILENAME,
    CONTAINER_SCHEMA,
    build_capture_container,
    csi_average_series,
    read_camera_frame,
    read_capture_metadata,
    update_capture_metadata,
)


def radar_packet(sequence: int, payload: bytes) -> bytes:
    return (
        (0).to_bytes(4, "little")
        + sequence.to_bytes(4, "little")
        + len(payload).to_bytes(4, "little")
        + payload
    )


class CaptureContainerTests(unittest.TestCase):
    def test_builds_pickle_free_synchronized_container_and_removes_fragments(self):
        with tempfile.TemporaryDirectory() as temporary:
            minute = Path(temporary) / "20260906_1200"
            minute.mkdir()
            radar = minute / "radar_000.bin"
            radar.write_bytes(radar_packet(10, b"radar-a") + radar_packet(11, b"radar-b"))
            camera = minute / "camera_000.jpg"
            camera.write_bytes(b"\xff\xd8frame\xff\xd9")
            csi = minute / "wifi_csi.csv"
            with csi.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["host_timestamp", "monotonic_ns", "serial_port", "raw_csi_line"])
                writer.writerow(["2026-09-06T12:00:00.200+00:00", "1000200000", "/dev/ttyACM0", "CSI_DATA,x,[3 4 5 12]"])
                writer.writerow(["2026-09-06T12:00:01.100+00:00", "2100000000", "/dev/ttyACM0", "CSI_DATA,x,[8 15]"])

            manifest = {
                "capture_started": "2026-09-06T12:00:00+00:00",
                "capture_started_monotonic_ns": 1_000_000_000,
                "duration_seconds": 2,
                "labels": ["baseline"],
                "device_id": "device-1",
                "capture_settings": {"sensors": {"usb_camera": True}},
                "outputs": {
                    "radar": {"chunks": [{
                        "chunk_index": 0,
                        "bin_path": str(radar),
                        "started": "2026-09-06T12:00:00+00:00",
                        "finished_capture": "2026-09-06T12:00:00.900+00:00",
                        "frame_monotonic_ns": [1_100_000_000, 1_900_000_000],
                    }]},
                    "camera": {"frames": [{
                        "second_index": 0,
                        "path": str(camera),
                        "captured_at": "2026-09-06T12:00:00.300+00:00",
                        "monotonic_ns": 1_300_000_000,
                    }]},
                    "wifi_csi": {"receivers": [{
                        "path": str(csi), "device": "/dev/ttyACM0", "device_id": "csi-main", "baud": 115200,
                    }]},
                },
            }

            info = build_capture_container(minute, manifest)
            container = minute / CONTAINER_FILENAME
            self.assertEqual(info["schema"], CONTAINER_SCHEMA)
            self.assertEqual(info["bytes"], container.stat().st_size)
            self.assertEqual(info["second_count"], 2)
            self.assertEqual(info["camera_frames"], 1)
            self.assertEqual(info["radar_samples"], 2)
            self.assertEqual(info["csi_samples"], 2)
            self.assertFalse(radar.exists() or camera.exists() or csi.exists())
            self.assertEqual(read_camera_frame(container, 0), b"\xff\xd8frame\xff\xd9")
            self.assertIsNone(read_camera_frame(container, 1))
            self.assertEqual(csi_average_series(container), [9.0, 17.0])
            metadata = read_capture_metadata(container)
            self.assertEqual(metadata["seconds"][0]["radar_samples"], 2)
            self.assertEqual(metadata["seconds"][0]["csi_samples"], 1)
            self.assertEqual(metadata["seconds"][1]["csi_samples"], 1)
            with np.load(container, allow_pickle=False) as archive:
                self.assertEqual(archive["radar_sample_second_index"].tolist(), [0, 0])
                self.assertEqual(archive["csi_sample_second_index"].tolist(), [0, 1])

    def test_metadata_label_update_is_atomic_and_pickle_free(self):
        with tempfile.TemporaryDirectory() as temporary:
            minute = Path(temporary) / "20260906_1201"
            minute.mkdir()
            manifest = {
                "capture_started": "2026-09-06T12:01:00+00:00",
                "duration_seconds": 1,
                "labels": ["before"],
                "outputs": {},
            }
            info = build_capture_container(minute, manifest, remove_fragments=False)
            container = minute / info["filename"]
            update_capture_metadata(container, {"labels": ["after", "participant-1"]})
            metadata = read_capture_metadata(container)
            self.assertEqual(metadata["labels"], ["after", "participant-1"])
            self.assertEqual(metadata["manifest"]["labels"], ["after", "participant-1"])
            with np.load(container, allow_pickle=False) as archive:
                self.assertIn("metadata_json", archive.files)


if __name__ == "__main__":
    unittest.main()
