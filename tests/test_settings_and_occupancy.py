import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

from backend.device_manager import DeviceManager
from backend import home_assistant
from backend.radar_analysis import StreamingChunkAnalyzer, occupancy_label


class _Config:
    def __init__(self, root: str):
        self.CONFIG_DIR = root
        self.DATA_DIR = root
        self.BASE_DIR = root
        self.MODELS_DIR = os.path.join(root, "models")
        self.DEVICE_ID_FILE = os.path.join(root, "device_id.txt")
        self.LEGACY_DEVICE_ID_FILE = self.DEVICE_ID_FILE
        self.BRAIN_SERVER_URL = ""


class SettingsTests(unittest.TestCase):
    def test_local_save_is_canonical_revisioned_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            saved = manager.save_capture_settings({
                "radar_detection_threshold_db": 12.5,
                "occupancy_threshold_percent": 62,
                "auto_occupancy_label_enabled": False,
            })
            self.assertEqual(saved["revision"], 1)
            self.assertIsNotNone(saved["updated_at"])
            reloaded = DeviceManager(_Config(root)).load_capture_settings()
            self.assertEqual(reloaded["radar_detection_threshold_db"], 12.5)
            self.assertEqual(reloaded["occupancy_threshold_percent"], 62.0)
            self.assertFalse(reloaded["auto_occupancy_label_enabled"])

    def test_persistence_failure_is_reported(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            with mock.patch("backend.device_manager.os.replace", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "disk full"):
                    manager.save_capture_settings({"occupancy_threshold_percent": 75})

    def test_older_remote_revision_does_not_overwrite_offline_edit(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            local = manager.save_capture_settings({"occupancy_threshold_percent": 75})
            manager._apply_response_settings({"capture_settings": {
                **local,
                "revision": 0,
                "occupancy_threshold_percent": 10,
            }})
            self.assertEqual(manager.load_capture_settings()["occupancy_threshold_percent"], 75.0)


class OccupancyTests(unittest.TestCase):
    def test_threshold_is_per_chunk_and_inclusive(self):
        self.assertEqual(occupancy_label(50, 100, 50), "occupied")
        self.assertEqual(occupancy_label(49, 100, 50), "empty")
        self.assertEqual(occupancy_label(0, 0, 0), "empty")

    def test_streaming_analyzer_processes_each_live_frame_and_finalizes_csv(self):
        class Processor:
            threshold_db = 8.0
            last_detection = {}
            last_motion_shadow = {"points": [], "intensity": []}

            def __init__(self):
                self.calls = 0

            def update(self, _frame):
                self.calls += 1
                self.last_detection = {"detected": self.calls == 1, "threshold_db": self.threshold_db}
                return []

        room = {"width_m": 4, "depth_m": 5, "height_m": 3, "sensor_wall": "Back", "sensor_position_m": 2, "sensor_height_m": 1}
        processor = Processor()
        with tempfile.TemporaryDirectory() as root, mock.patch("backend.radar_analysis.load_radar_config", return_value={}):
            csv_path = Path(root) / "chunk.csv"
            analyzer = StreamingChunkAnalyzer(processor, csv_path, 0, 10, room, 11, 50)
            with mock.patch("backend.radar_analysis.decode_radar_frame", return_value=(1, object())):
                analyzer.process(b"frame-1")
                analyzer.process(b"frame-2")
            result = analyzer.finish()
            self.assertEqual(processor.calls, 2)
            self.assertEqual(result["occupancy"]["label"], "occupied")
            self.assertEqual(result["occupancy"]["evaluated_frames"], 2)
            self.assertTrue(csv_path.exists())
            self.assertFalse(csv_path.with_suffix(".csv.tmp").exists())


class HomeAssistantTests(unittest.TestCase):
    def test_blank_token_preserves_secret_and_payload_has_chunk_attributes(self):
        with tempfile.TemporaryDirectory() as root:
            config_path = Path(root) / "home_assistant.json"
            with mock.patch.object(home_assistant, "CONFIG_PATH", config_path):
                home_assistant.save_home_assistant_config({"token": "secret", "enabled": True})
                home_assistant.save_home_assistant_config({"token": "", "entity_id": "thoth"})
                self.assertTrue(home_assistant.load_home_assistant_config()["configured"])
                response = mock.Mock()
                response.raise_for_status.return_value = None
                with mock.patch("backend.home_assistant.requests.post", return_value=response) as post:
                    result = home_assistant.publish_occupancy(
                        {"label": "occupied", "detected_frames": 8, "evaluated_frames": 10, "ratio": .8, "threshold_percent": 50},
                        "20260713_1200", chunk_index=2, location=[1.2, 3.4], confidence=.9,
                    )
                self.assertTrue(result["success"])
                payload = post.call_args.kwargs["json"]
                self.assertEqual(payload["state"], "on")
                self.assertEqual(payload["attributes"]["chunk_index"], 2)
                self.assertEqual(payload["attributes"]["coordinates"], {"x": 1.2, "y": 3.4})


if __name__ == "__main__":
    unittest.main()
