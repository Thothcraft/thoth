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
from backend.radar_analysis import PersistentTargetIdentity, StreamingChunkAnalyzer, occupancy_label
from backend.minute_collector import annotate_chunk_result, minute_start, summarize_minute_results
from collector import next_minute_boundary


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
    def test_next_capture_is_aligned_to_wall_clock(self):
        from datetime import datetime
        current = datetime.fromisoformat("2026-07-13T12:34:17-04:00")
        self.assertEqual(next_minute_boundary(current).isoformat(), "2026-07-13T12:35:00-04:00")

    def test_scheduled_capture_keeps_supervisor_minute(self):
        scheduled = minute_start(False, "2026-07-13T12:35:00-04:00")
        self.assertEqual(scheduled.isoformat(), "2026-07-13T12:35:00-04:00")

    def test_local_save_is_canonical_revisioned_and_survives_reload(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            saved = manager.save_capture_settings({
                "radar_detection_threshold_db": 12.5,
                "occupancy_threshold_percent": 62,
                "auto_occupancy_label_enabled": False,
                "chunk_seconds": 5,
                "system_mode": "responsive",
                "occupancy_vote_chunks": 3,
                "prediction_label_style": "presence",
                "people_count_label_enabled": True,
                "sleep_study_enabled": True,
            })
            self.assertEqual(saved["revision"], 1)
            self.assertIsNotNone(saved["updated_at"])
            reloaded = DeviceManager(_Config(root)).load_capture_settings()
            self.assertEqual(reloaded["radar_detection_threshold_db"], 12.5)
            self.assertEqual(reloaded["occupancy_threshold_percent"], 62.0)
            self.assertFalse(reloaded["auto_occupancy_label_enabled"])
            self.assertEqual(reloaded["chunk_seconds"], 5.0)
            self.assertEqual(reloaded["system_mode"], "responsive")
            self.assertEqual(reloaded["occupancy_vote_chunks"], 3)
            self.assertEqual(reloaded["prediction_label_style"], "presence")
            self.assertTrue(reloaded["people_count_label_enabled"])
            self.assertTrue(reloaded["sleep_study_enabled"])

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
    def test_target_ids_persist_across_minutes_with_position_error(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "target_ids.json"
            first = PersistentTargetIdentity(path, mode="balanced")
            targets = first.assign([{"id": 1, "position": [1.0, 2.0, 1.0]}])
            first.save()
            second = PersistentTargetIdentity(path, mode="balanced")
            continued = second.assign([{"id": 99, "position": [1.1, 2.05, 1.0]}])
            self.assertEqual(targets[0]["id"], continued[0]["id"])
            self.assertGreater(continued[0]["position_error_m"], 0)

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
            analyzer = StreamingChunkAnalyzer(processor, csv_path, 0, 10, room, 11, 50, live_state_path=None)
            with mock.patch("backend.radar_analysis.decode_radar_frame", return_value=(1, object())):
                analyzer.process(b"frame-1")
                analyzer.process(b"frame-2")
            result = analyzer.finish()
            self.assertEqual(processor.calls, 2)
            self.assertEqual(result["occupancy"]["label"], "occupied")
            self.assertEqual(result["occupancy"]["evaluated_frames"], 2)
            self.assertEqual(len(result["frames"]), 2)
            self.assertEqual(result["frame_interval_ms"], 5000)
            self.assertTrue(csv_path.exists())
            self.assertFalse(csv_path.with_suffix(".csv.tmp").exists())

    def test_chunk_labels_join_metadata_and_minute_vote(self):
        room = {"sleep_anchor": {"x": 1.0, "y": 1.0, "radius_m": 0.5}, "zones": [{"id": "bed", "label": "Bedroom", "x": 0.5, "y": 0.5, "width": 1.5, "depth": 1.5}]}
        settings = {"auto_occupancy_label_enabled": True, "prediction_label_style": "presence", "people_count_label_enabled": True, "sleep_study_enabled": True, "occupancy_vote_chunks": 2}
        chunks = []
        for index, label in enumerate(("occupied", "empty", "occupied")):
            result = {"chunk_index": index, "chunk_seconds": 10, "occupancy": {"label": label, "detected_frames": 8 if label == "occupied" else 1, "evaluated_frames": 10, "threshold_percent": 50}, "targets": [{"id": 4, "position": [1.1, 1.1, 1]}] if label == "occupied" else [], "bin_path": f"raw_{index}.bin", "csv_path": f"xy_{index}.csv"}
            chunks.append(annotate_chunk_result(result, settings, room, ["sleep-study", "participant-1"], "20260713_1200", 3, index * 10))
        self.assertIn("present", chunks[0]["labels"])
        self.assertIn("people_count:1", chunks[0]["labels"])
        self.assertIn("zone:Bedroom", chunks[0]["labels"])
        self.assertEqual(chunks[0]["targets"][0]["zones"], ["Bedroom"])
        self.assertEqual(chunks[1]["join"]["previous_chunk_id"], "20260713_1200:00")
        summary = summarize_minute_results(chunks, settings, ["sleep-study", "participant-1"])
        self.assertEqual(summary["occupancy"]["label"], "occupied")
        self.assertEqual(summary["occupancy"]["vote_required_chunks"], 2)
        self.assertEqual(summary["people_count"], 1)


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
                        targets=[{"id": 7, "position": [1.2, 3.4, 1.0], "position_error_m": .18}],
                    )
                self.assertTrue(result["success"])
                self.assertEqual(post.call_count, 6)
                payload = post.call_args_list[0].kwargs["json"]
                self.assertEqual(payload["state"], "on")
                self.assertEqual(payload["attributes"]["chunk_index"], 2)
                self.assertEqual(payload["attributes"]["coordinates"], {"x": 1.2, "y": 3.4})
                self.assertEqual(post.call_args_list[1].kwargs["json"]["state"], 1)
                self.assertEqual(post.call_args_list[2].kwargs["json"]["attributes"]["targets"][0]["target_id"], 7)
                self.assertEqual(post.call_args_list[3].kwargs["json"]["state"], "none")
                with mock.patch("backend.home_assistant.requests.post", return_value=response) as minute_post:
                    minute_result = home_assistant.publish_occupancy(
                        {"label": "occupied", "occupied_chunks": 3, "evaluated_chunks": 6, "vote_required_chunks": 3},
                        "20260713_1200", scope="minute", people_count=2,
                        labels=["sleep-study", "present", "people_count:2"],
                        sleep_proximity={"in_zone": True, "nearest_target_m": .25},
                    )
                self.assertIn("binary_sensor.thoth_minute_occupancy", minute_result["entity_ids"])
                self.assertTrue(any("sensor.thoth_minute_labels" in call.args[0] for call in minute_post.call_args_list))


if __name__ == "__main__":
    unittest.main()
