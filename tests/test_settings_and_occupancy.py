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
from backend.radar_analysis import PersistentTargetIdentity, StreamingChunkAnalyzer, occupancy_label, occupancy_region
from backend.calibration import derive_thresholds
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
                "yellow_threshold_percent": 27,
                "green_threshold_percent": 71,
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
            self.assertEqual(reloaded["yellow_threshold_percent"], 27.0)
            self.assertEqual(reloaded["green_threshold_percent"], 71.0)
            self.assertFalse(reloaded["auto_occupancy_label_enabled"])
            self.assertEqual(reloaded["chunk_seconds"], 5.0)
            self.assertEqual(reloaded["system_mode"], "responsive")
            self.assertEqual(reloaded["occupancy_vote_chunks"], 3)
            self.assertEqual(reloaded["prediction_label_style"], "presence")
            self.assertTrue(reloaded["people_count_label_enabled"])
            self.assertTrue(reloaded["sleep_study_enabled"])

    def test_dashboard_persists_labels_and_sensor_toggles_as_capture_settings(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            saved = manager.save_device_settings({
                "labels": ["bedroom", "participant-1"],
                "sensors": {"usb_camera": False, "dreamhat_radar": True},
            })
            self.assertEqual(saved["labels"], ["bedroom", "participant-1"])
            self.assertFalse(saved["sensors"]["usb_camera"])
            self.assertTrue(saved["sensors"]["dreamhat_radar"])

    def test_dashboard_rebases_detection_regions_onto_brain_revision(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            manager.registered = True
            manager.auth_token = "device-token"
            manager.device_id = "device-1"
            remote = {**manager.default_capture_settings(), "revision": 7}
            canonical = {
                **remote,
                "yellow_threshold_percent": 31.0,
                "green_threshold_percent": 74.0,
                "revision": 8,
                "updated_at": "2026-07-14T12:00:00+00:00",
            }
            get_response = mock.Mock(status_code=200)
            get_response.json.return_value = {"capture_settings": remote}
            put_response = mock.Mock(status_code=200)
            put_response.json.return_value = {"capture_settings": canonical}
            with mock.patch.object(manager.session, "get", return_value=get_response), mock.patch.object(
                manager.session, "put", return_value=put_response
            ) as put, mock.patch.object(manager, "update_status", return_value=True):
                saved = manager.save_device_settings({
                    "yellow_threshold_percent": 31,
                    "green_threshold_percent": 74,
                })
            sent = put.call_args.kwargs["json"]
            self.assertEqual(sent["revision"], 7)
            self.assertEqual(sent["yellow_threshold_percent"], 31.0)
            self.assertEqual(sent["green_threshold_percent"], 74.0)
            self.assertEqual(saved["revision"], 8)
            self.assertEqual(saved["yellow_threshold_percent"], 31.0)
            self.assertEqual(saved["green_threshold_percent"], 74.0)
            self.assertFalse(saved["sync_pending"])

    def test_frequent_heartbeat_does_not_rescan_complete_inventory(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            manager._last_file_report_at = 100.0
            with mock.patch("backend.device_manager.time.time", return_value=120.0), mock.patch.object(
                manager, "_get_data_files_list", side_effect=AssertionError("unexpected inventory scan")
            ):
                self.assertIsNone(manager._heartbeat_file_payload())

    def test_region_thresholds_are_strictly_validated(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            with self.assertRaisesRegex(ValueError, "yellow < green"):
                manager.save_capture_settings({"yellow_threshold_percent": 70, "green_threshold_percent": 60})

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
        self.assertEqual(occupancy_region(19, 100, 20, 60), "red")
        self.assertEqual(occupancy_region(20, 100, 20, 60), "yellow")
        self.assertEqual(occupancy_region(59, 100, 20, 60), "yellow")
        self.assertEqual(occupancy_region(60, 100, 20, 60), "green")

    def test_calibration_uses_adjacent_median_midpoints(self):
        result = derive_thresholds({"red": [4, 6, 8], "yellow": [38, 40, 42], "green": [84, 86, 88]})
        self.assertEqual(result["yellow_threshold_percent"], 23.0)
        self.assertEqual(result["green_threshold_percent"], 63.0)
        with self.assertRaisesRegex(ValueError, "ordered"):
            derive_thresholds({"red": [30], "yellow": [20], "green": [90]})

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
                return [{"id": 1, "lateral_m": 0.1, "forward_m": 1.0, "vertical_m": 1.0}] if self.calls == 1 else []

        room = {"width_m": 4, "depth_m": 5, "height_m": 3, "sensor_wall": "Back", "sensor_position_m": 2, "sensor_height_m": 1}
        processor = Processor()
        with tempfile.TemporaryDirectory() as root, mock.patch("backend.radar_analysis.load_radar_config", return_value={}):
            csv_path = Path(root) / "chunk.csv"
            analyzer = StreamingChunkAnalyzer(processor, csv_path, 0, 10, room, 11, 50, 20, 50, live_state_path=None)
            with mock.patch("backend.radar_analysis.decode_radar_frame", return_value=(1, object())):
                analyzer.process(b"frame-1")
                analyzer.process(b"frame-2")
            result = analyzer.finish()
            self.assertEqual(processor.calls, 2)
            self.assertEqual(result["occupancy"]["label"], "occupied")
            self.assertEqual(result["occupancy"]["classification"], "green")
            self.assertEqual(result["occupancy"]["evaluated_frames"], 2)
            self.assertEqual(len(result["frames"]), 2)
            self.assertEqual(result["frame_interval_ms"], 5000)
            self.assertTrue(csv_path.exists())
            self.assertFalse(csv_path.with_suffix(".csv.tmp").exists())

    def test_chunk_labels_join_metadata_and_minute_vote(self):
        room = {"zones": [{"id": "bed", "label": "Bedroom", "x": 0.5, "y": 0.5, "width": 1.5, "depth": 1.5}]}
        settings = {"auto_occupancy_label_enabled": True, "prediction_label_style": "presence", "people_count_label_enabled": True, "occupancy_vote_chunks": 2}
        chunks = []
        for index, label in enumerate(("occupied", "empty", "occupied")):
            tracked = {"id": 4, "position": [1.1, 1.1, 1]}
            frames = [{"targets": [tracked] if label == "occupied" and frame < 6 else []} for frame in range(10)]
            result = {"chunk_index": index, "chunk_seconds": 10, "occupancy": {"label": label, "detected_frames": 8 if label == "occupied" else 1, "evaluated_frames": 10, "threshold_percent": 50}, "targets": [dict(tracked)] if label == "occupied" else [], "frames": frames, "bin_path": f"raw_{index}.bin", "csv_path": f"xy_{index}.csv"}
            chunks.append(annotate_chunk_result(result, settings, room, ["care", "participant-1"], "20260713_1200", 3, index * 10))
        self.assertIn("present", chunks[0]["labels"])
        self.assertIn("people_count:1", chunks[0]["labels"])
        self.assertIn("zone:Bedroom", chunks[0]["labels"])
        self.assertEqual(chunks[0]["activity_labels"], ["present", "occupied", "zone:Bedroom"])
        self.assertEqual(chunks[0]["activity"]["targets"][0]["zone_frames"]["Bedroom"], 6)
        self.assertEqual(chunks[0]["targets"][0]["zones"], ["Bedroom"])
        self.assertEqual(chunks[1]["join"]["previous_chunk_id"], "20260713_1200:00")
        summary = summarize_minute_results(chunks, settings, ["care", "participant-1"])
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
                self.assertEqual(post.call_count, 7)
                payload = post.call_args_list[0].kwargs["json"]
                self.assertEqual(payload["state"], "on")
                self.assertEqual(payload["attributes"]["chunk_index"], 2)
                self.assertEqual(payload["attributes"]["coordinates"], {"x": 1.2, "y": 3.4})
                self.assertEqual(post.call_args_list[1].kwargs["json"]["state"], "green")
                self.assertEqual(post.call_args_list[2].kwargs["json"]["state"], 1)
                self.assertEqual(post.call_args_list[3].kwargs["json"]["attributes"]["targets"][0]["target_id"], 7)
                self.assertEqual(post.call_args_list[4].kwargs["json"]["state"], "none")
                with mock.patch("backend.home_assistant.requests.post", return_value=response) as minute_post:
                    minute_result = home_assistant.publish_occupancy(
                        {"label": "occupied", "occupied_chunks": 3, "evaluated_chunks": 6, "vote_required_chunks": 3},
                        "20260713_1200", scope="minute", people_count=2,
                        labels=["care", "present", "people_count:2", "zone:Bedroom"],
                        activity_labels=["present", "occupied", "zone:Bedroom"],
                        activity={"state": "occupied", "zones": ["Bedroom"]},
                    )
                self.assertIn("binary_sensor.thoth_minute_occupancy", minute_result["entity_ids"])
                self.assertTrue(any("sensor.thoth_minute_labels" in call.args[0] for call in minute_post.call_args_list))
                self.assertTrue(any("sensor.thoth_minute_activity" in call.args[0] and call.kwargs["json"]["attributes"]["labels"] == ["present", "occupied", "zone:Bedroom"] for call in minute_post.call_args_list))


if __name__ == "__main__":
    unittest.main()
