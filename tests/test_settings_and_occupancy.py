import json
import os
import queue
import sys
import tempfile
import types
import unittest
from collections import deque
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
if "dotenv" not in sys.modules:
    dotenv = types.ModuleType("dotenv")
    dotenv.load_dotenv = lambda *args, **kwargs: None
    sys.modules["dotenv"] = dotenv

from backend.device_manager import DeviceManager
from backend import home_assistant
from backend.capture_manager import minute_summary
from backend.radar_analysis import (
    PersistentTargetIdentity,
    SigProc,
    StreamingChunkAnalyzer,
    _update_example2_xy_plot,
    occupancy_label,
    occupancy_region,
)
from backend.calibration import derive_thresholds
from backend.minute_collector import (
    annotate_chunk_result,
    enqueue_latest_chunk_frame,
    live_chunk_statistics,
    minute_start,
    summarize_minute_results,
)
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
                "radar_detection_threshold_normalized": 0.52,
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
            self.assertEqual(reloaded["radar_detection_threshold_normalized"], 0.52)
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

    def test_minute_summary_never_returns_an_unlabeled_minute(self):
        with tempfile.TemporaryDirectory() as root:
            minute_dir = Path(root) / "20260716_1200"
            minute_dir.mkdir()
            (minute_dir / "manifest.json").write_text(json.dumps({
                "capture_finished": "2026-07-16T12:01:00Z",
                "expected_chunks": 1,
                "labels": [],
                "outputs": {
                    "radar": {
                        "chunks": [{
                            "chunk_index": 0,
                            "status": "occupied",
                            "labels": [],
                            "detected_frames": 8,
                            "evaluated_frames": 10,
                        }],
                    },
                },
            }), encoding="utf-8")
            self.assertEqual(minute_summary(minute_dir)["labels"], ["occupied", "present"])

            empty_dir = Path(root) / "20260716_1201"
            empty_dir.mkdir()
            (empty_dir / "manifest.json").write_text(json.dumps({
                "capture_finished": "2026-07-16T12:02:00Z",
                "labels": [],
                "outputs": {},
            }), encoding="utf-8")
            self.assertEqual(minute_summary(empty_dir)["labels"], ["no-radar-data"])

    def test_presence_example2_plot_uses_native_processor_and_grid(self):
        native_map = np.zeros((200, 400), dtype=float)
        native_map[40:48, 216:224] = 1.0
        exact = types.SimpleNamespace(
            dbf=types.SimpleNamespace(run=lambda _spectrum: np.ones((8, 4, 55))),
            num_beams=55,
            target_detection=mock.Mock(return_value=(native_map, np.array([1.0, 0.5]), 1.0)),
            x_bin=np.arange(0.0, 5.0, 0.025),
            y_bin=np.arange(-5.0, 5.0, 0.025),
            xy_map_buffer=deque(maxlen=10),
            buffer_decay=0.8,
            xy_marker_half_width_cells=4,
            processing_config={"spatial_resolution": 0.025},
        )
        processor = types.SimpleNamespace(
            _rd_spectrum=np.ones((8, 4, 3)),
            _thoth_example2_processor=exact,
        )
        first = _update_example2_xy_plot(processor)
        self.assertEqual((first["rows"], first["columns"]), (200, 400))
        self.assertEqual(first["levels"], [0.0, 1.0])
        self.assertTrue(first["transpose"])
        self.assertTrue(first["mirror_x"])
        self.assertTrue(first["mirror_y"])
        self.assertEqual(max(value for _index, value in first["values_sparse"]), 255)
        exact.target_detection.assert_called_once()
        self.assertEqual(first["source"], "example_2_track/location_gui.py")

    def test_dashboard_saves_detection_regions_without_waiting_for_brain(self):
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
            with mock.patch.object(manager.session, "get", return_value=get_response) as get, mock.patch.object(
                manager.session, "put", return_value=put_response
            ) as put:
                saved = manager.save_device_settings({
                    "yellow_threshold_percent": 31,
                    "green_threshold_percent": 74,
                })
                get.assert_not_called()
                put.assert_not_called()
                self.assertEqual(saved["yellow_threshold_percent"], 31.0)
                self.assertEqual(saved["green_threshold_percent"], 74.0)
                self.assertTrue(saved["sync_pending"])

                self.assertTrue(manager._sync_capture_settings_to_brain(manager.load_capture_settings()))
                sent = put.call_args.kwargs["json"]
                self.assertEqual(sent["revision"], 7)
                self.assertEqual(sent["yellow_threshold_percent"], 31.0)
                self.assertEqual(sent["green_threshold_percent"], 74.0)
                self.assertEqual(manager.load_capture_settings()["revision"], 8)

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

    def test_newer_remote_revision_does_not_overwrite_pending_detection_regions(self):
        with tempfile.TemporaryDirectory() as root:
            manager = DeviceManager(_Config(root))
            local = manager.save_capture_settings({
                "yellow_threshold_percent": 31,
                "green_threshold_percent": 74,
            })
            manager._apply_response_settings({"capture_settings": {
                **local,
                "revision": int(local["revision"]) + 5,
                "updated_at": "2999-01-01T00:00:00+00:00",
                "yellow_threshold_percent": 20,
                "green_threshold_percent": 60,
            }})
            persisted = manager.load_capture_settings()
            self.assertEqual(persisted["yellow_threshold_percent"], 31.0)
            self.assertEqual(persisted["green_threshold_percent"], 74.0)
            self.assertTrue(manager._settings_sync_pending)


class OccupancyTests(unittest.TestCase):
    def test_range_angle_products_returns_computed_static_views(self):
        import numpy as np

        class Doppler:
            range_window = np.ones(3)

            @staticmethod
            def compute_doppler_map(_frame, _antenna):
                return np.zeros((4, 3), dtype=complex)

        class Beamformer:
            @staticmethod
            def run(_spectrum):
                return np.ones((4, 3, 5), dtype=complex)

        processor = SigProc.__new__(SigProc)
        processor._rd_spectrum = np.zeros((4, 3, 3), dtype=complex)
        processor._static_spectrum = np.zeros((4, 3, 3), dtype=complex)
        processor.doppler = Doppler()
        processor.azimuth_dbf = Beamformer()
        processor.elevation_dbf = Beamformer()
        processor.num_azimuth_beams = 5
        frame = np.zeros((3, 3, 4), dtype=float)

        with mock.patch("signal_proc.fft_spectrum", return_value=np.zeros((3, 4))):
            products = processor.range_angle_products(frame)

        self.assertEqual(len(products), 6)
        self.assertEqual(products[3].shape, (4, 5))
        self.assertEqual(products[4].shape, (4, 3, 5))

    def test_live_chunk_statistics_classifies_in_progress_frames(self):
        analyzer = types.SimpleNamespace(
            evaluated_frames=5,
            detected_frames=3,
            yellow_threshold_percent=20,
            green_threshold_percent=60,
            occupancy_threshold_percent=50,
            last_position=(1.25, 2.5),
            last_score=0.87,
            last_targets=[{"id": 4}, {"id": 8}],
        )

        statistics = live_chunk_statistics(analyzer)

        self.assertEqual(statistics["status"], "collecting")
        self.assertEqual(statistics["classification"], "green")
        self.assertTrue(statistics["occupied"])
        self.assertEqual(statistics["detected_frames"], 3)
        self.assertEqual(statistics["evaluated_frames"], 5)
        self.assertEqual(statistics["ratio"], 0.6)
        self.assertEqual(statistics["people_count"], 2)
        self.assertEqual(statistics["location"], [1.25, 2.5])

    def test_analysis_backlog_keeps_latest_frames_within_each_chunk(self):
        jobs: queue.Queue = queue.Queue(maxsize=96)
        entry = {"chunk_index": 2}
        jobs.put(("start", entry, {}))
        replacements = [
            enqueue_latest_chunk_frame(jobs, entry, bytes([index]), float(index))
            for index in range(7)
        ]
        jobs.put(("end", entry))
        queued = list(jobs.queue)
        frames = [item[2][0] for item in queued if item[0] == "frame"]
        self.assertEqual(frames, [6])
        self.assertEqual(sum(replacements), 6)
        self.assertEqual(queued[0][0], "start")
        self.assertEqual(queued[-1][0], "end")

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
        self.assertEqual(occupancy_region(20, 100, 20, 60), "red")
        self.assertEqual(occupancy_region(59, 100, 20, 60), "red")
        self.assertEqual(occupancy_region(60, 100, 20, 60), "green")

    def test_radar_peak_gate_uses_normalized_presence_value(self):
        import numpy as np

        processor = SigProc.__new__(SigProc)
        processor.range_bin = np.linspace(0.0, 1.0, 6)
        processor.azimuth_bin = np.linspace(-20.0, 20.0, 9)
        processor.dead_zone = 0.0
        processor.cfar_training_range = 3
        processor.cfar_training_angle = 3
        processor.cfar_guard_range = 1
        processor.cfar_guard_angle = 1
        processor.normalized_threshold = 0.6
        processor.min_peak_separation_m = 0.1
        processor.min_peak_separation_deg = 2.0
        processor.max_candidates = 2
        processor.max_targets = 2
        processor.cluster_min_cells = 0
        processor.cluster_min_intensity = 0.0
        processor._merge_person_detections = lambda detections: detections
        processor._measure_detection = lambda *args: {
            "range_m": float(processor.range_bin[args[3]]),
            "azimuth_deg": float(processor.azimuth_bin[args[4]]),
            "cluster_cells": 1,
            "cluster_intensity": 1.0,
            "normalized_peak": float(args[-1]),
        }
        energy_db = np.zeros((6, 9), dtype=float)
        energy_db[3, 4] = 30.0
        normalized = np.full((6, 9), 0.1, dtype=float)
        normalized[3, 4] = 0.59
        cube = np.zeros((6, 4, 9), dtype=complex)

        self.assertEqual(
            processor._candidate_peaks(energy_db, 0.0, cube, cube, normalized), []
        )
        normalized[3, 4] = 0.6
        detections = processor._candidate_peaks(
            energy_db, 0.0, cube, cube, normalized
        )
        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0]["normalized_peak"], 0.6)

    def test_calibration_uses_adjacent_median_midpoints(self):
        result = derive_thresholds({"red": [4, 6, 8], "yellow": [38, 40, 42], "green": [84, 86, 88]})
        self.assertEqual(result["yellow_threshold_percent"], 23.0)
        self.assertEqual(result["green_threshold_percent"], 63.0)
        with self.assertRaisesRegex(ValueError, "ordered"):
            derive_thresholds({"red": [30], "yellow": [20], "green": [90]})

    def test_streaming_analyzer_processes_each_live_frame_and_finalizes_csv(self):
        class Processor:
            normalized_threshold = 0.45
            last_detection = {}
            last_motion_shadow = {"points": [], "intensity": []}

            def __init__(self):
                self.calls = 0

            def update(self, _frame):
                self.calls += 1
                self.last_detection = {
                    "detected": self.calls == 1,
                    "threshold_normalized": self.normalized_threshold,
                    "normalized_peak": 0.8 if self.calls == 1 else 0.0,
                }
                return [{"id": 1, "lateral_m": 0.1, "forward_m": 1.0, "vertical_m": 1.0}] if self.calls == 1 else []

        room = {"width_m": 4, "depth_m": 5, "height_m": 3, "sensor_wall": "Back", "sensor_position_m": 2, "sensor_height_m": 1}
        processor = Processor()
        with tempfile.TemporaryDirectory() as root, mock.patch("backend.radar_analysis.load_radar_config", return_value={}):
            csv_path = Path(root) / "chunk.csv"
            analyzer = StreamingChunkAnalyzer(processor, csv_path, 0, 10, room, 0.5, 50, 20, 50, live_state_path=None)
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

    def test_streaming_analyzer_can_analyze_without_creating_radar_csv(self):
        class Processor:
            normalized_threshold = 0.45
            last_detection = {}
            last_motion_shadow = {"points": [], "intensity": []}
            last_intensity_views = {}

            def update(self, _frame):
                self.last_detection = {
                    "detected": False,
                    "threshold_normalized": self.normalized_threshold,
                    "normalized_peak": 0.0,
                }
                return []

        room = {"width_m": 4, "depth_m": 5, "height_m": 3, "sensor_wall": "Back", "sensor_position_m": 2, "sensor_height_m": 1}
        with mock.patch("backend.radar_analysis.load_radar_config", return_value={}):
            analyzer = StreamingChunkAnalyzer(Processor(), None, 0, 1, room, 0.5, 50, 0, 50, live_state_path=None)
            with mock.patch("backend.radar_analysis.decode_radar_frame", return_value=(1, object())):
                for _ in range(10):
                    self.assertTrue(analyzer.process(b"frame"))
            result = analyzer.finish()
        self.assertEqual(result["occupancy"]["evaluated_frames"], 10)
        self.assertEqual(result["occupancy"]["classification"], "red")

    def test_occupancy_does_not_count_a_track_without_current_map_candidate(self):
        class Processor:
            normalized_threshold = 0.45
            last_detection = {}
            last_motion_shadow = {"points": [], "intensity": []}
            last_intensity_views = {}

            def update(self, _frame):
                self.last_detection = {
                    "detected": True,
                    "candidate_count": 0,
                    "threshold_normalized": self.normalized_threshold,
                }
                return [{"id": 1, "lateral_m": 0.2, "forward_m": 1.2, "vertical_m": 1.0}]

        room = {"width_m": 4, "depth_m": 5, "height_m": 3, "sensor_wall": "Back", "sensor_position_m": 2, "sensor_height_m": 1}
        with mock.patch("backend.radar_analysis.load_radar_config", return_value={}):
            analyzer = StreamingChunkAnalyzer(Processor(), None, 0, 1, room, 0.5, 50, 0, 50, live_state_path=None)
            with mock.patch("backend.radar_analysis.decode_radar_frame", return_value=(1, object())):
                for _ in range(10):
                    analyzer.process(b"frame")
            result = analyzer.finish()
        self.assertEqual(result["occupancy"]["detected_frames"], 0)
        self.assertEqual(result["occupancy"]["label"], "empty")

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
