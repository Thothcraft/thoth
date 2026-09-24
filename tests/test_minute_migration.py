"""Fixture-based minute migration tests (§10).

Legacy chunk-era manifests must normalize into thoth-minute/v1 without
losing timestamps, sources, predictions, or labels — and migration must
be explicit + non-destructive.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from whispy.minutes import (  # noqa: E402
    iter_minute_dirs, read_minute, write_minute_manifest,
)
from whispy.contracts import MinuteManifest  # noqa: E402

from tools.migrate_minutes import migrate  # noqa: E402
from tools.verify_minute_migration import verify  # noqa: E402
from tools.audit_minutes import audit  # noqa: E402


LEGACY_V7 = {
    "schema": "thoth-minute-manifest/v7",
    "folder_minute": "20260915_1200",
    "scheduled_start": "2026-09-15T12:00:00",
    "capture_started": "2026-09-15T12:00:00.5",
    "capture_finished": "2026-09-15T12:01:00.2",
    "duration_seconds": 59.7,
    "chunk_seconds": 1.0,
    "expected_chunks": 60,
    "status": "complete",
    "host": "pi2",
    "device_id": "pi2",
    "device_name": "Thoth Pi 2",
    "labels": ["occupied", "zone:desk"],
    "sensors_enabled": ["radar", "wifi_csi"],
    "warnings": [],
    "errors": [],
    "container": {"file": "capture.npz", "second_count": 60,
                  "radar_samples": 600, "csi_samples": 12000},
    "outputs": {
        "radar": {
            "sample_count": 600, "average_sampling_rate_hz": 10.0,
            "chunks": [
                {"chunk_index": 0, "status": "occupied", "score": 0.9},
                {"chunk_index": 1, "status": "empty", "score": 0.1},
            ],
        },
        "wifi_csi": {
            "receivers": [
                {"device_id": "csi-1", "device": "COM4",
                 "sample_count": 6000, "average_sampling_rate_hz": 100.0},
                {"device_id": "csi-2", "device": "COM5",
                 "sample_count": 6000, "average_sampling_rate_hz": 100.0},
            ],
        },
        "camera": {"type": "usb", "device": "/dev/video0",
                   "sample_count": 60, "average_sampling_rate_hz": 1.0},
    },
    "model_predictions": [{"model": "occ-v1", "prediction": "occupied",
                           "confidence": 0.9}],
}


@pytest.fixture
def legacy_minute(tmp_path):
    d = tmp_path / "20260915_1200"
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps(LEGACY_V7))
    (d / "capture.npz").write_bytes(b"NPZ-FIXTURE")
    (d / "predictions.json").write_text(json.dumps({
        "generated_at": "2026-09-15T12:01:05",
        "timeline": [
            {"chunk_index": 0, "occupied": True, "classification": "green"},
            {"chunk_index": 1, "occupied": False, "classification": "red"},
        ],
    }))
    return d


def test_legacy_manifest_normalizes(legacy_minute):
    m = read_minute(legacy_minute)
    assert m.format == "thoth-minute/v1"
    assert m.minute_id == "20260915_1200"
    assert m.device_id == "pi2"
    assert m.start_timestamp > 0
    assert abs(m.duration_seconds - 59.7) < 0.01
    # sources: radar + camera + 2 csi receivers
    modalities = sorted(s.modality for s in m.sources)
    assert modalities == ["camera", "csi", "csi", "radar"]
    # chunk_index normalized to second_index in predictions
    assert m.predictions[0]["second_index"] == 0
    assert "chunk_index" not in m.predictions[0]
    # legacy chunk entries normalized into quality.seconds
    seconds = m.quality["seconds"]
    assert seconds[0]["second_index"] == 0
    assert seconds[0]["status"] == "occupied"
    assert m.quality["expected_seconds"] == 60
    assert m.labels["labels"] == ["occupied", "zone:desk"]
    assert m.files["npz"] == "capture.npz"


def test_migration_is_explicit_and_nondestructive(legacy_minute, tmp_path):
    report = migrate(tmp_path)
    assert report["totals"]["migrated"] == 1
    canonical = legacy_minute / "minute.json"
    assert canonical.exists()
    # legacy files untouched
    assert (legacy_minute / "manifest.json").exists()
    assert (legacy_minute / "capture.npz").read_bytes() == b"NPZ-FIXTURE"
    # idempotent: second run skips
    report2 = migrate(tmp_path)
    assert report2["totals"]["skipped"] == 1
    # verify passes
    v = verify(tmp_path)
    assert v["totals"] == {"verified": 1, "failed": 0}


def test_audit_counts(legacy_minute, tmp_path):
    report = audit(tmp_path)
    assert report["totals"]["minutes"] == 1
    assert report["totals"]["legacy"] == 1
    migrate(tmp_path)
    report = audit(tmp_path)
    assert report["totals"]["canonical"] == 1


def test_canonical_round_trip(legacy_minute):
    m = read_minute(legacy_minute)
    path = write_minute_manifest(legacy_minute, m)
    loaded = MinuteManifest.from_dict(json.loads(path.read_text()))
    assert loaded.minute_id == m.minute_id
    assert loaded.format == "thoth-minute/v1"


def test_iter_minute_dirs_finds_label_subdirs(tmp_path):
    (tmp_path / "sleep" / "20260915_0300").mkdir(parents=True)
    (tmp_path / "20260915_0400").mkdir()
    (tmp_path / "not-a-minute").mkdir()
    found = {d.name for d in iter_minute_dirs(tmp_path)}
    assert found == {"20260915_0300", "20260915_0400"}
