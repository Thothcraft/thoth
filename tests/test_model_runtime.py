import json

import numpy as np
import pytest
import torch
from unittest.mock import Mock

from src.backend.model_runtime import (
    ModelRegistry,
    ModelValidationError,
    append_manifest_predictions,
    validate_torchscript,
)
from src.backend import home_assistant


class RadarClassifier(torch.nn.Module):
    def forward(self, radar):
        total = radar.reshape(-1).sum()
        return torch.stack((total, -total))


class MultimodalClassifier(torch.nn.Module):
    def forward(self, radar, csi):
        return {"head": (torch.stack((radar.sum(), csi.sum())),)}


def metadata(inputs, *, path=None, classes=None, output_kind="logits"):
    return {
        "schema": "thoth-model/v1",
        "name": "test classifier",
        "version": "1.0",
        "inputs": inputs,
        "output": {"kind": output_kind, "path": path or []},
        "class_names": classes or ["first", "second"],
    }


def radar_input(representation="raw_adc", frames=2):
    return {
        "sensor": "radar",
        "representation": representation,
        "frames": frames,
        "shape": [4],
        "fit": "left_pad_latest",
        "normalization": {"kind": "zscore", "mean": 0, "std": 1},
    }


def csi_input(representation="iq"):
    return {
        "sensor": "csi",
        "representation": representation,
        "samples": 2,
        "receivers": [0],
        "subcarriers": [0, 1],
        "shape": [8],
        "fit": "right_pad_earliest",
        "normalization": {"kind": "minmax", "min": -20, "max": 20},
    }


def save_radar_model(path):
    traced = torch.jit.trace(RadarClassifier(), torch.zeros(4))
    torch.jit.save(traced, str(path))


def save_multimodal_model(path):
    traced = torch.jit.trace(MultimodalClassifier(), (torch.zeros(4), torch.zeros(8)), strict=False)
    torch.jit.save(traced, str(path))


@pytest.mark.parametrize("representation", ["raw_adc", "fft_power"])
def test_radar_only_upload_is_disabled_then_runs(tmp_path, representation):
    artifact = tmp_path / "radar.pt"
    save_radar_model(artifact)
    registry = ModelRegistry(tmp_path / "models")
    saved = registry.add(artifact, metadata([radar_input(representation)]))
    frames = [np.array([1, 2], dtype="<i2").tobytes()] * 2

    assert saved["enabled"] is False
    assert registry.run_enabled(frames, [], 0, "now") == []
    registry.set_enabled(saved["id"], True)
    result = registry.run_enabled(frames, [], 0, "now")[0]
    assert result["status"] == "ok"
    assert set(result["scores"]) == {"first", "second"}


@pytest.mark.parametrize("representation", ["iq", "magnitude_phase"])
def test_csi_only_representations_and_missing_samples(tmp_path, representation):
    artifact = tmp_path / "csi.pt"
    traced = torch.jit.trace(RadarClassifier(), torch.zeros(8))
    torch.jit.save(traced, str(artifact))
    registry = ModelRegistry(tmp_path / "models")
    saved = registry.add(artifact, metadata([csi_input(representation)]))
    registry.set_enabled(saved["id"], True)
    assert registry.run_enabled([], [(0, "CSI_DATA [1,2,3,4]")], 0, "now")[0]["status"] == "skipped"
    result = registry.run_enabled([], [(0, "CSI_DATA [1,2,3,4]")] * 2, 1, "later")[0]
    assert result["status"] == "ok"


def test_multimodal_output_selector_and_multiple_models(tmp_path):
    artifact = tmp_path / "multi.pth"
    save_multimodal_model(artifact)
    registry = ModelRegistry(tmp_path / "models")
    first = registry.add(artifact, metadata([radar_input(), csi_input()], path=["head", 0]))
    second = registry.add(artifact, metadata([radar_input(), csi_input()], path=["head", 0]))
    registry.set_enabled(first["id"], True)
    registry.set_enabled(second["id"], True)
    frames = [np.array([1, 2], dtype="<i2").tobytes()] * 2
    csi = [(0, "CSI_DATA [1,2,3,4]")] * 2
    assert [item["status"] for item in registry.run_enabled(frames, csi, 2, "now")] == ["ok", "ok"]


def test_invalid_artifact_dimension_and_malformed_input(tmp_path):
    artifact = tmp_path / "radar.pt"
    save_radar_model(artifact)
    with pytest.raises(ModelValidationError, match="class_names has 3"):
        validate_torchscript(artifact, metadata([radar_input()], classes=["a", "b", "c"]))
    registry = ModelRegistry(tmp_path / "models")
    saved = registry.add(artifact, metadata([radar_input()]))
    registry.set_enabled(saved["id"], True)
    result = registry.run_enabled([b"", b""], [], 0, "now")[0]
    assert result["status"] == "error"


def test_manifest_append_preserves_human_labels(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"schema": "thoth-minute-manifest/v7", "labels": ["reading"]}))
    append_manifest_predictions(manifest_path, [{"model_id": "m1", "model_name": "M", "model_version": "1", "status": "skipped", "chunk_index": 0, "timestamp": "now"}])
    saved = json.loads(manifest_path.read_text())
    assert saved["labels"] == ["reading"]
    assert saved["model_predictions"][0]["timeline"][0]["status"] == "skipped"


def test_model_occupancy_publishes_only_explicit_binary_class(tmp_path, monkeypatch):
    monkeypatch.setattr(home_assistant, "CONFIG_PATH", tmp_path / "ha.json")
    home_assistant.save_home_assistant_config({"enabled": True, "base_url": "http://ha", "token": "secret", "entity_id": "binary_sensor.room"})
    response = Mock(status_code=200)
    response.raise_for_status.return_value = None
    post = Mock(return_value=response)
    monkeypatch.setattr(home_assistant.requests, "post", post)
    published = home_assistant.publish_model_occupancy({"class": "occupied", "confidence": 0.9, "model_id": "m"}, "20260915_1200", chunk_index=2)
    assert published["status"] == "published"
    assert post.call_args.kwargs["json"]["state"] == "on"
    assert home_assistant.publish_model_occupancy({"class": "walking"}, "20260915_1200")["status"] == "not_occupancy_class"
