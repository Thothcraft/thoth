"""Thoth node tests — registry, deployment state machine, daemon SMA loop."""
import time

import pytest

from thoth.deployment import DeploymentManager
from thoth.models import ModelRegistry
from thoth.settings import ConfigStore


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def test_config_store(tmp_home):
    cfg = ConfigStore()
    assert cfg.device_id
    assert cfg.local_token
    cfg.set("foo", "bar")
    assert ConfigStore().get("foo") == "bar"


def test_model_registry_persists_runtime_id(tmp_home):
    reg = ModelRegistry()
    rm = reg.install(name="occ", processor="rule",
                     config={"rules": [{"when": "x > 0", "label": "y"}]})
    assert rm.runtime_model_id.startswith("rm-")
    # New registry instance reads the same persisted ID
    reg2 = ModelRegistry()
    assert reg2.get(rm.runtime_model_id) is not None
    reg2.activate(rm.runtime_model_id, True)
    assert ModelRegistry().active()[0].runtime_model_id == rm.runtime_model_id


def test_deployment_state_machine(tmp_home):
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    dep = dm.process("dep-1", {
        "manifest": {"format": "whispy-model/v1", "processor": "rule"},
        "name": "occ", "processor": "rule",
        "config": {"rules": [{"when": "x > 0", "label": "y"}]},
    })
    assert dep["state"] == "acknowledged"
    assert dep["runtime_model_id"].startswith("rm-")
    dm.activate("dep-1")
    assert dm.state("dep-1")["state"] == "active"


def test_deployment_rejects_bad_processor(tmp_home):
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    dm.receive("dep-2", {"processor": "onnx"})
    rec = dm.validate("dep-2")
    assert rec["state"] == "failed"
    assert rec["failure"]["code"] == "manifest_invalid"


def test_deployment_rejects_wrong_format(tmp_home):
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    dm.receive("dep-fmt", {
        "manifest": {"format": "thoth-model/v0", "processor": "rule"},
        "processor": "rule"})
    rec = dm.validate("dep-fmt")
    assert rec["state"] == "failed"
    assert rec["failure"]["code"] == "manifest_invalid"


def test_deployment_accepts_legacy_format(tmp_home):
    """Pre-rename ``thoth-model/v1`` manifests still deploy."""
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    dep = dm.process("dep-legacy", {
        "manifest": {"format": "thoth-model/v1", "processor": "rule"},
        "name": "occ", "processor": "rule",
        "config": {"rules": [{"when": "x > 0", "label": "y"}]},
    })
    assert dep["state"] == "acknowledged"


def test_deployment_torchscript_requires_artifact(tmp_home):
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    dm.receive("dep-ts", {
        "manifest": {"format": "whispy-model/v1", "processor": "torchscript"},
        "processor": "torchscript"})
    rec = dm.validate("dep-ts")
    assert rec["state"] == "failed"
    assert rec["failure"]["code"] == "manifest_invalid"


def test_deployment_artifact_hash_mismatch(tmp_home):
    import base64
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    blob = b"fake-torchscript-bytes"
    dm.receive("dep-hash", {
        "manifest": {
            "format": "whispy-model/v1", "processor": "torchscript",
            "artifact": {"sha256": "0" * 64},   # wrong hash
        },
        "processor": "torchscript",
        "artifact": base64.b64encode(blob).decode()})
    rec = dm.validate("dep-hash")
    assert rec["state"] == "failed"
    assert rec["failure"]["code"] == "manifest_invalid"


def test_deployment_redelivery_is_idempotent(tmp_home):
    """A lost ack must not reinstall a duplicate runtime model."""
    reg = ModelRegistry()
    dm = DeploymentManager(reg)
    payload = {
        "manifest": {"format": "whispy-model/v1", "processor": "rule"},
        "name": "occ", "processor": "rule",
        "config": {"rules": [{"when": "x > 0", "label": "y"}]},
    }
    first = dm.process("dep-idem", payload)
    assert first["state"] == "acknowledged"
    rmid = first["runtime_model_id"]
    # Brain redelivers the same deployment_id after a lost ack.
    second = dm.process("dep-idem", payload)
    assert second["state"] == "acknowledged"
    assert second["runtime_model_id"] == rmid
    # Still exactly one runtime model installed.
    assert len(reg.list()) == 1


def test_capture_resolve_sensors(tmp_home):
    from thoth.capture.manager import CaptureManager
    available = ["system-0", "radar-0", "camera-0"]
    # Omitted → all available sensors.
    assert CaptureManager.resolve_sensors(None, available) == available
    assert CaptureManager.resolve_sensors([], available) == available
    # Explicit subset passes through.
    assert CaptureManager.resolve_sensors(["radar-0"], available) == ["radar-0"]
    # Unknown id is rejected, never silently recorded.
    with pytest.raises(ValueError):
        CaptureManager.resolve_sensors(["lidar-9"], available)


def test_capture_start_rejects_unknown_sensor(tmp_home):
    from thoth.capture.manager import CaptureManager
    cm = CaptureManager(root=tmp_home / "caps")
    with pytest.raises(ValueError):
        cm.start("dev-1", ["nope-0"], available=["system-0"])
    rec = cm.start("dev-1", None, available=["system-0", "radar-0"])
    assert rec["sensors"] == ["system-0", "radar-0"]


def _fake_stream(sensor_id="system-0"):
    """A started SampleStream over an infinite fake source."""
    import itertools
    from whispy.contracts import SensorSample
    from whispy.streams import SampleStream

    def gen():
        for i in itertools.count():
            yield SensorSample.now("dev-1", sensor_id, "system",
                                   {"x": i}, sequence=i)
            time.sleep(0.005)
    stream = SampleStream(gen(), name=sensor_id)
    stream.start()
    return stream


def test_tail_sensor_cursor(tmp_home):
    """LAN tail returns seq-filtered samples; cursor is per-client."""
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon

    daemon = ThothDaemon(config=ConfigStore())
    stream = _fake_stream("system-0")
    daemon._streams["system-0"] = stream
    try:
        time.sleep(0.1)                       # let samples buffer
        r1 = daemon.tail_sensor("system-0", 0)
        assert r1["samples"], "expected buffered samples"
        cur = r1["cursor"]
        assert cur > 0
        # A second read at the cursor yields only newer samples.
        r2 = daemon.tail_sensor("system-0", cur)
        assert all(s["sequence"] is not None for s in r2["samples"])
        assert r2["cursor"] >= cur
        # Unknown sensor → None (route maps to 404).
        assert daemon.tail_sensor("nope-0", 0) is None
    finally:
        stream.close()


def test_tail_endpoint_over_http(tmp_home):
    """The authenticated /api/sensors/{id}/tail route serves samples."""
    pytest.importorskip("whispy")
    import json
    import urllib.error
    import urllib.request
    from thoth.daemon import ThothDaemon

    cfg = ConfigStore()
    cfg.set("local_port", 5997)
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    daemon.start()
    try:
        time.sleep(0.4)
        sid = next(iter(daemon._streams))
        url = f"http://127.0.0.1:5997/api/sensors/{sid}/tail?cursor=0"
        req = urllib.request.Request(url, headers={
            "Authorization": f"Bearer {cfg.local_token}"})
        with urllib.request.urlopen(req, timeout=5) as res:
            body = json.loads(res.read().decode())
        assert body["sensor_id"] == sid
        assert "cursor" in body and "samples" in body
        # Unknown sensor → 404.
        bad = urllib.request.Request(
            "http://127.0.0.1:5997/api/sensors/nope-9/tail",
            headers={"Authorization": f"Bearer {cfg.local_token}"})
        with pytest.raises(urllib.error.HTTPError) as ei:
            urllib.request.urlopen(bad, timeout=5)
        assert ei.value.code == 404
    finally:
        daemon.stop()


def test_daemon_sma_loop(tmp_home):
    """Daemon opens whispy sensors, runs the loop, serves the local API."""
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon
    from thoth.ipc import DaemonClient

    cfg = ConfigStore()
    cfg.set("local_port", 5999)
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    daemon.start()
    try:
        time.sleep(0.6)  # let streams + a few ticks run
        st = daemon.status()
        assert st["running"]
        assert st["sensors"], "expected at least the system sensor"
        # local API reachable with the configured token
        client = DaemonClient(cfg, port=5999)
        assert client.is_running()
        api_status = client.status()
        assert api_status["device_id"] == st["device_id"]
    finally:
        daemon.stop()


def test_daemon_prediction_injection_drives_actuators(tmp_home):
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon

    cfg = ConfigStore()
    cfg.set("local_port", 5998)
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    # Install an active model with a webhook action pointing nowhere —
    # the action must produce an explicit result, not a crash.
    rm = daemon.registry.install(
        name="test", processor="rule",
        config={"rules": [{"when": "1 > 0", "label": "occupied"}],
                "actions": [{"type": "webhook",
                             "config": {"url": "http://127.0.0.1:1/x"},
                             "min_confidence": 0.0}]})
    daemon.registry.activate(rm.runtime_model_id, True)
    daemon.start()
    try:
        # The SMA loop is async — the first rolling window needs samples to
        # accumulate before predict+dispatch fires. Poll rather than sleep.
        deadline = time.time() + 8.0
        results = []
        while time.time() < deadline:
            results = list(daemon.dispatcher.results)
            if results:
                break
            time.sleep(0.1)
        assert results, "expected at least one dispatched action"
        # Webhook to a dead port → explicit failed result, never fake success
        assert results[-1]["result"]["status"] in ("failed", "unsupported")
    finally:
        daemon.stop()
