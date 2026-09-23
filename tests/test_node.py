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
        "manifest": {"format": "thoth-model/v1", "processor": "rule"},
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
