"""Thoth LAN actuator API — inventory, execution, exposure filtering."""
import json
import time
import urllib.error
import urllib.request

import pytest

from thoth.settings import ConfigStore


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


class _FakeSpeakerAdapter:
    """whispy ActuatorAdapter duck-type for tests."""

    def metadata(self):
        from whispy.actuators.base import ActuatorMeta
        return ActuatorMeta(name="fake-speaker", kinds=("speaker",))

    def discover(self):
        from whispy.contracts import ActuatorDescriptor
        return [ActuatorDescriptor(
            id="speaker-t1", kind="speaker", adapter="fake-speaker",
            operations=["speak", "stop"], stable=True)]

    def connect(self, descriptor, config=None):
        from whispy.actuators.base import ActuatorHandle
        from whispy.contracts import ActionResult, ActionStatus

        class _H(ActuatorHandle):
            @property
            def info(self):
                return descriptor

            def execute(self, command):
                if command.operation == "speak":
                    return ActionResult(status=ActionStatus.SUCCEEDED,
                                        action_type="speaker",
                                        detail="spoke")
                return ActionResult(status=ActionStatus.UNSUPPORTED,
                                    action_type="speaker")

        return _H()


def _daemon(tmp_home, port, exposed=None):
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon
    from whispy.devices.local import LocalDevice

    cfg = ConfigStore()
    cfg.set("local_port", port)
    if exposed is not None:
        cfg.set("exposed", exposed)
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    daemon._device = LocalDevice(
        device_id="rpi1", drivers={}, adapters={},
        actuator_adapters={"fake-speaker": _FakeSpeakerAdapter()})
    return daemon


def _req(port, path, token, method="GET", body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=5)


def test_actuator_inventory_endpoint(tmp_home):
    daemon = _daemon(tmp_home, 5991)
    daemon.start()
    try:
        time.sleep(0.3)
        with _req(5991, "/api/actuators", daemon.config.local_token) as res:
            body = json.loads(res.read().decode())
        ids = [a["id"] for a in body["actuators"]]
        assert "speaker-t1" in ids
        assert body["actuators"][0]["operations"] == ["speak", "stop"]
    finally:
        daemon.stop()


def test_actuator_execute_endpoint(tmp_home):
    daemon = _daemon(tmp_home, 5992)
    daemon.start()
    try:
        time.sleep(0.3)
        with _req(5992, "/api/actuators/speaker-t1/actions",
                  daemon.config.local_token, method="POST",
                  body={"operation": "speak",
                        "params": {"text": "hi"}}) as res:
            body = json.loads(res.read().decode())
        assert body["status"] == "succeeded"
        assert body["detail"] == "spoke"
    finally:
        daemon.stop()


def test_actuator_execute_unknown_is_explicit(tmp_home):
    daemon = _daemon(tmp_home, 5993)
    daemon.start()
    try:
        time.sleep(0.3)
        with _req(5993, "/api/actuators/nope-9/actions",
                  daemon.config.local_token, method="POST",
                  body={"operation": "speak"}) as res:
            body = json.loads(res.read().decode())
        assert body["status"] == "unsupported"
    finally:
        daemon.stop()


def test_actuator_exposure_filter(tmp_home):
    """``thoth expose --actuator X`` hides every other actuator."""
    daemon = _daemon(tmp_home, 5994,
                     exposed={"actuators": ["other-1"], "sensors": []})
    daemon.start()
    try:
        time.sleep(0.3)
        with _req(5994, "/api/actuators", daemon.config.local_token) as res:
            body = json.loads(res.read().decode())
        assert body["actuators"] == []
        with _req(5994, "/api/actuators/speaker-t1/actions",
                  daemon.config.local_token, method="POST",
                  body={"operation": "speak"}) as res:
            body = json.loads(res.read().decode())
        assert body["status"] == "unsupported"
        assert "not exposed" in body["detail"]
    finally:
        daemon.stop()


def test_sensor_exposure_filter(tmp_home):
    """``thoth expose --sensor X`` hides other sensors from /api/sensors."""
    daemon = _daemon(tmp_home, 5995,
                     exposed={"sensors": ["only-this"], "actuators": []})
    daemon.start()
    try:
        time.sleep(0.3)
        with _req(5995, "/api/sensors", daemon.config.local_token) as res:
            body = json.loads(res.read().decode())
        assert body["sensors"] == []
    finally:
        daemon.stop()
