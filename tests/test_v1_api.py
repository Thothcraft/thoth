"""Canonical /api/v1/* surface on the node local API (§12)."""
import json
import urllib.error
import urllib.request

import pytest

from thoth.settings import ConfigStore


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def _daemon(tmp_home, port):
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon
    from whispy.devices.local import LocalDevice
    from whispy.sensors import FixtureDriver

    cfg = ConfigStore()
    cfg.set("local_port", port)
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    daemon._device = LocalDevice(
        device_id="rpi1",
        drivers={"fixture": FixtureDriver()})
    # Low rate: 6 fixture streams at 100Hz starve the HTTP handler's GIL
    # share on slow machines.
    daemon._device.open({"fixture": {"sensor_type": "microphone",
                                     "payloads": [[0.5]],
                                     "sample_rate": 10}})
    return daemon


def _req(port, path, token, method="GET", body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    return urllib.request.urlopen(req, timeout=20)


def test_v1_device_and_compute(tmp_home):
    daemon = _daemon(tmp_home, 5981)
    daemon.start()
    try:
        with _req(5981, "/api/v1/device", daemon.config.local_token) as res:
            body = json.loads(res.read())
        assert body["id"] == "rpi1"
        assert body["compute"]["architecture"]
        assert body["compute"]["logical_cpu_count"] > 0
        with _req(5981, "/api/v1/compute", daemon.config.local_token) as res:
            compute = json.loads(res.read())
        assert "memory_total_mb" in compute
    finally:
        daemon.stop()


def test_v1_health_and_sources(tmp_home):
    daemon = _daemon(tmp_home, 5982)
    daemon.start()
    try:
        with _req(5982, "/api/v1/health", daemon.config.local_token) as res:
            health = json.loads(res.read())
        assert health["running"] is True
        assert health["sources"][0]["id"]
        assert "uptime_s" in health
        with _req(5982, "/api/v1/sources", daemon.config.local_token) as res:
            sources = json.loads(res.read())["sources"]
        # FixtureDriver exposes several modalities; use the first.
        assert sources
        sid = sources[0]["id"]
        with _req(5982, f"/api/v1/sources/{sid}",
                  daemon.config.local_token) as res:
            assert json.loads(res.read())["id"] == sid
        with _req(5982, f"/api/v1/sources/{sid}/observations",
                  daemon.config.local_token) as res:
            out = json.loads(res.read())
        assert "cursor" in out and "samples" in out
    finally:
        daemon.stop()


def test_v1_inference_unknown_model_is_explicit(tmp_home):
    daemon = _daemon(tmp_home, 5983)
    daemon.start()
    try:
        with _req(5983, "/api/v1/inference", daemon.config.local_token,
                  method="POST", body={"model_id": "nope"}) as res:
            out = json.loads(res.read())
        assert out["status"] == "failed"
        assert "unknown model" in out["error"]
    finally:
        daemon.stop()


def test_v1_minutes_empty_and_privacy(tmp_home):
    daemon = _daemon(tmp_home, 5984)
    daemon.start()
    try:
        with _req(5984, "/api/v1/minutes", daemon.config.local_token) as res:
            assert json.loads(res.read())["minutes"] == []
        with _req(5984, "/api/v1/privacy", daemon.config.local_token) as res:
            privacy = json.loads(res.read())
        assert privacy["local_api_host"] == "127.0.0.1"
        assert "exposed" in privacy
        with _req(5984, "/api/v1/sync", daemon.config.local_token) as res:
            sync = json.loads(res.read())
        assert "captures_total" in sync
    finally:
        daemon.stop()


def test_v1_requires_auth(tmp_home):
    daemon = _daemon(tmp_home, 5985)
    daemon.start()
    try:
        with pytest.raises(urllib.error.HTTPError) as exc:
            urllib.request.urlopen(
                f"http://127.0.0.1:5985/api/v1/health", timeout=5)
        assert exc.value.code == 401
    finally:
        daemon.stop()
