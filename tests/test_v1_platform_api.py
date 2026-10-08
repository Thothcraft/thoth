"""v1 API additions: /api/v1/net, /api/v1/ble/*, /api/v1/entities,
/api/v1/relations, /api/v1/context/calibrate, conformance."""
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
    # /net and /ble surface tests need those subsystems alive (they
    # fast-fail without hardware) — but fixture pumps must be paced:
    # without realtime=true they free-run at 100% GIL and starve the
    # HTTP handler thread (~100s+ per request on slow machines).
    cfg.set("dashboard_enabled", False)
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    daemon._device = LocalDevice(
        device_id="rpi1",
        drivers={"fixture": FixtureDriver()})
    daemon._device.open({"fixture": {"sensor_type": "microphone",
                                     "payloads": [[0.5]],
                                     "sample_rate": 10,
                                     "realtime": True}})
    return daemon


def _req(port, path, token, method="GET", body=None):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode() if body is not None else None,
        method=method,
        headers={"Authorization": f"Bearer {token}",
                 "Content-Type": "application/json"})
    try:
        res = urllib.request.urlopen(req, timeout=120)
        return res.status, json.loads(res.read())
    except urllib.error.HTTPError as exc:
        try:
            return exc.code, json.loads(exc.read())
        except Exception:
            return exc.code, {}


def test_net_endpoints(tmp_home):
    d = _daemon(tmp_home, 5997)
    d.start()
    t = d.config.local_token
    try:
        code, body = _req(5997, "/api/v1/net", t)
        assert code == 200 and "state" in body and "wifi" in body
        code, body = _req(5997, "/api/v1/net/scan", t)
        assert code == 200 and body["networks"] == []
        code, body = _req(5997, "/api/v1/net", t, "POST",
                          {"ssid": "X", "psk": "y"})
        # no wifi backend on CI/Windows — queued but state machine will
        # fail cleanly; accepted reflects the request was well-formed
        assert code in (200, 202)
        code, _ = _req(5997, "/api/v1/net", t, "POST", {"psk": "y"})
        assert code == 400
    finally:
        d.stop()


def test_entities_and_relations_round_trip(tmp_home):
    d = _daemon(tmp_home, 5997)
    d.start()
    t = d.config.local_token
    try:
        code, person = _req(5997, "/api/v1/entities", t, "POST",
                            {"type": "person", "name": "alice"})
        assert code == 200 and person["id"].startswith("person:")
        code, dev = _req(5997, "/api/v1/entities", t, "POST",
                         {"type": "device", "name": "watch"})
        code, edge = _req(5997, "/api/v1/relations", t, "POST",
                          {"from": person["id"], "rel": "wears",
                           "to": dev["id"]})
        assert code == 200 and edge["rel"] == "wears"
        code, doc = _req(5997, "/api/v1/entities", t)
        ids = {e["id"] for e in doc["entities"]}
        assert person["id"] in ids and dev["id"] in ids
        assert any(r["from"] == person["id"] and r["to"] == dev["id"]
                   for r in doc["relations"])
        code, _ = _req(5997, "/api/v1/relations", t, "POST",
                       {"from": person["id"], "rel": "wears"})
        assert code == 400
    finally:
        d.stop()


def test_ble_devices_surface(tmp_home):
    d = _daemon(tmp_home, 5997)
    d.start()
    t = d.config.local_token
    try:
        code, body = _req(5997, "/api/v1/ble/devices", t)
        assert code == 200 and "known" in body and "seen_now" in body
        # enroll through the API — no BlueZ on CI, but the known-device
        # map is backend-agnostic
        code, out = _req(5997, "/api/v1/ble/enroll", t, "POST",
                         {"address": "AA:BB:CC:DD:EE:01",
                          "kind": "watch", "name": "pt",
                          "person": "alice"})
        assert code == 200 and out["ok"] and \
            out["device_id"].startswith("device:")
        code, body = _req(5997, "/api/v1/ble/devices", t)
        assert any(r["device_id"] == out["device_id"]
                   for r in body["known"])
        code, un = _req(5997, "/api/v1/ble/unenroll", t, "POST",
                        {"address": "AA:BB:CC:DD:EE:01"})
        assert un["ok"] is True
    finally:
        d.stop()


def test_calibrate_and_context_estimates(tmp_home):
    d = _daemon(tmp_home, 5997)
    d.start()
    t = d.config.local_token
    try:
        # no evidence yet → calibrate is a clean no-op error
        code, out = _req(5997, "/api/v1/context/calibrate", t, "POST",
                         {"zone": "kitchen"})
        assert code == 200 and out["ok"] is False
        # feed an observation → estimate lands in /api/v1/context
        d.emit_observation({"schema": "ble.rssi.v1",
                            "source_id": "ble:hci0",
                            "subject": "device:x",
                            "value": {"rssi_dbm": -50}})
        code, ctx = _req(5997, "/api/v1/context", t)
        assert code == 200
        keys = {s["key"] for s in ctx["estimates"]}
        assert "occupancy.v1" in keys
    finally:
        d.stop()


def test_capabilities_domains_and_bluetooth(tmp_home):
    d = _daemon(tmp_home, 5997)
    d.start()
    t = d.config.local_token
    try:
        code, caps = _req(5997, "/api/v1/capabilities", t)
        assert code == 200
        assert "domains" in caps and "bluetooth" in caps["domains"]
        assert caps["bluetooth"]["present"] in (True, False)
        assert "roles" in caps["bluetooth"]
        code, conf = _req(5997, "/api/v1/sources/conformance", t)
        assert code == 200 and "adapters" in conf
    finally:
        d.stop()
