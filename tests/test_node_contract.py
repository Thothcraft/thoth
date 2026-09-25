"""CONTRACT v1 node surface: metadata, room, dashboard :80, brain WS."""
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from thoth.settings import ConfigStore


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def _daemon(tmp_home, port, dash_port=0, dash_dir=None):
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon
    from whispy.devices.local import LocalDevice
    from whispy.sensors import FixtureDriver

    cfg = ConfigStore()
    cfg.set("local_port", port)
    cfg.set("dashboard_port", dash_port)
    if dash_dir is not None:
        cfg.set("dashboard_dir", str(dash_dir))
    daemon = ThothDaemon(config=cfg, window_seconds=0.5, tick_hz=4.0)
    daemon._device = LocalDevice(
        device_id="rpi1",
        drivers={"fixture": FixtureDriver()})
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


# -- MetadataManager ---------------------------------------------------------

def test_metadata_document_shape(tmp_home):
    from thoth.metadata import MetadataManager
    mgr = MetadataManager(path=tmp_home / "metadata.json")
    doc = mgr.document()
    assert set(doc) == {"inferred", "manual"}
    assert set(doc["inferred"]) == {"location", "activity",
                                    "battery", "application"}
    assert set(doc["manual"]) == {"room_name", "friendly_name", "room_id"}


def test_metadata_manual_put_persists(tmp_home):
    from thoth.metadata import MetadataManager
    mgr = MetadataManager(path=tmp_home / "metadata.json")
    doc = mgr.set_manual({"room_name": "Living Room",
                          "ignored_key": "dropped"})
    assert doc["manual"]["room_name"] == "Living Room"
    assert "ignored_key" not in doc["manual"]
    # reload → manual survives
    mgr2 = MetadataManager(path=tmp_home / "metadata.json")
    assert mgr2.document()["manual"]["room_name"] == "Living Room"


def test_metadata_inferred_refresh(tmp_home, monkeypatch):
    from thoth.metadata import MetadataManager
    import whispy.geo
    monkeypatch.setattr(whispy.geo, "public_geo", lambda *a, **k: {
        "latitude": 43.0, "longitude": -79.0,
        "postal_code": "M5V", "city": "Toronto"})
    mgr = MetadataManager(path=tmp_home / "metadata.json")
    changed = mgr.refresh_inferred(
        predictions=[{"label": "occupied", "confidence": 0.9,
                      "timestamp": __import__("time").time()}])
    doc = mgr.document()
    assert changed["location"]["city"] == "Toronto"
    assert doc["inferred"]["location"]["postal_code"] == "M5V"
    assert doc["inferred"]["activity"]["kind"] == "occupied"
    assert doc["inferred"]["application"]["platform"] in (
        "windows", "linux", "android")


def test_metadata_activity_idle_when_stale(tmp_home):
    from thoth.metadata import MetadataManager
    mgr = MetadataManager(path=tmp_home / "metadata.json")
    mgr.refresh_inferred(predictions=[
        {"label": "occupied", "confidence": 0.9, "timestamp": 0.0}])
    assert mgr.document()["inferred"]["activity"]["kind"] == "idle"


# -- RoomManager -------------------------------------------------------------

def test_room_put_stamps_and_emits(tmp_home):
    from thoth.room import RoomManager
    seen = []
    mgr = RoomManager(path=tmp_home / "room.json",
                      on_change=seen.append)
    doc = mgr.put({"room_id": "living", "name": "Living Room",
                   "dims": {"w": 7.2, "d": 5.4, "h": 2.8},
                   "devices": [{"device_id": "chen", "pos": [1, 2, 0],
                                "rot_y": 1.57, "mount": "wall",
                                "sensors": [{"type": "radar",
                                             "pos": [0, 0, 0.06],
                                             "fov_deg": 60,
                                             "range_m": 6}]}]})
    assert doc["updated_at"] > 0
    assert len(seen) == 1                       # room_changed fired
    assert seen[0]["devices"][0]["sensors"][0]["fov_deg"] == 60.0
    # persisted + normalized on reload
    mgr2 = RoomManager(path=tmp_home / "room.json")
    doc2 = mgr2.document()
    assert doc2["room_id"] == "living"
    assert doc2["devices"][0]["mount"] == "wall"


def test_room_defaults_valid(tmp_home):
    from thoth.room import RoomManager
    mgr = RoomManager(path=tmp_home / "room.json")
    doc = mgr.document()
    assert doc["format"] == "room/v1"
    assert doc["dims"]["w"] > 0
    assert doc["devices"] == []


# -- HTTP surface --------------------------------------------------------------

def test_metadata_room_endpoints(tmp_home):
    daemon = _daemon(tmp_home, 5986)
    daemon.start()
    try:
        tok = daemon.config.local_token
        with _req(5986, "/api/v1/metadata", tok) as res:
            meta = json.loads(res.read())
        assert "inferred" in meta and "manual" in meta
        with _req(5986, "/api/v1/metadata", tok, method="PUT",
                  body={"friendly_name": "April", "room_name": "Lab"}) as res:
            meta = json.loads(res.read())
        assert meta["manual"]["friendly_name"] == "April"
        with _req(5986, "/api/v1/room", tok) as res:
            room = json.loads(res.read())
        assert room["format"] == "room/v1"
        before = room["updated_at"]
        with _req(5986, "/api/v1/room", tok, method="PUT",
                  body={"room_id": "lab", "name": "Lab",
                        "dims": {"w": 5, "d": 4, "h": 2.6},
                        "devices": []}) as res:
            room = json.loads(res.read())
        assert room["room_id"] == "lab"
        assert room["updated_at"] > before
    finally:
        daemon.stop()


def test_dashboard_served_on_dash_port(tmp_home):
    # A fake dist/ stands in for the real Vite build.
    dist = tmp_home / "dist"
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html>dash</html>")
    (dist / "assets" / "app.js").write_text("console.log(1)")
    daemon = _daemon(tmp_home, 5987, dash_port=5988, dash_dir=dist)
    daemon.start()
    try:
        tok = daemon.config.local_token
        # :5988 serves the built UI with ?token=
        res = urllib.request.urlopen(
            f"http://127.0.0.1:5988/?token={tok}", timeout=10)
        assert res.status == 200
        assert "dash" in res.read().decode()
        # assets ride the same auth (cookie or ?token)
        res = urllib.request.urlopen(
            f"http://127.0.0.1:5988/assets/app.js?token={tok}", timeout=10)
        assert res.status == 200
        assert res.headers["Content-Type"].startswith("text/javascript")
        # API answers on :5988 too (CONTRACT §1 — http://<node>:80/api/v1/*)
        with _req(5988, "/api/v1/room", tok) as res:
            assert json.loads(res.read())["format"] == "room/v1"
        # API port / redirects to the dashboard port
        class _NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        opener = urllib.request.build_opener(_NoRedirect)
        with pytest.raises(urllib.error.HTTPError) as exc:
            opener.open(f"http://127.0.0.1:5987/?token={tok}", timeout=10)
        assert exc.value.code == 302
        assert ":5988" in exc.value.headers["Location"]
        # and stays fully functional
        with _req(5987, "/api/status", tok) as res:
            assert json.loads(res.read())["device_id"] == "rpi1"
    finally:
        daemon.stop()


# -- BrainWSClient --------------------------------------------------------------

def test_dispatch_api_executes_over_local_api(tmp_home):
    daemon = _daemon(tmp_home, 5989)
    daemon.start()
    try:
        from thoth.daemon.brain_ws import BrainWSClient
        client = BrainWSClient(
            daemon, api_base="http://127.0.0.1:5989",
            local_token=daemon.config.local_token,
            brain_url="https://api.example.test",
            device_id="rpi1", device_token="dev-tok")
        # api_request frame → api_response body (no WS needed to test
        # the dispatch half).
        out = client.dispatch_api("GET", "/api/v1/metadata", None)
        assert out["status"] == 200
        assert "manual" in out["body"]
        out = client.dispatch_api("PUT", "/api/v1/metadata",
                                  {"room_name": "Den"})
        assert out["body"]["manual"]["room_name"] == "Den"
        out = client.dispatch_api("GET", "/nope", None)
        assert out["status"] == 404
    finally:
        daemon.stop()


def test_event_frame_falls_back_to_rest(tmp_home):
    # Tiny Brain stub capturing POST /v1/events.
    hits = []

    class _Stub(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            hits.append((self.path, json.loads(self.rfile.read(n))))
            self.send_response(200)
            self.end_headers()
        def log_message(self, *a):
            pass

    stub = ThreadingHTTPServer(("127.0.0.1", 0), _Stub)
    port = stub.server_address[1]
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    try:
        from thoth.daemon.brain_ws import BrainWSClient
        client = BrainWSClient(
            daemon=None, api_base="http://127.0.0.1:1",
            local_token="x", brain_url=f"http://127.0.0.1:{port}",
            device_id="rpi1", device_token="dev-tok")
        assert client.send_event("trigger_fired",
                                 {"automation_id": "a1"}) is False
        for _ in range(50):
            if hits:
                break
            __import__("time").sleep(0.05)
        assert hits and hits[0][0] == "/v1/events"
        body = hits[0][1]
        assert body["device_id"] == "rpi1"
        assert body["kind"] == "trigger_fired"
        assert body["data"]["automation_id"] == "a1"
    finally:
        stub.shutdown()


def test_ws_url_scheme():
    from thoth.daemon.brain_ws import _ws_url
    assert _ws_url("https://api.thothcraft.com", "d", "t").startswith(
        "wss://api.thothcraft.com/v1/node/ws?device_id=d")
    assert _ws_url("http://127.0.0.1:8000", "d", "t").startswith(
        "ws://127.0.0.1:8000/v1/node/ws")


# -- automation trigger_fired ----------------------------------------------------

def test_automation_fire_emits_event(tmp_home):
    daemon = _daemon(tmp_home, 5990)
    sent = []
    daemon.emit_event = lambda kind, data: sent.append((kind, data))
    daemon.automations.on_fire = daemon._automation_fired
    daemon.automations.upsert({
        "id": "a1", "name": "on gad", "enabled": True,
        "trigger": {"type": "event", "on": "label", "label": "gad"},
        "action": {"type": "webhook",
                   "config": {"url": "http://127.0.0.1:1/hook"}}})
    from whispy.contracts import Prediction
    daemon.automations.on_prediction(
        Prediction(label="gad", confidence=0.9, device_id="rpi1"))
    assert sent and sent[0][0] == "trigger_fired"
    assert sent[0][1]["automation_id"] == "a1"
    assert sent[0][1]["label"] == "gad"
