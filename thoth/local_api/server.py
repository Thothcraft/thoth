"""Authenticated local API — loopback HTTP for the CLI and LAN clients.

Bound to localhost by default. Every request requires
``Authorization: Bearer <local_token>`` (the token lives in
``~/.thoth/config.json``). This is the node's local control surface —
it is never exposed to the public Internet (§24).
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs


class _Handler(BaseHTTPRequestHandler):
    server_version = "ThothLocal/0.1"

    # -- helpers -------------------------------------------------------------
    def _json(self, code: int, body: Any) -> None:
        data = json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _authorized(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {token}":
            return True
        self._json(401, {"error": "unauthorized"})
        return False

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length:
                return json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception:
            pass
        return {}

    def log_message(self, *args):  # quiet
        pass

    @property
    def daemon(self):
        return self.server.daemon_ref  # type: ignore[attr-defined]

    # -- routes ---------------------------------------------------------------
    def do_GET(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        d = self.daemon
        if path == "/api/status":
            return self._json(200, d.status())
        if path == "/api/device":
            return self._json(200, d._device.info.to_dict() if d._device else {})
        if path == "/api/sensors":
            if qs.get("descriptors") and hasattr(d._device, "sensor_descriptors"):
                try:
                    descs = d._device.sensor_descriptors()
                    return self._json(200, {"descriptors": [
                        x.to_dict() for x in descs
                        if d.sensor_exposed(x.id)]})
                except Exception:
                    pass
            sensors = d._device.sensors() if d._device else []
            return self._json(200, {"sensors": [
                s.to_dict() for s in sensors if d.sensor_exposed(s.id)]})
        if path == "/api/actuators":
            return self._json(200, {"actuators": d.actuators()})
        if path.startswith("/api/sensors/") and path.endswith("/tail"):
            sensor_id = path[len("/api/sensors/"):-len("/tail")]
            try:
                cursor = int(qs.get("cursor", [0])[0] or 0)
            except (TypeError, ValueError):
                cursor = 0
            out = d.tail_sensor(sensor_id, cursor)
            if out is None:
                return self._json(404, {"error": f"unknown sensor {sensor_id}"})
            return self._json(200, out)
        if path == "/api/predictions":
            limit = int(qs.get("limit", [50])[0])
            return self._json(200, {"predictions": d.recent_predictions(limit)})
        if path == "/api/models":
            return self._json(200, {"models": [m.to_dict() for m in d.registry.list()]})
        if path == "/api/captures":
            return self._json(200, {"captures": d.captures.list()})
        if path == "/api/actions":
            return self._json(200, {"actions": list(d.dispatcher.results)})
        if path == "/api/deployments":
            return self._json(200, {"deployments": d.deployments._states})

        # -- canonical v1 surface (§12) ---------------------------------------
        if path == "/api/v1/device":
            info = d._device.info.to_dict() if d._device else {}
            info["compute"] = d.compute()
            return self._json(200, info)
        if path == "/api/v1/health":
            return self._json(200, d.health())
        if path == "/api/v1/compute":
            return self._json(200, d.compute())
        if path == "/api/v1/sources":
            return self._json(200, {"sources": d.sources()})
        if path == "/api/v1/actuators":
            return self._json(200, {"actuators": d.actuators()})
        if path == "/api/v1/models":
            return self._json(200, {"models": [m.to_dict() for m in d.registry.list()]})
        if path == "/api/v1/model-deployments":
            return self._json(200, {"deployments": d.deployments._states})
        if path == "/api/v1/minutes":
            return self._json(200, {"minutes": d.minutes()})
        if path == "/api/v1/privacy":
            return self._json(200, d.privacy())
        if path == "/api/v1/sync":
            return self._json(200, d.sync_state())
        if path.startswith("/api/v1/minutes/"):
            rest = path[len("/api/v1/minutes/"):]
            parts = rest.split("/")
            minute_id = parts[0]
            if len(parts) == 3 and parts[1] == "seconds":
                try:
                    idx = int(parts[2])
                except ValueError:
                    return self._json(400, {"error": "second index must be an integer"})
                out = d.minute_second(minute_id, idx)
                return self._json(200 if out else 404, out or {"error": "not found"})
            out = d.minute(minute_id)
            return self._json(200 if out else 404, out or {"error": "not found"})
        if path.startswith("/api/v1/sources/"):
            rest = path[len("/api/v1/sources/"):]
            if rest.endswith("/observations"):
                source_id = rest[:-len("/observations")]
                try:
                    cursor = int(qs.get("cursor", [0])[0] or 0)
                except (TypeError, ValueError):
                    cursor = 0
                out = d.source_observations(source_id, cursor)
                if out is None:
                    return self._json(404, {"error": f"unknown source {source_id}"})
                return self._json(200, out)
            desc = d.source(rest)
            return self._json(200 if desc else 404,
                            desc or {"error": "not found"})
        return self._json(404, {"error": "not found"})

    def do_POST(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        body = self._body()
        d = self.daemon
        if path == "/api/captures/start":
            try:
                rec = d.captures.start(
                    d.device_id, body.get("sensors"),
                    available=list(d._streams))
            except ValueError as exc:
                return self._json(422, {"error": str(exc)})
            return self._json(201, rec)
        if path == "/api/captures/stop":
            rec = d.captures.stop(str(body.get("capture_id") or ""))
            return self._json(200, rec or {"error": "no such capture"})
        if path == "/api/models/install":
            rm = d.registry.install(
                name=body.get("name", "model"),
                processor=body.get("processor", "rule"),
                config=body.get("config") or {},
                deployment_id=body.get("deployment_id"))
            return self._json(201, rm.to_dict())
        if path == "/api/models/activate":
            ok = d.registry.activate(str(body.get("runtime_model_id") or ""),
                                     bool(body.get("active", True)))
            return self._json(200 if ok else 404, {"ok": ok})
        if path == "/api/deployments":
            dep = d.deployments.process(
                str(body.get("deployment_id") or ""), body)
            return self._json(200, dep)
        if path.startswith("/api/actuators/") and path.endswith("/actions"):
            actuator_id = path[len("/api/actuators/"):-len("/actions")]
            result = d.execute_actuator(actuator_id, body)
            return self._json(200, result)
        # -- canonical v1 surface (§12) ---------------------------------------
        if path == "/api/v1/inference":
            return self._json(200, d.infer(body))
        if path.startswith("/api/v1/actuators/") and path.endswith("/actions"):
            actuator_id = path[len("/api/v1/actuators/"):-len("/actions")]
            return self._json(200, d.execute_actuator(actuator_id, body))
        if path == "/api/internal/prediction":
            # Inject a prediction — drives linked actuators (test hook).
            from whispy.contracts import Prediction
            pred = Prediction(label=body.get("class", "unknown"),
                              confidence=float(body.get("confidence", 1.0)),
                              device_id=d.device_id)
            d.predictions.append(pred.to_dict())
            for model in d.registry.active():
                d._fire_actions(model, pred)
            return self._json(200, {"ok": True, "label": pred.label})
        return self._json(404, {"error": "not found"})


class _FastBindHTTPServer(ThreadingHTTPServer):
    """ThreadingHTTPServer without the reverse-DNS stall.

    ``HTTPServer.server_bind`` calls ``socket.getfqdn(host)`` which can
    block ~20s on Windows when the resolver is slow — the node API must
    come up immediately.
    """

    def server_bind(self) -> None:
        import socketserver
        socketserver.TCPServer.server_bind(self)
        host, port = self.server_address[:2]
        self.server_name = host
        self.server_port = port


class LocalAPIServer:
    """Threaded loopback HTTP server hosting the node's local API."""

    def __init__(self, daemon, host: str = "127.0.0.1", port: int = 5000,
                 token: str = ""):
        self.daemon = daemon
        self.host = host
        self.port = port
        self.token = token
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> "LocalAPIServer":
        httpd = _FastBindHTTPServer((self.host, self.port), _Handler)
        httpd.daemon_ref = self.daemon  # type: ignore[attr-defined]
        httpd.token = self.token        # type: ignore[attr-defined]
        self._httpd = httpd
        self._thread = threading.Thread(target=httpd.serve_forever,
                                        name="thoth-local-api", daemon=True)
        self._thread.start()
        return self

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def stop(self) -> None:
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
