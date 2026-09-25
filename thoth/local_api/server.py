"""Authenticated local API — loopback HTTP for the CLI and LAN clients.

Bound to localhost by default. Every request requires
``Authorization: Bearer <local_token>`` (the token lives in
``~/.thoth/config.json``). This is the node's local control surface —
it is never exposed to the public Internet (§24).

The same handler class also serves the node dashboard (React ``dist/``
on port 80 per CONTRACT §5): document and asset fetches authenticate via
``?token=`` once, which sets a ``thoth_dash`` session cookie, or via the
Bearer header/XHR path. When the privileged port cannot be bound the
daemon falls back to serving the UI on the API port.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

_DASHBOARD_DIR = Path(__file__).resolve().parents[1] / "dashboard" / "dist"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".json": "application/json",
    ".map": "application/json",
    ".svg": "image/svg+xml",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".ico": "image/x-icon",
    ".woff": "font/woff",
    ".woff2": "font/woff2",
    ".ttf": "font/ttf",
    ".txt": "text/plain; charset=utf-8",
}


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

    def _token_qs_ok(self, qs) -> bool:
        """Browser-doc auth: ``?token=<local_token>`` for page/asset fetches
        where no Authorization header can be set (initial GET /, downloads)."""
        token = self.server.token  # type: ignore[attr-defined]
        return bool(token) and (qs.get("token", [""])[0] == token)

    def _html(self, code: int, body: str, set_cookie: bool = False) -> None:
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        if set_cookie:
            self.send_header(
                "Set-Cookie",
                f"thoth_dash={self.server.token}; Path=/; "   # type: ignore[attr-defined]
                "SameSite=Strict")
        self.end_headers()
        self.wfile.write(data)

    def _redirect(self, location: str) -> None:
        self.send_response(302)
        self.send_header("Location", location)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _static(self, rel: str) -> None:
        """Serve a file from the built dashboard ``dist/`` tree."""
        root = getattr(self.server, "dashboard_dir", None) or _DASHBOARD_DIR
        try:
            target = (root / rel.lstrip("/")).resolve()
            if not str(target).startswith(str(root.resolve())) \
                    or not target.is_file():
                return self._json(404, {"error": "not found"})
            data = target.read_bytes()
        except OSError:
            return self._json(404, {"error": "not found"})
        ctype = _CONTENT_TYPES.get(target.suffix.lower(),
                                   "application/octet-stream")
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _dashboard_authorized(self, qs) -> bool:
        """Auth for document/asset fetches: ``?token=`` once (cookie is
        set), the Bearer header on XHR, or the session cookie."""
        if self._token_qs_ok(qs) or self._authorized_header():
            return True
        cookie = self.headers.get("Cookie", "")
        token = self.server.token  # type: ignore[attr-defined]
        for part in cookie.split(";"):
            part = part.strip()
            if part.startswith("thoth_dash="):
                return bool(token) and part[len("thoth_dash="):] == token
        return False

    def _dashboard_index(self) -> Optional[Path]:
        root = getattr(self.server, "dashboard_dir", None) or _DASHBOARD_DIR
        index = root / "index.html"
        return index if index.is_file() else None

    def _file(self, path, download_name: Optional[str] = None) -> None:
        try:
            data = path.read_bytes()
        except OSError:
            return self._json(404, {"error": "not found"})
        self.send_response(200)
        self.send_header("Content-Type", "application/zip")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Content-Disposition",
                         f'attachment; filename="{download_name or path.name}"')
        self.end_headers()
        self.wfile.write(data)
        try:
            path.unlink()                    # export artifact is ephemeral
        except OSError:
            pass

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
        path = urlparse(self.path).path
        qs = parse_qs(urlparse(self.path).query)
        serve_ui = bool(getattr(self.server, "serve_ui", True))
        # The dashboard page + capture downloads authenticate via ?token=
        # because the browser cannot set Authorization on document fetches.
        if path in ("/", "/dashboard", "/index.html"):
            if not serve_ui:
                # API-only port: hand the browser over to the dashboard
                # port when one is bound (CONTRACT §5 — UI on :80).
                dash = getattr(self.daemon, "_dash", None)
                if dash is not None:
                    host = (self.headers.get("Host") or "").split(":")[0] \
                        or dash.host
                    port = dash.port
                    tok = qs.get("token", [""])[0]
                    loc = (f"http://{host}:{port}/" if port != 80
                           else f"http://{host}/")
                    if tok:
                        loc += f"?token={tok}"
                    return self._redirect(loc)
                return self._json(401, {"error": "unauthorized; "
                                        "open /?token=<local_token>"})
            if not self._dashboard_authorized(qs):
                return self._json(401, {"error": "unauthorized; "
                                        "open /?token=<local_token>"})
            index = self._dashboard_index()
            if index is not None:
                data = index.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type",
                                 "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(data)))
                if self._token_qs_ok(qs):
                    self.send_header(
                        "Set-Cookie",
                        f"thoth_dash={self.server.token}; Path=/; "  # type: ignore[attr-defined]
                        "SameSite=Strict")
                self.end_headers()
                return self.wfile.write(data)
            from .dashboard import PAGE   # dist absent → legacy page
            return self._html(200, PAGE, set_cookie=self._token_qs_ok(qs))
        if serve_ui and path.startswith("/assets/"):
            if not self._dashboard_authorized(qs):
                return self._json(401, {"error": "unauthorized"})
            return self._static(path)
        if path.startswith("/api/captures/") and path.endswith("/download") \
                and not self._authorized_header():
            if not self._token_qs_ok(qs):
                return self._json(401, {"error": "unauthorized"})
            d = self.daemon
            cap_id = path[len("/api/captures/"):-len("/download")]
            out = d.captures.export(cap_id)
            if out is None:
                return self._json(404, {"error": "no such capture"})
            return self._file(out)
        if not self._authorized():
            return
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
        if path.startswith("/api/captures/") and path.endswith("/download"):
            cap_id = path[len("/api/captures/"):-len("/download")]
            out = d.captures.export(cap_id)
            if out is None:
                return self._json(404, {"error": "no such capture"})
            return self._file(out)
        if path.startswith("/api/captures/"):
            rec = d.captures.get(path[len("/api/captures/"):])
            return self._json(200 if rec else 404,
                            rec or {"error": "no such capture"})
        if path == "/api/automations":
            return self._json(200, {"automations": d.automations.list()})
        if path == "/api/v1/model-catalog":
            return self._json(200, {"models": d.model_catalog()})
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
        if path == "/api/v1/location":
            # Node resolves its own public egress IP → postal metadata.
            from whispy.geo import public_geo
            return self._json(200, public_geo() or {"error": "unresolved"})
        if path == "/api/v1/sync":
            return self._json(200, d.sync_state())
        if path == "/api/v1/metadata":
            return self._json(200, d.metadata.document())
        if path == "/api/v1/context":
            return self._json(200, d.context())
        if path == "/api/v1/room":
            return self._json(200, d.room.document())
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
                if qs.get("latest"):
                    out = d.latest_observation(source_id)
                else:
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
        # -- automations + capture annotation ------------------------------------
        if path == "/api/automations":
            auto = d.automations.upsert(body)
            return self._json(201, auto.to_dict())
        if path.startswith("/api/automations/"):
            auto = d.automations.upsert({"id": path[len("/api/automations/"):],
                                         **body})
            return self._json(200, auto.to_dict())
        if path.startswith("/api/captures/"):
            rest = path[len("/api/captures/"):]
            parts = rest.rsplit("/", 1)
            cap_id, op = parts[0], (parts[1] if len(parts) == 2 else "")
            if op == "label":
                entry = d.captures.add_label(cap_id, body.get("label") or "",
                                             source="manual")
                return self._json(201 if entry else 404,
                                entry or {"error": "no such capture"})
            if op == "autolabel":
                rec = d.captures.autolabel(cap_id, list(d.predictions))
                return self._json(200 if rec else 404,
                                rec or {"error": "no such capture"})
            if op == "clear-labels":
                rec = d.captures.clear_labels(
                    cap_id, source=body.get("source") or None)
                return self._json(200 if rec else 404,
                                rec or {"error": "no such capture"})
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
            d.automations.on_prediction(pred)
            return self._json(200, {"ok": True, "label": pred.label})
        return self._json(404, {"error": "not found"})

    def do_PUT(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        body = self._body()
        d = self.daemon
        if path == "/api/v1/metadata":
            # Body is a subset of the manual section (CONTRACT §1.1);
            # accept {"manual": {...}} too for relay symmetry.
            manual = body.get("manual") if isinstance(
                body.get("manual"), dict) else body
            return self._json(200, d.metadata.set_manual(manual or {}))
        if path == "/api/v1/room":
            return self._json(200, d.room.put(body))
        return self._json(404, {"error": "not found"})

    def _authorized_header(self) -> bool:
        token = self.server.token  # type: ignore[attr-defined]
        return self.headers.get("Authorization", "") == f"Bearer {token}"

    def do_DELETE(self):
        if not self._authorized():
            return
        path = urlparse(self.path).path
        d = self.daemon
        if path.startswith("/api/automations/"):
            ok = d.automations.remove(path[len("/api/automations/"):])
            return self._json(200 if ok else 404, {"ok": ok})
        if path.startswith("/api/captures/"):
            ok = d.captures.delete(path[len("/api/captures/"):])
            return self._json(200 if ok else 404,
                            {"ok": ok, "error": None if ok else
                             "active or missing capture"})
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


class _DualStackHTTPServer(_FastBindHTTPServer):
    """IPv6 wildcard socket with V6ONLY=0 — accepts v4-mapped clients too.

    Avahi/Bonjour advertise AAAA records for ``thoth-*.local``; an
    IPv4-only listener makes the dashboard unreachable for clients that
    prefer the v6 address.
    """

    address_family = __import__("socket").AF_INET6

    def server_bind(self) -> None:
        import socket
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6,
                                   socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


class LocalAPIServer:
    """Threaded loopback HTTP server hosting the node's local API.

    ``serve_ui`` gates the dashboard surface (GET ``/`` + ``/assets/*``);
    API routes are always available on whatever port the server binds.
    """

    def __init__(self, daemon, host: str = "127.0.0.1", port: int = 5000,
                 token: str = "", serve_ui: bool = True,
                 dashboard_dir: Optional[Path] = None):
        self.daemon = daemon
        self.host = host
        self.port = port
        self.token = token
        self.serve_ui = serve_ui
        self.dashboard_dir = dashboard_dir
        self._httpd: Optional[ThreadingHTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def _make_server(self):
        """Wildcard binds prefer a dual-stack socket; explicit/loopback
        hosts stay on the default IPv4 class."""
        if self.host in ("0.0.0.0", "::", ""):
            try:
                return _DualStackHTTPServer(("::", self.port), _Handler)
            except OSError:
                # No IPv6 on this host — plain IPv4 wildcard.
                pass
        return _FastBindHTTPServer((self.host, self.port), _Handler)

    def start(self) -> "LocalAPIServer":
        httpd = self._make_server()
        httpd.daemon_ref = self.daemon  # type: ignore[attr-defined]
        httpd.token = self.token        # type: ignore[attr-defined]
        httpd.serve_ui = self.serve_ui  # type: ignore[attr-defined]
        if self.dashboard_dir is not None:
            httpd.dashboard_dir = self.dashboard_dir  # type: ignore[attr-defined]
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
