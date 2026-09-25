"""Outbound Brain WebSocket — the node's cloud control channel.

One connection to ``wss://<brain>/v1/node/ws?device_id=…&token=…``
(CONTRACT §2):

- ``api_request`` frames are executed against the node's own local API
  over loopback HTTP (auth: ``local_token``) and answered with
  ``api_response`` — the portal↔node command tunnel.
- ``event``/``room_changed``/``metadata`` frames are pushed upstream;
  when the socket is down the daemon falls back to ``POST /v1/events``
  via :meth:`BrainWSClient.send_event`.

Reconnects with 1→30 s backoff. The whole channel is best-effort: an
unreachable Brain must never disturb local sensing, so every failure is
absorbed at debug level.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import urllib.request
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_BACKOFF_MIN_S = 1.0
_BACKOFF_MAX_S = 30.0


def _ws_url(brain_url: str, device_id: str, token: str) -> str:
    base = brain_url.rstrip("/")
    if base.startswith("https://"):
        base = "wss://" + base[len("https://"):]
    elif base.startswith("http://"):
        base = "ws://" + base[len("http://"):]
    elif not base.startswith(("ws://", "wss://")):
        base = "wss://" + base
    return f"{base}/v1/node/ws?device_id={device_id}&token={token}"


class BrainWSClient:
    """Daemon-owned WS client; started only when a device token exists."""

    def __init__(self, daemon: Any, api_base: str, local_token: str,
                 brain_url: str, device_id: str, device_token: str):
        self._daemon = daemon
        self._api_base = api_base.rstrip("/")
        self._local_token = local_token
        self._brain_url = brain_url
        self._device_id = device_id
        self._device_token = device_token
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._outbox: Optional[asyncio.Queue] = None
        self._wake: Optional[asyncio.Event] = None
        self._ws = None

    # -- lifecycle -------------------------------------------------------------
    def start(self) -> "BrainWSClient":
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="thoth-brain-ws")
        self._thread.start()
        return self

    @property
    def connected(self) -> bool:
        return self._ws is not None

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            def _wake() -> None:
                if self._wake is not None:
                    self._wake.set()
                if self._outbox is not None:
                    # unblock a sender stuck awaiting outbox.get() is not
                    # needed — it is cancelled on session teardown
                    pass
            try:
                loop.call_soon_threadsafe(_wake)
                ws = self._ws
                if ws is not None:
                    asyncio.run_coroutine_threadsafe(ws.close(), loop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- outbound frames ----------------------------------------------------------
    def send_event(self, kind: str, data: Dict[str, Any]) -> bool:
        """Push a node→Brain frame; REST ``POST /v1/events`` fallback.

        Frame shape per CONTRACT §2: trigger fires ride
        ``{"type":"event","kind":…}``; room/metadata deltas ride
        ``{"type":"room_changed"|"metadata"}``.
        Returns True when the frame went over the socket.
        """
        ftype = {"room_changed": "room_changed",
                 "metadata": "metadata"}.get(kind, "event")
        frame: Dict[str, Any] = {"type": ftype, "data": data}
        if ftype == "event":
            frame["kind"] = kind
        if self._enqueue(frame):
            return True
        self._post_event(kind, data)
        return False

    def _enqueue(self, frame: Dict[str, Any]) -> bool:
        if not self.connected or self._outbox is None or \
                self._loop is None:
            return False
        try:
            self._loop.call_soon_threadsafe(
                self._outbox.put_nowait, frame)
            return True
        except Exception:
            return False

    def _post_event(self, kind: str, data: Dict[str, Any]) -> None:
        """Detached REST fallback — never stalls the SMA loop."""
        def _go() -> None:
            try:
                body = {"device_id": self._device_id, "kind": kind,
                        "data": data}
                req = urllib.request.Request(
                    f"{self._brain_url.rstrip('/')}/v1/events",
                    data=json.dumps(body).encode(), method="POST",
                    headers={"Content-Type": "application/json",
                             "Authorization":
                                 f"Bearer {self._device_token}"})
                urllib.request.urlopen(req, timeout=8).read()
            except Exception as exc:
                logger.debug("event REST fallback failed: %s", exc)
        threading.Thread(target=_go, name="thoth-event-post",
                         daemon=True).start()

    # -- api_request dispatch ----------------------------------------------------
    def dispatch_api(self, method: str, path: str,
                     body: Optional[Any]) -> Dict[str, Any]:
        """Execute a relayed request against the local API over loopback.

        Runs the real HTTP surface (auth, exposure filters and all), so a
        relayed GET is byte-identical to a local one.
        """
        method = (method or "GET").upper()
        url = f"{self._api_base}{path}"
        try:
            data = None if body is None else json.dumps(body).encode()
            req = urllib.request.Request(
                url, data=data, method=method,
                headers={"Authorization": f"Bearer {self._local_token}",
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=20) as res:
                return {"status": res.status,
                        "body": json.loads(res.read() or b"null")}
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read() or b"null")
            except Exception:
                payload = {"error": str(exc)}
            return {"status": exc.code, "body": payload}
        except Exception as exc:
            return {"status": 502, "body": {"error": str(exc)}}

    # -- asyncio machinery ---------------------------------------------------------
    def _run(self) -> None:
        try:
            asyncio.run(self._main())
        except Exception as exc:
            logger.debug("brain ws loop exited: %s", exc)

    async def _main(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._outbox = asyncio.Queue()
        self._wake = asyncio.Event()
        delay = _BACKOFF_MIN_S
        while not self._stop.is_set():
            try:
                await self._session()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("brain ws connect failed: %s", exc)
            self._ws = None
            if self._stop.is_set():
                break
            try:
                # Wake early on stop() instead of sleeping the full delay.
                await asyncio.wait_for(self._wake.wait(), delay)
                self._wake.clear()
            except asyncio.TimeoutError:
                pass
            delay = min(_BACKOFF_MAX_S, delay * 2.0)

    async def _session(self) -> None:
        import websockets  # lazy — optional until the node is paired
        url = _ws_url(self._brain_url, self._device_id, self._device_token)
        async with websockets.connect(url, ping_interval=20,
                                      ping_timeout=20) as ws:
            self._ws = ws
            logger.info("brain ws connected: %s", url.split("?")[0])
            # CONTRACT §1.2 — re-sync room + metadata on (re)connect so
            # Brain's cache converges even if change events were lost.
            daemon = self._daemon
            if daemon is not None and self._outbox is not None:
                try:
                    self._outbox.put_nowait({
                        "type": "metadata",
                        "data": daemon.metadata.document()})
                    room_doc = daemon.room.document()
                    if room_doc.get("updated_at"):
                        self._outbox.put_nowait({
                            "type": "room_changed", "data": room_doc})
                except Exception as exc:
                    logger.debug("connect resync failed: %s", exc)
            sender = asyncio.create_task(self._drain(ws))
            try:
                async for raw in ws:
                    if self._stop.is_set():
                        break
                    await self._on_message(ws, raw)
            finally:
                sender.cancel()
                try:
                    await sender
                except Exception:
                    pass

    async def _drain(self, ws) -> None:
        while True:
            frame = await self._outbox.get()
            try:
                await ws.send(json.dumps(frame))
            except Exception:
                # Socket died mid-send — hand the frame off to REST so it
                # isn't lost, then let the session tear down.
                if frame.get("type") in ("event", "room_changed",
                                         "metadata"):
                    self._post_event(
                        frame.get("kind") or frame.get("type") or "event",
                        frame.get("data") or {})
                raise

    async def _on_message(self, ws, raw: Any) -> None:
        try:
            frame = json.loads(raw)
        except Exception:
            return
        if frame.get("type") != "api_request":
            return
        req_id = frame.get("id")
        resp = await asyncio.get_running_loop().run_in_executor(
            None, self.dispatch_api, frame.get("method"),
            frame.get("path"), frame.get("body"))
        await ws.send(json.dumps(
            {"type": "api_response", "id": req_id, **resp}))


__all__ = ["BrainWSClient"]
