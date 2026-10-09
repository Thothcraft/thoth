"""ThothDaemon — the persistent node service.

Runs the Sensor→Model→Actuator loop on Whispy abstractions:

    sensors (whispy.local) → SampleStream (bounded, non-blocking)
        → WindowSynchronizer (missing/stale markers)
        → active processors → Prediction
        → action dispatcher → actuators (explicit ActionResult)

Also records captures, serves the authenticated local API, and reports
health. Sensor ingestion never blocks the loop (§55).
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Optional

from ..capture import CaptureManager
from ..deployment import DeploymentManager
from ..metadata import MetadataManager
from ..models import ModelRegistry
from ..events import EventHub
from ..observations import (BATCH_MAX_ITEMS, Observation,
                            ObservationSpool, build_batch)
from ..room import RoomManager
from ..settings import ConfigStore, config_dir
from .actions import ActionScheduler

logger = logging.getLogger(__name__)


def _probe_bluetooth() -> bool:
    """Cheap BlueZ presence probe for the capabilities advertisement —
    no daemon ownership; the BLE subsystem owns the controller."""
    try:
        import shutil
        return shutil.which("bluetoothctl") is not None or \
            shutil.which("btmgmt") is not None
    except Exception:
        return False


def _lan_ip() -> Optional[str]:
    """Best-effort primary LAN IPv4 (UDP connect — no traffic is sent)."""
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = str(s.getsockname()[0])
        s.close()
        return ip
    except Exception:
        return None


class ThothDaemon:
    """Persistent node service — owns the SMA loop and local API."""

    def __init__(self, config: Optional[ConfigStore] = None,
                 window_seconds: float = 2.0, tick_hz: float = 2.0,
                 serve_ui: Optional[bool] = None,
                 dashboard_port: Optional[int] = None):
        self.config = config or ConfigStore()
        self.registry = ModelRegistry()
        self.deployments = DeploymentManager(self.registry)
        self.captures = CaptureManager()
        self.dispatcher = ActionScheduler(executor=self._execute_action)
        from ..automation import AutomationManager
        self.automations = AutomationManager(self.dispatcher.fire_now)
        self.automations.on_fire = self._automation_fired
        self.metadata = MetadataManager()
        self.room = RoomManager(on_change=self._room_changed)
        self.window_seconds = window_seconds
        self.tick_hz = tick_hz

        self._device = None
        self._streams: Dict[str, Any] = {}
        self._sync = None
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._api = None
        # action_id → ActionResult dict — idempotent actuator dispatch
        self._action_results: Dict[str, Dict[str, Any]] = {}
        self.predictions: Deque[Dict[str, Any]] = deque(maxlen=500)
        # model_id → last emitted label — prediction events are
        # edge-triggered (label changes only, not every tick)
        self._last_pred_labels: Dict[str, str] = {}
        self._cap_subs: Dict[str, Dict[str, Any]] = {}
        # observation/v1 uplink path — durable bounded outbox + a small
        # local ring so the dashboard/API can introspect recent envelopes
        # without touching the spool.
        self._obs_spool = ObservationSpool(config_dir() / "spool")
        self._obs_inflight: Dict[str, List[str]] = {}
        self.observations: Deque[Dict[str, Any]] = deque(maxlen=500)
        from ..radio_evidence import RadioEvidenceBridge
        self._radio_evidence = RadioEvidenceBridge(self.device_id)
        # Node-side context estimators — observations → versioned state
        # keys (contract §4/§8); transitions uplink as context.state.v1.
        from ..estimators import EstimatorHub
        self.estimators = EstimatorHub(
            device_id=self.device_id,
            fingerprints_path=config_dir() / "fingerprints.json")
        # Entities + relationships — persons/devices/spaces share the
        # observation subject keyspace so evidence joins to identity.
        from ..entities import EntityStore
        self.entities = EntityStore(config_dir() / "entities.json")
        # Context uplink — descriptors/predictions/metadata to Brain at
        # the configured rate + detail level (config["context"]).
        from ..context_uplink import ContextUplink
        self.context_uplink = ContextUplink(
            lambda: self.config.get("context") or {}, self.device_id)
        # Built-in discriminators over physical descriptors (guided/auto
        # calibration persisted under ~/.thoth/calibration).
        from ..discriminators import DiscriminatorRuntime
        self.discriminators = DiscriminatorRuntime(
            lambda: self.config.get("discriminators") or {},
            config_dir() / "calibration")
        # Local live transport — SSE fanout for the dashboard/SDK
        # (observations + prediction edges; never the uplink path).
        self.events = EventHub()
        self._started_at: Optional[float] = None
        self._last_heartbeat = 0.0
        self._last_metadata = 0.0
        self._dash = None            # dashboard server on :80 (may be None)
        self._brain_ws = None        # outbound Brain WS client
        self._ws_lock = threading.Lock()  # serializes _start_brain_ws races
        self._ble = None             # BlueZ subsystem (observer/central/peripheral)
        self._prov = None            # provisioning state machine
        self._watches: Dict[str, Any] = {}   # device_id → PinetimeLink
        # None → follow config["dashboard_enabled"] (default on);
        # --no-dashboard passes False for this run.
        self._serve_ui = serve_ui
        self._dashboard_port = dashboard_port

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> "ThothDaemon":
        import whispy
        # One installation identity: the persisted config device_id is the
        # node's identity everywhere (pairing, captures, telemetry, Brain).
        # A pre-set _device (tests, embedded hosts) is used as-is.
        if self._device is None:
            self._device = whispy.local(device_id=self.config.device_id)
        # Warm the lazy subsystem imports *before* sensor streams and the
        # SMA thread start contending for the GIL — under stream load each
        # deferred import crawls; uncontended they cost ~1s total.
        try:
            from ..local_api import LocalAPIServer  # noqa: F401
            from ..bluetooth import BluetoothSubsystem  # noqa: F401
            from ..provisioning import ProvisionManager  # noqa: F401
            from .brain_ws import BrainWSClient  # noqa: F401
        except Exception:
            pass
        self._open_streams()
        from whispy.synchronization import WindowSynchronizer
        self._sync = WindowSynchronizer(
            self._streams, expected=list(self._streams), stale_after_s=5.0)
        self._started_at = time.time()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="thoth-sma",
                                        daemon=True)
        self._thread.start()
        self._start_api()
        self._start_brain_ws()
        self._start_bluetooth()
        self._start_provisioning()
        # First inferred refresh off the SMA thread — geo probes egress.
        self._maybe_refresh_metadata(force=True)
        logger.info("ThothDaemon started: %d sensor stream(s)", len(self._streams))
        return self

    def _open_streams(self) -> None:
        from whispy.streams import SampleStream
        for sensor in self._device.sensors():
            try:
                handle = self._device.sensor(sensor.id)
                # Pass the callable — SampleStream retries by calling it
                # again after a raised exception; a bare iterator is dead
                # forever once its pump loop exits.
                stream = SampleStream(handle.stream, maxlen=4096,
                                      name=sensor.id)
                stream.start()
                self._streams[sensor.id] = stream
            except Exception as exc:
                logger.warning("sensor %s failed to open: %s", sensor.id, exc)

    def _attach_watches(self) -> None:
        """Central-role links to enrolled wearables — subscribe to
        motion, push context back. Retried from ``_tick`` while the
        BLE loop isn't up yet (``attach()`` is a no-op once attached)."""
        ble = self._ble
        if ble is None:
            return
        try:
            from ..bluetooth.pinetime import PinetimeLink
        except Exception:
            return
        for rec in ble.known.list():
            if rec.get("kind") != "watch":
                continue
            did = rec["device_id"]
            link = self._watches.get(did)
            if link is None:
                link = self._watches[did] = PinetimeLink(
                    ble, rec["address"], did, self.emit_observation,
                    ble.source_id)
            if not link._attached:
                try:
                    link.attach()
                except Exception:
                    pass

    def _start_api(self) -> None:
        from ..local_api import LocalAPIServer
        # Loopback by default; set local_host=0.0.0.0 to opt into LAN access.
        host = str(self.config.get("local_host", "127.0.0.1"))
        api_port = int(self.config.get("local_port", 5000))
        token = self.config.local_token
        from pathlib import Path
        dash_dir_cfg = self.config.get("dashboard_dir")
        dash_dir = Path(dash_dir_cfg) if dash_dir_cfg else None

        # Dashboard surface: one port for UI + API (local_port, 5000).
        # A separate dashboard_port is opt-in only.
        # Same exposure policy as the API port (LAN only when opted in).
        # On by default; `--no-dashboard` or config dashboard_enabled=false
        # runs API-only.
        serve_ui = self._serve_ui
        if serve_ui is None:
            serve_ui = bool(self.config.get("dashboard_enabled", True))
        dash_port = (self._dashboard_port
                     if self._dashboard_port is not None
                     else int(self.config.get("dashboard_port", api_port)))
        ui_on_api = False
        if not serve_ui:
            logger.info("dashboard disabled — API-only mode")
        elif dash_port and dash_port != api_port:
            try:
                self._dash = LocalAPIServer(
                    self, host=host, port=dash_port, token=token,
                    serve_ui=True, dashboard_dir=dash_dir).start()
            except OSError as exc:
                # Privileged/port-in-use — degrade gracefully: the API
                # port serves the dashboard instead (Pi without root).
                logger.warning("dashboard port %d unavailable (%s); "
                               "serving UI on :%d", dash_port, exc,
                               api_port)
                self._dash = None
                ui_on_api = True
        else:
            ui_on_api = True
        self._api = LocalAPIServer(self, host=host, port=api_port,
                                   token=token, serve_ui=ui_on_api,
                                   dashboard_dir=dash_dir)
        self._api.start()
        self._start_mdns(dash_port if self._dash is not None
                         else api_port)

    def _start_mdns(self, port: int) -> None:
        """Advertise <device_name>.local + <hostname>.local over mDNS so
        nodes on OSes without a responder (Windows) resolve .local too.
        Only meaningful on a LAN-facing bind; loopback skips it."""
        host = str(self.config.get("local_host", "127.0.0.1"))
        if host != "0.0.0.0":
            return
        try:
            from ..mdns import MdnsAdvertiser
            import socket as _s
            names = {self.config.device_name, _s.gethostname()}
            names |= {n if n.lower().startswith("thoth-") else f"thoth-{n}"
                      for n in list(names) if n}
            self._mdns = MdnsAdvertiser().start(
                names, port, device_id=self.device_id)
        except Exception as exc:
            logger.debug("mDNS advertiser not started: %s", exc)

    def _start_bluetooth(self) -> None:
        """BlueZ subsystem — observer/central roles feeding the
        observation uplink; peripheral role is added by provisioning."""
        try:
            if not bool(self.config.get("ble.enabled", True)):
                self._ble = None
                return
            from ..bluetooth import BluetoothSubsystem
            self._ble = BluetoothSubsystem(
                self.config, emit=self.emit_observation,
                device_id=self.device_id).start()
            self._attach_watches()
        except Exception as exc:
            logger.debug("bluetooth subsystem not started: %s", exc)
            self._ble = None

    def _start_provisioning(self) -> None:
        """Network bring-up: idle → provisioning → online/failed,
        with the BLE commissioning GATT surface when unprovisioned."""
        try:
            if not bool(self.config.get("provisioning.enabled", True)):
                self._prov = None
                return
            from ..provisioning import ProvisionManager
            self._prov = ProvisionManager(
                self.config, emit=self.emit_observation,
                device_id=self.device_id, ble=self._ble).start()
        except Exception as exc:
            logger.debug("provisioning not started: %s", exc)
            self._prov = None

    def net_status(self) -> Dict[str, Any]:
        if self._prov is None:
            return {"enabled": False, "state": "disabled"}
        return self._prov.status()

    def net_provision(self, ssid: str, psk: str = "",
                      hidden: bool = False) -> Dict[str, Any]:
        if self._prov is None:
            return {"accepted": False, "error": "provisioning disabled"}
        return self._prov.provision(ssid, psk, hidden)

    def net_scan(self) -> List[Dict[str, Any]]:
        if self._prov is None:
            return []
        return self._prov.scan_payload()

    def calibrate_zone(self, zone: str) -> Dict[str, Any]:
        """RSSI fingerprint calibration — record the current per-subject
        RSSI vector as ``zone``'s fingerprint (Phase 7)."""
        snap = self.estimators.calibrate_zone(str(zone))
        if snap is None:
            return {"ok": False, "error": "no ble.rssi.v1 evidence yet"}
        return {"ok": True, "zone": str(zone), "anchors": snap}

    # -- Phase 9: wearable enrollment + entity relations -------------------------
    def ble_enroll(self, address: str, kind: str = "device",
                   name: Optional[str] = None,
                   person: Optional[str] = None) -> Dict[str, Any]:
        """Bind a BLE address to a stable ``device:<uuid>`` subject and
        optionally a ``person:<uuid>`` via a ``wears`` edge."""
        ble = self._ble
        if ble is None:
            return {"ok": False, "error": "bluetooth unavailable"}
        from ..bluetooth import BluetoothSubsystem
        ah = BluetoothSubsystem.addr_hash(address, ble._salt())
        ent = self.entities.upsert(
            type="device", name=name,
            attributes={"ble": True, "kind": kind})
        rec = ble.known.enroll(ah, address, kind=kind, name=name,
                               device_id=ent["id"])
        out: Dict[str, Any] = {"ok": True, "device_id": ent["id"],
                               "addr_hash": ah, "record": rec}
        if person:
            p = self._find_or_create_person(person)
            edge = self.entities.relate(p["id"], ent["id"], "wears")
            out["person_id"] = p["id"]
            out["relation"] = edge
        if kind == "watch":
            self._attach_watches()
        return out

    def _find_or_create_person(self, name: str) -> Dict[str, Any]:
        for p in self.entities.list(type="person"):
            if p.get("name") == name:
                return p
        return self.entities.upsert(type="person", name=name)

    def ble_unenroll(self, address: str) -> Dict[str, Any]:
        ble = self._ble
        if ble is None:
            return {"ok": False, "error": "bluetooth unavailable"}
        from ..bluetooth import BluetoothSubsystem
        ah = BluetoothSubsystem.addr_hash(address, ble._salt())
        return {"ok": ble.known.unenroll(ah), "addr_hash": ah}

    def ble_known(self) -> Dict[str, Any]:
        ble = self._ble
        if ble is None:
            return {"known": [], "seen_now": []}
        return {"known": ble.known.list(), "seen_now": ble.seen_now()}

    def _start_brain_ws(self) -> None:
        """Outbound Brain channel (CONTRACT §2) — only when paired."""
        token = self.config.device_token
        if not token:
            return
        with self._ws_lock:
            if self._brain_ws is not None:
                return
            self._start_brain_ws_locked(token)

    def _start_brain_ws_locked(self, token: str) -> None:
        """Launch the WS client; caller holds ``_ws_lock``."""
        try:
            from .brain_ws import BrainWSClient
            host = str(self.config.get("local_host", "127.0.0.1"))
            loop_host = "127.0.0.1" if host == "0.0.0.0" else host
            api_base = (f"http://{loop_host}:"
                        f"{int(self.config.get('local_port', 5000))}")
            self._brain_ws = BrainWSClient(
                self, api_base=api_base,
                local_token=self.config.local_token,
                brain_url=self.config.brain_url,
                device_id=self.device_id,
                device_token=token).start()
        except Exception as exc:
            logger.debug("brain ws client not started: %s", exc)

    def emit_event(self, kind: str, data: Dict[str, Any]) -> None:
        """Push a node→Brain frame (WS when connected, REST fallback)."""
        ws = self._brain_ws
        if ws is not None:
            try:
                ws.send_event(kind, data)
                return
            except Exception as exc:
                logger.debug("event send failed: %s", exc)
        # Unpaired or no WS client — nothing to do locally.

    # -- observations (observation/v1) -----------------------------------------
    def emit_observation(self, obs: Observation | Dict[str, Any]) -> str:
        """Producer API: validate + enqueue one observation for uplink.

        Sources call this for *context-relevant* envelopes only — raw
        high-rate data stays on the capture/tail path. Returns the
        observation_id (stable across reconnect redelivery).
        """
        item = obs.validate().to_dict() if isinstance(obs, Observation) else dict(obs)
        obs_id = self._obs_spool.append(item)
        item["observation_id"] = obs_id
        self.observations.append(item)
        try:
            self.events.publish("observation", item)
        except Exception:
            pass
        # estimators consume the stream and may emit context.state.v1
        for state in self.estimators.consume(item):
            self._emit_state(state)
        return obs_id

    def _emit_state(self, state: Dict[str, Any]) -> None:
        """One estimator state transition → context.state.v1 on the
        uplink (evidence-linked) and the local event fanout."""
        try:
            self.emit_observation(Observation(
                schema="context.state.v1",
                source_id="estimators",
                subject=state.get("entity_id"),
                value=state).validate())
            # context-device channel — push the new state to watches
            for link in self._watches.values():
                try:
                    link.push_context({"k": state.get("key"),
                                       "v": state.get("value"),
                                       "c": state.get("confidence")})
                except Exception:
                    pass
        except Exception:
            pass

    def _flush_observations(self) -> None:
        """Drain the spool into ``observation_batch`` WS frames.

        Peek-then-ack: items leave the spool only after the socket
        confirms the send (``_observation_batch_sent``); a failed frame
        keeps its items pending so reconnect redelivers them — Brain
        dedupes on observation_id (contract §2).
        """
        ws = self._brain_ws
        if ws is None or not ws.connected or len(self._obs_spool) == 0:
            return
        if len(self._obs_inflight) > 16:
            # Outbound queue is saturated — let the drain task catch up.
            return
        items = self._obs_spool.pending(BATCH_MAX_ITEMS)
        if not items:
            return
        frame = build_batch(items)
        self._obs_inflight[frame["id"]] = [
            str(i.get("observation_id")) for i in frame["items"]
            if i.get("observation_id")]
        if not ws.send_frame(frame):
            self._obs_inflight.pop(frame["id"], None)

    def _observation_batch_sent(self, batch_id: str) -> None:
        """WS drain confirmation → ack items out of the spool."""
        ids = self._obs_inflight.pop(batch_id, None)
        if ids:
            self._obs_spool.ack(ids)

    def _observation_batch_failed(self, batch_id: str) -> None:
        """Send died — drop the inflight marker; items stay pending and
        are re-sent on the next flush."""
        self._obs_inflight.pop(batch_id, None)

    def _automation_fired(self, auto: Any, action_cfg: Dict[str, Any],
                        prediction: Any) -> None:
        """AutomationManager → CONTRACT §6 trigger_fired event."""
        self.emit_event("trigger_fired", {
            "automation_id": getattr(auto, "id", ""),
            "name": getattr(auto, "name", ""),
            "device_id": self.device_id,
            "action_type": (action_cfg or {}).get("type", ""),
            "label": getattr(prediction, "label", None),
            "confidence": getattr(prediction, "confidence", None),
            "at": time.time(),
        })

    def _room_changed(self, doc: Dict[str, Any]) -> None:
        """Room PUT → broadcast room_changed + authoritative REST sync
        (CONTRACT §1.2 — node POSTs the doc to Brain on every change)."""
        self.emit_event("room_changed", doc)
        token = self.config.device_token
        if not token:
            return
        def _push() -> None:
            try:
                import json as _json
                import urllib.request as _req
                req = _req.Request(
                    f"{self.config.brain_url.rstrip('/')}/v1/nodes/"
                    f"{self.device_id}/room",
                    data=_json.dumps(doc).encode(), method="PUT",
                    headers={"Content-Type": "application/json",
                             "Authorization": f"Bearer {token}"})
                _req.urlopen(req, timeout=8).read()
            except Exception as exc:
                logger.debug("room push failed: %s", exc)
        threading.Thread(target=_push, name="thoth-room-push",
                         daemon=True).start()

    def _maybe_refresh_metadata(self, force: bool = False,
                                inline: bool = False) -> None:
        """Refresh inferred metadata (≥60 s cadence) off the SMA thread.

        ``public_geo`` does a network call — run it on a helper thread so
        an unreachable egress never stalls inference. ``inline`` is the
        synchronous path used by tests.
        """
        interval = float(self.config.get("metadata_interval_s", 60.0))
        now = time.time()
        if not force and now - self._last_metadata < interval:
            return
        self._last_metadata = now

        def _run() -> None:
            try:
                changed = self.metadata.refresh_inferred(
                    predictions=list(self.predictions))
            except Exception as exc:
                logger.debug("metadata refresh failed: %s", exc)
                return
            if changed:
                self.emit_event("metadata", self.metadata.document())

        if inline:
            _run()
        else:
            threading.Thread(target=_run, name="thoth-metadata",
                             daemon=True).start()

    def stop(self) -> None:
        self._stop.set()
        if getattr(self, "_mdns", None) is not None:
            try:
                self._mdns.close()
            except Exception:
                pass
        if self._brain_ws is not None:
            try:
                self._brain_ws.stop()
            except Exception:
                pass
            self._brain_ws = None
        if self._ble is not None:
            try:
                self._ble.stop()
            except Exception:
                pass
            self._ble = None
        if self._prov is not None:
            try:
                self._prov.stop()
            except Exception:
                pass
            self._prov = None
        if self._dash is not None:
            try:
                self._dash.stop()
            except Exception:
                pass
            self._dash = None
        if self._thread:
            self._thread.join(timeout=3.0)
        try:
            self.dispatcher.stop()
        except Exception:
            pass
        for cap_id, subs in self._cap_subs.items():
            for sid, sub in subs.items():
                stream = self._streams.get(sid)
                if stream:
                    try:
                        stream.unsubscribe(sub)
                    except Exception:
                        pass
        self._cap_subs.clear()
        for stream in self._streams.values():
            try:
                stream.close()
            except Exception:
                pass
        if self._device:
            try:
                self._device.close()
            except Exception:
                pass
        if self._api:
            self._api.stop()
        logger.info("ThothDaemon stopped")

    # -- SMA loop -----------------------------------------------------------------
    def _loop(self) -> None:
        period = 1.0 / self.tick_hz if self.tick_hz > 0 else 0.5
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.warning("SMA tick error: %s", exc)
            time.sleep(period)

    def _tick(self) -> None:
        self._maybe_heartbeat()
        self._maybe_refresh_metadata()
        # Observations must flush even when no sensor streams exist —
        # sources like the BLE observer produce them independently.
        self._flush_observations()
        # estimator decay — silent sources produce no new observations
        for state in self.estimators.tick():
            self._emit_state(state)
        self._attach_watches()
        if self._sync is None:
            # No streams: still drive time/schedule triggers (no features).
            self.automations.tick()
            self._maybe_context_uplink(None)
            return
        window = self._sync.rolling(self.window_seconds)
        self._record_captures()
        for observation in self._radio_evidence.consume(window):
            self.emit_observation(observation)
        from whispy.windows import WindowFeatures
        feats = WindowFeatures(window)
        last_pred: Optional[Any] = None
        for model in self.registry.active():
            try:
                proc = model.processor_impl()
                prediction = proc.predict(window)
                prediction.device_id = self.device_id
                prediction.runtime_model_id = model.runtime_model_id
            except Exception as exc:
                logger.warning("model %s predict failed: %s",
                               model.runtime_model_id, exc)
                continue
            self.predictions.append(prediction.to_dict())
            self._emit_prediction_edge(model, prediction)
            self._fire_actions(model, prediction)
            self.automations.on_prediction(prediction, features=feats)
            last_pred = prediction
        for prediction in self._run_discriminators(window):
            self.automations.on_prediction(prediction, features=feats)
            last_pred = prediction
        self.automations.tick(prediction=last_pred, features=feats)
        self._maybe_context_uplink(window)

    def _run_discriminators(self, window: Any) -> List[Any]:
        """Built-in discriminators: record calibration windows, predict
        when calibrated, and publish like any model prediction."""
        try:
            from types import SimpleNamespace
            from whispy.descriptors import window_descriptors
            preds = self.discriminators.observe(
                window_descriptors(window), device_id=self.device_id)
        except Exception as exc:
            logger.warning("discriminators failed: %s", exc)
            return []
        for p in preds:
            self.predictions.append(p.to_dict())
            self._emit_prediction_edge(
                SimpleNamespace(runtime_model_id=p.runtime_model_id,
                                config={}), p)
        return preds

    def _maybe_context_uplink(self, window: Any) -> None:
        if not self.context_uplink.due():
            return
        try:
            self.context_uplink.maybe_emit(
                self.emit_observation, window=window,
                predictions=list(self.predictions)[-50:],
                estimates=self.estimators.states(),
                room=self.room.document())
        except Exception as exc:
            logger.warning("context uplink failed: %s", exc)

    def _emit_prediction_edge(self, model: Any, prediction: Any) -> None:
        """Emit a ``prediction`` event on each label transition.

        Brain projects these into ContextState (``key=prediction``) →
        ContextEvent → server-side rules. Edge-triggered so a steady
        label never floods the channel.
        """
        mid = model.runtime_model_id
        label = prediction.label
        if self._last_pred_labels.get(mid) == label:
            return
        previous = self._last_pred_labels.get(mid)
        self._last_pred_labels[mid] = label
        try:
            self.events.publish("prediction", prediction.to_dict())
        except Exception:
            pass
        self.emit_event("prediction", {
            **prediction.to_dict(),
            "model_id": mid,
            "previous_label": previous})

    def _execute_action(self, key: str, action: Any, pred: Any) -> None:
        """Dispatcher executor — runs the actuator, then bridges
        ``notification`` results onto the node→Brain event channel so a
        triggered rule surfaces as a push notification on the app."""
        from whispy.contracts import ActionResult, ActionStatus
        from whispy.actuators import create_actuator
        try:
            actuator = create_actuator(action)
            result = actuator.trigger(action, pred)
        except Exception as exc:
            result = ActionResult(status=ActionStatus.FAILED,
                                  action_type=action.type, detail=str(exc))
        self.dispatcher._record(key, action, pred, result=result)
        if action.type == "notification":
            note = ((result.response or {}).get("notification")
                    if result.response else None)
            if note and result.status == ActionStatus.SUCCEEDED:
                self.emit_event("notification", note)

    def _fire_actions(self, model: Any, prediction: Any) -> None:
        for action_cfg in model.config.get("actions") or []:
            self.dispatcher.submit(action_cfg, prediction,
                                   model_id=model.runtime_model_id)

    # -- Brain heartbeat ---------------------------------------------------------
    def _maybe_heartbeat(self) -> None:
        """POST /api/device/heartbeat so the portal sees this node online.

        Runs at most every ``heartbeat_interval_s`` (default 30s — Brain's
        online timeout is 90s). No-op until a device token is configured.
        Reloads config each tick so ``thoth pair`` on a running daemon
        takes effect without a restart; the WS tunnel is (re)started the
        first time a token appears.
        Failures are logged at debug level: an unreachable Brain must never
        disturb local sensing.
        """
        try:
            self.config.reload()
        except Exception:
            pass
        token = self.config.device_token
        if not token:
            return
        if self._brain_ws is None:
            logger.info("device token found — starting Brain channel")
            self._start_brain_ws()
        interval = float(self.config.get("heartbeat_interval_s", 30.0))
        now = time.time()
        if now - self._last_heartbeat < interval:
            return
        self._last_heartbeat = now
        try:
            import json as _json
            import socket as _socket
            import urllib.request as _req
            body = {
                "device_id": self.device_id,
                "device_name": self.config.device_name,
                "device_type": "thoth",
                "device_hostname": _socket.gethostname(),
                "online": True,
                "hardware_info": self._heartbeat_inventory(),
            }
            req = _req.Request(
                f"{self.config.brain_url.rstrip('/')}/api/device/heartbeat",
                data=_json.dumps(body).encode(), method="POST",
                headers={"Content-Type": "application/json",
                         "Authorization": f"Bearer {token}"})
            with _req.urlopen(req, timeout=10) as resp:
                if resp.status != 200:
                    logger.debug("heartbeat rejected: %s", resp.status)
        except Exception as exc:
            logger.debug("heartbeat failed: %s", exc)

    def _heartbeat_inventory(self) -> Dict[str, Any]:
        """Sensor/actuator inventory + LAN endpoint for the registry.

        Brain stores this under ``device.hardware_info``: ``local_api``
        lets same-LAN clients (and Brain-side actuation) reach the node;
        ``sensors`` feeds ``/v1/devices/{id}/sensors``.
        """
        sensors: List[Dict[str, Any]] = []
        actuators: List[Dict[str, Any]] = []
        if self._device is not None:
            try:
                sensors = [s.to_dict() for s in self._device.sensors()]
            except Exception:
                pass
            try:
                actuators = [a.to_dict() for a in self._device.actuators()]
            except Exception:
                pass
        local_api: Dict[str, Any] = {}
        host = _lan_ip()
        if host:
            local_api = {
                "host": host,
                "port": int(self.config.get("local_port", 5000)),
                "token": self.config.local_token,
            }
        name = str(self.config.device_name or "").strip().lower()
        if name and not name.startswith("thoth-"):
            name = f"thoth-{name}"
        hostname = f"{name}.local" if name else None
        if local_api and hostname:
            local_api["hostname"] = hostname
        return {"sensors": sensors, "actuators": actuators,
                "local_api": local_api, "hostname": hostname,
                "activity": self._activity()}

    def _activity(self) -> Dict[str, Any]:
        """Live "what is this node doing" snapshot for the fleet UI.

        Lands under ``device.hardware_info.activity`` via the heartbeat
        merge — mode + the sensors/captures/models/watch links actually
        live right now, so the app can answer "idle? collecting through
        the dashboard? streaming which sensors?" without guessing.
        """
        now = time.time()
        streams: List[Dict[str, Any]] = []
        for sid, st in (self._streams or {}).items():
            try:
                last = getattr(st, "_last", None)
                age = (now - last.timestamp) if last is not None else None
                subs = [getattr(s, "_name", "") or ""
                        for s in getattr(st, "_subs", [])]
                streams.append({
                    "id": sid,
                    "fresh": age is not None and age < 10.0,
                    "last_age_s": round(age, 1) if age is not None else None,
                    "subscribers": [n for n in subs if n],
                })
            except Exception:
                pass
        captures = [{"id": c.get("id"),
                     "sensors": list(c.get("sensors") or [])}
                    for c in getattr(self.captures, "_active", {}).values()]
        try:
            models = [m.runtime_model_id for m in self.registry.active()]
        except Exception:
            models = []
        watches = [{"subject": w.subject, "connected": w.connected}
                   for w in self._watches.values()]
        brain = bool(self._brain_ws is not None
                     and getattr(self._brain_ws, "connected", False))
        if captures:
            mode = "capturing"
        elif models:
            mode = "inferring"
        elif any(s["fresh"] for s in streams):
            mode = "sensing"
        else:
            mode = "idle"
        return {"mode": mode, "captures": captures, "models": models,
                "streams": streams, "watches": watches,
                "brain_ws": brain}

    def _record_captures(self) -> None:
        """Flush each capture's private subscription queue to disk.

        Subscriptions are non-destructive: recording never removes samples
        from the shared buffer the synchronizer reads, and concurrent
        captures each receive their own copy of every sample.
        """
        active_ids = set(self.captures._active)
        # Drop subscriptions for captures that have stopped.
        for cap_id in [c for c in self._cap_subs if c not in active_ids]:
            for sid, sub in self._cap_subs.pop(cap_id).items():
                stream = self._streams.get(sid)
                if stream:
                    stream.unsubscribe(sub)

        for cap in self.captures._active.values():
            subs = self._cap_subs.setdefault(cap["id"], {})
            for sid in cap["sensors"]:
                stream = self._streams.get(sid)
                if not stream:
                    continue
                sub = subs.get(sid)
                if sub is None:
                    sub = stream.subscribe(name=f"capture-{cap['id']}")
                    subs[sid] = sub
                for sample in sub.read():
                    self.captures.record(cap["id"], sample)

    # -- LAN tail (§24) ---------------------------------------------------------
    def _exposure(self) -> Dict[str, Any]:
        """Capability exposure lists set by ``thoth expose``.

        ``{"sensors": [...], "actuators": [...]}`` — empty/absent lists
        mean "all". Permissions describe capabilities, not HTTP routes.
        """
        exp = self.config.get("exposed") or {}
        return {"sensors": list(exp.get("sensors") or []),
                "actuators": list(exp.get("actuators") or [])}

    def sensor_exposed(self, sensor_id: str) -> bool:
        allowed = self._exposure()["sensors"]
        return not allowed or sensor_id in allowed

    def actuator_exposed(self, actuator_id: str) -> bool:
        allowed = self._exposure()["actuators"]
        return not allowed or actuator_id in allowed

    def tail_sensor(self, sensor_id: str, cursor: int = 0,
                    limit: Optional[int] = None) -> Optional[Dict[str, Any]]:
        """Return buffered samples newer than ``cursor`` for one sensor.

        Reads the stream's ring buffer non-destructively (``snapshot``), so
        tailing never disturbs inference or captures. Each sample's own
        ``sequence`` is the ordering key, so a client's cursor is just the
        last sequence it saw — stateless and safe for concurrent clients.
        Returns ``None`` for unknown or non-exposed sensors.

        ``limit`` bounds the response: ``0`` returns only the latest cursor
        (a cheap "jump to live edge" probe), and ``N`` returns the newest
        ``N`` samples — without it a fast sensor can dump the whole 4096-
        sample ring on the first poll and stall the client.
        """
        if not self.sensor_exposed(sensor_id):
            return None
        stream = self._streams.get(sensor_id)
        if stream is None:
            return None
        snap = stream.snapshot()
        latest = max((s.sequence for s in snap), default=cursor)
        # Slice the backlog BEFORE serializing — a fast sensor can hold
        # thousands of samples and to_dict() on each one is the expensive
        # part, so dropping them post-serialize wasted real CPU per poll.
        backlog = [s for s in snap if s.sequence > cursor]
        skipped = 0
        if limit is not None:
            if limit <= 0:
                backlog = []          # cursor-only probe
            elif len(backlog) > limit:
                skipped, backlog = len(backlog) - limit, backlog[-limit:]
        samples = [s.to_dict() for s in backlog]
        out = {"sensor_id": sensor_id, "cursor": latest,
               "samples": samples}
        if skipped:
            out["skipped"] = skipped
        return out

    # -- actuators ---------------------------------------------------------------
    def actuators(self) -> List[Dict[str, Any]]:
        """Actuator inventory as descriptor dicts (exposure-filtered)."""
        if self._device is None:
            return []
        try:
            descriptors = self._device.actuators()
        except Exception:
            return []
        return [d.to_dict() for d in descriptors
                if self.actuator_exposed(d.id)]

    def execute_actuator(self, actuator_id: str,
                         command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an ActuatorCommand on a local actuator by id/kind.

        Returns an ActionResult dict — explicit outcomes only (§7.6).
        """
        from whispy.contracts import (
            ActionResult, ActionStatus, ActuatorCommand)
        action_id = command.get("action_id")
        expires_at = command.get("expires_at")
        # Idempotent dispatch: a replayed action_id returns the stored
        # result instead of re-executing (at-least-once → exactly-once).
        if action_id and action_id in self._action_results:
            out = dict(self._action_results[action_id])
            out["deduplicated"] = True
            return out
        # Expired commands never execute — explicit terminal status.
        if expires_at is not None and float(expires_at) < time.time():
            out = ActionResult(
                status=ActionStatus.EXPIRED, action_type="actuator",
                detail="command expired before dispatch").to_dict()
            if action_id:
                self._action_results[action_id] = dict(out)
            return out
        if not self.actuator_exposed(actuator_id):
            return ActionResult(
                status=ActionStatus.UNSUPPORTED,
                action_type="actuator",
                detail=f"actuator {actuator_id!r} is not exposed").to_dict()
        if self._device is None:
            return ActionResult(status=ActionStatus.FAILED,
                                action_type="actuator",
                                detail="device not started").to_dict()
        try:
            handle = self._device.actuator(actuator_id)
        except KeyError:
            return ActionResult(
                status=ActionStatus.UNSUPPORTED, action_type="actuator",
                detail=f"unknown actuator {actuator_id!r}").to_dict()
        try:
            result = handle.execute(ActuatorCommand.from_dict(command))
            out = result.to_dict() if hasattr(result, "to_dict") \
                else dict(result)
        except Exception as exc:
            out = ActionResult(status=ActionStatus.FAILED,
                               action_type="actuator",
                               detail=str(exc)).to_dict()
        if action_id:
            self._action_results[action_id] = dict(out)
        return out

    # -- introspection -------------------------------------------------------------
    @property
    def device_id(self) -> str:
        return self._device.info.id if self._device else self.config.device_id

    def status(self) -> Dict[str, Any]:
        return {
            "device_id": self.device_id,
            "device_name": self.config.device_name,
            "uptime_s": (time.time() - self._started_at) if self._started_at else 0,
            "sensors": [s.to_dict() for s in (self._device.sensors() if self._device else [])
                        if self.sensor_exposed(s.id)],
            "actuators": self.actuators(),
            "streams": {sid: {"dropped": st.dropped,
                              "last_ts": st.last_timestamp}
                        for sid, st in self._streams.items()},
            "models": [m.to_dict() for m in self.registry.list()],
            "active_models": len(self.registry.active()),
            "predictions": len(self.predictions),
            "captures": len(self.captures.list()),
            "capture": {
                "active": bool(self.captures._active),
                "capture_id": next(iter(self.captures._active), None),
            },
            "running": not self._stop.is_set(),
        }

    def recent_predictions(self, limit: int = 50) -> List[Dict[str, Any]]:
        return list(self.predictions)[-limit:]

    def model_catalog(self) -> List[Dict[str, Any]]:
        """Runnable models: local whispy inventory + Brain's cloud catalog
        when the node is paired. Cloud failures degrade to local-only."""
        from whispy.models.registry import models as whispy_models
        out: List[Dict[str, Any]] = []
        try:
            out.extend(whispy_models())
        except Exception as exc:
            logger.debug("local model inventory failed: %s", exc)
        token = self.config.device_token
        if token and self.config.brain_url:
            try:
                import json as _json
                import urllib.request as _req
                req = _req.Request(
                    f"{self.config.brain_url.rstrip('/')}/v1/models",
                    headers={"Authorization": f"Bearer {token}"})
                with _req.urlopen(req, timeout=8) as res:
                    for m in _json.loads(res.read().decode()).get("models", []):
                        m = dict(m); m.setdefault("source", "cloud")
                        out.append(m)
            except Exception as exc:
                logger.debug("cloud model catalog unreachable: %s", exc)
        return out

    # -- v1 API surface (§12) ---------------------------------------------------
    def compute(self) -> Dict[str, Any]:
        """Compute capability advertisement — probed, never fabricated.

        Cached for 60s: probing spawns subprocesses (nvidia-smi) that can
        be slow, and compute capability is semi-static.
        """
        now = time.time()
        cached = getattr(self, "_compute_cache", None)
        if cached and now - cached[0] < 60.0:
            return dict(cached[1])
        from whispy.compute import probe_compute
        result = probe_compute().to_dict()
        self._compute_cache = (now, result)
        return dict(result)

    def health(self) -> Dict[str, Any]:
        """Structured health: per-source state, models, actuators, uptime."""
        sources = []
        for s in (self._device.sensors() if self._device else []):
            stream = self._streams.get(s.id)
            sources.append({
                "id": s.id, "type": s.type, "online": s.online,
                "exposed": self.sensor_exposed(s.id),
                "last_sample_ts": getattr(stream, "last_timestamp", None)
                    if stream else None,
                "dropped": getattr(stream, "dropped", 0) if stream else 0,
                "expected_rate": getattr(s, "sample_rate", None),
                "actual_rate": self._measured_rate(stream),
            })
        return {
            "device_id": self.device_id,
            "running": not self._stop.is_set(),
            "uptime_s": (time.time() - self._started_at)
                if self._started_at else 0,
            "observations_pending": len(self._obs_spool),
            "sources": sources,
            "actuators": self.actuators(),
            "models": {"installed": len(self.registry.list()),
                       "active": len(self.registry.active())},
            "captures_active": len(self.captures._active),
        }

    def capabilities(self) -> Dict[str, Any]:
        """Node capability advertisement (§12 /api/v1/capabilities).

        Generic capability names — hardware appears as source types, not
        product APIs. Probed, never fabricated: optional hardware that
        isn't present simply doesn't appear.
        """
        import platform
        caps: Dict[str, Any] = {
            "device_id": self.device_id,
            "platform": platform.system().lower(),
            "compute": self.compute(),
            "features": [],
            "sources": {},
        }
        feats = caps["features"]
        if self._brain_ws is not None:
            feats.append("cloud_link")
        feats.append("observation_uplink")
        if self._streams:
            feats.append("streaming")
        try:
            for d in self.sources():
                key = d.get("modality") or d.get("type") or "unknown"
                caps["sources"][key] = caps["sources"].get(key, 0) + 1
        except Exception:
            pass
        if caps["sources"]:
            feats.append("sensing")
        if self.actuators():
            feats.append("actuation")
        caps["domains"] = self._detect_domains()
        # Bluetooth capability — the BLE subsystem reports presence even
        # before it is enabled, so provisioning surfaces can react.
        ble = getattr(self, "_ble", None)
        if ble is not None:
            try:
                caps["bluetooth"] = ble.capability()
                feats.append("bluetooth")
            except Exception:
                caps["bluetooth"] = {"present": False}
        else:
            caps["bluetooth"] = {"present": _probe_bluetooth()}
            if caps["bluetooth"]["present"]:
                feats.append("bluetooth")
        return caps

    _RATE_WINDOW_S = 20.0

    def _measured_rate(self, stream: Any) -> Optional[float]:
        """Observed sample rate (Hz) over the last _RATE_WINDOW_S of the
        ring buffer — the 'actual' half of device inspection (G11)."""
        if stream is None:
            return None
        try:
            n = len(stream.since(time.time() - self._RATE_WINDOW_S))
        except Exception:
            return None
        return round(n / self._RATE_WINDOW_S, 2) if n else 0.0

    def _annotate_rate(self, desc: Dict[str, Any],
                       sensor: Any = None) -> Dict[str, Any]:
        """Attach expected_rate (declared sample_rate) + actual_rate
        (measured) to a source descriptor dict."""
        stream = self._streams.get(desc.get("id"))
        desc.setdefault("expected_rate",
                        desc.get("sample_rate")
                        or getattr(sensor, "sample_rate", None))
        desc["actual_rate"] = self._measured_rate(stream)
        return desc

    def sources(self) -> List[Dict[str, Any]]:
        """All observation-source descriptors (exposure-filtered)."""
        if self._device is None:
            return []
        sensors = {s.id: s for s in self._device.sensors()}
        try:
            descriptors = self._device.sensor_descriptors()
        except Exception:
            descriptors = []
        if not descriptors:
            # Older devices only expose the Sensor inventory contract.
            return [self._annotate_rate(s.to_dict(), s)
                    for s in self._device.sensors()
                    if self.sensor_exposed(s.id)]
        return [self._annotate_rate(d.to_dict(), sensors.get(d.id))
                for d in descriptors
                if self.sensor_exposed(d.id)]

    def source(self, key: str) -> Optional[Dict[str, Any]]:
        for desc in self.sources():
            if key in (desc.get("id"), desc.get("name")):
                return desc
        matches = [d for d in self.sources()
                   if d.get("modality") == key or d.get("type") == key]
        return matches[0] if len(matches) == 1 else None

    # -- Phase 6: conformance + capability detection ----------------------------
    _CONF_CACHE_S = 30.0

    def source_conformance(self) -> Dict[str, Any]:
        """whispy conformance gate over every discovered sensor adapter.
        Cached — these checks touch hardware and must not be cheap."""
        now = time.time()
        cached = getattr(self, "_conf_cache", None)
        if cached and now - cached[0] < self._CONF_CACHE_S:
            return dict(cached[1])
        reports: Dict[str, Any] = {}
        device = self._device
        adapters = getattr(device, "_adapters", {}) if device else {}
        try:
            from whispy.conformance import check_sensor_adapter
        except Exception:
            check_sensor_adapter = None
        for name, adapter in adapters.items():
            if check_sensor_adapter is None:
                reports[name] = {"passed": None,
                                 "error": "whispy.conformance unavailable"}
                continue
            try:
                reports[name] = check_sensor_adapter(
                    adapter, max_samples=1, timeout_s=3.0)
            except Exception as exc:
                reports[name] = {"passed": False,
                                 "error": f"{type(exc).__name__}: {exc}"}
        out = {"adapters": reports,
               "passed": all(r.get("passed") for r in reports.values())
                         if reports else None}
        self._conf_cache = (now, out)
        return dict(out)

    def _detect_domains(self) -> Dict[str, bool]:
        """Canonical sensing domains probed from descriptors, adapters
        and subsystems — honest presence bits for the capabilities ad."""
        modalities: set = set()
        adapter_names: set = set()
        device = self._device
        if device is not None:
            try:
                for d in device.sensor_descriptors() or []:
                    m = getattr(d, "modality", None) or \
                        (d.to_dict().get("modality")
                         if hasattr(d, "to_dict") else None)
                    if m:
                        modalities.add(str(m).lower())
            except Exception:
                pass
            adapter_names = {str(n).lower()
                             for n in getattr(device, "_adapters", {})}
        haystack = modalities | adapter_names
        ble = getattr(self, "_ble", None)
        ble_present = bool(getattr(ble, "present", False)) or \
            _probe_bluetooth()
        return {
            "camera": bool(haystack & {"camera", "video", "vision"}),
            "radar": bool(haystack & {"radar", "csi", "mmwave"}),
            "bluetooth": ble_present or "bluetooth" in haystack
                          or "ble" in haystack,
            "zigbee": bool(haystack & {"zigbee", "z2m", "zb"}),
            "imu": bool(haystack & {"imu", "accel", "gyro"}),
            "environmental": bool(haystack & {"environmental", "env",
                                              "temperature", "humidity"}),
        }

    def source_observations(self, source_id: str,
                            cursor: int = 0) -> Optional[Dict[str, Any]]:
        """Canonical name for the sensor tail endpoint."""
        return self.tail_sensor(source_id, cursor)

    def latest_observation(self, source_id: str) -> Optional[Dict[str, Any]]:
        """Just the newest sample — what live views actually want.

        ``tail`` over a whole buffer is for history; streaming UIs poll
        ``?latest=1`` and get one sample, so the response stays small
        even when payloads are heavy (radar xy_map frames).
        """
        if not self.sensor_exposed(source_id):
            return None
        stream = self._streams.get(source_id)
        if stream is None:
            return None
        snap = stream.snapshot()
        if not snap:
            return {"sensor_id": source_id, "cursor": 0, "samples": []}
        last = max(snap, key=lambda s: s.sequence)
        return {"sensor_id": source_id, "cursor": last.sequence,
                "samples": [last.to_dict()]}

    def _minutes_root(self):
        from pathlib import Path
        from ..settings import config_dir
        configured = self.config.get("data_dir")
        return Path(configured).expanduser() if configured \
            else config_dir() / "minutes"

    def minutes(self) -> List[Dict[str, Any]]:
        """Minute summaries under the node's data root."""
        from whispy.minutes import iter_minute_dirs, read_minute
        out = []
        for d in iter_minute_dirs(self._minutes_root()):
            try:
                m = read_minute(d)
                out.append({
                    "minute_id": m.minute_id, "device_id": m.device_id,
                    "start_timestamp": m.start_timestamp,
                    "end_timestamp": m.end_timestamp,
                    "sources": len(m.sources),
                    "predictions": len(m.predictions),
                    "labels": m.labels.get("labels") or [],
                    "canonical": (d / "minute.json").exists(),
                })
            except Exception as exc:
                out.append({"minute_id": d.name, "error": str(exc)})
        return out

    def minute(self, minute_id: str) -> Optional[Dict[str, Any]]:
        from whispy.minutes import iter_minute_dirs, read_minute
        for d in iter_minute_dirs(self._minutes_root()):
            if d.name == minute_id:
                return read_minute(d).to_dict()
        return None

    def minute_second(self, minute_id: str, index: int
                      ) -> Optional[Dict[str, Any]]:
        """Second-level view of one minute (legacy chunk data normalized)."""
        manifest = self.minute(minute_id)
        if manifest is None:
            return None
        seconds = (manifest.get("quality") or {}).get("seconds") or []
        for entry in seconds:
            if entry.get("second_index") == index:
                return entry
        return {"minute_id": minute_id, "second_index": index,
                "status": "missing"}

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """One-shot inference on the live window → canonical InferenceResult."""
        from whispy.contracts import (
            InferenceRequest, InferenceResult, InferenceTrace)
        req = InferenceRequest.from_dict(request)
        started = time.time()
        model = None
        for m in self.registry.list():
            if req.model_id in (m.runtime_model_id, m.name):
                model = m
                break
        if model is None:
            return InferenceResult(
                request_id=req.request_id, status="failed",
                error=f"unknown model {req.model_id!r}").to_dict()
        if self._sync is None:
            return InferenceResult(
                request_id=req.request_id, status="failed",
                error="daemon not started").to_dict()
        try:
            window = self._sync.rolling(
                req.window_seconds or self.window_seconds)
            pred = model.processor_impl().predict(window)
            pred.device_id = self.device_id
            pred.runtime_model_id = model.runtime_model_id
            latency_ms = (time.time() - started) * 1000.0
            trace = InferenceTrace(
                model_id=req.model_id, runtime_id=model.runtime_model_id,
                execution_device=self.device_id, execution_class="local",
                input_interval={"start": window.start_timestamp,
                                "end": window.end_timestamp},
                inference_timestamp=started, latency_ms=latency_ms,
                confidence=pred.confidence)
            self.predictions.append(pred.to_dict())
            self._emit_prediction_edge(model, pred)
            self._fire_actions(model, pred)
            return InferenceResult(request_id=req.request_id,
                                   status="succeeded", prediction=pred,
                                   trace=trace).to_dict()
        except Exception as exc:
            return InferenceResult(request_id=req.request_id,
                                   status="failed", error=str(exc)).to_dict()

    def context(self) -> Dict[str, Any]:
        """Current device context — the same normalized shape Brain's
        ``/v1/context/state`` stores: latest prediction per model as a
        state entry, plus room + metadata docs.

        One context model backs every surface (local API, Brain
        projection, SDK, MCP) — this method is the node-side view.
        """
        states = [
            {
                "key": "prediction",
                "entity_id": self.device_id,
                "value": p.get("label"),
                "confidence": p.get("confidence"),
                "estimator": p.get("runtime_model_id"),
                "ts": p.get("timestamp"),
            }
            for p in list(self.predictions)[-50:]
        ]
        return {
            "device_id": self.device_id,
            "states": states,
            "observations": list(self.observations)[-50:],
            "observations_pending": len(self._obs_spool),
            "estimates": self.estimators.states(),
            "room": self.room.document(),
            "metadata": self.metadata.document(),
            "online": bool(self._brain_ws is not None),
        }

    def privacy(self) -> Dict[str, Any]:
        """Current exposure/privacy posture — what leaves this node."""
        return {
            "exposed": self._exposure(),
            "local_api_host": str(self.config.get("local_host", "127.0.0.1")),
            "brain_url": self.config.brain_url,
            "paired": bool(self.config.device_token),
        }

    def sync_state(self) -> Dict[str, Any]:
        """Upload/sync posture for captures and minutes."""
        captures = self.captures.list()
        return {
            "captures_total": len(captures),
            "captures_active": len(self.captures._active),
            "minutes_root": str(self._minutes_root()),
            "brain_url": self.config.brain_url,
            "paired": bool(self.config.device_token),
        }

    def run_forever(self) -> None:
        self.start()
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
