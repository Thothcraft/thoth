"""Provisioning state machine — credentials in (BLE GATT / local API),
Wi-Fi out, ``net.link.v1`` observations on every edge.

States: ``disabled → idle → provisioning → online | failed`` (+ ``ap``
when AP fallback is engaged, Phase 4). The manager serializes credential
requests on one worker thread so a BLE write and an API POST can't
interleave nmcli calls.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..observations import Observation
from ..settings import config_dir
from .wifi import WifiStatus, default_wifi_manager, ssid_hash

logger = logging.getLogger(__name__)

STATES = ("disabled", "idle", "provisioning", "online", "failed", "ap")
_PROV_FILE = "provisioning.json"


class ProvisionManager:
    """Owns the bring-up loop. ``wifi`` is injectable — the state machine
    is fully testable with a fake; production uses
    :func:`~thoth.provisioning.wifi.default_wifi_manager`."""

    def __init__(self, config: Any,
                 emit: Callable[[Observation], Any],
                 device_id: str,
                 wifi: Any = None,
                 ble: Any = None,
                 path: Optional[Path] = None):
        self._config = config
        self._emit = emit
        self._device_id = device_id
        self.wifi = wifi if wifi is not None else default_wifi_manager()
        self._ble = ble            # BluetoothSubsystem — GATT peripheral glue
        self._gatt = None          # registered GattApplication
        self._lock = threading.Lock()
        self._requests: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._path = path or (config_dir() / _PROV_FILE)
        self._state = "idle" if self.enabled else "disabled"
        self._last_error: Optional[str] = None
        self._last_attempt: float = 0.0
        self._attempts = 0
        self._ap: Optional[Dict[str, str]] = None  # {ssid, psk} while up
        self._loaded = self._load()

    # -- persistence --------------------------------------------------------------
    def _load(self) -> Dict[str, Any]:
        try:
            return json.loads(self._path.read_text())
        except Exception:
            return {}

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps(self._loaded, indent=2))
            try:
                import os
                os.chmod(self._path, 0o600)
            except OSError:
                pass
        except OSError:
            pass

    # -- config ------------------------------------------------------------------
    def _conf(self, key: str, default: Any) -> Any:
        try:
            return self._config.get(key, default)
        except Exception:
            return default

    @property
    def enabled(self) -> bool:
        return bool(self._conf("provisioning.enabled", True))

    # -- lifecycle ---------------------------------------------------------------
    def start(self) -> "ProvisionManager":
        if not self.enabled:
            self._set_state("disabled")
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop,
                                        name="thoth-prov", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        try:
            self._requests.put_nowait({"op": "stop"})
        except Exception:
            pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def _loop(self) -> None:
        """Worker: derive state from wifi status, then service queued
        credential requests. Polls at ~1Hz; requests preempt the poll."""
        while not self._stop.is_set():
            try:
                self._tick()
            except Exception as exc:
                logger.debug("provisioning tick failed: %s", exc)
            try:
                req = self._requests.get(timeout=1.0)
            except queue.Empty:
                continue
            if req.get("op") == "stop":
                break
            try:
                self._apply(req)
            except Exception as exc:
                self._fail(f"{type(exc).__name__}: {exc}")

    def _tick(self) -> None:
        if not self.enabled:
            self._set_state("disabled")
            return
        status = self._safe_status()
        with self._lock:
            state = self._state
        if status is None or not status.present:
            if state not in ("failed", "disabled"):
                self._set_state("failed", "no wifi interface")
            return
        if status.connected:
            if self._ap is not None:
                try:
                    self.wifi.ap_stop()
                except Exception:
                    pass
                self._ap = None
            if state not in ("online",):
                self._set_state("online")
                self._link_obs(status)
            return
        # not connected
        if state == "online":
            self._set_state("idle")
            self._link_obs(status)
            return
        if state == "disabled":
            self._set_state("idle")
        elif state == "idle":
            # unprovisioned node — engage the BLE commissioning surface
            self._start_gatt()
        elif state == "failed":
            # repeated failures → AP fallback so the setup flow stays
            # reachable over the node's own network
            max_fails = int(self._conf("provisioning.ap_after_failures", 3))
            if self._attempts >= max_fails and self._ap is None:
                self._enter_ap()

    # -- public API ----------------------------------------------------------------
    def provision(self, ssid: str, psk: str = "",
                  hidden: bool = False) -> Dict[str, Any]:
        """Queue a credential application (BLE write or API POST)."""
        if not self.enabled:
            return {"state": self._state, "accepted": False,
                    "error": "provisioning disabled"}
        self._requests.put({"op": "provision", "ssid": ssid,
                            "psk": psk, "hidden": hidden})
        return {"state": "provisioning", "accepted": True}

    def _apply(self, req: Dict[str, Any]) -> None:
        ssid, psk = req.get("ssid"), req.get("psk") or ""
        if not ssid:
            self._fail("ssid required")
            return
        # credentials while the AP is up → tear it down and rejoin STA
        if self._ap is not None:
            try:
                self.wifi.ap_stop()
            except Exception:
                pass
            self._ap = None
        self._set_state("provisioning")
        self._attempts += 1
        self._last_attempt = time.time()
        self._notify_gatt()   # push "provisioning" to any BLE central
        try:
            status = self.wifi.connect(
                ssid, psk, hidden=bool(req.get("hidden")))
        except Exception as exc:
            self._fail(str(exc))
            return
        if status.connected:
            self._set_state("online")
            self._loaded = {"ssid_hash": ssid_hash(ssid),
                            "provisioned_at": time.time()}
            self._save()
            self._link_obs(status)
            self._stop_gatt()   # done — retract the commissioning surface
        else:
            self._fail("wifi did not connect")

    def _fail(self, err: str) -> None:
        self._set_state("failed", err)
        # engage the AP fallback immediately at the threshold instead
        # of waiting for the next tick
        max_fails = int(self._conf("provisioning.ap_after_failures", 3))
        if self._attempts >= max_fails and self._ap is None:
            self._enter_ap()

    def _enter_ap(self) -> None:
        """AP fallback — ``thoth-setup-<id6>`` so first-boot setup stays
        reachable when STA credentials fail."""
        import secrets
        ssid = f"thoth-setup-{self._device_id[-6:]}"
        psk = secrets.token_hex(4)
        try:
            ok = bool(self.wifi.ap_start(ssid, psk))
        except Exception as exc:
            logger.debug("ap_start failed: %s", exc)
            ok = False
        if ok:
            self._ap = {"ssid": ssid, "psk": psk}
            self._set_state("ap")
            logger.info("provisioning: AP fallback up as %s", ssid)

    def _set_state(self, state: str, error: Optional[str] = None) -> None:
        with self._lock:
            changed = state != self._state
            self._state = state
            if error is not None:
                self._last_error = error
            elif state not in ("failed",):
                self._last_error = None
        if changed:
            logger.info("provisioning → %s%s", state,
                        f" ({error})" if error else "")
            self._notify_gatt()

    def _safe_status(self) -> Optional[WifiStatus]:
        try:
            return self.wifi.status()
        except Exception as exc:
            logger.debug("wifi.status failed: %s", exc)
            return None

    def _link_obs(self, status: WifiStatus) -> None:
        """``net.link.v1`` edge — hashed SSID, never the raw string."""
        try:
            self._emit(Observation(
                schema="net.link.v1",
                source_id=f"wifi:{status.iface}",
                value={
                    "iface": status.iface,
                    "state": "connected" if status.connected else "down",
                    "ssid_hash": ssid_hash(status.ssid)
                               if status.ssid else None,
                }).validate())
        except Exception as exc:
            logger.debug("net.link emit failed: %s", exc)

    # -- BLE GATT commissioning glue -------------------------------------------------
    def _start_gatt(self) -> None:
        """Register the commissioning service on the BLE peripheral role —
        only when the node is unprovisioned and the role exists."""
        if self._gatt is not None:
            return
        ble = self._ble
        loop = getattr(ble, "_loop", None) if ble is not None else None
        if ble is None or loop is None or not ble.present:
            return
        try:
            import asyncio
            from ..bluetooth.gatt import GattApplication
            name = str(self._conf("device_name",
                                  self._config.get("device_name", "thoth-node"))
                       or "thoth-node")
            app = GattApplication(
                name=name, device_id=self._device_id,
                on_configure=lambda p: self.provision(
                    p.get("ssid", ""), p.get("psk", ""),
                    bool(p.get("hidden"))),
                on_scan=self.scan_payload,
                get_status=self.status_payload)
            asyncio.run_coroutine_threadsafe(
                app.register(ble._adapter or "hci0"), loop)
            self._gatt = app
        except Exception as exc:
            logger.debug("provisioning GATT not started: %s", exc)

    def _stop_gatt(self) -> None:
        app, self._gatt = self._gatt, None
        if app is None:
            return
        ble = self._ble
        loop = getattr(ble, "_loop", None) if ble is not None else None
        if loop is not None:
            try:
                import asyncio
                asyncio.run_coroutine_threadsafe(app.unregister(), loop)
            except Exception:
                pass

    def _notify_gatt(self) -> None:
        app = self._gatt
        if app is not None:
            ble = self._ble
            loop = getattr(ble, "_loop", None) if ble is not None else None
            if loop is not None:
                try:
                    loop.call_soon_threadsafe(app.notify_status)
                except Exception:
                    pass

    # -- status --------------------------------------------------------------------
    def scan_payload(self) -> List[Dict[str, Any]]:
        try:
            return [{"ssid": n.ssid, "signal": n.signal,
                     "security": n.security} for n in self.wifi.scan()]
        except Exception:
            return []

    def status_payload(self) -> Dict[str, Any]:
        return self.status()

    def status(self) -> Dict[str, Any]:
        """Local-facing status — SSID is fine here (never uplinked raw)."""
        wifi = self._safe_status()
        with self._lock:
            return {
                "enabled": self.enabled,
                "state": self._state,
                "last_error": self._last_error,
                "attempts": self._attempts,
                "provisioned": bool(self._loaded.get("ssid_hash")),
                "ssid_hash": self._loaded.get("ssid_hash"),
                "wifi": {
                    "present": wifi.present if wifi else False,
                    "connected": wifi.connected if wifi else False,
                    "ssid": wifi.ssid if wifi else None,
                    "iface": wifi.iface if wifi else None,
                    "ip": wifi.ip if wifi else None,
                },
                "ble_gatt": self._gatt is not None,
                "ap": dict(self._ap) if self._ap else None,
            }


__all__ = ["ProvisionManager", "STATES"]
