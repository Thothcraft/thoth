"""Bluetooth subsystem — the node's BLE observer / central / peripheral
roles on BlueZ, feeding the observation/v1 uplink.

Roles (CONTRACT §0): *observer* duty-cycles advertisement scanning and
emits ``ble.rssi.v1`` / ``ble.presence.v1``; *central* keeps managed
connections to enrolled peripherals (provisioning, wearable IMU) with
backoff; *peripheral* hosts the commissioning GATT service (Phase 3).

Privacy (contract §1): raw MACs never leave the node. Unknown
advertisers get a node-scoped anonymous subject
(``device:ble:<hmac12>``); enrolled devices get their stable
``device:<uuid>`` from :class:`KnownDeviceStore`.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import secrets
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..observations import Observation
from ..settings import config_dir
from .backend import (Advertisement, BackendUnavailable, NullBackend,
                      default_backend)
from .known import KnownDeviceStore

logger = logging.getLogger(__name__)


def _norm_mac(address: str) -> str:
    return "".join(c for c in str(address).lower() if c in "0123456789abcdef")


class _CentralSession:
    """State for one managed central-role connection."""

    __slots__ = ("address", "addr_hash", "subscriptions", "on_state",
                 "handle", "task", "closed")

    def __init__(self, address: str, addr_hash: str,
                 subscriptions: Dict[str, Callable[[bytes], None]],
                 on_state: Optional[Callable[[str, bool], None]]):
        self.address = address
        self.addr_hash = addr_hash
        self.subscriptions = subscriptions
        self.on_state = on_state
        self.handle = None
        self.task: Optional[asyncio.Task] = None
        self.closed = False


class BluetoothSubsystem:
    """Owns the controller, the scan duty cycle, and BLE identity.

    ``config`` is a :class:`~thoth.settings.ConfigStore`-like mapping;
    ``emit`` is ``ThothDaemon.emit_observation`` (thread-safe). ``backend``
    is injectable — tests drive a fake; production uses
    :func:`~thoth.bluetooth.backend.default_backend`.
    """

    def __init__(self, config: Any, emit: Callable[[Observation], Any],
                 device_id: str, *, backend: Any = None,
                 adapter: Optional[str] = None,
                 known: Optional[KnownDeviceStore] = None):
        self._config = config
        self._emit = emit
        self._device_id = device_id
        self._backend = backend if backend is not None \
            else default_backend(adapter)
        self._adapter = adapter
        self.known = known or KnownDeviceStore(config_dir() / "ble_devices.json")

        self._lock = threading.Lock()
        self._devices: Dict[str, Dict[str, Any]] = {}   # addr-hash → state
        self._centrals: Dict[str, _CentralSession] = {}  # addr-hash → session
        self._scanning = False
        self._scan_error: Optional[str] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = False

    # -- config -----------------------------------------------------------------
    def _conf(self, key: str, default: Any) -> Any:
        try:
            return self._config.get(key, default)
        except Exception:
            return default

    @property
    def enabled(self) -> bool:
        return bool(self._conf("ble.enabled", True))

    def _salt(self) -> bytes:
        """Per-node HMAC key for anonymous subject ids — lazily persisted."""
        salt = self._conf("ble.salt", None)
        if not salt:
            salt = secrets.token_hex(16)
            try:
                self._config.set("ble.salt", salt)
            except Exception:
                pass
        return salt.encode() if isinstance(salt, str) else bytes(salt)

    @staticmethod
    def addr_hash(address: str, salt: bytes) -> str:
        """Node-scoped anonymous id — unlinkable off-node, stable on-node."""
        return hmac.new(salt, _norm_mac(address).encode(),
                        hashlib.sha256).hexdigest()[:16]

    # -- lifecycle --------------------------------------------------------------
    def start(self) -> "BluetoothSubsystem":
        if not self.enabled or not self._backend.available():
            logger.info("bluetooth: %s",
                        "disabled in config" if not self.enabled
                        else "no BlueZ backend — observer/central inactive")
            return self
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True,
                                        name="thoth-ble")
        self._thread.start()
        return self

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        try:
            loop.create_task(self._observer_task())
            loop.create_task(self._sweep_task())
            loop.run_forever()
        except Exception as exc:
            logger.debug("ble loop exited: %s", exc)
        finally:
            try:
                loop.close()
            except Exception:
                pass
            self._loop = None

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            try:
                loop.call_soon_threadsafe(
                    lambda: [t.cancel() for t in asyncio.all_tasks(loop)])
                loop.call_soon_threadsafe(loop.stop)
            except Exception:
                pass
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._started = False
        self._scanning = False

    # -- observer role ------------------------------------------------------------
    async def _observer_task(self) -> None:
        """Duty-cycled scan: on-window / off-window, or continuous when
        ``ble.duty_on_s`` is 0. Survives transient adapter errors."""
        duty_on = float(self._conf("ble.duty_on_s", 0.0))
        duty_off = float(self._conf("ble.duty_off_s", 60.0))
        while not self._stop.is_set():
            try:
                await self._backend.scan_start(self._on_adv)
                with self._lock:
                    self._scanning = True
                    self._scan_error = None
                logger.info("bluetooth: observer scanning")
            except Exception as exc:
                with self._lock:
                    self._scanning = False
                    self._scan_error = str(exc)
                logger.debug("bluetooth: scan_start failed: %s", exc)
                await asyncio.sleep(5.0)
                continue
            # on-window
            if duty_on > 0:
                await asyncio.sleep(duty_on)
                if self._stop.is_set():
                    break
                try:
                    await self._backend.scan_stop()
                finally:
                    with self._lock:
                        self._scanning = False
                await asyncio.sleep(max(duty_off, 1.0))
            else:
                # continuous — park until stop; scan errors surface via
                # the backend's own callbacks/exceptions on next ops
                while not self._stop.is_set():
                    await asyncio.sleep(1.0)
        try:
            await self._backend.scan_stop()
        except Exception:
            pass
        with self._lock:
            self._scanning = False

    def _on_adv(self, adv: Advertisement) -> None:
        """Backend → normalized observation (runs on the BLE loop)."""
        ah = self.addr_hash(adv.address, self._salt())
        now = time.time()
        presence: Optional[Observation] = None
        with self._lock:
            dev = self._devices.get(ah)
            if dev is None:
                rec = self.known.get(ah)
                dev = self._devices[ah] = {
                    "addr_hash": ah,
                    "enrolled": rec is not None,
                    "subject": rec["device_id"] if rec
                               else f"device:ble:{ah[:12]}",
                    "kind": (rec or {}).get("kind", "device"),
                    "name": adv.name,
                    "rssi_ema": float(adv.rssi),
                    "last_seen": now,
                    "last_emit": 0.0,
                    "last_presence_hb": 0.0,
                    "seq": 0,
                    "seen": True,
                    "present_reported": False,
                }
                # first sighting — always emit immediately
                dev["last_emit"] = -1.0
            else:
                dev["rssi_ema"] += 0.35 * (adv.rssi - dev["rssi_ema"])
                dev["last_seen"] = now
                if adv.name and not dev.get("name"):
                    dev["name"] = adv.name
                if not dev["seen"]:
                    dev["seen"] = True
                    dev["present_reported"] = False
            # sighting = the presence edge — report enrolled devices
            # here so a slow sweep can never skip their first "seen"
            if dev["enrolled"] and not dev["present_reported"]:
                dev["present_reported"] = True
                dev["last_presence_hb"] = now
                dev["seq"] += 1
                presence = self._presence(
                    dev, True,
                    float(self._conf("ble.gone_after_s", 45.0)), now)
            rssi = int(round(dev["rssi_ema"]))
            should_emit = (
                dev["last_emit"] < 0.0
                or abs(rssi - int(round(dev.get("emit_rssi", rssi)))) >= 4
                or now - dev["last_emit"] >=
                float(self._conf("ble.obs_min_interval_s", 10.0)))
            if should_emit:
                dev["last_emit"] = now
                dev["emit_rssi"] = rssi
                dev["seq"] += 1
                seq = dev["seq"]
                subject = dev["subject"]
            else:
                seq = subject = None
        if seq is not None:
            self._emit_observation(Observation(
                schema="ble.rssi.v1",
                source_id=self.source_id,
                subject=subject,
                value={"rssi_dbm": rssi,
                       "tx_power_dbm": adv.tx_power,
                       "addr_type": adv.addr_type,
                       # Account-scoped correlation key: the app emits
                       # ble:<MAC> for the same device — lets Brain's
                       # proximity map merge edges seen by node + phone
                       # into one node instead of two anonymous bubbles.
                       **({"mac": str(adv.address).upper()}
                          if adv.addr_type == "public" else {}),
                       "name": dev.get("name")},
                units={"rssi_dbm": "dBm", "tx_power_dbm": "dBm"},
                sequence=seq,
                timestamp=now,
            ))
        if presence is not None:
            self._emit_observation(presence)
        # enrolled last_seen refresh — keeps the map fresh even when
        # the RSSI emission was throttled
        rec_dev = self._devices.get(ah)
        if rec_dev and rec_dev.get("enrolled"):
            self.known.touch(ah, now)

    async def _sweep_task(self) -> None:
        """Presence edges: devices silent for ``gone_after_s`` flip to
        seen:false (enrolled only — anonymous churn stays off the wire)."""
        sweep_s = float(self._conf("ble.sweep_s", 10.0))
        gone_after = float(self._conf("ble.gone_after_s", 45.0))
        hb_s = float(self._conf("ble.presence_hb_s", 60.0))
        while not self._stop.is_set():
            await asyncio.sleep(max(sweep_s, 1.0))
            now = time.time()
            emissions: List[Observation] = []
            with self._lock:
                for dev in self._devices.values():
                    if dev["seen"] and now - dev["last_seen"] > gone_after:
                        dev["seen"] = False
                        if dev["enrolled"] and dev["present_reported"]:
                            dev["present_reported"] = False
                            dev["seq"] += 1
                            emissions.append(self._presence(dev, False,
                                                            gone_after, now))
                    elif (dev["enrolled"] and dev["seen"] and
                          dev["present_reported"] and
                          now - dev["last_presence_hb"] >= hb_s):
                        dev["last_presence_hb"] = now
                        dev["seq"] += 1
                        emissions.append(self._presence(dev, True,
                                                        gone_after, now))
            for obs in emissions:
                self._emit_observation(obs)

    def _presence(self, dev: Dict[str, Any], seen: bool, window_s: float,
                  now: float) -> Observation:
        return Observation(
            schema="ble.presence.v1",
            source_id=self.source_id,
            subject=dev["subject"],
            value={"seen": seen, "window_s": window_s},
            sequence=dev["seq"],
            timestamp=now)

    # -- central role ---------------------------------------------------------------
    def central_connect(self, address: str,
                        subscriptions: Optional[Dict[str, Callable]] = None,
                        on_state: Optional[Callable[[str, bool], None]] = None
                        ) -> bool:
        """Open a managed central-role link: connect now, reconnect with
        backoff until ``central_disconnect``. Returns False when no loop."""
        loop = self._loop
        if loop is None:
            return False
        ah = self.addr_hash(address, self._salt())
        session = _CentralSession(address, ah, dict(subscriptions or {}),
                                  on_state)
        with self._lock:
            old = self._centrals.pop(ah, None)
            self._centrals[ah] = session
        if old is not None:
            old.closed = True
            if old.task is not None:
                loop.call_soon_threadsafe(old.task.cancel)
        loop.call_soon_threadsafe(self._spawn_central, session)
        return True

    def _spawn_central(self, session: _CentralSession) -> None:
        session.task = asyncio.create_task(self._central_task(session))

    def central_disconnect(self, address: str) -> bool:
        ah = self.addr_hash(address, self._salt())
        with self._lock:
            session = self._centrals.pop(ah, None)
        if session is None:
            return False
        session.closed = True
        loop = self._loop
        if loop is not None and session.task is not None:
            loop.call_soon_threadsafe(session.task.cancel)
        return True

    def central_write(self, address: str, char_uuid: str, data: bytes
                      ) -> bool:
        """Write ``data`` to ``char_uuid`` on a live central link.
        Returns False when no such session/loop — callers retry later."""
        loop = self._loop
        if loop is None:
            return False
        ah = self.addr_hash(address, self._salt())
        with self._lock:
            session = self._centrals.get(ah)
        handle = session.handle if session else None
        if handle is None or not handle.connected:
            return False

        async def _w() -> None:
            try:
                await self._backend.write_char(handle, char_uuid, data)
            except Exception as exc:
                logger.debug("ble central write %s failed: %s",
                             char_uuid, exc)
        loop.call_soon_threadsafe(
            lambda: asyncio.ensure_future(_w()))
        return True

    async def _central_task(self, session: _CentralSession) -> None:
        backoff = 1.0
        while not session.closed and not self._stop.is_set():
            try:
                handle = await self._backend.connect(session.address)
                session.handle = handle
                for char_uuid, cb in session.subscriptions.items():
                    try:
                        await self._backend.subscribe(handle, char_uuid,
                                                      cb)
                    except Exception as exc:
                        # Missing char (stock firmware lacks the fork
                        # 00030003/0004 chars) — skip it, don't drop the
                        # whole link into a reconnect loop.
                        logger.debug("ble subscribe %s skipped: %s",
                                     char_uuid, exc)
                backoff = 1.0
                self._state_cb(session, True)
                while (handle.connected and not session.closed
                       and not self._stop.is_set()):
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.debug("ble central %s failed: %s",
                             session.addr_hash[:8], exc)
            finally:
                try:
                    await self._backend.disconnect(session.handle)
                except Exception:
                    pass
                session.handle = None
                self._state_cb(session, False)
            if not session.closed and not self._stop.is_set():
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, 30.0)

    def _state_cb(self, session: _CentralSession, up: bool) -> None:
        cb = session.on_state
        if cb is None:
            return
        try:
            cb(session.address, up)
        except Exception:
            pass

    # -- introspection ------------------------------------------------------------
    @property
    def source_id(self) -> str:
        return f"ble:{self._adapter or self._backend.adapter_name()}"

    @property
    def present(self) -> bool:
        try:
            return bool(self._backend.available())
        except Exception:
            return False

    def seen_now(self) -> List[Dict[str, Any]]:
        """Devices currently inside the presence window."""
        now = time.time()
        gone_after = float(self._conf("ble.gone_after_s", 45.0))
        with self._lock:
            return [{
                "subject": d["subject"], "enrolled": d["enrolled"],
                "rssi_dbm": int(round(d["rssi_ema"])),
                "last_seen": d["last_seen"], "kind": d["kind"],
                "name": d.get("name"),
            } for d in self._devices.values()
                if d["seen"] and now - d["last_seen"] <= gone_after]

    def capability(self) -> Dict[str, Any]:
        """``/api/v1/capabilities`` bluetooth block."""
        with self._lock:
            known_n = len(self.known.list())
            seen_n = sum(1 for d in self._devices.values() if d["seen"])
            central_n = len(self._centrals)
        try:
            peripheral = bool(self._backend.peripheral_supported())
        except Exception:
            peripheral = False
        return {
            "present": self.present,
            "enabled": self.enabled,
            "adapter": self._adapter
                       or (self._backend.adapter_name()
                           if self.present else None),
            "source_id": self.source_id,
            "roles": {"observer": self.present, "central": self.present,
                      "peripheral": peripheral},
            "scanning": self._scanning,
            "scan_error": self._scan_error,
            "known_devices": known_n,
            "seen_now": seen_n,
            "central_links": central_n,
            "duty": {"on_s": self._conf("ble.duty_on_s", 0.0),
                     "off_s": self._conf("ble.duty_off_s", 60.0)},
        }

    def _emit_observation(self, obs: Observation) -> None:
        try:
            obs.validate()
            self._emit(obs)
        except Exception as exc:
            logger.debug("ble observation dropped: %s", exc)


__all__ = ["BluetoothSubsystem"]
