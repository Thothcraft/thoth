"""BLE subsystem — observer emissions, privacy, known map, central links."""
import re
import threading
import time
from pathlib import Path

import pytest

from thoth.bluetooth import (Advertisement, BluetoothSubsystem,
                             KnownDeviceStore)
from thoth.settings import ConfigStore

MAC_RE = re.compile(r"\b([0-9a-fA-F]{2}:){5}[0-9a-fA-F]{2}\b")


class FakeClient:
    def __init__(self, address):
        self.address = address
        self.is_connected = False
        self.notified = {}

    async def connect(self):
        self.is_connected = True

    async def disconnect(self):
        self.is_connected = False

    async def start_notify(self, uuid, cb):
        self.notified[uuid] = cb


class FakeBackend:
    """Drives the subsystem like BlueZ would — but synchronously, in-test."""

    def __init__(self):
        self.on_adv = None
        self.scan_calls = 0
        self.scan_stops = 0
        self.connects = []
        self.clients = []

    @staticmethod
    def available():
        return True

    @staticmethod
    def peripheral_supported():
        return False

    @staticmethod
    def adapter_name(default="hci0"):
        return default

    async def scan_start(self, on_adv):
        self.on_adv = on_adv
        self.scan_calls += 1

    async def scan_stop(self):
        self.scan_stops += 1

    async def connect(self, address):
        self.connects.append(address)
        client = FakeClient(address)
        await client.connect()
        self.clients.append(client)
        from thoth.bluetooth.backend import CentralHandle
        return CentralHandle(client=client, address=address)

    async def subscribe(self, handle, char_uuid, on_data):
        await handle.client.start_notify(char_uuid, on_data)

    async def disconnect(self, handle):
        if handle is not None:
            await handle.client.disconnect()

    # test helpers -------------------------------------------------------
    def feed(self, **kw):
        assert self.on_adv is not None
        self.on_adv(Advertisement(
            address=kw.pop("address"), rssi=kw.pop("rssi", -60),
            ts=time.time(), **kw))


def _sub(tmp_home, backend, **conf):
    cfg = ConfigStore()
    for k, v in conf.items():
        cfg.set(k, v)
    seen = []
    sub = BluetoothSubsystem(cfg, emit=lambda o: seen.append(o.to_dict()),
                             device_id="node1", backend=backend,
                             known=KnownDeviceStore(tmp_home / "ble.json"))
    return sub, seen


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def adv(addr="AA:BB:CC:DD:EE:01", **kw):
    return Advertisement(address=addr, rssi=kw.pop("rssi", -55),
                         ts=time.time(), **kw)


def test_rssi_observation_emitted_on_first_sighting(tmp_home):
    sub, seen = _sub(tmp_home, FakeBackend())
    sub._on_adv(adv(rssi=-58, tx_power=-59, addr_type="random"))
    assert len(seen) == 1
    obs = seen[0]
    assert obs["schema"] == "ble.rssi.v1"
    assert obs["source_id"] == "ble:hci0"
    assert obs["value"]["rssi_dbm"] == -58
    assert obs["value"]["tx_power_dbm"] == -59
    assert obs["value"]["addr_type"] == "random"
    assert obs["subject"].startswith("device:ble:")


def test_no_raw_mac_in_emitted_subjects(tmp_home):
    sub, seen = _sub(tmp_home, FakeBackend())
    for i in range(5):
        sub._on_adv(adv(addr=f"AA:BB:CC:DD:EE:{i:02X}", rssi=-50 - i))
    assert len(seen) == 5
    for obs in seen:
        assert not MAC_RE.search(obs["subject"])
        assert not MAC_RE.search(str(obs["value"]))


def test_anonymous_subject_is_stable_and_salt_scoped(tmp_home):
    sub, _ = _sub(tmp_home, FakeBackend())
    h1 = sub._devices  # populated via _on_adv
    sub._on_adv(adv(addr="11:22:33:44:55:66"))
    s1 = list(h1.values())[0]["subject"]
    # second sighting → same subject
    sub._on_adv(adv(addr="11:22:33:44:55:66", rssi=-70))
    assert list(h1.values())[0]["subject"] == s1
    # different salt → different subject for the same MAC
    sub._config.set("ble.salt", "0" * 32)
    sub._on_adv(adv(addr="99:88:77:66:55:44", rssi=-70))
    s2 = [d for d in h1.values() if d["subject"] != s1][0]["subject"]
    assert s2.startswith("device:ble:") and s2 != s1


def test_enrolled_device_subject_is_device_uuid(tmp_home):
    backend = FakeBackend()
    sub, seen = _sub(tmp_home, backend)
    ah = sub.addr_hash("AA:BB:CC:DD:EE:09", sub._salt())
    rec = sub.known.enroll(ah, "AA:BB:CC:DD:EE:09", kind="watch")
    sub._on_adv(adv(addr="AA:BB:CC:DD:EE:09"))
    assert seen[0]["subject"] == rec["device_id"]
    assert rec["device_id"].startswith("device:")
    sub._on_adv(adv(addr="AA:BB:CC:DD:EE:09", rssi=-80))
    assert sub._devices[ah]["enrolled"] is True


def test_rssi_emission_throttled_but_change_forces_emit(tmp_home):
    sub, seen = _sub(tmp_home, FakeBackend(),
                     **{"ble.obs_min_interval_s": 30.0})
    sub._on_adv(adv(rssi=-55))
    sub._on_adv(adv(rssi=-56))   # within interval, small change — throttled
    sub._on_adv(adv(rssi=-57))
    assert len(seen) == 1
    sub._on_adv(adv(rssi=-80))   # ≥4 dB jump — emits even inside interval
    assert len(seen) == 2


def test_presence_edges_emitted_for_enrolled_only(tmp_home):
    backend = FakeBackend()
    sub, seen = _sub(tmp_home, backend, **{"ble.gone_after_s": 0.05,
                                         "ble.sweep_s": 0.05})
    ah = sub.addr_hash("AA:BB:CC:DD:EE:10", sub._salt())
    rec = sub.known.enroll(ah, "AA:BB:CC:DD:EE:10")
    sub.start()
    try:
        for _ in range(100):
            if backend.on_adv is not None:
                break
            time.sleep(0.02)
        backend.feed(address="AA:BB:CC:DD:EE:10", rssi=-55)
        backend.feed(address="AA:BB:CC:DD:EE:99", rssi=-55)
        # let the sweep mark both gone (0.05s window)
        deadline = time.time() + 3.0
        while time.time() < deadline:
            if any(o["schema"] == "ble.presence.v1" and
                   o["value"]["seen"] is False for o in seen):
                break
            time.sleep(0.05)
        presence = [o for o in seen if o["schema"] == "ble.presence.v1"]
        assert any(o["value"]["seen"] is True
                   and o["subject"] == rec["device_id"] for o in presence)
        assert any(o["value"]["seen"] is False
                   and o["subject"] == rec["device_id"] for o in presence)
        # unknown device: RSSI only, never a presence observation
        assert all(o["subject"].startswith("device:ble:") is False
                   or o["schema"] == "ble.rssi.v1" for o in seen)
    finally:
        sub.stop()


def test_capability_shape(tmp_home):
    sub, _ = _sub(tmp_home, FakeBackend())
    cap = sub.capability()
    assert cap["present"] is True
    assert cap["roles"]["observer"] is True
    assert cap["roles"]["central"] is True
    assert cap["roles"]["peripheral"] is False
    assert cap["source_id"] == "ble:hci0"
    assert cap["scanning"] is False


def test_central_connect_subscribes_and_disconnects(tmp_home):
    backend = FakeBackend()
    sub, _ = _sub(tmp_home, backend)
    sub.start()
    try:
        # wait for the asyncio loop
        for _ in range(100):
            if sub._loop is not None:
                break
            time.sleep(0.02)
        got = []
        ok = sub.central_connect(
            "AA:BB:CC:DD:EE:20",
            subscriptions={"uuid-1": lambda b: got.append(b)})
        assert ok is True
        for _ in range(100):
            if backend.clients and backend.clients[-1].notified:
                break
            time.sleep(0.02)
        assert backend.connects == ["AA:BB:CC:DD:EE:20"]
        assert "uuid-1" in backend.clients[-1].notified
        assert sub.central_disconnect("AA:BB:CC:DD:EE:20") is True
    finally:
        sub.stop()


def test_disabled_and_unavailable_backends(tmp_home):
    sub, _ = _sub(tmp_home, FakeBackend(), **{"ble.enabled": False})
    assert sub.start() is sub
    assert sub._thread is None
    from thoth.bluetooth import NullBackend
    sub2, _ = _sub(tmp_home, NullBackend())
    assert sub2.start() is sub2
    assert sub2.capability()["present"] is False
