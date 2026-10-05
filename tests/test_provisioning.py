"""Provisioning state machine — wifi backend injected, state edges and
net.link.v1 emissions verified without hardware."""
import re
import threading
import time

import pytest

from thoth.provisioning import ProvisionManager, ssid_hash
from thoth.provisioning.wifi import WifiNetwork, WifiStatus
from thoth.settings import ConfigStore

SSID_RE = re.compile(r"MyHomeNet")


class FakeWifi:
    def __init__(self, present=True, connected=False, ssid=None):
        self._status = WifiStatus(present=present, connected=connected,
                                  ssid=ssid, ip="10.0.0.5" if connected else None)
        self.connect_calls = []
        self.connect_result = None      # WifiStatus | Exception
        self.ap_started = []

    @staticmethod
    def available():
        return True

    def status(self):
        return self._status

    def scan(self):
        return [WifiNetwork(ssid="MyHomeNet", signal=-40, security="WPA2")]

    def connect(self, ssid, psk, *, hidden=False):
        self.connect_calls.append((ssid, psk, hidden))
        if isinstance(self.connect_result, Exception):
            raise self.connect_result
        if self.connect_result is not None:
            self._status = self.connect_result
            return self.connect_result
        self._status = WifiStatus(present=True, connected=True,
                                  ssid=ssid, ip="10.0.0.5")
        return self._status

    def disconnect(self):
        self._status = WifiStatus(present=True, connected=False)

    def ap_start(self, ssid, psk):
        self.ap_started.append((ssid, psk))
        return True

    def ap_stop(self):
        pass


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def _pm(tmp_home, wifi, **conf):
    cfg = ConfigStore()
    for k, v in conf.items():
        cfg.set(k, v)
    emitted = []
    pm = ProvisionManager(cfg, emit=lambda o: emitted.append(o.to_dict()),
                          device_id="node1", wifi=wifi,
                          path=tmp_home / "prov.json")
    return pm, emitted


def test_idle_when_unprovisioned(tmp_home):
    pm, _ = _pm(tmp_home, FakeWifi())
    pm._tick()
    assert pm.status()["state"] == "idle"
    assert pm.status()["wifi"]["present"] is True


def test_online_when_already_connected(tmp_home):
    pm, emitted = _pm(tmp_home, FakeWifi(connected=True, ssid="MyHomeNet"))
    pm._tick()
    assert pm.status()["state"] == "online"
    link = [o for o in emitted if o["schema"] == "net.link.v1"]
    assert link and link[-1]["value"]["state"] == "connected"
    # privacy — hashed ssid on the wire, raw never
    assert link[-1]["value"]["ssid_hash"] == ssid_hash("MyHomeNet")
    assert not SSID_RE.search(str(link[-1]))


def test_link_down_edge_emits_observation(tmp_home):
    wifi = FakeWifi(connected=True, ssid="MyHomeNet")
    pm, emitted = _pm(tmp_home, wifi)
    pm._tick()
    wifi.disconnect()
    pm._tick()
    assert pm.status()["state"] == "idle"
    link = [o for o in emitted if o["schema"] == "net.link.v1"]
    assert link[-1]["value"]["state"] == "down"
    assert link[-1]["value"]["ssid_hash"] is None


def test_provision_success_reaches_online_and_persists(tmp_home):
    wifi = FakeWifi()
    pm, emitted = _pm(tmp_home, wifi)
    pm._apply({"ssid": "MyHomeNet", "psk": "secret123"})
    assert wifi.connect_calls == [("MyHomeNet", "secret123", False)]
    assert pm.status()["state"] == "online"
    assert pm.status()["provisioned"] is True
    assert pm.status()["ssid_hash"] == ssid_hash("MyHomeNet")
    # persisted + emitted, never raw
    assert "MyHomeNet" not in (tmp_home / "prov.json").read_text()
    assert all("MyHomeNet" not in str(o["value"]) for o in emitted)


def test_provision_failure_state_and_error(tmp_home):
    wifi = FakeWifi()
    wifi.connect_result = RuntimeError("psk rejected")
    pm, _ = _pm(tmp_home, wifi)
    pm._apply({"ssid": "MyHomeNet", "psk": "bad"})
    assert pm.status()["state"] == "failed"
    assert "psk rejected" in pm.status()["last_error"]


def test_provision_connect_but_not_connected(tmp_home):
    wifi = FakeWifi()
    wifi.connect_result = WifiStatus(present=True, connected=False)
    pm, _ = _pm(tmp_home, wifi)
    pm._apply({"ssid": "MyHomeNet", "psk": "x"})
    assert pm.status()["state"] == "failed"


def test_no_wifi_interface_fails_cleanly(tmp_home):
    pm, _ = _pm(tmp_home, FakeWifi(present=False))
    pm._tick()
    st = pm.status()
    assert st["state"] == "failed"
    assert "wifi interface" in st["last_error"]
    assert st["wifi"]["present"] is False


def test_disabled_via_config(tmp_home):
    pm, _ = _pm(tmp_home, FakeWifi(), **{"provisioning.enabled": False})
    pm.start()
    pm.stop()
    assert pm.status()["state"] == "disabled"


def test_provision_queue_and_worker_loop(tmp_home):
    wifi = FakeWifi()
    pm, _ = _pm(tmp_home, wifi)
    pm.start()
    try:
        # let the first tick run: unprovisioned → idle
        for _ in range(100):
            if pm.status()["state"] == "idle":
                break
            time.sleep(0.02)
        out = pm.provision("MyHomeNet", "secret123")
        assert out["accepted"] is True
        deadline = time.time() + 5.0
        while time.time() < deadline and \
                pm.status()["state"] != "online":
            time.sleep(0.05)
        assert pm.status()["state"] == "online"
    finally:
        pm.stop()


def test_ap_fallback_after_failures(tmp_home):
    wifi = FakeWifi()
    wifi.connect_result = WifiStatus(present=True, connected=False)
    pm, _ = _pm(tmp_home, wifi,
                **{"provisioning.ap_after_failures": 2})
    pm._apply({"ssid": "MyHomeNet", "psk": "x"})
    assert pm.status()["state"] == "failed"
    pm._apply({"ssid": "MyHomeNet", "psk": "x"})
    st = pm.status()
    assert st["state"] == "ap"
    assert st["ap"]["ssid"].startswith("thoth-setup-")
    assert wifi.ap_started and wifi.ap_started[0][0] == st["ap"]["ssid"]


def test_ap_torn_down_on_credentials(tmp_home):
    wifi = FakeWifi()
    wifi.connect_result = WifiStatus(present=True, connected=False)
    pm, _ = _pm(tmp_home, wifi,
                **{"provisioning.ap_after_failures": 1})
    pm._apply({"ssid": "bad", "psk": "x"})
    assert pm.status()["state"] == "ap"
    # now good creds → STA rejoin wins over AP
    wifi.connect_result = None
    pm._apply({"ssid": "MyHomeNet", "psk": "good"})
    assert pm.status()["state"] == "online"
    assert pm.status()["ap"] is None


def test_ap_torn_down_when_link_comes_up(tmp_home):
    wifi = FakeWifi()
    wifi.connect_result = WifiStatus(present=True, connected=False)
    pm, _ = _pm(tmp_home, wifi,
                **{"provisioning.ap_after_failures": 1})
    pm._apply({"ssid": "bad", "psk": "x"})
    assert pm.status()["state"] == "ap"
    wifi._status = WifiStatus(present=True, connected=True,
                              ssid="MyHomeNet")
    pm._tick()
    assert pm.status()["state"] == "online"
    assert pm.status()["ap"] is None


def test_scan_payload(tmp_home):
    pm, _ = _pm(tmp_home, FakeWifi())
    nets = pm.scan_payload()
    assert nets == [{"ssid": "MyHomeNet", "signal": -40,
                     "security": "WPA2"}]
