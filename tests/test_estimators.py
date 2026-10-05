"""EstimatorHub — occupancy/motion/zone state derivation + versioning."""
import time

from thoth.estimators import EstimatorHub


def _obs(schema, subject=None, value=None, ts=None, oid="o1"):
    return {"schema": schema, "subject": subject, "value": value or {},
            "timestamp": ts if ts is not None else time.time(),
            "observation_id": oid}


def _rssi(subject, rssi, ts=None, oid="o1"):
    return _obs("ble.rssi.v1", subject,
                {"rssi_dbm": rssi, "tx_power_dbm": -59,
                 "addr_type": "random"}, ts, oid)


def test_occupancy_occupied_on_ble_evidence(tmp_path):
    hub = EstimatorHub("node1", fingerprints_path=tmp_path / "fp.json")
    changed = hub.consume(_rssi("device:x", -55))
    occ = [c for c in changed if c["key"] == "occupancy.v1"]
    assert occ and occ[0]["value"]["occupied"] is True
    assert occ[0]["value"]["distinct_subjects"] == 1
    assert occ[0]["evidence_ids"] == ["o1"]
    assert occ[0]["version"] == 1


def test_occupancy_decays_when_silent(tmp_path):
    hub = EstimatorHub("node1", fingerprints_path=tmp_path / "fp.json")
    t0 = time.time() - 300.0
    hub.consume(_rssi("device:x", -55, ts=t0))
    # device aged out of the 120s window — tick produces the gone edge
    changed = hub.tick()
    occ = [c for c in changed if c["key"] == "occupancy.v1"]
    assert occ and occ[0]["value"]["occupied"] is False


def test_no_version_churn_on_heartbeats(tmp_path):
    hub = EstimatorHub("node1", fingerprints_path=tmp_path / "fp.json")
    hub.consume(_rssi("device:x", -55))
    hub.consume(_rssi("device:x", -60, oid="o2"))
    st = hub.state_for("occupancy.v1")
    assert st["version"] == 1          # value unchanged → no new version
    assert "o2" in st["evidence_ids"]  # evidence refreshed though


def test_motion_stationary_vs_moving(tmp_path):
    hub = EstimatorHub("node1", fingerprints_path=tmp_path / "fp.json")
    flat = {"rate_hz": 50, "samples": 16,
            "axes": {"x": [0.0] * 16, "y": [9.8] * 16, "z": [0.0] * 16}}
    hub.consume(_obs("imu.window.v1", "device:w", flat, oid="m1"))
    st = hub.state_for("activity.motion.v1")
    assert st["value"]["motion"] == "stationary"
    jitter = {"rate_hz": 50, "samples": 16,
              "axes": {"x": [i * 0.1 for i in range(16)],
                       "y": [9.8 + ((-1) ** i) * 0.3 for i in range(16)],
                       "z": [0.0] * 16}}
    changed = hub.consume(_obs("imu.window.v1", "device:w", jitter,
                               oid="m2"))
    st = hub.state_for("activity.motion.v1")
    assert st["value"]["motion"] == "moving"
    assert any(c["key"] == "activity.motion.v1" for c in changed)


def test_zone_fingerprint_calibration_and_match(tmp_path):
    fp_path = tmp_path / "fp.json"
    hub = EstimatorHub("node1", fingerprints_path=fp_path)
    # calibrate "kitchen" while anchors a,b are near
    hub.consume(_rssi("device:a", -40, oid="a1"))
    hub.consume(_rssi("device:b", -50, oid="b1"))
    assert hub.calibrate_zone("kitchen") is not None
    # move: a weaker, b stronger → nearer the "hall" fingerprint
    hub2 = EstimatorHub("node1", fingerprints_path=fp_path)  # persisted
    hub2._fingerprints["hall"] = {"device:a": -75, "device:b": -45}
    hub2.consume(_rssi("device:a", -70, oid="a2"))
    hub2.consume(_rssi("device:b", -44, oid="b2"))
    st = hub2.state_for("location.zone.v1")
    assert st is not None and st["value"]["zone"] == "hall"


def test_states_only_report_after_first_value(tmp_path):
    hub = EstimatorHub("node1", fingerprints_path=tmp_path / "fp.json")
    assert hub.states() == []
    hub.consume(_rssi("device:x", -55))
    keys = {s["key"] for s in hub.states()}
    assert "occupancy.v1" in keys
