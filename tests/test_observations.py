"""observation/v1 — envelope validation, spool durability, daemon flush."""
import json

import pytest

from thoth.observations import (Observation, ObservationError,
                                ObservationSpool, build_batch)


@pytest.fixture
def tmp_home(tmp_path, monkeypatch):
    monkeypatch.setenv("THOTH_HOME", str(tmp_path))
    return tmp_path


def _obs(**kw):
    args = dict(schema="ble.rssi.v1", source_id="ble:hci0",
                value={"rssi_dbm": -48}, subject="device:watch-1",
                observer="device:node-1", sequence=7)
    args.update(kw)
    return Observation(**args)


# -- envelope -----------------------------------------------------------------

def test_envelope_wire_shape():
    o = _obs().validate()
    d = o.to_dict()
    assert d["schema"] == "ble.rssi.v1"
    assert d["observation_id"]
    assert d["source_id"] == "ble:hci0"
    assert d["subject"] == "device:watch-1"
    assert d["provenance"]["observer"] == "device:node-1"
    assert d["sequence"] == 7
    assert d["timestamp"] > 0
    json.dumps(d)  # wire-safe


def test_envelope_roundtrip():
    d = _obs(batch_id="b1").to_dict()
    o2 = Observation.from_dict(d).validate()
    assert o2.observation_id == d["observation_id"]
    assert o2.batch_id == "b1"
    assert o2.subject == "device:watch-1"


@pytest.mark.parametrize("bad", [
    dict(schema="RSSI"),                      # no version suffix
    dict(schema="ble.rssi"),                  # unversioned
    dict(source_id=""),                       # missing source
    dict(value=None),                         # missing value
    dict(confidence=1.5),                     # out of range
    dict(timestamp=-1),                       # bad ts
])
def test_envelope_rejects_invalid(bad):
    with pytest.raises(ObservationError):
        _obs(**bad).validate()


def test_envelope_rejects_non_json_value():
    with pytest.raises(ObservationError):
        _obs(value=object()).validate()


def test_build_batch_stamps_batch_id():
    items = [_obs().to_dict() for _ in range(3)]
    frame = build_batch(items, batch_id="batch-1")
    assert frame["type"] == "observation_batch"
    assert frame["id"] == "batch-1"
    assert all(i["batch_id"] == "batch-1" for i in frame["items"])


# -- spool --------------------------------------------------------------------

def test_spool_ack_and_durability(tmp_path):
    spool = ObservationSpool(tmp_path)
    oid = spool.append(_obs().to_dict())
    spool.append(_obs(sequence=8).to_dict())
    assert len(spool) == 2
    # Restart durability — a new spool over the same dir sees pending items.
    spool2 = ObservationSpool(tmp_path)
    assert len(spool2) == 2
    assert spool2.ack([oid]) == 1
    assert len(spool2) == 1
    assert spool2.ack(["nonexistent"]) == 0


def test_spool_overflow_drops_oldest_and_marks(tmp_path):
    spool = ObservationSpool(tmp_path, mem_max=3)
    for i in range(5):
        spool.append(_obs(sequence=i).to_dict())
    pending = spool.pending()
    # One dropped-marker observation + the 3 newest items.
    assert pending[0]["schema"] == "observation.dropped.v1"
    assert pending[0]["value"]["dropped"] == 2
    assert [p["sequence"] for p in pending[1:]] == [2, 3, 4]
    # Marker is emitted once — next pending() has no marker.
    assert spool.pending()[0]["schema"] == "ble.rssi.v1"


def test_spool_requeue_front(tmp_path):
    spool = ObservationSpool(tmp_path, mem_max=4)
    spool.append(_obs(sequence=1).to_dict())
    failed = [_obs(sequence=9).to_dict(), _obs(sequence=10).to_dict()]
    spool.requeue_front(failed)
    seqs = [p.get("sequence") for p in spool.pending()
            if p["schema"] != "observation.dropped.v1"]
    assert seqs[:2] == [9, 10]


# -- daemon flush --------------------------------------------------------------

class _FakeWS:
    def __init__(self):
        self.connected = True
        self.frames = []

    def send_frame(self, frame):
        self.frames.append(frame)
        return True


def _daemon(tmp_home):
    pytest.importorskip("whispy")
    from thoth.daemon import ThothDaemon
    from thoth.settings import ConfigStore
    return ThothDaemon(config=ConfigStore(), window_seconds=0.5,
                       tick_hz=4.0)


def test_daemon_emit_and_flush(tmp_home):
    daemon = _daemon(tmp_home)
    oid = daemon.emit_observation(_obs(schema="test.probe.v1"))
    assert oid
    assert len(daemon._obs_spool) == 1
    # No WS → stays pending (offline durabilty path).
    daemon._flush_observations()
    assert len(daemon._obs_spool) == 1

    ws = _FakeWS()
    daemon._brain_ws = ws
    daemon._flush_observations()
    assert len(ws.frames) == 1
    frame = ws.frames[0]
    assert frame["type"] == "observation_batch"
    assert frame["items"][0]["observation_id"] == oid
    # Still pending until the drain confirms the send.
    assert len(daemon._obs_spool) == 1
    daemon._observation_batch_sent(frame["id"])
    assert len(daemon._obs_spool) == 0


def test_daemon_failed_send_stays_pending(tmp_home):
    daemon = _daemon(tmp_home)
    daemon.emit_observation(_obs(schema="test.probe.v1"))

    class _DeadWS(_FakeWS):
        def send_frame(self, frame):
            return False

    daemon._brain_ws = _DeadWS()
    daemon._flush_observations()
    assert len(daemon._obs_spool) == 1
    assert not daemon._obs_inflight

    ws = _FakeWS()
    daemon._brain_ws = ws
    daemon._flush_observations()
    daemon._observation_batch_sent(ws.frames[0]["id"])
    assert len(daemon._obs_spool) == 0
