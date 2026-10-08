"""Context uplink — rate-limited context.descriptors.v1 observations."""
import json

from whispy.contracts import SensorSample, SensorWindow

from thoth.context_uplink import SCHEMA, ContextUplink, normalize_config


def _window():
    samples = [SensorSample(device_id="d", sensor_id="radar-1",
                            sensor_type="radar", timestamp=100 + i * 0.1,
                            sequence=i, payload_type="auto",
                            payload={"snr_db": 12.0 + i}) for i in range(5)]
    return SensorWindow(start_timestamp=100.0, end_timestamp=102.0,
                        samples={"radar-1": samples})


PRED = [{"runtime_model_id": "rm-occ", "label": "occupied",
         "confidence": 0.9, "timestamp": 101.0},
        {"runtime_model_id": "rm-occ", "label": "empty",
         "confidence": 0.6, "timestamp": 99.0}]


def test_normalize_clamps_and_defaults():
    assert normalize_config(None) == {
        "enabled": True, "rate_s": 60.0, "detail": "descriptors",
        "text": {"speech": True, "person": True, "face": True,
                 "face_recognizer": None}}
    assert normalize_config({"text": {"speech": 0, "bogus": 1}})["text"] \
        == {"speech": False, "person": True, "face": True,
            "face_recognizer": None}
    cfg = normalize_config({"rate_s": 1, "detail": "raw", "junk": 1})
    assert cfg["rate_s"] == 5.0 and cfg["detail"] == "descriptors"
    assert "junk" not in cfg
    assert normalize_config({"rate_s": "x"})["rate_s"] == 60.0


def test_emits_once_per_rate_window():
    cfg = {"rate_s": 30}
    up = ContextUplink(lambda: cfg, "dev-1")
    sent = []
    assert up.maybe_emit(sent.append, window=_window(), predictions=PRED,
                         now=1000.0) is not None
    assert up.maybe_emit(sent.append, window=_window(), now=1010.0) is None
    assert up.maybe_emit(sent.append, window=_window(), now=1031.0) is not None
    assert len(sent) == 2
    obs = sent[0].validate().to_dict()
    assert obs["schema"] == SCHEMA
    assert obs["source_id"] == "node:dev-1"
    assert obs["subject"] == "device:dev-1"
    v = obs["value"]
    # latest prediction per model wins (list order = oldest → newest)
    assert v["predictions"]["rm-occ"]["label"] == "empty"
    assert v["sensors"]["radar-1"]["fields"]["snr_db"]["mean"] == 14.0
    json.dumps(obs)


def test_detail_levels():
    cfg = {"detail": "minimal"}
    up = ContextUplink(lambda: cfg, "dev-1")
    v = up.build(window=_window(), predictions=PRED, now=1.0)
    assert "sensors" not in v and v["predictions"]
    cfg["detail"] = "full"
    v = up.build(window=_window(), room={"room_id": "lr", "name": "Living",
                                         "rooms": [{"name": "Kitchen"}]},
                 now=1.0)
    assert v["sensors"]["radar-1"]["n"] == 5
    assert v["room"] == {"room_id": "lr", "name": "Living",
                         "rooms": ["Kitchen"]}


def test_disabled_never_emits():
    up = ContextUplink(lambda: {"enabled": False}, "dev-1")
    sent = []
    assert up.maybe_emit(sent.append, now=10_000.0) is None
    assert sent == []


class _FakeDescriber:
    def __init__(self, cfg):
        self.cfg = cfg

    def describe(self, window, physical, now):
        return {sid: {"text": f"{sid}: busy", "cues": {"motion": "high"}}
                for sid in physical}

    def summary(self, described):
        return " | ".join(v["text"] for v in described.values())


def test_textual_cues_and_scene_line():
    made = []

    def factory(cfg):
        made.append(cfg)
        return _FakeDescriber(cfg)

    cfg = {"text": {"speech": False}}
    up = ContextUplink(lambda: cfg, "dev-1", describer_factory=factory)
    v = up.build(window=_window(), now=1.0)
    assert v["sensors"]["radar-1"]["text"] == "radar-1: busy"
    assert v["sensors"]["radar-1"]["cues"] == {"motion": "high"}
    assert v["scene"] == "radar-1: busy"
    up.build(window=_window(), now=2.0)
    assert len(made) == 1 and made[0]["speech"] is False   # cached
    cfg["text"] = {"speech": True}
    up.build(window=_window(), now=3.0)
    assert len(made) == 2                                    # rebuilt


def test_describer_failure_does_not_drop_descriptors():
    def boom(cfg):
        raise RuntimeError("no models")
    up = ContextUplink(lambda: {}, "dev-1", describer_factory=boom)
    v = up.build(window=_window(), now=1.0)
    assert v["sensors"]["radar-1"]["n"] == 5
    assert "no models" in v["text_error"]


def test_no_window_still_reports_predictions():
    up = ContextUplink(lambda: {}, "dev-1")
    v = up.build(window=None, predictions=PRED, now=1.0)
    assert "sensors" not in v and "rm-occ" in v["predictions"]
