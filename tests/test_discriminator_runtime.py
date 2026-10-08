"""Node discriminator runtime — guided/auto calibration + gated predictions."""
import random

import pytest

from thoth.discriminators import DiscriminatorRuntime


def _desc(std, mean=10.0):
    return {"radar-1": {"type": "radar", "n": 20, "fields": {
        "snr_db": {"mean": mean, "std": std, "min": 0, "max": 20}}}}


@pytest.fixture
def rt(tmp_path):
    cfg = {"interval_s": 0}
    return DiscriminatorRuntime(lambda: cfg, tmp_path), cfg


def test_guided_flow_then_predicts(rt):
    runtime, _ = rt
    rng = random.Random(0)
    st = runtime.start_guided("occupancy-radar")
    assert st["current"]["label"] == "empty"
    t = 0.0
    for _ in range(20):
        t += 1
        assert runtime.observe(_desc(rng.gauss(0.5, 0.1)), now=t) == []
    runtime.next_step()
    for _ in range(20):
        t += 1
        runtime.observe(_desc(rng.gauss(3.0, 0.2), 12.0), now=t)
    out = runtime.finish()
    assert out["calibrated"] and out["calibration"]["method"] == "guided"
    preds = runtime.observe(_desc(3.0, 12.0), device_id="n1", now=t + 1)
    assert [p.label for p in preds] == ["occupied"]
    assert runtime.status()["session"] is None


def test_auto_calibration_from_history(rt):
    runtime, _ = rt
    rng = random.Random(1)
    for i in range(80):
        runtime.observe(_desc(rng.gauss(0.5 if i % 2 else 3.0, 0.2)),
                        now=float(i))
    out = runtime.calibrate_auto("occupancy-radar")
    assert out["calibration"]["method"] == "auto"


def test_disabled_and_interval_gate(rt):
    runtime, cfg = rt
    rng = random.Random(2)
    for i in range(80):
        runtime.observe(_desc(rng.gauss(0.5 if i % 2 else 3.0, 0.2)),
                        now=float(i))
    runtime.calibrate_auto("occupancy-radar")
    cfg["disabled"] = ["occupancy-radar"]
    assert runtime.observe(_desc(3.0), now=1000.0) == []
    cfg["disabled"] = []
    cfg["interval_s"] = 10
    assert runtime.observe(_desc(3.0), now=2000.0)
    assert runtime.observe(_desc(3.0), now=2005.0) == []


def test_session_errors(rt):
    runtime, _ = rt
    with pytest.raises(ValueError):
        runtime.finish()
    with pytest.raises(KeyError):
        runtime.start_guided("nope")
