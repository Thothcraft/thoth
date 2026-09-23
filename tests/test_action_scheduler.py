"""ActionScheduler timing policy — delay, debounce, cooldown, cancel.

These tests pin the §17/§36 semantics that the old synchronous dispatcher
ignored: a 180 s occupancy delay must actually wait, a cleared condition
must cancel a pending fire, and cooldown may only be consumed by a real
dispatch — never by a gated/rejected prediction.
"""
import time

import pytest

from thoth.daemon.actions import ActionScheduler
from whispy.contracts import Prediction


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, seconds):
        self.t += seconds


def _sched(fired, clock):
    """Deterministic scheduler: inline executor, manual fire_due driver."""
    def rec(key, action, pred):
        fired.append({"key": key, "type": action.type,
                      "label": pred.label})
    return ActionScheduler(workers=0, autostart=False, clock=clock,
                           executor=rec)


def _pred(label="empty", confidence=1.0):
    return Prediction(label=label, confidence=confidence)


def _cfg(**kw):
    cfg = {"type": "webhook", "config": {"url": "http://x"},
           "min_confidence": 0.0}
    cfg.update(kw)
    return cfg


# -- delay -------------------------------------------------------------------

def test_delay_waits_full_duration():
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    s.submit(_cfg(delay_seconds=180.0), _pred("empty"), model_id="m1")
    s.fire_due()
    assert not fired                      # armed but not yet due
    clk.advance(179.0)
    s.fire_due()
    assert not fired                      # still inside the 180 s window
    clk.advance(2.0)                      # t = 181
    s.fire_due()
    assert len(fired) == 1                # fired only after the delay


def test_delay_counts_from_condition_start_not_latest_tick():
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    s.submit(_cfg(delay_seconds=180.0), _pred("empty"), model_id="m1")
    # Repeated qualifying ticks must NOT push the due time out.
    for _ in range(5):
        clk.advance(30.0)
        s.submit(_cfg(delay_seconds=180.0), _pred("empty"), model_id="m1")
    # t = 150 now; due is still t=180 from the first qualifying tick.
    clk.advance(31.0)                     # t = 181
    s.fire_due()
    assert len(fired) == 1


# -- cancellation --------------------------------------------------------------

def test_condition_clearing_cancels_pending_fire():
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    # Only the "empty" label triggers this off-action.
    cfg = _cfg(delay_seconds=180.0, trigger_labels=["empty"])
    s.submit(cfg, _pred("empty"), model_id="m1")
    clk.advance(60.0)
    # Occupancy returns before the delay elapses → cancel the off-action.
    s.submit(cfg, _pred("occupied"), model_id="m1")
    clk.advance(200.0)
    s.fire_due()
    assert not fired


# -- debounce ------------------------------------------------------------------

def test_debounce_requires_continuous_qualification():
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    cfg = _cfg(debounce_seconds=5.0, trigger_labels=["occupied"])
    s.submit(cfg, _pred("occupied"), model_id="m1")   # t=0, due=5
    clk.advance(2.0)
    s.submit(cfg, _pred("empty"), model_id="m1")      # flicker → reset
    clk.advance(1.0)
    s.submit(cfg, _pred("occupied"), model_id="m1")   # t=3, due=8
    clk.advance(4.0)                                  # t=7
    s.fire_due()
    assert not fired                                  # debounce not met
    clk.advance(1.0)                                  # t=8
    s.fire_due()
    assert len(fired) == 1


# -- cooldown ------------------------------------------------------------------

def test_cooldown_only_consumed_by_real_dispatch():
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    cfg = _cfg(cooldown_seconds=60.0, min_confidence=0.5)
    s.submit(cfg, _pred("occupied"), model_id="m1")
    s.fire_due()                                       # fires at t=0
    assert len(fired) == 1

    # A non-qualifying prediction must NOT consume the cooldown window.
    s.submit(cfg, _pred("occupied", confidence=0.1), model_id="m1")
    clk.advance(30.0)                                  # t=30, within cooldown
    s.submit(cfg, _pred("occupied"), model_id="m1")
    s.fire_due()
    assert len(fired) == 1                             # suppressed by cooldown

    clk.advance(40.0)                                  # t=70, cooldown elapsed
    s.submit(cfg, _pred("occupied"), model_id="m1")
    s.fire_due()
    assert len(fired) == 2


def test_cooldown_does_not_suppress_unrelated_action():
    """Two distinct actions of the same type have independent timers."""
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    a = _cfg(cooldown_seconds=60.0, config={"url": "http://a"})
    b = _cfg(cooldown_seconds=60.0, config={"url": "http://b"})
    s.submit(a, _pred("occupied"), model_id="m1")
    s.submit(b, _pred("occupied"), model_id="m1")
    s.fire_due()
    assert len(fired) == 2                             # both fired
    keys = {f["key"] for f in fired}
    assert len(keys) == 2                              # distinct identities


def test_same_action_different_models_independent():
    clk = FakeClock()
    fired = []
    s = _sched(fired, clk)
    cfg = _cfg()
    s.submit(cfg, _pred("occupied"), model_id="m1")
    s.submit(cfg, _pred("occupied"), model_id="m2")
    s.fire_due()
    assert len(fired) == 2


# -- non-blocking ---------------------------------------------------------------

def test_submit_is_non_blocking():
    fired = []
    s = ActionScheduler(workers=2, autostart=True,
                        executor=lambda k, a, p: time.sleep(0.5))
    try:
        start = time.time()
        s.submit(_cfg(), _pred("occupied"), model_id="m1")
        assert time.time() - start < 0.2               # never blocks the loop
    finally:
        s.stop()
