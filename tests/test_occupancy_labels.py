"""Regression tests for the generic presence-label vocabulary.

The positive class is resolved from each model's own class_names instead of
being hard-coded to "occupied", so models exported as absent/present (or any
synonym) must work end to end. These helpers are pure Python and need no
torch runtime.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from backend.model_runtime import (  # noqa: E402
    is_occupancy_result,
    occupancy_probability,
    positive_class_name,
)


def test_positive_class_name_resolves_synonyms():
    assert positive_class_name(["empty", "occupied"]) == "occupied"
    assert positive_class_name(["absent", "present"]) == "present"
    assert positive_class_name(["clear", "person"]) == "person"


def test_positive_class_name_binary_fallback():
    # Unknown binary labels: the positive class is conventionally last.
    assert positive_class_name(["foo", "bar"]) == "bar"
    assert positive_class_name(["only"]) is None
    assert positive_class_name([]) is None


def test_is_occupancy_result():
    assert is_occupancy_result({"class": "present"})
    assert is_occupancy_result({"class": "empty"})
    assert is_occupancy_result({"class": "x", "scores": {"present": 0.9, "absent": 0.1}})
    assert not is_occupancy_result({"class": "cat"})
    assert not is_occupancy_result({})


def test_occupancy_probability():
    assert occupancy_probability({"class": "present", "confidence": 0.8}) == 0.8
    assert abs(occupancy_probability({"class": "absent", "confidence": 0.8}) - 0.2) < 1e-9
    assert occupancy_probability({"scores": {"present": 0.7, "absent": 0.3}}) == 0.7
    assert occupancy_probability({"scores": {"empty": 0.4, "occupied": 0.6}}) == 0.6
    assert 0.0 <= occupancy_probability({"class": "weird"}) <= 1.0
