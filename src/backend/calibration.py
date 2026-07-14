"""Pure calculations for the guided three-region calibration workflow."""

from __future__ import annotations

import statistics
from typing import Iterable

REGIONS = ("red", "yellow", "green")


def _clean_samples(values: Iterable[float]) -> list[float]:
    samples = [float(value) for value in values]
    if not samples or any(value < 0.0 or value > 100.0 for value in samples):
        raise ValueError("each region requires detection-ratio samples from 0 to 100")
    return samples


def derive_thresholds(samples_by_region: dict[str, Iterable[float]], *, max_spread: float = 35.0) -> dict:
    """Derive midpoint thresholds, rejecting incomplete, unstable, or crossed samples."""
    samples = {region: _clean_samples(samples_by_region.get(region, [])) for region in REGIONS}
    summaries = {}
    medians = []
    for region in REGIONS:
        values = samples[region]
        spread = max(values) - min(values)
        if spread > max_spread:
            raise ValueError(f"{region} calibration is unstable (spread {spread:.1f}%)")
        median = float(statistics.median(values))
        medians.append(median)
        summaries[region] = {
            "sample_count": len(values), "median_percent": median,
            "minimum_percent": min(values), "maximum_percent": max(values),
            "spread_percent": spread,
        }
    if not medians[0] < medians[1] < medians[2]:
        raise ValueError("calibration medians must be ordered red < yellow < green")
    yellow_threshold = (medians[0] + medians[1]) / 2.0
    green_threshold = (medians[1] + medians[2]) / 2.0
    if not 0.0 <= yellow_threshold < green_threshold <= 100.0:
        raise ValueError("derived calibration thresholds overlap")
    return {
        "yellow_threshold_percent": yellow_threshold,
        "green_threshold_percent": green_threshold,
        "sample_summaries": summaries,
    }
