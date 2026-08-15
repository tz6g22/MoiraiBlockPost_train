from __future__ import annotations

import statistics
from typing import Sequence

import numpy as np


def distortion_summary(
    values: Sequence[float],
) -> dict[str, float | int | list[float]]:
    if not values:
        raise ValueError("Distortion summary requires at least one value")
    numeric = [float(value) for value in values]
    if not np.isfinite(numeric).all():
        raise FloatingPointError("Distortion summary contains NaN or Inf")
    return {
        "count": len(numeric),
        "mean": statistics.fmean(numeric),
        "std": statistics.stdev(numeric) if len(numeric) > 1 else 0.0,
        "median": statistics.median(numeric),
    }

