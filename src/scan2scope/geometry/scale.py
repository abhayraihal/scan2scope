"""Metric-scale cues, fused in log space.

A cue (log_scale, sigma) says: multiply the scene by exp(log_scale) to make it metric, with 1-sigma
uncertainty sigma in log units.
"""

from __future__ import annotations

import math
from collections.abc import Iterable


def fuse_scale(cues: Iterable[tuple[float, float]]) -> tuple[float, float]:
    """Inverse-variance fusion of independent log-scale cues.

    Cues with a non-finite value or a negative or non-finite sigma are ignored. Exact cues (sigma 0) win and are
    averaged. With no usable cue the result is (0.0, inf): no correction, no information.
    """
    usable = [(float(v), float(s)) for v, s in cues
              if math.isfinite(float(v)) and math.isfinite(float(s)) and float(s) >= 0.0]
    if not usable:
        return 0.0, math.inf
    exact = [v for v, s in usable if s == 0.0]
    if exact:
        return sum(exact) / len(exact), 0.0
    w = [1.0 / (s * s) for _, s in usable]
    total = sum(w)
    return sum(wi * v for wi, (v, _) in zip(w, usable)) / total, 1.0 / math.sqrt(total)


def door_height_cue(height_m: float, prior_m: float = 2.05, prior_sigma: float = 0.06,
                    meas_sigma_m: float = 0.0) -> tuple[float, float]:
    """Scale cue from a full-height door opening measured as height_m in the current scene units.

    The cue rescales the scene so the door matches the prior height; sigma combines the prior spread with the
    measurement noise, both as relative errors. An unusable height gives (0.0, inf).
    """
    if not (math.isfinite(height_m) and height_m > 0.0 and prior_m > 0.0):
        return 0.0, math.inf
    sigma = math.hypot(prior_sigma / prior_m, max(meas_sigma_m, 0.0) / height_m)
    return math.log(prior_m) - math.log(height_m), sigma
