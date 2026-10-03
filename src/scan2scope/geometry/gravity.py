"""Gravity (up) estimation and Manhattan wall direction."""

from __future__ import annotations

import numpy as np

from scan2scope.geometry.se3 import rotation_between


def estimate_up(normals: np.ndarray, weights: np.ndarray, up_hint: np.ndarray, max_angle_deg: float = 35.0,
                iters: int = 3) -> np.ndarray:
    """Refine an up direction from surface normals of floors and ceilings near up_hint."""
    up = up_hint / np.linalg.norm(up_hint)
    cos_t = np.cos(np.radians(max_angle_deg))
    for _ in range(iters):
        d = normals @ up
        sel = np.abs(d) > cos_t
        if sel.sum() < 50:
            break
        n = normals[sel] * np.sign(d[sel])[:, None]
        up = (n * weights[sel, None]).sum(0)
        up /= np.linalg.norm(up)
        cos_t = np.cos(np.radians(max(8.0, max_angle_deg / 2)))
    return up


def align_up_to_z(up: np.ndarray) -> np.ndarray:
    """Rotation mapping `up` to +z."""
    return rotation_between(up, np.array([0.0, 0.0, 1.0]))


def manhattan_angle(normals_xy: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Dominant wall direction modulo 90 degrees, in radians in [0, pi/2)."""
    n = normals_xy / np.maximum(np.linalg.norm(normals_xy, axis=1, keepdims=True), 1e-12)
    ang = np.arctan2(n[:, 1], n[:, 0])
    w = np.ones(len(ang)) if weights is None else weights
    z = (w * np.exp(4j * ang)).sum()
    return float((np.angle(z) / 4.0) % (np.pi / 2))
