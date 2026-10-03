"""Organised point maps: back-projection, normals, downsampling."""

from __future__ import annotations

import numpy as np


def backproject_depth(depth: np.ndarray, K: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    """Depth (h, w) in metres with intrinsics K at the same resolution -> world point map (h, w, 3)."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w) + 0.0, np.arange(h) + 0.0)
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    pc = np.stack([x, y, depth], -1)
    return pc @ T_wc[:3, :3].T + T_wc[:3, 3]


def normals_from_pointmap(pm: np.ndarray, valid: np.ndarray, cam_center: np.ndarray | None = None,
                          step: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Normals from central differences on the pixel grid, oriented towards cam_center when given."""
    p = pm.astype(np.float64)
    dx = np.zeros_like(p)
    dy = np.zeros_like(p)
    dx[:, step:-step] = p[:, 2 * step:] - p[:, :-2 * step]
    dy[step:-step, :] = p[2 * step:, :] - p[:-2 * step, :]
    n = np.cross(dx, dy)
    norm = np.linalg.norm(n, axis=-1)
    ok = valid.copy()
    ok[:, :step] = ok[:, -step:] = False
    ok[:step, :] = ok[-step:, :] = False
    for sh in ((0, step), (0, -step), (step, 0), (-step, 0)):
        ok &= np.roll(valid, sh, axis=(0, 1))
    ok &= norm > 1e-12
    n = n / np.maximum(norm, 1e-12)[..., None]
    if cam_center is not None:
        flip = ((cam_center - p) * n).sum(-1) < 0
        n[flip] *= -1
    return n.astype(np.float32), ok


def voxel_downsample(points: np.ndarray, voxel: float, *attrs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Average points (and per-point attributes) inside each voxel."""
    if len(points) == 0:
        return (points, *attrs)
    keys = np.floor(points / voxel).astype(np.int64)
    _, inv, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    out = []
    for a in (points, *attrs):
        a2 = a.reshape(len(points), -1).astype(np.float64)
        acc = np.zeros((len(counts), a2.shape[1]))
        np.add.at(acc, inv, a2)
        acc /= counts[:, None]
        out.append(acc.reshape((len(counts),) + a.shape[1:]).astype(a.dtype))
    return tuple(out)
