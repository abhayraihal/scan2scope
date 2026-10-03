"""Rigid and similarity transforms (4x4 matrices, column vectors)."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def make_T(R: np.ndarray, t: np.ndarray, s: float = 1.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    return T


def decompose_sim3(T: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    A = T[:3, :3]
    s = float(np.cbrt(np.linalg.det(A)))
    return s, A / s, T[:3, 3].copy()


def invert(T: np.ndarray) -> np.ndarray:
    s, R, t = decompose_sim3(T)
    Ri = R.T / s
    return make_T(Ri, -Ri @ t)


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Transform points of shape (..., 3)."""
    return P @ T[:3, :3].T + T[:3, 3]


def umeyama(src: np.ndarray, dst: np.ndarray, with_scale: bool = True,
            weights: np.ndarray | None = None) -> np.ndarray:
    """Least-squares T with dst ~= T(src). Returns a 4x4 similarity (or rigid) transform."""
    src = np.asarray(src, float)
    dst = np.asarray(dst, float)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    mu_s = (w[:, None] * src).sum(0)
    mu_d = (w[:, None] * dst).sum(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = (w[:, None] * xd).T @ xs
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = 1.0
    if with_scale:
        var_s = (w * (xs ** 2).sum(1)).sum()
        s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - s * R @ mu_s
    return make_T(R, t, s)


def rot_z(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rotvec_to_R(v: np.ndarray) -> np.ndarray:
    return Rotation.from_rotvec(v).as_matrix()


def R_to_rotvec(R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(R).as_rotvec()


def rotation_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Smallest rotation R with R @ a parallel to b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if np.linalg.norm(v) < 1e-9:
        if c > 0:
            return np.eye(3)
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        return rotvec_to_R(np.pi * axis / np.linalg.norm(axis))
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))
