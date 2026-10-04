"""Sim(3) registration of overlapping reconstruction chunks and a small Sim(3) pose graph.

A Sim(3) is a 4x4 matrix with T[:3, :3] = s R. T_ij maps coordinates of frame j into frame i (x_i = T_ij x_j).
Pose-graph nodes X_k map chunk k into the world.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from scan2scope.geometry.se3 import apply, decompose_sim3, invert, make_T, umeyama

log = logging.getLogger("scan2scope.geometry")


@dataclass
class Sim3Fit:
    T: np.ndarray  # dst ~= T(src)
    residual_m: float  # median residual of the inliers, in dst units
    inlier_frac: float
    n: int  # finite correspondences considered
    ok: bool

    @property
    def scale(self) -> float:
        return decompose_sim3(self.T)[0]


def _so3_exp(v: np.ndarray) -> np.ndarray:
    """Batched rotation vectors (N, 3) -> matrices (N, 3, 3)."""
    return Rotation.from_rotvec(np.atleast_2d(v)).as_matrix()


def _so3_log(R: np.ndarray) -> np.ndarray:
    """Batched matrices (N, 3, 3) -> rotation vectors (N, 3)."""
    return Rotation.from_matrix(np.asarray(R).reshape(-1, 3, 3)).as_rotvec()


def rotation_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.linalg.norm(_so3_log(R)[0])))


def robust_sim3(src: np.ndarray, dst: np.ndarray, weights: np.ndarray | None = None, *, thresh: float | None = None,
                iters: int = 128, refine: int = 4, trim: float = 0.9, min_points: int = 20,
                max_points: int = 20000, seed: int = 0) -> Sim3Fit:
    """RANSAC Umeyama on 4-point samples, then inlier refits and a trimmed final fit.

    thresh is the inlier distance in dst units; by default 4% of the median distance of dst from its median.
    """
    src = np.asarray(src, float).reshape(-1, 3)
    dst = np.asarray(dst, float).reshape(-1, 3)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, float).reshape(-1)
    good = np.isfinite(src).all(1) & np.isfinite(dst).all(1) & np.isfinite(w) & (w > 0)
    src, dst, w = src[good], dst[good], w[good]
    n = len(src)
    fail = Sim3Fit(np.eye(4), float("inf"), 0.0, n, False)
    if n < max(min_points, 4):
        return fail
    rng = np.random.default_rng(seed)
    if n > max_points:
        keep = rng.choice(n, max_points, replace=False)
        src, dst, w = src[keep], dst[keep], w[keep]
        n = max_points
    if thresh is None:
        spread = float(np.median(np.linalg.norm(dst - np.median(dst, 0), axis=1)))
        thresh = float(np.clip(0.04 * spread, 0.01, 0.25))

    best_score, best_r = -1.0, None
    for _ in range(iters):
        idx = rng.choice(n, 4, replace=False)
        try:
            T = umeyama(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        s = decompose_sim3(T)[0]
        if not (np.isfinite(T).all() and s > 1e-6):
            continue
        r = np.linalg.norm(apply(T, src) - dst, axis=1)
        score = float(w[r < thresh].sum())
        if score > best_score:
            best_score, best_r = score, r
    if best_r is None:
        return fail

    inl = best_r < thresh
    T = np.eye(4)
    for _ in range(max(1, refine)):
        if inl.sum() < max(min_points, 4):
            return fail
        T = umeyama(src[inl], dst[inl], weights=w[inl])
        r = np.linalg.norm(apply(T, src) - dst, axis=1)
        new = r < thresh
        if np.array_equal(new, inl):
            break
        inl = new
    if inl.sum() < max(min_points, 4):
        return fail
    if 0.0 < trim < 1.0:
        cut = np.quantile(r[inl], trim)
        core = inl & (r <= cut)
        if core.sum() >= max(min_points, 4):
            T = umeyama(src[core], dst[core], weights=w[core])
            r = np.linalg.norm(apply(T, src) - dst, axis=1)
            inl = r < thresh
    frac = float(inl.mean())
    resid = float(np.median(r[inl])) if inl.any() else float("inf")
    ok = bool(np.isfinite(T).all() and decompose_sim3(T)[0] > 1e-6 and inl.sum() >= min_points and frac >= 0.25)
    return Sim3Fit(T, resid, frac, n, ok)


def chain(relative: list[np.ndarray]) -> list[np.ndarray]:
    """relative[k] = T_{k, k+1}; returns X_k = T_{0, k} for k = 0..len(relative)."""
    out = [np.eye(4)]
    for T in relative:
        out.append(out[-1] @ T)
    return out


def sim3_error(A: np.ndarray, B: np.ndarray) -> dict[str, float]:
    """Size of inv(A) @ B: translation norm, rotation angle (degrees) and log scale."""
    s, R, t = decompose_sim3(invert(A) @ B)
    return {"trans": float(np.linalg.norm(t)), "rot_deg": rotation_angle_deg(R), "log_scale": float(np.log(s))}


@dataclass
class Edge:
    """Relative measurement x_i = T_ij x_j with 1-sigma noise per component."""

    i: int
    j: int
    T_ij: np.ndarray
    sigma_rot: float = 0.01  # rad
    sigma_trans: float = 0.03  # units of frame i
    sigma_log_scale: float = 0.01
    kind: str = "seq"


@dataclass
class Prior:
    """Unary constraint on node k: world rotation target and/or a node-frame point whose world z is known."""

    node: int
    R_world: np.ndarray | None = None
    sigma_rot: float = 0.002
    point: np.ndarray | None = None
    z_world: float = 0.0
    sigma_z: float = 0.005


@dataclass
class GraphResult:
    nodes: list[np.ndarray]
    cost_before: float
    cost_after: float
    edge_residuals: list[dict[str, float]] = field(default_factory=list)
    success: bool = True


def optimize_pose_graph(nodes: list[np.ndarray], edges: list[Edge], priors: list[Prior] | tuple = (),
                        fixed: tuple[int, ...] | list[int] = (0,), max_nfev: int = 200) -> GraphResult:
    """Least-squares Sim(3) pose graph. Free nodes are parameterised as a local rotation update, a translation
    and a log scale around their initial values."""
    K = len(nodes)
    dec = [decompose_sim3(np.asarray(T, float)) for T in nodes]
    s0 = np.array([d[0] for d in dec])
    R0 = np.stack([d[1] for d in dec])
    t0 = np.stack([d[2] for d in dec])
    free = [k for k in range(K) if k not in set(fixed)]
    slot = {k: n for n, k in enumerate(free)}
    if not free or (not edges and not priors):
        return GraphResult([np.asarray(T, float).copy() for T in nodes], 0.0, 0.0)

    ei = np.array([e.i for e in edges], int)
    ej = np.array([e.j for e in edges], int)
    edec = [decompose_sim3(np.asarray(e.T_ij, float)) for e in edges]
    es = np.log(np.array([d[0] for d in edec])) if edges else np.zeros(0)
    eR = np.stack([d[1] for d in edec]) if edges else np.zeros((0, 3, 3))
    et = np.stack([d[2] for d in edec]) if edges else np.zeros((0, 3))
    sig = np.array([[e.sigma_rot, e.sigma_trans, e.sigma_log_scale] for e in edges]).reshape(-1, 3)
    rot_pri = [p for p in priors if p.R_world is not None]
    z_pri = [p for p in priors if p.point is not None]

    def unpack(x: np.ndarray):
        logs, R, t = np.log(s0).copy(), R0.copy(), t0.copy()
        if free:
            xf = x.reshape(len(free), 7)
            fi = np.array(free)
            R[fi] = _so3_exp(xf[:, :3]) @ R0[fi]
            t[fi] = xf[:, 3:6]
            logs[fi] = xf[:, 6]
        return logs, R, t

    def residuals(x: np.ndarray) -> np.ndarray:
        logs, R, t = unpack(x)
        out = []
        if len(edges):
            Ri, Rj = R[ei], R[ej]
            Rpred = np.transpose(Ri, (0, 2, 1)) @ Rj
            rr = _so3_log(np.transpose(eR, (0, 2, 1)) @ Rpred) / sig[:, :1]
            tpred = np.einsum("nji,nj->ni", Ri, t[ej] - t[ei]) / np.exp(logs[ei])[:, None]
            rt = (tpred - et) / sig[:, 1:2]
            rs = ((logs[ej] - logs[ei]) - es) / sig[:, 2]
            out += [rr.ravel(), rt.ravel(), rs]
        for p in rot_pri:
            out.append(_so3_log(p.R_world.T @ R[p.node])[0] / p.sigma_rot)
        for p in z_pri:
            zw = np.exp(logs[p.node]) * (R[p.node] @ p.point)[2] + t[p.node][2]
            out.append(np.array([(zw - p.z_world) / p.sigma_z]))
        return np.concatenate(out) if out else np.zeros(1)

    x0 = np.zeros(7 * len(free))
    for k in free:
        x0[7 * slot[k] + 3: 7 * slot[k] + 6] = t0[k]
        x0[7 * slot[k] + 6] = np.log(s0[k])
    r0 = residuals(x0)
    try:
        sol = least_squares(residuals, x0, method="trf", max_nfev=max_nfev)
        x, success = sol.x, bool(sol.success)
    except (ValueError, np.linalg.LinAlgError) as exc:
        log.warning("pose graph failed: %s", exc)
        x, success = x0, False
    r1 = residuals(x)
    logs, R, t = unpack(x)
    out_nodes = [make_T(R[k], t[k], float(np.exp(logs[k]))) for k in range(K)]
    per_edge = []
    for e in edges:
        err = sim3_error(np.asarray(e.T_ij, float), invert(out_nodes[e.i]) @ out_nodes[e.j])
        per_edge.append({"i": e.i, "j": e.j, "kind": e.kind, **err})
    return GraphResult(out_nodes, float(0.5 * r0 @ r0), float(0.5 * r1 @ r1), per_edge, success)
