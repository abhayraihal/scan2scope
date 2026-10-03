"""Drift correction: loop closure, pose graph, plane anchoring and Manhattan yaw anchoring.

Poses are camera-to-world with OpenCV camera axes in a gravity-aligned z-up world (types.py). Keyframes are
grouped into segments of about 3 s that are treated as rigid; each segment gets a world-frame correction C_s,
applied on the left (corrected = C_s @ raw). The VIO observes gravity, so the pose graph solves yaw and
translation per segment (4 DoF); tilt is handled by plane anchoring. Corrections are interpolated in time to
every keyframe.

Stages, each with a switch for the ablation:
1. Loop closure: point-to-plane ICP between segments more than 20 s apart whose clouds overlap, always
   including first versus last, then a pose graph with odometry edges from the raw poses and accepted loop edges.
2. Plane anchoring: per-segment floor height (as a prior in a second pose-graph solve) and floor tilt.
3. Manhattan yaw anchoring: segments whose dominant wall angle is within 5 degrees of the global one get a yaw
   prior that snaps them to it, in the same second solve.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation, Slerp

from scan2scope.geometry.se3 import apply, rotation_between, rotvec_to_R

log = logging.getLogger("scan2scope.geometry.drift")

SEGMENT_S = 3.0
SEG_VOXEL = 0.05
LOOP_MIN_GAP_S = 20.0
MAX_LOOP_CANDIDATES = 60  # per round
LOOPS_PER_SEGMENT = 2
LOOP_ROUNDS = 2  # the second round registers from the corrected poses and finds pairs drift had hidden
SCREEN_RADIUS = 0.3  # candidate screening: share of source points with a target point this close
SCREEN_OVERLAP = 0.3
ICP_MAX_DIST = 0.10
ICP_COARSE_DIST = 0.30
ICP_MAX_ITER = 30
ACCEPT_RMSE = 0.02
ACCEPT_OVERLAP = 0.30
DEGENERATE_EIG = 0.03  # eigenvalue of the scaled per-point ICP information below which a direction is free
MIN_LOOP_DOF = 4  # a floor alone constrains 3 (z, roll, pitch); a wall adds yaw and one horizontal direction
LOOP_SIGMA = 0.005  # metres of loop-edge uncertainty along a direction with information LOOP_REF_EIG
LOOP_REF_EIG = 1.0 / 3.0  # information per direction when three orthogonal planes share the points
MANHATTAN_SNAP_DEG = 5.0
FLOOR_BELOW = (0.5, 2.2)  # a segment's floor is the lowest dominant up-facing plane this far below its camera
FLOOR_ANCHOR_MAX = 0.6  # floors further than this from the common level are left alone (another storey)
FLOOR_OUTLIER = 0.10  # a floor this far from the median of its neighbours in time is a table, not the floor
TILT_DEADBAND_DEG = 0.5
TILT_MAX_DEG = 3.0
JUMP_SPEED = 3.0  # m/s between consecutive keyframes marks a relocalisation jump

_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class KeyFrame:
    frame_id: int
    timestamp: float


@dataclass
class FrameCloud:
    """Points of one keyframe in its camera frame."""

    points: np.ndarray  # (M, 3)
    normals: np.ndarray  # (M, 3) unit, towards the camera
    weights: np.ndarray  # (M,) in (0, 1]


DepthReader = Callable[[int], "FrameCloud | None"]


@dataclass
class Segment:
    members: np.ndarray  # keyframe positions
    anchor: int  # keyframe position whose pose carries the segment
    points: np.ndarray  # (M, 3) in the anchor camera frame
    normals: np.ndarray
    weights: np.ndarray


@dataclass
class IcpResult:
    T: np.ndarray  # (4, 4) world-frame transform applied to the source
    rmse: float  # point-to-plane RMS over inliers
    overlap: float  # inliers / source points
    n_inliers: int
    center: np.ndarray  # (3,) centroid of the aligned inliers, the pivot of `info`
    info: np.ndarray  # (6, 6) per-point information of (rotation about center, translation), weak directions zeroed
    dof: int  # number of well-constrained directions
    eig: np.ndarray  # eigenvalues of the scaled information, ascending
    iterations: int


@dataclass
class Edge:
    """Relative-pose edge: inv(X_a) @ X_b should equal Z, isotropic sigmas in the frame of b."""

    a: int
    b: int
    Z: np.ndarray
    sigma_t: float  # metres
    sigma_r: float  # radians


@dataclass
class LoopEdge:
    """ICP loop edge between segments a (target) and b (source), registered with their clouds placed at Ga, Gb.

    T moved b's cloud onto a's in that registration world. The residual is the remaining motion of b's cloud
    about `center` (rotation vector, shift), weighted by sqrt_info, so directions the registration left free
    cost nothing.
    """

    a: int
    b: int
    T: np.ndarray
    center: np.ndarray
    sqrt_info: np.ndarray  # (6, 6)
    Ga: np.ndarray  # (4, 4) pose of segment a during the registration
    Gb: np.ndarray


# ---------------------------------------------------------------------------------------------- voxel clouds

_OFF = 1 << 20


def voxel_keys(points: np.ndarray, voxel: float) -> np.ndarray:
    ijk = np.clip(np.floor(points / voxel).astype(np.int64) + _OFF, 0, (1 << 21) - 1)
    return (ijk[:, 0] << 42) | (ijk[:, 1] << 21) | ijk[:, 2]


class VoxelAccumulator:
    """Streams weighted points into voxels.

    Per voxel: weighted mean position, normalised weighted normal sum, mean weight, and the tag of its heaviest
    point.
    """

    def __init__(self, voxel: float, compact_rows: int = 3_000_000) -> None:
        self.voxel = voxel
        self.compact_rows = compact_rows
        self._parts: list[tuple[np.ndarray, ...]] = []
        self._rows = 0

    def add(self, points: np.ndarray, normals: np.ndarray, weights: np.ndarray, tag: int | np.ndarray = 0) -> None:
        if len(points) == 0:
            return
        w = np.asarray(weights, np.float64)
        tags = np.broadcast_to(np.asarray(tag, np.int64), w.shape)
        part = self._reduce(voxel_keys(points, self.voxel), w, points * w[:, None], normals * w[:, None],
                            np.ones(len(w)), w, tags)
        self._parts.append(part)
        self._rows += len(part[0])
        if self._rows > self.compact_rows and len(self._parts) > 1:
            self._compact()

    @staticmethod
    def _reduce(keys, sw, swp, swn, cnt, best_w, tags) -> tuple[np.ndarray, ...]:
        order = np.lexsort((-best_w, keys))
        keys = keys[order]
        start = np.flatnonzero(np.r_[True, keys[1:] != keys[:-1]])
        return (keys[start], np.add.reduceat(sw[order], start), np.add.reduceat(swp[order], start, axis=0),
                np.add.reduceat(swn[order], start, axis=0), np.add.reduceat(cnt[order], start),
                best_w[order][start], tags[order][start])

    def _compact(self) -> None:
        cat = [np.concatenate(x) for x in zip(*self._parts)]
        self._parts = [self._reduce(*cat)]
        self._rows = len(self._parts[0][0])

    def result(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(points, normals, weights, tags)."""
        if not self._parts:
            return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), np.zeros(0, np.int64)
        if len(self._parts) > 1:
            self._compact()
        _, sw, swp, swn, cnt, _, tags = self._parts[0]
        pts = swp / np.maximum(sw, 1e-12)[:, None]
        nrm = swn / np.maximum(np.linalg.norm(swn, axis=1), 1e-12)[:, None]
        return pts, nrm, sw / cnt, tags.copy()


def voxel_reduce(points, normals, weights, voxel) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc = VoxelAccumulator(voxel)
    acc.add(points, normals, weights)
    p, n, w, _ = acc.result()
    return p, n, w


# ------------------------------------------------------------------------------------------- small helpers

def _inv(T: np.ndarray) -> np.ndarray:
    """Inverse of rigid transforms (..., 4, 4)."""
    R = T[..., :3, :3]
    Ti = np.zeros_like(T)
    Ti[..., :3, :3] = np.swapaxes(R, -1, -2)
    Ti[..., :3, 3] = -np.einsum("...ji,...j->...i", R, T[..., :3, 3])
    Ti[..., 3, 3] = 1.0
    return Ti


def _yaw_of(R: np.ndarray) -> np.ndarray:
    return np.arctan2(R[..., 1, 0], R[..., 0, 0])


def _corrections(params: np.ndarray) -> np.ndarray:
    """(S, 4) yaw, tx, ty, tz -> (S, 4, 4) world-frame transforms."""
    c, s = np.cos(params[:, 0]), np.sin(params[:, 0])
    C = np.zeros((len(params), 4, 4))
    C[:, 0, 0], C[:, 0, 1], C[:, 1, 0], C[:, 1, 1] = c, -s, s, c
    C[:, 2, 2] = C[:, 3, 3] = 1.0
    C[:, :3, 3] = params[:, 1:]
    return C


def _wrap(a: np.ndarray | float, period: float = 2 * np.pi) -> np.ndarray | float:
    return (np.asarray(a) + period / 2) % period - period / 2


def _pose_jumps(t: np.ndarray, poses: np.ndarray) -> np.ndarray:
    """Bool (K,), True at k when keyframe k-1 -> k moves faster than a person can carry a phone."""
    if len(t) < 2:
        return np.zeros(len(t), bool)
    d = np.linalg.norm(np.diff(poses[:, :3, 3], axis=0), axis=1)
    dt = np.maximum(np.diff(t), 1e-3)
    return np.r_[False, (d / dt > JUMP_SPEED) & (d > 0.15)]


# ------------------------------------------------------------------------------------------------ segments

def build_segments(t: np.ndarray, poses: np.ndarray, depth_reader: DepthReader, *, segment_s: float = SEGMENT_S,
                   jumps: np.ndarray | None = None, voxel: float = SEG_VOXEL) -> list[Segment]:
    """Group keyframes into segments of segment_s seconds (breaking at pose jumps) with one cloud each."""
    if len(t) == 0:
        return []
    bin_id = np.floor((t - t[0]) / segment_s).astype(np.int64)
    if jumps is not None:
        bin_id = bin_id * (len(t) + 1) + np.cumsum(jumps)
    starts = np.flatnonzero(np.r_[True, bin_id[1:] != bin_id[:-1]])
    bounds = np.r_[starts, len(t)]
    segs = []
    for a, b in itertools.pairwise(bounds):
        members = np.arange(a, b)
        mid = 0.5 * (t[a] + t[b - 1])
        anchor = int(members[np.argmin(np.abs(t[members] - mid))])
        Tinv = _inv(poses[anchor])
        acc = VoxelAccumulator(voxel)
        for k in members:
            fc = depth_reader(int(k))
            if fc is None or len(fc.points) == 0:
                continue
            A = Tinv @ poses[k]
            acc.add(apply(A, fc.points), fc.normals @ A[:3, :3].T, fc.weights)
        p, n, w, _ = acc.result()
        segs.append(Segment(members, anchor, p, n, w))
    return segs


def _world(seg: Segment, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return apply(X, seg.points), seg.normals @ X[:3, :3].T


# ---------------------------------------------------------------------------------------------------- ICP

def _normal_equations(p: np.ndarray, q: np.ndarray, n: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    """Point-to-plane residuals and the per-point normal equations in scaled coordinates.

    Unknowns are (rotation about the centroid c times L, translation), L the RMS radius about c, so the six
    eigenvalues of the information are comparable. Residuals get Huber weights at 2 cm.
    """
    r = ((p - q) * n).sum(1)
    c = p.mean(0)
    L = max(float(np.sqrt(((p - c) ** 2).sum(1).mean())), 1e-3)
    J = np.hstack([np.cross(p - c, n) / L, n])
    a = np.abs(r)
    w = np.where(a < 0.02, 1.0, 0.02 / np.maximum(a, 1e-12))
    sw = float(w.sum())
    H = (J * w[:, None]).T @ J / sw
    g = (J * w[:, None]).T @ r / sw
    return r, c, L, H, g


def icp_point_to_plane(src: np.ndarray, src_n: np.ndarray, tgt: np.ndarray, tgt_n: np.ndarray, *,
                       T0: np.ndarray | None = None, tree: cKDTree | None = None, max_iter: int = ICP_MAX_ITER,
                       max_dist: float = ICP_MAX_DIST, coarse_dist: float = ICP_COARSE_DIST,
                       max_points: int = 4000, seed: int = 0) -> IcpResult:
    """Rigid T with T(src) on the target surface: Gauss-Newton on SE(3) over point-to-plane residuals.

    Correspondences are nearest neighbours with compatible normals, trimmed at a distance that shrinks from
    coarse_dist to max_dist over the first iterations. Steps are taken only along well-constrained directions
    (a wall plus floor leaves sliding along the wall free), and the result carries the information matrix so a
    pose graph can use the constrained directions of a partly degenerate registration.
    """
    src = np.asarray(src, float)
    src_n = np.asarray(src_n, float)
    if len(src) > max_points:
        sel = np.random.default_rng(seed).choice(len(src), max_points, replace=False)
        src, src_n = src[sel], src_n[sel]
    tree = cKDTree(tgt) if tree is None else tree
    T = np.eye(4) if T0 is None else np.array(T0, float)
    cos_n = math.cos(math.radians(45.0))
    iters = 0

    def matches(T: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = apply(T, src)
        d, j = tree.query(p, distance_upper_bound=thr)
        m = np.flatnonzero(np.isfinite(d))
        jm = j[m]
        ok = ((src_n[m] @ T[:3, :3].T) * tgt_n[jm]).sum(1) > cos_n
        return p[m[ok]], tgt[jm[ok]], tgt_n[jm[ok]]

    for it in range(max_iter):
        thr = max(max_dist, coarse_dist * 0.7 ** it)
        p, q, n = matches(T, thr)
        if len(p) < 30:
            break
        _, c, L, H, g = _normal_equations(p, q, n)
        ev, U = np.linalg.eigh(H)
        good = ev > DEGENERATE_EIG
        if not good.any():
            break
        xin = -U[:, good] @ ((U[:, good].T @ g) / ev[good])
        xi = np.r_[xin[:3] / L, xin[3:]]
        dT = np.eye(4)
        dR = rotvec_to_R(xi[:3])
        dT[:3, :3] = dR
        dT[:3, 3] = c + xi[3:] - dR @ c
        T = dT @ T
        iters = it + 1
        if np.linalg.norm(xi[:3]) < 1e-5 and np.linalg.norm(xi[3:]) < 1e-5 and thr <= max_dist:
            break

    p, q, n = matches(T, max_dist)
    if len(p) < 30:
        return IcpResult(T, math.inf, len(p) / max(len(src), 1), len(p), np.zeros(3), np.zeros((6, 6)), 0,
                         np.zeros(6), iters)
    r, c, L, H, _ = _normal_equations(p, q, n)
    ev, U = np.linalg.eigh(H)
    good = ev > DEGENERATE_EIG
    D = np.diag([L, L, L, 1.0, 1.0, 1.0])
    info = D @ ((U * np.where(good, ev, 0.0)) @ U.T) @ D
    return IcpResult(T, float(np.sqrt(np.mean(r ** 2))), len(p) / len(src), len(p), c, info, int(good.sum()), ev,
                     iters)


def _motion(T: np.ndarray, centroid: np.ndarray) -> tuple[float, float]:
    """(displacement of centroid in metres, rotation angle in degrees) of a world-frame transform."""
    ang = float(np.degrees(np.linalg.norm(Rotation.from_matrix(T[:3, :3]).as_rotvec())))
    return float(np.linalg.norm(apply(T, centroid) - centroid)), ang


def _accept(res: IcpResult, centroid: np.ndarray) -> str | None:
    """None when the registration is good enough for a loop edge, else the reason."""
    if not math.isfinite(res.rmse) or res.rmse >= ACCEPT_RMSE:
        return "rmse"
    if res.overlap <= ACCEPT_OVERLAP:
        return "overlap"
    if res.dof < MIN_LOOP_DOF:
        return "degenerate"
    dist, ang = _motion(res.T, centroid)
    if ang > 15.0 or dist > 1.5:
        return "too_large"
    return None


def _loop_candidates(t_mid: np.ndarray, min_gap_s: float, clouds: list[tuple[np.ndarray, np.ndarray]],
                     skip: set[tuple[int, int]] = frozenset()) -> list[tuple[int, int, float]]:
    """Segment pairs more than min_gap_s apart whose clouds overlap: the best few per segment, then first-last."""
    S = len(clouds)
    has = np.array([len(c[0]) >= 50 for c in clouds])
    lo = np.array([c[0].min(0) if h else np.full(3, np.inf) for c, h in zip(clouds, has)])
    hi = np.array([c[0].max(0) if h else np.full(3, -np.inf) for c, h in zip(clouds, has)])
    trees: dict[int, cKDTree] = {}
    rng = np.random.default_rng(0)
    out = []
    for i in range(S):
        if not has[i]:
            continue
        for j in range(i + 1, S):
            if not has[j] or t_mid[j] - t_mid[i] <= min_gap_s or (i, j) in skip:
                continue
            if (np.minimum(hi[i], hi[j]) - np.maximum(lo[i], lo[j]) < -SCREEN_RADIUS).any():
                continue
            if i not in trees:
                trees[i] = cKDTree(clouds[i][0])
            pj = clouds[j][0]
            if len(pj) > 400:
                pj = pj[rng.choice(len(pj), 400, replace=False)]
            d, _ = trees[i].query(pj, distance_upper_bound=SCREEN_RADIUS)
            ov = float(np.isfinite(d).mean())
            if ov > SCREEN_OVERLAP:
                out.append((i, j, ov))
    out.sort(key=lambda x: -x[2])
    per: dict[int, int] = {}
    chosen = []
    for i, j, ov in out:  # spread registrations over the trajectory instead of one revisited room
        if per.get(i, 0) < LOOPS_PER_SEGMENT or per.get(j, 0) < LOOPS_PER_SEGMENT:
            chosen.append((i, j, ov))
            per[i] = per.get(i, 0) + 1
            per[j] = per.get(j, 0) + 1
    chosen = chosen[:MAX_LOOP_CANDIDATES]
    if (S >= 2 and has[0] and has[S - 1] and (0, S - 1) not in skip
            and not any(i == 0 and j == S - 1 for i, j, _ in chosen)):
        chosen.append((0, S - 1, 0.0))
    return chosen


# --------------------------------------------------------------------------------------------- pose graph

def loop_edge(a: int, b: int, res: IcpResult, Ga: np.ndarray, Gb: np.ndarray) -> LoopEdge:
    ev, U = np.linalg.eigh(res.info / (LOOP_REF_EIG * LOOP_SIGMA ** 2))
    return LoopEdge(a, b, res.T, res.center, np.sqrt(np.maximum(ev, 0.0))[:, None] * U.T, Ga.copy(), Gb.copy())


def solve_pose_graph(anchors: np.ndarray, edges: Sequence[Edge], *, loops: Sequence[LoopEdge] = (),
                     init: np.ndarray | None = None, z_priors: dict[int, tuple[float, float]] | None = None,
                     yaw_priors: dict[int, tuple[float, float]] | None = None) -> np.ndarray:
    """Corrections (S, 4) = (yaw, tx, ty, tz) with X_s = C_s @ anchors[s] that best satisfy the edges.

    X_s = C_s @ anchors[s]; loop edges carry the poses they were registered at. z_priors map a node to (anchor z, sigma),
    yaw_priors to (yaw correction, sigma). Node 0 keeps its x and y; its yaw and z are fixed unless priors on
    that quantity exist.
    """
    S = len(anchors)
    params0 = np.zeros((S, 4)) if init is None else np.array(init, float)
    z_priors = z_priors or {}
    yaw_priors = yaw_priors or {}
    if S < 2 or not (edges or loops):
        return params0
    free = np.ones((S, 4), bool)
    free[0, 1:3] = False
    if not yaw_priors:
        free[0, 0] = False
    if not z_priors:
        free[0, 3] = False
    fidx = np.flatnonzero(free.ravel())
    col = -np.ones(4 * S, np.int64)
    col[fidx] = np.arange(len(fidx))

    ea = np.array([e.a for e in edges], np.int64)
    eb = np.array([e.b for e in edges], np.int64)
    Zinv = _inv(np.stack([e.Z for e in edges])) if edges else np.zeros((0, 4, 4))
    st = np.array([e.sigma_t for e in edges])[:, None]
    sr = np.array([e.sigma_r for e in edges])[:, None]
    la = np.array([e.a for e in loops], np.int64)
    lb = np.array([e.b for e in loops], np.int64)
    if loops:
        Pa = np.stack([e.Ga for e in loops])
        Pb_inv = _inv(np.stack([e.Gb for e in loops]))
        T_inv = _inv(np.stack([e.T for e in loops]))
        lc = np.stack([e.center for e in loops])
        lS = np.stack([e.sqrt_info for e in loops])
    zn = np.array(sorted(z_priors), np.int64)
    zt = np.array([z_priors[i][0] for i in zn])
    zs = np.array([z_priors[i][1] for i in zn])
    yn = np.array(sorted(yaw_priors), np.int64)
    yt = np.array([yaw_priors[i][0] for i in yn])
    ys = np.array([yaw_priors[i][1] for i in yn])

    def unpack(x: np.ndarray) -> np.ndarray:
        p = params0.ravel().copy()
        p[fidx] = x
        return p.reshape(S, 4)

    def fun(x: np.ndarray) -> np.ndarray:
        p = unpack(x)
        X = _corrections(p) @ anchors
        res = []
        if len(ea):
            E = Zinv @ _inv(X[ea]) @ X[eb]
            rv = Rotation.from_matrix(E[:, :3, :3]).as_rotvec()
            res += [(rv / sr).ravel(), (E[:, :3, 3] / st).ravel()]
        if len(la):
            # motion, in the registration world, from where the registration put b's cloud to where X puts it
            D = Pa @ _inv(X[la]) @ X[lb] @ Pb_inv @ T_inv
            rv = Rotation.from_matrix(D[:, :3, :3]).as_rotvec()
            shift = np.einsum("nij,nj->ni", D[:, :3, :3], lc) + D[:, :3, 3] - lc
            res.append(np.einsum("nij,nj->ni", lS, np.hstack([rv, shift])).ravel())
        if len(zn):
            res.append((X[zn, 2, 3] - zt) / zs)
        if len(yn):
            res.append(_wrap(p[yn, 0] - yt) / ys)
        return np.concatenate(res)

    n_e, n_l = len(ea), len(la)
    n_res = 6 * n_e + 6 * n_l + len(zn) + len(yn)
    sp = lil_matrix((n_res, len(fidx)), dtype=np.int8)
    for k in range(n_e):
        for node in (ea[k], eb[k]):
            for q in range(4):
                c = col[4 * node + q]
                if c >= 0:
                    sp[3 * k:3 * k + 3, c] = 1
                    sp[3 * n_e + 3 * k:3 * n_e + 3 * k + 3, c] = 1
    for k in range(n_l):
        for node in (la[k], lb[k]):
            for q in range(4):
                c = col[4 * node + q]
                if c >= 0:
                    sp[6 * n_e + 6 * k:6 * n_e + 6 * k + 6, c] = 1
    off = 6 * (n_e + n_l)
    for m, node in enumerate(zn):
        if col[4 * node + 3] >= 0:
            sp[off + m, col[4 * node + 3]] = 1
    for m, node in enumerate(yn):
        if col[4 * node] >= 0:
            sp[off + len(zn) + m, col[4 * node]] = 1
    x0 = params0.ravel()[fidx]
    if len(x0) == 0:
        return params0
    sol = least_squares(fun, x0, jac_sparsity=sp.tocsr(), method="trf", loss="huber", f_scale=3.0,
                        x_scale="jac", max_nfev=200)
    return unpack(sol.x)


def interpolate_corrections(t_anchor: np.ndarray, C: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-frame corrections at times t: slerp of rotations, linear translations, held constant at the ends."""
    if len(C) == 1:
        return np.repeat(C, len(t), axis=0)
    t_anchor = np.maximum.accumulate(np.asarray(t_anchor, float)) + 1e-9 * np.arange(len(t_anchor))
    tq = np.clip(t, t_anchor[0], t_anchor[-1])
    R = Slerp(t_anchor, Rotation.from_matrix(C[:, :3, :3]))(tq).as_matrix()
    out = np.tile(np.eye(4), (len(t), 1, 1))
    out[:, :3, :3] = R
    for a in range(3):
        out[:, a, 3] = np.interp(tq, t_anchor, C[:, a, 3])
    return out


# ----------------------------------------------------------------------------------------------- anchoring

def _segment_floor(p: np.ndarray, n: np.ndarray, w: np.ndarray, cam_z: float
                   ) -> tuple[float, np.ndarray | None, np.ndarray, int] | None:
    """(height, normal or None when the patch is too small for a tilt, centroid, points) of a segment's floor.

    The floor is the lowest up-facing plane with at least 30% of the strongest peak's weight between
    FLOOR_BELOW metres under the camera, which follows the floor through vertical drift.
    """
    zr = p[:, 2] - cam_z
    m = (n[:, 2] > math.cos(math.radians(15.0))) & (zr > -FLOOR_BELOW[1]) & (zr < -FLOOR_BELOW[0])
    if m.sum() < 40:
        return None
    z = p[m, 2]
    lo = cam_z - FLOOR_BELOW[1]
    nb = round((FLOOR_BELOW[1] - FLOOR_BELOW[0]) / 0.01)
    hist, _ = np.histogram(z, bins=nb, range=(lo, lo + nb * 0.01), weights=w[m])
    hist = np.convolve(hist, [1, 2, 3, 2, 1], mode="same")
    pad = np.r_[-1.0, hist, -1.0]
    peaks = np.flatnonzero((hist >= pad[:-2]) & (hist >= pad[2:]) & (hist >= 0.3 * hist.max()) & (hist > 0))
    if len(peaks) == 0:
        return None
    h0 = lo + (peaks[0] + 0.5) * 0.01
    near = np.abs(z - h0) < 0.04
    h = float(np.median(z[near]))
    q = p[m][np.abs(z - h) < 0.03]
    c = q.mean(0) if len(q) else p[m].mean(0)
    if len(q) < 80:
        return h, None, c, int(near.sum())
    ev, evec = np.linalg.eigh(np.cov((q - c).T))
    if math.sqrt(max(ev[1], 0.0)) < 0.25:  # the patch must span about a metre in two directions
        return h, None, c, int(near.sum())
    nrm = evec[:, 0] * np.sign(evec[2, 0] or 1.0)
    return h, nrm, c, int(near.sum())


def _weighted_median(v: np.ndarray, w: np.ndarray) -> float:
    o = np.argsort(v)
    cw = np.cumsum(w[o])
    return float(v[o][np.searchsorted(cw, 0.5 * cw[-1])])


def _manhattan(n: np.ndarray, w: np.ndarray) -> tuple[float, float, float]:
    """(angle mod 90 deg in radians, concentration in [0, 1], wall weight) of vertical-surface normals."""
    m = np.abs(n[:, 2]) < math.sin(math.radians(20.0))
    if m.sum() < 20:
        return 0.0, 0.0, 0.0
    ang = np.arctan2(n[m, 1], n[m, 0])
    ww = w[m] * (1.0 - n[m, 2] ** 2)
    z = (ww * np.exp(4j * ang)).sum()
    tot = float(ww.sum())
    return float((np.angle(z) / 4.0) % (np.pi / 2)), float(abs(z) / max(tot, 1e-12)), tot


def _spread(values: list[float]) -> float | None:
    return round(float(max(values) - min(values)), 4) if len(values) >= 2 else None


# ---------------------------------------------------------------------------------------------- main entry

def correct_lidar_poses(frames: Sequence[KeyFrame], poses: np.ndarray, depth_reader: DepthReader, *,
                        enabled: bool = True, loop_closure: bool = True, plane_anchoring: bool = True,
                        manhattan_anchoring: bool = True, segment_s: float = SEGMENT_S,
                        min_loop_gap_s: float = LOOP_MIN_GAP_S) -> tuple[np.ndarray, dict]:
    """Drift-corrected keyframe poses and a JSON-ready record of what was done.

    frames[k] and poses[k] (camera-to-world, z-up world) describe keyframe k; depth_reader(k) returns its cloud
    in camera coordinates or None. With enabled=False the poses come back unchanged with {"enabled": False}.
    """
    poses = np.asarray(poses, float)
    if not enabled:
        return poses.copy(), {"enabled": False}
    rec: dict = {"enabled": True, "segments": 0, "loop_closures_tried": 0, "loop_closures_accepted": 0,
                 "max_translation_correction_m": 0.0, "max_yaw_correction_deg": 0.0,
                 "floor_z_spread_before_m": None, "floor_z_spread_after_m": None,
                 "stages": {"loop_closure": loop_closure, "plane_anchoring": plane_anchoring,
                            "manhattan_anchoring": manhattan_anchoring},
                 "loop_closures": [], "flags": []}
    K = len(frames)
    if K < 2:
        rec["flags"].append("too_few_keyframes")
        return poses.copy(), rec
    t = np.array([f.timestamp for f in frames], float)
    jumps = _pose_jumps(t, poses)
    if jumps.any():
        rec["flags"].append(f"pose_jumps:{int(jumps.sum())}")
    segs = build_segments(t, poses, depth_reader, segment_s=segment_s, jumps=jumps)
    S = len(segs)
    rec["segments"] = S
    a_idx = np.array([s.anchor for s in segs])
    anchors = poses[a_idx]
    t_anchor = t[a_idx]

    edges: list[Edge] = []
    for s in range(S - 1):
        k0, k1 = a_idx[s], a_idx[s + 1]
        path = float(np.linalg.norm(np.diff(poses[k0:k1 + 1, :3, 3], axis=0), axis=1).sum())
        if jumps[k0 + 1:k1 + 1].any():
            sig_t, sig_r = 0.5, math.radians(10.0)
        else:
            sig_t, sig_r = 0.01 + 0.01 * path, math.radians(0.15 + 0.1 * path)
        edges.append(Edge(s, s + 1, _inv(anchors[s]) @ anchors[s + 1], sig_t, sig_r))

    def floors_of(X: np.ndarray) -> tuple[list[tuple[int, float, np.ndarray | None, np.ndarray]], float | None]:
        """Per-segment floors that agree with their neighbours in time, and their common level."""
        found = []
        for s, seg in enumerate(segs):
            p, n = _world(seg, X[s])
            f = _segment_floor(p, n, seg.weights, float(X[s, 2, 3]))
            if f is not None:
                found.append((s, *f))
        if not found:
            return [], None
        hs = np.array([f[1] for f in found])
        idx = np.array([f[0] for f in found])
        keep = [f for f, h, s in zip(found, hs, idx)
                if abs(h - np.median(hs[np.abs(idx - s) <= 3])) <= FLOOR_OUTLIER]
        if not keep:
            return [], None
        level = _weighted_median(np.array([f[1] for f in keep]), np.array([f[4] for f in keep], float))
        return [f[:4] for f in keep if abs(f[1] - level) < FLOOR_ANCHOR_MAX], level

    floors0, _ = floors_of(anchors)
    rec["floor_z_spread_before_m"] = _spread([f[1] for f in floors0])

    params = np.zeros((S, 4))
    loops: list[LoopEdge] = []
    if loop_closure and S >= 2:
        done: set[tuple[int, int]] = set()
        for rnd in range(LOOP_ROUNDS):
            X = _corrections(params) @ anchors
            clouds = [_world(sg, X[s]) for s, sg in enumerate(segs)]
            cands = _loop_candidates(t_anchor, min_loop_gap_s, clouds, skip=done)
            n_before = len(loops)
            for i, j, ov in cands:
                res = icp_point_to_plane(clouds[j][0], clouds[j][1], clouds[i][0], clouds[i][1])
                centroid = clouds[j][0].mean(0)
                why = _accept(res, centroid)
                rec["loop_closures_tried"] += 1
                if len(rec["loop_closures"]) < 60:
                    dist, _ = _motion(res.T, centroid)
                    rec["loop_closures"].append({
                        "round": rnd + 1, "segments": [int(i), int(j)], "screen_overlap": round(ov, 3),
                        "rmse_m": round(res.rmse, 4) if math.isfinite(res.rmse) else None,
                        "overlap": round(res.overlap, 3), "shift_m": round(dist, 4),
                        "yaw_deg": round(float(np.degrees(_yaw_of(res.T[:3, :3]))), 3), "dof": res.dof,
                        "accepted": why is None, "reason": why})
                if why is None:
                    loops.append(loop_edge(i, j, res, X[i], X[j]))
                    done.add((i, j))
            if len(loops) == n_before:
                break
            params = solve_pose_graph(anchors, edges, loops=loops, init=params)
        rec["loop_closures_accepted"] = len(loops)

    X = _corrections(params) @ anchors
    z_priors: dict[int, tuple[float, float]] = {}
    yaw_priors: dict[int, tuple[float, float]] = {}
    h_g = None
    if plane_anchoring:
        floors, h_g = floors_of(X)
        if h_g is None:
            rec["flags"].append("no_floor_found")
        else:
            for s, h, _, _ in floors:
                z_priors[s] = (float(X[s, 2, 3] + h_g - h), 0.01)
            rec["plane_anchoring"] = {"floor_z_m": round(h_g, 4), "segments_anchored": len(z_priors)}
    if manhattan_anchoring:
        allp = [_world(sg, X[s]) for s, sg in enumerate(segs)]
        n_all = np.concatenate([n for _, n in allp]) if allp else np.zeros((0, 3))
        w_all = np.concatenate([sg.weights for sg in segs]) if segs else np.zeros(0)
        th_g, conc_g, _ = _manhattan(n_all, w_all)
        snapped = []
        if conc_g > 0.4:
            for s, sg in enumerate(segs):
                th_s, conc_s, tot = _manhattan(allp[s][1], sg.weights)
                if conc_s < 0.5 or tot < 20.0:
                    continue
                dev = float(_wrap(th_s - th_g, np.pi / 2))
                if abs(dev) < math.radians(MANHATTAN_SNAP_DEG):
                    yaw_priors[s] = (float(params[s, 0] - dev), math.radians(0.5))
                    snapped.append(s)
        else:
            rec["flags"].append("manhattan_weak")
        rec["manhattan_anchoring"] = {"global_angle_deg": round(math.degrees(th_g), 3),
                                      "concentration": round(conc_g, 3), "segments_snapped": len(snapped)}
    if z_priors or yaw_priors:
        params = solve_pose_graph(anchors, edges, loops=loops, init=params, z_priors=z_priors,
                                  yaw_priors=yaw_priors)

    C = _corrections(params)
    if plane_anchoring and h_g is not None:
        X = C @ anchors
        tilt = np.zeros((S, 3))
        pivots = X[:, :3, 3].copy()
        have = np.zeros(S, bool)
        for s, h, nrm, c in floors_of(X)[0]:
            pivots[s] = c
            if nrm is not None:
                tilt[s] = Rotation.from_matrix(rotation_between(nrm, _UP)).as_rotvec()
                have[s] = True
        if have.any():
            sm = np.zeros_like(tilt)
            for s in range(S):
                nb = [q for q in (s - 1, s, s + 1) if 0 <= q < S and have[q]]
                if nb:
                    sm[s] = tilt[nb].mean(0)
            ang = np.degrees(np.linalg.norm(sm, axis=1))
            apply_tilt = (ang > TILT_DEADBAND_DEG) & (ang < TILT_MAX_DEG)
            for s in np.flatnonzero(apply_tilt):
                Tt = np.eye(4)
                Tt[:3, :3] = rotvec_to_R(sm[s])
                Tt[:3, 3] = pivots[s] - Tt[:3, :3] @ pivots[s]
                C[s] = Tt @ C[s]
            rec["plane_anchoring"]["segments_tilt_corrected"] = int(apply_tilt.sum())
            rec["plane_anchoring"]["max_tilt_deg"] = round(float(ang.max()), 3)

    Ck = interpolate_corrections(t_anchor, C, t)
    corrected = Ck @ poses
    rec["max_translation_correction_m"] = round(float(np.linalg.norm(corrected[:, :3, 3] - poses[:, :3, 3],
                                                                     axis=1).max()), 4)
    rec["max_yaw_correction_deg"] = round(float(np.degrees(np.abs(_yaw_of(Ck[:, :3, :3]))).max()), 3)
    rec["floor_z_spread_after_m"] = _spread([f[1] for f in floors_of(C @ anchors)[0]])
    log.info("drift: %d segments, loop closures %d/%d, max correction %.3f m / %.2f deg, floor spread %s -> %s",
             S, rec["loop_closures_accepted"], rec["loop_closures_tried"], rec["max_translation_correction_m"],
             rec["max_yaw_correction_deg"], rec["floor_z_spread_before_m"], rec["floor_z_spread_after_m"])
    return corrected, rec
