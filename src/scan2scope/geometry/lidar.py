"""LiDAR tier geometry: a Stray Scanner capture becomes a gravity-aligned Scene (z up)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np

from scan2scope.geometry.drift import FrameCloud, KeyFrame, VoxelAccumulator, correct_lidar_poses
from scan2scope.geometry.pointmaps import backproject_depth, normals_from_pointmap
from scan2scope.ingest.stray import StrayCapture, load_stray, scale_intrinsics
from scan2scope.types import CameraView, Scene

log = logging.getLogger("scan2scope.geometry.lidar")

# ARKit world (x, y up, z) -> z-up world (x, -z, y): a +90 degree rotation about x, so det = +1
C_ZUP = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
DEPTH_RANGE = (0.2, 4.5)
CONF_WEIGHTS = np.array([0.0, 0.3, 1.0], np.float32)  # per ARKit confidence level 0, 1, 2
NO_CONF_WEIGHT = 0.3  # recordings without confidence maps
EDGE_JUMP = 0.1  # a relative depth step this large to a 4-neighbour marks a depth edge (flying pixels)
VOXEL = 0.015
KEYFRAME_TRANS = 0.05
KEYFRAME_ROT_DEG = 5.0
MAX_KEYFRAMES = 400
VIEW_SPACING_S = 1.0
MAX_VIEWS = 60
SCALE_LOG_SIGMA = 0.003
NORMAL_STEP = 2
FUSE_STRIDE = 2
DRIFT_STRIDE = 4


def arkit_to_zup(T_wc: np.ndarray) -> np.ndarray:
    """Camera-to-world poses from ARKit's y-up world to the z-up world; camera axes are unchanged."""
    return C_ZUP @ T_wc


def _rot_angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(Ra.T @ Rb) - 1.0) / 2.0
    return math.degrees(math.acos(min(1.0, max(-1.0, c))))


def select_keyframes(T_wc: np.ndarray, usable: np.ndarray, *, trans: float = KEYFRAME_TRANS,
                     rot_deg: float = KEYFRAME_ROT_DEG, max_keyframes: int = MAX_KEYFRAMES) -> np.ndarray:
    """Row indices where the camera moved trans metres or turned rot_deg since the last keyframe.

    The first usable frame is always kept. When there would be more than max_keyframes, both thresholds grow
    until they fit.
    """
    idx = np.flatnonzero(usable)
    if len(idx) == 0:
        return idx

    def pick(tr: float, rot: float) -> np.ndarray:
        keep = [idx[0]]
        last = T_wc[idx[0]]
        for i in idx[1:]:
            if (np.linalg.norm(T_wc[i, :3, 3] - last[:3, 3]) >= tr
                    or _rot_angle_deg(last[:3, :3], T_wc[i, :3, :3]) >= rot):
                keep.append(i)
                last = T_wc[i]
        return np.array(keep, np.int64)

    kf = pick(trans, rot_deg)
    for _ in range(20):
        if len(kf) <= max_keyframes:
            return kf
        f = max(1.05, (len(kf) / max_keyframes) ** 0.75)
        trans, rot_deg = trans * f, rot_deg * f
        kf = pick(trans, rot_deg)
    sel = np.unique(np.linspace(0, len(kf) - 1, max_keyframes).round().astype(np.int64))
    return kf[sel]


def depth_mask(depth: np.ndarray, conf: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(valid, weight) per pixel: depth in DEPTH_RANGE, confidence 2 -> 1.0, 1 -> 0.3, 0 dropped, no depth edge."""
    valid = (depth >= DEPTH_RANGE[0]) & (depth <= DEPTH_RANGE[1])
    if conf is not None and conf.shape == depth.shape:
        w = CONF_WEIGHTS[np.clip(conf, 0, 2)]
    else:
        w = np.full(depth.shape, NO_CONF_WEIGHT, np.float32)
    valid &= w > 0
    edge = np.zeros(depth.shape, bool)
    for ax in (0, 1):
        a = np.moveaxis(depth, ax, 0)
        lo = np.minimum(a[1:], a[:-1])
        jump = (np.abs(a[1:] - a[:-1]) > EDGE_JUMP * lo) & (lo > 0)
        e = np.moveaxis(edge, ax, 0)
        e[1:] |= jump
        e[:-1] |= jump
    valid &= ~edge
    return valid, np.where(valid, w, 0.0).astype(np.float32)


def _stride_mask(shape: tuple[int, int], stride: int) -> np.ndarray:
    m = np.zeros(shape, bool)
    m[::stride, ::stride] = True
    return m


def camera_cloud(depth: np.ndarray, conf: np.ndarray | None, K_d: np.ndarray, stride: int = DRIFT_STRIDE
                 ) -> FrameCloud:
    """Valid depth pixels of one frame as points, normals and weights in its camera frame, every stride-th pixel."""
    valid, w = depth_mask(depth, conf)
    pm = backproject_depth(depth, K_d, np.eye(4))
    n, ok = normals_from_pointmap(pm, valid, cam_center=np.zeros(3), step=NORMAL_STEP)
    sel = valid & ok & _stride_mask(depth.shape, stride)
    return FrameCloud(pm[sel], n[sel].astype(np.float64), w[sel].astype(np.float64))


def _motion_score(T_wc: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Angular speed (rad/s) plus half the linear speed (m/s) per frame, a proxy for motion blur."""
    n = len(t)
    if n < 2:
        return np.zeros(n)
    i0 = np.clip(np.arange(n) - 1, 0, n - 1)
    i1 = np.clip(np.arange(n) + 1, 0, n - 1)
    dt = np.maximum(t[i1] - t[i0], 1e-3)
    v = np.linalg.norm(T_wc[i1, :3, 3] - T_wc[i0, :3, 3], axis=1) / dt
    rel = np.einsum("nji,njk->nik", T_wc[i0, :3, :3], T_wc[i1, :3, :3])
    ang = np.arccos(np.clip((np.trace(rel, axis1=1, axis2=2) - 1) / 2, -1, 1)) / dt
    return ang + 0.5 * v


def select_views(t: np.ndarray, motion: np.ndarray, valid_fraction: np.ndarray, *,
                 spacing: float = VIEW_SPACING_S, max_views: int = MAX_VIEWS) -> np.ndarray:
    """Positions (into the keyframe arrays) of about one sharp, well-measured keyframe per spacing seconds.

    Keyframes with an infinite motion score are never chosen.
    """
    if len(t) == 0:
        return np.zeros(0, np.int64)
    dur = float(t.max() - t.min())
    n_bins = int(max(1, min(max_views, math.floor(dur / spacing) + 1)))
    edges = np.linspace(t.min(), t.max() + 1e-6, n_bins + 1)
    out = []
    for b in range(n_bins):
        cand = np.flatnonzero((t >= edges[b]) & (t < edges[b + 1]))
        if len(cand) == 0:
            continue
        cand = cand[np.isfinite(motion[cand])]
        if len(cand) == 0:
            continue
        good = cand[valid_fraction[cand] > 0.2]
        cand = good if len(good) else cand
        out.append(int(cand[np.argmin(motion[cand])]))
    return np.unique(np.array(out, np.int64))


def build_scene(root: str | Path, work_dir: str | Path, *, drift_correction: bool = True) -> Scene:
    """Scene (tier "lidar") from a Stray Scanner dataset folder: fused LiDAR points, normals and RGB views."""
    work_dir = Path(work_dir)
    cap = load_stray(root)
    flags = list(cap.flags)
    T = arkit_to_zup(cap.T_wc)
    usable = cap.has_depth & np.isfinite(T).all(axis=(1, 2))
    if not usable.any():
        raise ValueError(f"{cap.root}: no frame has both a pose and a depth map")
    rgb_size = cap.rgb_size
    depth_size = cap.depth_size or (256, 192)
    if abs(rgb_size[0] / rgb_size[1] - depth_size[0] / depth_size[1]) > 0.01:
        # depth intrinsics are the RGB ones rescaled, which assumes both images share a field of view
        flags.append(f"rgb_depth_aspect_mismatch:{rgb_size[0]}x{rgb_size[1]}/{depth_size[0]}x{depth_size[1]}")
    K_d = scale_intrinsics(cap.K, rgb_size, depth_size)
    kf = select_keyframes(T, usable)
    t_kf = cap.timestamps[kf]
    h_d, w_d = depth_size[1], depth_size[0]
    bad_shape = 0

    def load(i: int) -> tuple[np.ndarray, np.ndarray | None] | None:
        nonlocal bad_shape
        d = cap.read_depth(int(i))
        if d is None:
            return None
        if d.shape != (h_d, w_d):
            bad_shape += 1
            return None
        return d, cap.read_conf(int(i))

    def drift_reader(k: int) -> FrameCloud | None:
        dc = load(kf[k])
        return None if dc is None else camera_cloud(dc[0], dc[1], K_d[kf[k]])

    frames = [KeyFrame(int(cap.frame_ids[i]), float(cap.timestamps[i])) for i in kf]
    poses, drift = correct_lidar_poses(frames, T[kf], drift_reader, enabled=drift_correction)
    flags += [f"drift_{f}" for f in drift.get("flags", [])]

    acc = VoxelAccumulator(VOXEL)
    fuse_mask = _stride_mask((h_d, w_d), FUSE_STRIDE)
    valid_frac = np.zeros(len(kf))
    conf_sum, conf_n, n_unread = 0.0, 0, 0
    for k, i in enumerate(kf):
        dc = load(i)
        if dc is None:
            n_unread += 1
            continue
        depth, conf = dc
        valid, w = depth_mask(depth, conf)
        pm = backproject_depth(depth, K_d[i], poses[k])
        n, ok = normals_from_pointmap(pm, valid, cam_center=poses[k][:3, 3], step=NORMAL_STEP)
        sel = valid & ok & fuse_mask
        acc.add(pm[sel], n[sel].astype(np.float64), w[sel], tag=k)
        valid_frac[k] = float(valid.mean())
        has = depth > 0
        if conf is not None and conf.shape == depth.shape:
            conf_sum += float(conf[has].sum())
            conf_n += int(has.sum())
    points, normals, weights, tags = acc.result()
    if n_unread:
        flags.append(f"keyframes_unreadable:{n_unread}/{len(kf)}")
    if bad_shape:
        flags.append(f"depth_shape_mismatch:{bad_shape}")
    if len(points) == 0:
        flags.append("no_valid_depth")

    motion = _motion_score(T, cap.timestamps)[kf]
    if cap.n_video_frames:
        motion[cap.frame_ids[kf] >= cap.n_video_frames] = np.inf  # beyond the end of a short rgb.mp4
    vpos = select_views(t_kf, motion, valid_frac)
    views = _make_views(cap, kf, vpos, poses, K_d, depth_size, work_dir / "lidar_views", load)
    if any(v.image_path is None for v in views):
        flags.append(f"views_without_image:{sum(v.image_path is None for v in views)}")
    for f in cap.flags:
        if f not in flags:
            flags.append(f)
    if views:
        t_views = np.array([v.timestamp for v in views])
        view_of_kf = np.argmin(np.abs(t_kf[:, None] - t_views[None, :]), axis=1)
        view_index = view_of_kf[tags] if len(tags) else np.zeros(0, np.int64)
    else:
        view_index = np.zeros(len(points), np.int64)

    quality = {
        "n_frames": len(cap),
        "n_keyframes": len(kf),
        "duration_s": round(cap.duration_s, 3),
        # mean ARKit level / 2, in [0, 1]; 0.5 (medium) when the capture has no confidence maps (flagged)
        "mean_conf": round(conf_sum / conf_n / 2.0, 4) if conf_n else 0.5,
        "valid_depth_fraction": round(float(valid_frac.mean()), 4),
    }
    meta = {
        "drift": drift,
        "quality": quality,
        "flags": flags,
        "capture": {"format": cap.format, "fps": round(cap.fps, 3), "rgb_size": list(rgb_size),
                    "depth_size": list(depth_size), "n_video_frames": cap.n_video_frames,
                    "imu_samples": 0 if cap.imu is None else len(cap.imu), "imu_accel_unit": cap.imu_accel_unit},
        "keyframe_ids": [int(cap.frame_ids[i]) for i in kf],
        "world": "z-up; ARKit (x, y, z) -> (x, -z, y)",
    }
    log.info("lidar scene: %d points from %d keyframes, %d views, valid depth %.0f%%", len(points), len(kf),
             len(views), 100 * quality["valid_depth_fraction"])
    return Scene(tier="lidar", views=views, points=points, normals=normals, weights=weights,
                 view_index=view_index.astype(np.int64), scale_log_sigma=SCALE_LOG_SIGMA, meta=meta)


# columns: the camera axes after np.rot90(image, k), in the original camera coordinates
_TURN_AXES = (np.eye(3), np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]), np.diag([-1.0, -1.0, 1.0]),
              np.array([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))


def upright_turns(T_wc: np.ndarray) -> int:
    """Quarter turns (np.rot90 k) that put world down closest to image down; 0 when looking nearly straight down."""
    down = T_wc[:3, :3].T @ np.array([0.0, 0.0, -1.0])  # world down in camera coordinates
    if math.hypot(down[0], down[1]) < 0.3:
        return 0
    return int(np.argmax([A[:, 1] @ down for A in _TURN_AXES]))


def turn_camera(k: int, K: np.ndarray, T_wc: np.ndarray, size: tuple[int, int]
                ) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """(K, T_wc, (width, height)) of an image of `size` after np.rot90(image, k)."""
    k %= 4
    w, h = size
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    Kn = np.eye(3)
    if k == 0:
        return K.copy(), T_wc.copy(), size
    if k == 1:
        Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2], new = fy, fx, cy, w - 1 - cx, (h, w)
    elif k == 2:
        Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2], new = fx, fy, w - 1 - cx, h - 1 - cy, (w, h)
    else:
        Kn[0, 0], Kn[1, 1], Kn[0, 2], Kn[1, 2], new = fy, fx, h - 1 - cy, cx, (h, w)
    T = T_wc.copy()
    T[:3, :3] = T_wc[:3, :3] @ _TURN_AXES[k]
    return Kn, T, new


def _make_views(cap: StrayCapture, kf: np.ndarray, vpos: np.ndarray, poses: np.ndarray, K_d: np.ndarray,
                depth_size: tuple[int, int], out_dir: Path, load) -> list[CameraView]:
    """Views with upright images: frames, K, poses and point maps turned together by quarter turns."""
    rows = kf[vpos]
    turns = [upright_turns(poses[k]) for k in vpos]
    paths = cap.extract_rgb([int(cap.frame_ids[i]) for i in rows], out_dir, turns=turns)
    views = []
    for k, i, path, q in zip(vpos, rows, paths, turns):
        dc = load(i)
        pm = valid = conf = None
        if dc is not None:
            valid, conf = depth_mask(*dc)
            pm = backproject_depth(dc[0], K_d[i], poses[k]).astype(np.float32)
            pm, valid, conf = (np.ascontiguousarray(np.rot90(a, q)) for a in (pm, valid, conf))
        K, T_wc, (w, h) = turn_camera(q, cap.K[i], poses[k], cap.rgb_size)
        Kd, _, _ = turn_camera(q, K_d[i], poses[k], depth_size)
        fid = int(cap.frame_ids[i])
        views.append(CameraView(
            id=f"F{fid:06d}", image_path=path, width=int(w), height=int(h), K=K, T_wc=T_wc, pointmap=pm,
            valid=valid, conf=conf, timestamp=float(cap.timestamps[i]),
            meta={"frame_id": fid, "keyframe": int(k), "K_depth": Kd, "quarter_turns": q}))
    return views
