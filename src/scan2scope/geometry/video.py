"""Video tier: frames sampled from one walkthrough clip, MapAnything on overlapping chunks chained with Sim(3)
on the shared frames, then loop closure, gravity, floor and Manhattan anchoring, fused into one Scene.

A link between two chunks is refused when the two runs put the shared cameras far apart (align_runs); the loop
registration can stand in for one refused link, and chunks that stay unconnected are dropped with a
video_segment_dropped flag. The measured disagreement widens the capture's scale interval.

Frame sampling here is a thin stand-in for scan2scope.ingest.video.sample_frames; integration can switch to the
ingest version once it lands.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from scan2scope.geometry.chunk_align import (
    Edge,
    Prior,
    Sim3Fit,
    optimize_pose_graph,
    robust_sim3,
    sim3_error,
)
from scan2scope.geometry.mapanything_backend import (
    MAX_VIEWS,
    MapAnythingRunner,
    OutputCacheLike,
    ViewPrediction,
    focal_check,
)
from scan2scope.geometry.photo import file_sha256, fuse_points, gravity_alignment
from scan2scope.geometry.pointmaps import normals_from_pointmap
from scan2scope.geometry.se3 import apply, decompose_sim3, invert, make_T, rot_z, rotation_between
from scan2scope.types import CameraView, Scene

log = logging.getLogger("scan2scope.geometry")

VIDEO_SUFFIXES = {".mov", ".mp4", ".m4v", ".avi", ".mkv", ".webm", ".3gp", ".hevc"}
TARGET_FPS = 1.5
MAX_FRAMES = 120
FRAME_MAX_SIDE = 1280
CHUNK_SIZE = 24
OVERLAP = 5
LOOP_FRAMES = 6
KEYFRAME_EVERY = 3
VOXEL = 0.02
MAX_POINTS = 1_500_000
FUSE_STRIDE = 2
ANALYSIS_SAMPLES = 3000  # pixels per frame used for floor, wall and gravity statistics
# MapAnything's metric error is mostly per scene: the chunks of one clip share it, so more chunks do not average
# it out (home video_1: chunks within 2.5% of each other, clip 7.3% small; ARKitScenes kitchen: 1.7% apart,
# depth 6.5% small against LiDAR). The base is the error of one clip, not of one chunk.
SCALE_SIGMA_BASE = 0.08
SINGLE_RUN_SCALE_SIGMA = 0.08  # one MapAnything metric estimate
BLUR_REL = 0.35
TILT_MAX_DEG = 3.0
YAW_MAX_DEG = 5.0
FLOOR_MAX_DZ = 0.2
MANHATTAN_MIN_CONC = 0.3  # |sum w exp(4i angle)| / sum w; 1 for perfectly perpendicular walls
# A wrong loop closure bends the whole walk, so it must be consistent with the chain it corrects.
LOOP_MIN_OVERLAP = 0.2
LOOP_MAX_DRIFT = 0.10  # loop error as a share of the walked path
LOOP_MAX_ROT_DEG = 20.0
MIN_WEIGHT = 0.1
CORR_PER_FRAME = 3000
# Chunk-to-chunk links. A joint fit over the shared frames with at least ALIGN_GOOD_INLIERS is taken as is;
# below that the runs disagree about the motion inside the overlap and the link falls back to per-frame fits.
ALIGN_GOOD_INLIERS = 0.5
FRAME_MIN_INLIERS = 0.6  # a shared frame registers on its own (same pixels, both runs)
FRAME_AGREE = 0.04  # two per-frame transforms agree when they move points by under this share of the depth
EDGE_SIGMA_FLOOR = 0.01
SINGLE_FRAME_SIGMA = 0.03
# Two chunk runs that put the shared cameras further apart than this share of the scene depth are not chained.
# Consistent links on the real captures measured 0.007 to 0.07; broken ones 0.13 to 5.
LINK_MAX_DISAGREE = 0.10
GEO_SIGMA_CAP = 0.5  # log units; a capture whose runs disagree more than this has no usable metric geometry


@dataclass
class Frame:
    index: int  # position in the sampled sequence
    source_index: int  # frame number in the decoded stream
    timestamp: float  # seconds from the start of the clip
    path: Path  # upright frame image
    sharpness: float = float("nan")


@dataclass
class ChunkRun:
    start: int
    end: int
    preds: list[ViewPrediction]
    _normals: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def pred(self, f: int) -> ViewPrediction:
        return self.preds[f - self.start]

    def normals(self, f: int) -> tuple[np.ndarray, np.ndarray]:
        """Normals of frame f in the chunk frame and their validity, computed on first use."""
        if f not in self._normals:
            p = self.pred(f)
            n, ok = normals_from_pointmap(p.pts3d, p.mask, p.T_wc[:3, 3], step=2)
            self._normals[f] = (n.astype(np.float16), ok)
        return self._normals[f]


# ----------------------------------------------------------------------------------------------- sampling


def _sharpness(frame) -> float:
    import cv2

    w = 320
    h = max(2, round(w * frame.height / max(frame.width, 1)))
    g = frame.reformat(width=w, height=h, format="gray").to_ndarray()
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def _frame_rgb(frame, max_side: int) -> np.ndarray:
    rot = getattr(frame, "rotation", 0) or 0
    k = round(rot / 90.0) % 4
    w, h = frame.width, frame.height
    f = min(1.0, max_side / max(w, h))
    w2, h2 = max(2, round(w * f)), max(2, round(h * f))
    if (w2, h2) == (w, h):
        arr = frame.to_ndarray(format="rgb24")
    else:
        arr = frame.reformat(width=w2, height=h2, format="rgb24", interpolation="AREA").to_ndarray()
    return np.ascontiguousarray(np.rot90(arr, k)) if k else arr


def probe_video(video_path: Path) -> dict[str, Any]:
    import av

    with av.open(str(video_path)) as c:
        s = c.streams.video[0]
        dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else 0.0)
        rate = float(s.average_rate) if s.average_rate else 0.0
        return {"duration_s": dur, "fps": rate, "width": s.codec_context.width, "height": s.codec_context.height,
                "codec": s.codec_context.name, "frames": int(s.frames or 0)}


def sample_frames(video_path: Path, out_dir: Path, *, target_fps: float = TARGET_FPS, max_frames: int = MAX_FRAMES,
                  max_side: int = FRAME_MAX_SIDE) -> tuple[list[Frame], dict[str, Any]]:
    """Sharpest frame per time slot at about target_fps (fewer for long clips, at most max_frames), upright,
    saved as JPEG. Slots much blurrier than the median are dropped."""
    import av

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        info = probe_video(video_path)
    except (IndexError, av.error.FFmpegError) as exc:
        raise ValueError(f"{video_path} has no readable video stream: {exc}") from exc
    duration = info["duration_s"]
    fps = target_fps if duration <= 0 else min(target_fps, max_frames / duration)
    slot_len = 1.0 / max(fps, 1e-6)
    kept: list[Frame] = []
    rotation = 0

    def flush(best) -> None:
        nonlocal rotation
        sharp, n, t, frame = best
        rotation = getattr(frame, "rotation", 0) or 0
        arr = _frame_rgb(frame, max_side)
        path = out_dir / f"{len(kept):04d}_{n:06d}.jpg"
        Image.fromarray(arr).save(path, quality=95)
        kept.append(Frame(len(kept), n, float(t), path, sharp))

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        rate = info["fps"] or 30.0
        best, slot, n_decoded = None, None, 0
        try:
            for n, frame in enumerate(container.decode(stream)):
                n_decoded += 1
                t = float(frame.time) if frame.time is not None else n / rate
                s = int(t // slot_len)
                if slot is not None and s != slot and best is not None:
                    flush(best)
                    best = None
                slot = s
                sharp = _sharpness(frame)
                if best is None or sharp > best[0]:
                    best = (sharp, n, t, frame)
        except av.error.FFmpegError as exc:  # truncated clips still give the frames decoded so far
            log.warning("video decode stopped early: %s", exc)
            info["decode_error"] = str(exc)
        if best is not None:
            flush(best)

    n_slots = len(kept)
    if len(kept) > max_frames:
        keep = set(np.linspace(0, len(kept) - 1, max_frames).round().astype(int).tolist())
        for i, fr in enumerate(kept):
            if i not in keep:
                fr.path.unlink(missing_ok=True)
        kept = [fr for i, fr in enumerate(kept) if i in keep]
    dropped = 0
    if len(kept) > 2:
        med = float(np.median([fr.sharpness for fr in kept]))
        blurred = [fr for fr in kept if fr.sharpness < BLUR_REL * med]
        if len(kept) - len(blurred) >= 2:
            for fr in blurred:
                fr.path.unlink(missing_ok=True)
            kept = [fr for fr in kept if fr.sharpness >= BLUR_REL * med]
            dropped = len(blurred)
    for i, fr in enumerate(kept):
        fr.index = i
    if not duration and kept:
        duration = kept[-1].timestamp
    info.update({"duration_s": float(duration), "sample_fps": float(fps), "n_decoded": n_decoded,
                 "n_slots": n_slots, "n_frames": len(kept), "blur_dropped": dropped,
                 "blur_drop_rate": dropped / max(n_slots, 1), "rotation_deg": int(rotation)})
    return kept, info


# ----------------------------------------------------------------------------------------------- chunks


def plan_chunks(n: int, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    """Frame spans [start, end) of size chunk_size overlapping by at least overlap; the last chunk is pulled
    back to end at n."""
    if n <= chunk_size:
        return [(0, n)]
    stride = max(1, chunk_size - overlap)
    starts = list(range(0, n - chunk_size + 1, stride))
    if starts[-1] + chunk_size < n:
        starts.append(n - chunk_size)
    return [(s, s + chunk_size) for s in starts]


def assign_owners(n: int, spans: list[tuple[int, int]]) -> list[int]:
    """Each frame belongs to the chunk in which it sits farthest from the chunk ends."""
    owners = []
    for f in range(n):
        best, depth = 0, -1
        for c, (a, b) in enumerate(spans):
            if a <= f < b:
                d = min(f - a, b - 1 - f)
                if d > depth:
                    best, depth = c, d
        owners.append(best)
    return owners


def _correspondences(dst: ViewPrediction, src: ViewPrediction, rng: np.random.Generator,
                     per_frame: int = CORR_PER_FRAME) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if dst.pts3d.shape != src.pts3d.shape:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0)
    wd, ws = dst.weight, src.weight
    ok = dst.mask & src.mask & (wd >= MIN_WEIGHT) & (ws >= MIN_WEIGHT)
    idx = np.flatnonzero(ok.ravel())
    if len(idx) > per_frame:
        idx = rng.choice(idx, per_frame, replace=False)
    return (src.pts3d.reshape(-1, 3)[idx].astype(float), dst.pts3d.reshape(-1, 3)[idx].astype(float),
            np.sqrt(wd.ravel()[idx] * ws.ravel()[idx]))


def _median_depth(preds: list[ViewPrediction]) -> float:
    d = [np.median(np.linalg.norm(p.pts3d[p.mask] - p.T_wc[:3, 3], axis=1)) for p in preds if p.mask.any()]
    return float(np.median(d)) if d else 1.0


def register_views(dst: list[ViewPrediction], src: list[ViewPrediction], seed: int = 0) -> Sim3Fit:
    """Sim(3) mapping the src run onto the dst run from the same frames (same pixel grid, both valid)."""
    rng = np.random.default_rng(seed)
    parts = [_correspondences(d, s, rng) for d, s in zip(dst, src)]
    S = np.concatenate([p[0] for p in parts])
    D = np.concatenate([p[1] for p in parts])
    W = np.concatenate([p[2] for p in parts])
    thresh = float(np.clip(0.03 * _median_depth(dst), 0.02, 0.2))
    return robust_sim3(S, D, W, thresh=thresh, seed=seed)


@dataclass
class Alignment:
    """How chunk c was registered onto chunk c - 1 (or the loop run onto a chunk)."""

    T: np.ndarray  # x_prev = T x_cur
    ok: bool
    method: str  # points | points_subset | single_frame | failed
    inlier_frac: float  # of the fit behind T
    residual_m: float
    frames_used: list[int]
    n_shared: int
    n_frames_ok: int  # shared frames that register on their own
    sigma_log_scale: float  # spread of the per-frame scales behind the link, in log units
    frame_log_scales: list[float] = field(default_factory=list)
    joint_inlier_frac: float = 0.0  # of the joint fit over every shared frame
    camera_disagreement: float = 0.0  # rms distance between the two runs' shared cameras under T, / scene depth
    reason: str = ""

    def record(self) -> dict[str, Any]:
        return {"align_method": self.method, "align_ok": self.ok, "align_inlier_frac": float(self.inlier_frac),
                "align_joint_inlier_frac": float(self.joint_inlier_frac),
                "align_residual_m": float(self.residual_m) if math.isfinite(self.residual_m) else None,
                "align_frames_used": len(self.frames_used), "align_frames_ok": self.n_frames_ok,
                "align_sigma_log_scale": float(self.sigma_log_scale) if math.isfinite(self.sigma_log_scale) else None,
                "align_camera_disagreement": (float(self.camera_disagreement)
                                              if math.isfinite(self.camera_disagreement) else None),
                "align_frame_log_scales": [round(v, 4) for v in self.frame_log_scales],
                "align_reason": self.reason}


def _disagreement(Ta: np.ndarray, Tb: np.ndarray, pts: np.ndarray, depth: float) -> float:
    """Median distance between two transforms' images of the same points, as a share of the scene depth."""
    if len(pts) == 0:
        return math.inf
    return float(np.median(np.linalg.norm(apply(Ta, pts) - apply(Tb, pts), axis=1)) / max(depth, 1e-6))


def camera_disagreement(dst: list[ViewPrediction], src: list[ViewPrediction], T: np.ndarray) -> float:
    """RMS distance between where the two runs put the same cameras once src is mapped by T, as a share of the
    scene depth. Two runs that agree about the shared frames' depth but not about how the camera moved between
    them cannot both be right; the size of the disagreement bounds how wrong one of them is."""
    pairs = [(d.T_wc[:3, 3], s.T_wc[:3, 3]) for d, s in zip(dst, src) if d.pose_ok and s.pose_ok]
    if not pairs:
        return math.inf
    cd = np.array([p[0] for p in pairs])
    cs = apply(T, np.array([p[1] for p in pairs]))
    return float(np.sqrt(np.mean(np.sum((cd - cs) ** 2, axis=1))) / max(_median_depth(dst), 1e-6))


def align_runs(dst: list[ViewPrediction], src: list[ViewPrediction], frames: list[int], seed: int = 0
               ) -> Alignment:
    """Sim(3) taking the src run onto the dst run from their shared frames.

    The joint fit over all shared frames is used when it is good. When it is not, the two runs disagree about the
    camera motion inside the overlap (a fast turn at the chunk boundary): each shared frame is then registered on
    its own, the largest group of frames whose transforms agree is refitted jointly, and if no two frames agree the
    single best frame is used. The per-frame scales give the link's scale uncertainty. Camera poses are never used
    for the transform: their spacing is exactly what the two runs disagree about. They are used to judge the
    link: when the runs put the shared cameras more than LINK_MAX_DISAGREE of the scene depth apart, the link is
    refused, because chaining two runs that disagree that much corrupts the plan.
    """
    n_shared = len(frames)
    rng = np.random.default_rng(seed)
    joint = register_views(dst, src, seed=seed)
    per = [register_views([d], [s], seed=seed * 1009 + k) for k, (d, s) in enumerate(zip(dst, src))]
    good = [k for k, p in enumerate(per) if p.ok and p.inlier_frac >= FRAME_MIN_INLIERS]
    logs = [math.log(per[k].scale) for k in good]

    def spread(idx: list[int]) -> float:
        v = [math.log(per[k].scale) for k in idx]
        return float(np.std(v)) if len(v) > 1 else 0.0

    def done(T, method, fit, used, sigma) -> Alignment:
        dis = camera_disagreement(dst, src, T)
        ok, why = True, "ok"
        if dis > LINK_MAX_DISAGREE:
            ok, why = False, f"cameras_disagree:{dis:.3f}"
        return Alignment(T, ok, method, fit.inlier_frac, fit.residual_m, used, n_shared, len(good), sigma, logs,
                         joint.inlier_frac, dis, why)

    if joint.ok and joint.inlier_frac >= ALIGN_GOOD_INLIERS:
        return done(joint.T, "points", joint, list(frames),
                    max(spread(good) / math.sqrt(max(len(good), 1)), EDGE_SIGMA_FLOOR))
    if not good:
        return Alignment(np.eye(4), False, "failed", joint.inlier_frac, joint.residual_m, [], n_shared, 0, math.inf,
                         logs, joint.inlier_frac, math.inf, "no_frame_registers")
    depth = _median_depth(dst)
    pts = []
    for k in good:
        s = src[k]
        idx = np.flatnonzero(s.mask.ravel())
        pts.append(s.pts3d.reshape(-1, 3)[rng.choice(idx, min(len(idx), 500), replace=False)].astype(float))
    groups = []
    for a in good:
        members = [b for i, b in enumerate(good)
                   if b == a or _disagreement(per[a].T, per[b].T, np.concatenate([pts[good.index(a)], pts[i]]),
                                              depth) < FRAME_AGREE]
        groups.append((len(members), sum(per[b].inlier_frac for b in members), members))
    _, _, members = max(groups, key=lambda g: (g[0], g[1]))
    if len(members) >= 2:
        fit = register_views([dst[k] for k in members], [src[k] for k in members], seed=seed + 7)
        if fit.ok and fit.inlier_frac >= ALIGN_GOOD_INLIERS:
            return done(fit.T, "points_subset", fit, [frames[k] for k in members], max(spread(good), EDGE_SIGMA_FLOOR))
    best = max(members, key=lambda k: (per[k].inlier_frac, -abs(k - (n_shared - 1) / 2)))
    return done(per[best].T, "single_frame", per[best], [frames[best]], max(spread(good), SINGLE_FRAME_SIGMA))


def covisibility(a: list[ViewPrediction], b: list[ViewPrediction], rel_tol: float = 0.05,
                 n_samples: int = 2000, seed: int = 0) -> float:
    """Share of the points of runs-views a that land on consistent depth in at least one view of b (same frame)."""
    rng = np.random.default_rng(seed)
    pts = []
    for p in a:
        idx = np.flatnonzero(p.mask.ravel())
        if len(idx):
            pts.append(p.pts3d.reshape(-1, 3)[rng.choice(idx, min(len(idx), n_samples), replace=False)])
    if not pts:
        return 0.0
    P = np.concatenate(pts).astype(float)
    hit = np.zeros(len(P), bool)
    for q in b:
        R, c = q.T_wc[:3, :3], q.T_wc[:3, 3]
        depth = ((q.pts3d.reshape(-1, 3) - c) @ R)[:, 2].reshape(q.mask.shape)
        pc = (P - c) @ R
        z = pc[:, 2]
        front = z > 1e-3
        u = np.full(len(P), -1, int)
        v = np.full(len(P), -1, int)
        u[front] = np.round(q.K[0, 0] * pc[front, 0] / z[front] + q.K[0, 2]).astype(int)
        v[front] = np.round(q.K[1, 1] * pc[front, 1] / z[front] + q.K[1, 2]).astype(int)
        h, w = q.mask.shape
        inside = front & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        ii = np.flatnonzero(inside)
        dz = depth[v[ii], u[ii]]
        ok = q.mask[v[ii], u[ii]] & (np.abs(dz - z[ii]) < rel_tol * np.maximum(dz, 1e-3))
        hit[ii[ok]] = True
    return float(hit.mean())


# ----------------------------------------------------------------------------------------------- anchoring


def floor_plane(points: np.ndarray, normals: np.ndarray, weights: np.ndarray, cam_z: float | None, *,
                bin_m: float = 0.02, min_support: int = 150) -> dict[str, Any] | None:
    """Lowest well-supported upward-facing horizontal plane below the cameras."""
    up = normals[:, 2] > math.cos(math.radians(20.0))
    if up.sum() < min_support:
        return None
    z, w = points[up, 2], weights[up] + 1e-3
    lo, hi = np.percentile(z, [0.5, 99.5])
    edges = np.arange(lo, max(hi, lo + bin_m) + bin_m, bin_m)
    hist = np.convolve(np.histogram(z, edges, weights=w)[0], np.ones(3), "same")
    if hist.max() <= 0:
        return None
    peaks = [i for i in range(len(hist)) if hist[i] >= 0.25 * hist.max() and hist[i] == hist[max(0, i - 2):i + 3].max()]
    if not peaks:
        return None
    z0 = 0.5 * (edges[peaks[0]] + edges[peaks[0] + 1])
    inl = up & (np.abs(points[:, 2] - z0) < 0.05)
    if inl.sum() < min_support:
        return None
    P, wi = points[inl].astype(float), weights[inl] + 1e-3
    c = (P * wi[:, None]).sum(0) / wi.sum()
    n = np.linalg.svd((P - c) * np.sqrt(wi)[:, None], full_matrices=False)[2][-1]
    n = n if n[2] > 0 else -n
    if cam_z is not None and c[2] > cam_z - 0.4:
        return None
    tilt = float(np.degrees(np.arccos(np.clip(n[2], -1.0, 1.0))))
    return {"z": float(c[2]), "normal": n, "centroid": c, "support": int(inl.sum()), "tilt_deg": tilt}


def manhattan_vector(normals: np.ndarray, weights: np.ndarray, min_count: int = 200) -> tuple[complex, float]:
    """Resultant of exp(4i * wall angle) over wall normals, and the total weight."""
    wall = np.abs(normals[:, 2]) < 0.25
    if wall.sum() < min_count:
        return 0j, 0.0
    n = normals[wall, :2]
    w = weights[wall].astype(float)
    return complex((w * np.exp(4j * np.arctan2(n[:, 1], n[:, 0]))).sum()), float(w.sum())


def _wrap45(a: float) -> float:
    return (a + math.pi / 4) % (math.pi / 2) - math.pi / 4


# ----------------------------------------------------------------------------------------------- scene


def _load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"))


def _world_pose(X: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    s, R, t = decompose_sim3(X)
    return make_T(R @ T_wc[:3, :3], s * R @ T_wc[:3, 3] + t)


def _edge(i: int, j: int, T: np.ndarray, resid: float, ok: bool, kind: str = "seq") -> Edge:
    f = max(1.0, resid / 0.02) if math.isfinite(resid) else 3.0
    f *= 1.0 if ok else 3.0
    return Edge(i, j, T, sigma_rot=0.01 * f, sigma_trans=0.03 * f, sigma_log_scale=0.01 * f, kind=kind)


def _chunk_cloud(run: ChunkRun, X: np.ndarray, frames: list[int]):
    s, R, t = decompose_sim3(X)
    pts, nrm, ws, cz = [], [], [], []
    for f in frames:
        p = run.pred(f)
        normals, ok = run.normals(f)
        stride = max(1, int(math.sqrt(p.mask.size / ANALYSIS_SAMPLES)))
        sel = ok & (p.weight >= MIN_WEIGHT)
        grid = np.zeros_like(sel)
        grid[::stride, ::stride] = True
        sel &= grid
        pts.append(p.pts3d[sel] @ (s * R).T + t)
        nrm.append(normals[sel].astype(np.float32) @ R.T)
        ws.append(p.weight[sel])
        if p.pose_ok:
            cz.append((s * R @ p.T_wc[:3, 3] + t)[2])
    if not pts:
        return np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0), None
    return np.concatenate(pts), np.concatenate(nrm), np.concatenate(ws), float(np.mean(cz)) if cz else None


def _floor_spread(values: list[float]) -> float | None:
    """Std of per-chunk floor heights within FLOOR_MAX_DZ of their median (tables and beds left out)."""
    if not values:
        return None
    v = np.asarray(values, float)
    v = v[np.abs(v - np.median(v)) <= FLOOR_MAX_DZ]
    return float(np.std(v)) if len(v) >= 2 else 0.0


def resolve_video(path: Path) -> Path:
    """The clip itself, or the single video inside a folder."""
    path = Path(path)
    if not path.is_dir():
        return path
    clips = sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
                   and not p.name.startswith("."))
    if not clips:
        raise ValueError(f"no video file in {path}")
    if len(clips) > 1:
        log.warning("%d videos in %s, using the largest", len(clips), path)
    return max(clips, key=lambda p: p.stat().st_size)


def build_scene(video_path: Path, work_dir: Path, *, drift_correction: bool = True,
                cache: OutputCacheLike | None = None, runner: MapAnythingRunner | None = None,
                chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, target_fps: float = TARGET_FPS,
                max_frames: int = MAX_FRAMES, loop_frames: int = LOOP_FRAMES) -> Scene:
    video_path = resolve_video(video_path)
    frames, info = sample_frames(video_path, Path(work_dir) / "frames", target_fps=target_fps,
                                 max_frames=max_frames)
    if not frames:
        raise ValueError(f"no decodable video frames in {video_path}")
    source = {"tier": "video", "video_sha256": file_sha256(video_path),
              "sampler": {"fps": target_fps, "max_frames": max_frames, "max_side": FRAME_MAX_SIDE}}
    return build_scene_from_frames(frames, drift_correction=drift_correction, cache=cache, runner=runner,
                                   chunk_size=chunk_size, overlap=overlap, loop_frames=loop_frames,
                                   source_key=source, video_info=info)


def build_scene_from_frames(frames: list[Frame], *, drift_correction: bool = True,
                            cache: OutputCacheLike | None = None, runner: MapAnythingRunner | None = None,
                            chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP, loop_frames: int = LOOP_FRAMES,
                            source_key: dict | None = None, video_info: dict | None = None) -> Scene:
    runner = runner or MapAnythingRunner.get()
    info = dict(video_info or {})
    source_key = dict(source_key or {"tier": "video"})
    flags: list[str] = []
    n = len(frames)
    chunk_size = int(min(max(chunk_size, 2), MAX_VIEWS))
    overlap = int(min(max(overlap, 1), chunk_size - 1))
    spans = plan_chunks(n, chunk_size, overlap)
    owners = assign_owners(n, spans)
    if n < 3:
        flags.append("thin")
    if info.get("decode_error"):
        flags.append("video_decode_error")

    runs: list[ChunkRun] = []
    for c, (a, b) in enumerate(spans):
        imgs = [_load_rgb(frames[f].path) for f in range(a, b)]
        key = {**source_key, "chunk": [a, b], "frames": [frames[f].source_index for f in range(a, b)]}
        preds = runner.infer(imgs, None, key=key, cache=cache)
        runs.append(ChunkRun(a, b, preds))
        log.info("video chunk %d/%d: frames %d-%d, metric scale %.3f", c + 1, len(spans), a, b - 1,
                 preds[0].metric_scale)

    # link chunk c onto chunk c-1 on their shared frames; a failed link is never chained
    links: list[Alignment | None] = [None]
    for c in range(1, len(runs)):
        shared = list(range(runs[c].start, runs[c - 1].end))
        links.append(align_runs([runs[c - 1].pred(f) for f in shared], [runs[c].pred(f) for f in shared], shared,
                                seed=c))
        lk = links[c]
        if not lk.ok:
            flags.append(f"chunk_align_failed:{c}")
            log.warning("chunk %d: not chained to chunk %d (%s; joint inlier share %.2f, %d of %d shared frames "
                        "register alone)", c, c - 1, lk.reason, lk.joint_inlier_frac, lk.n_frames_ok, lk.n_shared)
        elif lk.method != "points":
            flags.append(f"chunk_align_{lk.method}:{c}")
            log.warning("chunk %d: joint inlier share %.2f, linked by %s (%d of %d shared frames register alone, "
                        "cameras %.3f of depth apart)", c, lk.joint_inlier_frac, lk.method, lk.n_frames_ok,
                        lk.n_shared, lk.camera_disagreement)
    broken = [c for c in range(1, len(runs)) if not links[c].ok]
    loop_reg = None
    if drift_correction and len(runs) >= 2:
        loop_reg = _loop_registration(frames, runs, runner, cache, source_key, loop_frames)
    bridge = loop_reg if broken and loop_reg is not None and loop_reg["usable"] else None
    kept, X_kept, used_links = _connect_chunks(runs, links, None if bridge is None else bridge["Z"])
    bridged = bridge is not None and 0 in kept and len(runs) - 1 in kept
    if bridged:
        flags.append("chunk_align_loop_bridge")
    dropped = [c for c in range(len(runs)) if c not in kept]
    covered = {f for c in kept for f in range(runs[c].start, runs[c].end)}
    dropped_ranges = _ranges([f for f in range(n) if f not in covered])
    for a, b in dropped_ranges:
        flags.append(f"video_segment_dropped:{a}-{b}")
        log.warning("video frames %d-%d (%.1f-%.1f s) dropped: their chunk could not be registered to the rest",
                    a, b, frames[a].timestamp, frames[b].timestamp)

    # from here on, work on the kept chunks only (index k into kept); owners[f] = -1 for dropped frames
    K = len(kept)
    sub_runs = [runs[c] for c in kept]
    kept_spans = [spans[c] for c in kept]
    owners = [-1] * n
    for f, o in enumerate(assign_owners(n, kept_spans)):
        if f in covered:
            owners[f] = o
    owned = [[f for f in range(run.start, run.end) if owners[f] == k] for k, run in enumerate(sub_runs)]
    k_of = {c: k for k, c in enumerate(kept)}
    seq_edges = [_edge(k_of[i], k_of[j], T, lk.residual_m, lk.method == "points") for i, j, T, lk in used_links]
    X_chain = [X_kept[c] for c in kept]
    X_base, scale, ginfo = _scale_and_gravity(sub_runs, X_chain, owners, owned)
    floors_chain = [_chunk_floor(sub_runs[k], X_base[k], owned[k]) for k in range(K)]

    loop: dict[str, Any] = {"attempted": False, "accepted": False, "residual_m": None}
    if loop_reg is not None:
        loop = {k: v for k, v in loop_reg.items() if k not in ("Z", "usable")}
    anchor: dict[str, Any] = {"max_yaw_correction_deg": 0.0, "max_tilt_correction_deg": 0.0}
    floors_mid = floors_chain
    X = X_base
    if drift_correction:
        X = X_chain
        if bridged:
            loop["reason"] = "used_as_link"
        elif dropped and loop_reg is not None and loop_reg["attempted"]:
            loop["reason"] = "not_used:segment_dropped"
        elif loop_reg is not None:
            X, loop = _loop_closure(frames, sub_runs, owners, X, seq_edges, loop_reg)
        if loop["accepted"]:
            X, scale, ginfo = _scale_and_gravity(sub_runs, X, owners, owned)
            floors_mid = [_chunk_floor(sub_runs[k], X[k], owned[k]) for k in range(K)]
        else:
            X = X_base
        X, anchor = _anchor_chunks(sub_runs, X, owned, floors_mid, seq_edges + loop.pop("_edges", []))
    loop.pop("_edges", None)
    floors_after = [_chunk_floor(sub_runs[k], X[k], owned[k]) for k in range(K)]
    chunk_info = []
    for c, run in enumerate(runs):
        ci = {"index": c, "frames": [run.start, run.end], "metric_scale": run.preds[0].metric_scale,
              "kept": c in k_of}
        if c == 0:
            ci.update({"align_method": "reference", "align_inlier_frac": 1.0, "align_residual_m": 0.0})
        else:
            ci.update(links[c].record())
            ci["shared_frames"] = links[c].n_shared
            ci["relative_scale"] = float(decompose_sim3(links[c].T)[0]) if links[c].ok else None
        if c in k_of:
            k = k_of[c]
            ci["world_scale"] = float(decompose_sim3(X[k])[0])
            ci["floor_z_chain"] = None if floors_chain[k] is None else floors_chain[k]["z"]
            ci["floor_z_final"] = None if floors_after[k] is None else floors_after[k]["z"]
            ci.update(anchor.get("per_chunk", {}).get(k, {}))
        chunk_info.append(ci)
    scale_log_sigma = math.sqrt(SCALE_SIGMA_BASE ** 2 + scale["spread"] ** 2 / K)
    if K < 3:  # one or two chunks cannot show their spread; fall back to the single-run photo prior
        scale_log_sigma = max(scale_log_sigma, SINGLE_RUN_SCALE_SIGMA / math.sqrt(K))
    # A link that rests on a subset of the shared frames carries its own scale uncertainty into every chunk after
    # it. Two runs that place the shared cameras apart by d (share of depth) disagree by about d about the
    # geometry, so each is off by about d / sqrt(2). A refused link between two kept chunks counts in full; one
    # whose other side was dropped counts by the dropped share of chunks, the odds that the kept side is the wrong
    # one. A link where no shared frame registers at all counts as the largest disagreement.
    link_var = sum(lk.sigma_log_scale ** 2 for _, _, _, lk in used_links if lk.method != "points")
    terms = []
    for c in range(1, len(runs)):
        lk = links[c]
        d = min(lk.camera_disagreement, GEO_SIGMA_CAP) if math.isfinite(lk.camera_disagreement) else GEO_SIGMA_CAP
        both = c in k_of and c - 1 in k_of
        terms.append(d * d if lk.ok or both else d * d * len(dropped) / len(runs))
    geo_sigma = math.sqrt(float(np.mean(terms)) / 2.0) if terms else 0.0
    # A predicted field of view the protocol's camera cannot have marks input MapAnything misreads (recompressed
    # frames); its log distance from the camera's range is one more scale term.
    focal, focal_flags = focal_check([p for run in sub_runs for p in run.preds])
    flags.extend(focal_flags)
    scale_log_sigma = math.sqrt(scale_log_sigma ** 2 + link_var + geo_sigma ** 2 + focal["sigma_log"] ** 2)
    scale.update({"sigma_log": scale_log_sigma, "link_sigma_log": math.sqrt(link_var),
                  "geometry_sigma_log": geo_sigma, "focal_sigma_log": focal["sigma_log"]})

    scene = _fuse_scene(frames, sub_runs, X, owners, flags)
    n_kept = sum(o >= 0 for o in owners)
    quality = {
        "n_frames": n, "n_frames_used": n_kept,
        "duration_s": float(info.get("duration_s", frames[-1].timestamp - frames[0].timestamp)),
        "blur_drop_rate": float(info.get("blur_drop_rate", 0.0)), "mean_conf": scene.meta.pop("_mean_conf"),
        "n_chunks": len(runs), "n_chunks_used": K, "sample_fps": float(info.get("sample_fps", 0.0)),
        "chunk_size": chunk_size, "overlap": overlap, "scale_spread_log": scale["spread"], "thin": n < 3,
        "chunk_links": [lk.method for lk in links[1:]], "frames_dropped": n - n_kept,
        "ma_focal_long": focal["ma_focal_long"], "focal_sigma_log": focal["sigma_log"],
    }
    applied = ["sim3_chain"] + (["loop_closure"] if loop["accepted"] else [])
    if anchor.get("floor_applied"):
        applied.append("floor_anchoring")
    if anchor.get("yaw_applied"):
        applied.append("manhattan_yaw")
    drift = {
        "enabled": bool(drift_correction), "method": "+".join(applied), "chunks": chunk_info, "loop_closure": loop,
        "max_yaw_correction_deg": float(anchor["max_yaw_correction_deg"]),
        "max_tilt_correction_deg": float(anchor["max_tilt_correction_deg"]),
        "floor_z_spread_before_m": _floor_spread([fl["z"] for fl in floors_chain if fl is not None]),
        "floor_z_spread_after_loop_m": _floor_spread([fl["z"] for fl in floors_mid if fl is not None]),
        "floor_z_spread_after_m": _floor_spread([fl["z"] for fl in floors_after if fl is not None]),
    }
    scene.scale_log_sigma = scale_log_sigma
    scene.meta.update({"quality": quality, "drift": drift, "gravity": ginfo, "video": info, "scale": scale})
    log.info("video scene: %d frames, %d chunks, %d points, loop %s, scale sigma %.3f", n, len(runs),
             len(scene.points), "accepted" if loop["accepted"] else "not used", scale_log_sigma)
    return scene


def _scale_and_gravity(runs: list[ChunkRun], X: list[np.ndarray], owners: list[int], owned: list[list[int]]):
    """Global metric scale (median of what each chunk's own metric estimate implies for the chain), then one
    global gravity alignment. Returns the new node poses, the scale record and the gravity record."""
    chunk_log = np.array([-math.log(decompose_sim3(T)[0]) for T in X])
    log_corr = float(np.median(chunk_log))
    spread = float(np.std(chunk_log)) if len(chunk_log) > 1 else 0.0
    S = np.diag([math.exp(log_corr)] * 3 + [1.0])
    X = [S @ T for T in X]
    poses = [_world_pose(X[owners[f]], runs[owners[f]].pred(f).T_wc) for f in range(len(owners))
             if owners[f] >= 0 and runs[owners[f]].pred(f).pose_ok]
    poses = poses or [np.eye(4)]
    nrm, w = [], []
    for c, run in enumerate(runs):
        _, nr, wc, _ = _chunk_cloud(run, X[c], owned[c])
        nrm.append(nr)
        w.append(wc)
    nrm, w = np.concatenate(nrm), np.concatenate(w)
    sub = slice(None, None, max(1, len(nrm) // 300_000))
    R_g, ginfo = gravity_alignment(poses, nrm[sub], w[sub])
    G = make_T(R_g, np.zeros(3))
    scale = {"log_correction": log_corr, "per_chunk_log": chunk_log.tolist(), "spread": spread}
    return [G @ T for T in X], scale, ginfo


def _chunk_floor(run: ChunkRun, X: np.ndarray, own: list[int]) -> dict[str, Any] | None:
    if not own:
        return None
    P, N, W, cz = _chunk_cloud(run, X, own)
    if len(P) == 0:
        return None
    fl = floor_plane(P, N, W, cz)
    if fl is not None:
        fl["normal"] = fl["normal"].tolist()
        fl["centroid"] = fl["centroid"].tolist()
    return fl


def _ranges(idx: list[int]) -> list[tuple[int, int]]:
    """Runs of consecutive integers as inclusive (first, last) pairs."""
    out: list[tuple[int, int]] = []
    for i in sorted(idx):
        if out and i == out[-1][1] + 1:
            out[-1] = (out[-1][0], i)
        else:
            out.append((i, i))
    return out


def _connect_chunks(runs: list[ChunkRun], links: list[Alignment | None], Z_loop: np.ndarray | None):
    """Chunks joined by usable links, as the largest connected group (by frames covered).

    links[c] takes chunk c onto chunk c - 1; Z_loop, when given, takes the last chunk onto the first (the walk
    ends where it started), which can stand in for one failed link. Returns the kept chunk indices, each kept
    chunk's transform into the frame of the first kept chunk, and the links used as (i, j, T_ij, Alignment).
    """
    K = len(runs)
    adj: dict[int, list[tuple[int, np.ndarray]]] = {c: [] for c in range(K)}
    for c in range(1, K):
        if links[c].ok:
            adj[c - 1].append((c, links[c].T))
            adj[c].append((c - 1, invert(links[c].T)))
    if Z_loop is not None and K >= 2:
        adj[0].append((K - 1, Z_loop))
        adj[K - 1].append((0, invert(Z_loop)))
    seen: set[int] = set()
    comps: list[dict[int, np.ndarray]] = []
    for root in range(K):
        if root in seen:
            continue
        X = {root: np.eye(4)}
        stack = [root]
        seen.add(root)
        while stack:
            u = stack.pop()
            for v, T in adj[u]:
                if v not in X:
                    X[v] = X[u] @ T
                    seen.add(v)
                    stack.append(v)
        comps.append(X)

    def cover(X: dict[int, np.ndarray]) -> int:
        return len({f for c in X for f in range(runs[c].start, runs[c].end)})

    X = max(comps, key=lambda X: (cover(X), 0 in X))
    kept = sorted(X)
    inv_root = invert(X[kept[0]])
    X = {c: inv_root @ T for c, T in X.items()}
    used = [(c - 1, c, links[c].T, links[c]) for c in range(1, K) if c in X and c - 1 in X and links[c].ok]
    if Z_loop is not None and 0 in X and K - 1 in X and K >= 2:
        used.append((0, K - 1, Z_loop, Alignment(Z_loop, True, "loop", 1.0, 0.0, [], 0, 0, EDGE_SIGMA_FLOOR)))
    return kept, X, used


def _loop_registration(frames, runs, runner, cache, source_key, loop_frames) -> dict[str, Any]:
    """MapAnything on the first and last frames together, registered onto the first and last chunks. "Z" takes the
    last chunk onto the first; "usable" says whether the registration alone is good enough to stand in for a
    failed chunk link (its residual and covisibility checks; the consistency checks need the chain)."""
    n = len(frames)
    L = int(min(loop_frames, runs[0].end - runs[0].start, runs[-1].end - runs[-1].start, n // 2))
    reg: dict[str, Any] = {"attempted": False, "accepted": False, "residual_m": None, "usable": False}
    if L < 2 or runs[-1].start < L:
        reg["reason"] = "too_few_frames"
        return reg
    first, last = list(range(L)), list(range(n - L, n))
    imgs = [_load_rgb(frames[f].path) for f in first + last]
    key = {**source_key, "loop": True, "frames": [frames[f].source_index for f in first + last]}
    preds = runner.infer(imgs, None, key=key, cache=cache)
    reg["attempted"] = True
    fit0 = align_runs([runs[0].pred(f) for f in first], preds[:L], first, seed=101)
    fitK = align_runs([runs[-1].pred(f) for f in last], preds[L:], last, seed=102)
    resid = max(fit0.residual_m, fitK.residual_m)
    over = min(covisibility(preds[L:], preds[:L]), covisibility(preds[:L], preds[L:]))
    reg.update({"residual_m": float(resid) if math.isfinite(resid) else None, "overlap": over,
                "frames": L, "inlier_frac": float(min(fit0.inlier_frac, fitK.inlier_frac)),
                "align_methods": [fit0.method, fitK.method]})
    if not (fit0.ok and fitK.ok):
        reg["reason"] = "registration_failed"
        return reg
    depth = _median_depth(runs[0].preds)
    reg["Z"] = fit0.T @ invert(fitK.T)
    reg["checks_local"] = {"residual": bool(resid <= max(0.03, 0.03 * depth)),
                           "overlap": bool(over >= LOOP_MIN_OVERLAP),
                           "joint": fit0.method != "single_frame" and fitK.method != "single_frame"}
    reg["usable"] = all(reg["checks_local"].values())
    reg["reason"] = "registered"
    return reg


def _loop_closure(frames, runs, owners, X, seq_edges, reg):
    """Accept the loop registration when it agrees with the chain, and distribute its error by the pose graph."""
    n = len(frames)
    loop = {k: v for k, v in reg.items() if k not in ("Z", "usable")}
    if "Z" not in reg:
        return X, loop
    Z, resid, over = reg["Z"], reg["residual_m"], reg["overlap"]
    err = sim3_error(Z, invert(X[0]) @ X[-1])
    centers = np.array([apply(X[owners[f]], runs[owners[f]].pred(f).T_wc[:3, 3]) for f in range(n)
                        if owners[f] >= 0 and runs[owners[f]].pred(f).pose_ok]).reshape(-1, 3)
    path = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0
    loop.update({"error_trans_m": err["trans"], "error_rot_deg": err["rot_deg"], "error_log_scale": err["log_scale"],
                 "path_length_m": path})
    checks = {
        "residual": reg["checks_local"]["residual"],
        "overlap": reg["checks_local"]["overlap"],
        "translation": err["trans"] <= max(0.5, LOOP_MAX_DRIFT * path),
        "rotation": err["rot_deg"] <= LOOP_MAX_ROT_DEG,
        "scale": abs(err["log_scale"]) <= 0.25,
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        loop["reason"] = "rejected:" + ",".join(failed)
        log.info("loop closure rejected (%s): residual %.3f m, overlap %.2f, error %.2f m / %.1f deg",
                 ",".join(failed), resid, over, err["trans"], err["rot_deg"])
        return X, loop
    loop_edge = _edge(0, len(runs) - 1, Z, resid, True, kind="loop")
    res = optimize_pose_graph(X, seq_edges + [loop_edge])
    after = next(e for e in res.edge_residuals if e["kind"] == "loop")
    loop.update({"accepted": True, "reason": "ok", "error_after_trans_m": after["trans"],
                 "error_after_rot_deg": after["rot_deg"], "_edges": [loop_edge]})
    log.info("loop closure accepted: %.2f m / %.1f deg distributed over %d chunks", err["trans"], err["rot_deg"],
             len(runs))
    return res.nodes, loop


def _anchor_chunks(runs, X, owned, floors, edges):
    """Level each chunk's floor (tilt under TILT_MAX_DEG, height to the median floor) and snap its wall yaw to the
    global Manhattan frame when within YAW_MAX_DEG, then re-solve the pose graph with these as priors."""
    K = len(runs)
    valid = [c for c in range(K) if floors[c] is not None]
    out = {"max_yaw_correction_deg": 0.0, "max_tilt_correction_deg": 0.0, "per_chunk": {}, "height_chunks": []}
    z_ref = float(np.median([floors[c]["z"] for c in valid])) if valid else 0.0
    height_ok = [c for c in valid if abs(floors[c]["z"] - z_ref) <= FLOOR_MAX_DZ]
    out["height_chunks"] = height_ok
    mvec = {}
    for c in range(K):
        own = owned[c]
        _, N, W, _ = _chunk_cloud(runs[c], X[c], own) if own else (None, np.zeros((0, 3)), np.zeros(0), None)
        z, wsum = manhattan_vector(N, W)
        if wsum > 0 and abs(z) / wsum >= MANHATTAN_MIN_CONC:
            mvec[c] = z
    theta_g = float(np.angle(sum(mvec.values())) / 4.0) if mvec else None

    corr: dict[int, np.ndarray] = {}
    for c in range(K):
        Rc, rec = np.eye(3), {}
        fl = floors[c]
        if fl is not None and 0.0 < fl["tilt_deg"] <= TILT_MAX_DEG:
            Rc = rotation_between(np.asarray(fl["normal"]), np.array([0.0, 0.0, 1.0]))
            rec["tilt_correction_deg"] = fl["tilt_deg"]
        if theta_g is not None and c in mvec:
            dpsi = _wrap45(theta_g - float(np.angle(mvec[c]) / 4.0))
            if abs(math.degrees(dpsi)) <= YAW_MAX_DEG:
                Rc = rot_z(dpsi) @ Rc
                rec["yaw_correction_deg"] = math.degrees(dpsi)
        if rec:
            corr[c] = Rc
        out["per_chunk"][c] = rec

    if not corr and not height_ok:
        return X, out
    priors: list[Prior] = []
    X = list(X)
    for c in range(1, K):
        R = decompose_sim3(X[c])[1]
        if c in corr:
            priors.append(Prior(c, R_world=corr[c] @ R))
        if c in height_ok:
            local = apply(invert(X[c]), np.asarray(floors[c]["centroid"]))
            priors.append(Prior(c, point=local, z_world=z_ref))
    # chunk 0 is the gauge: apply its own correction about its floor centroid (else its first camera), then fix it
    pivot = np.asarray(floors[0]["centroid"]) if floors[0] is not None else X[0][:3, 3]
    A = make_T(corr.get(0, np.eye(3)), np.zeros(3))
    A[:3, 3] = pivot - A[:3, :3] @ pivot
    dz0 = 0.0
    if 0 in height_ok:
        dz0 = z_ref - apply(A, np.asarray(floors[0]["centroid"]))[2]
        A[2, 3] += dz0
    X[0] = A @ X[0]
    res = optimize_pose_graph(X, edges, priors, fixed=(0,))
    yaws = [abs(r.get("yaw_correction_deg", 0.0)) for r in out["per_chunk"].values()]
    tilts = [abs(r.get("tilt_correction_deg", 0.0)) for r in out["per_chunk"].values()]
    out["floor_applied"] = bool(any(c >= 1 for c in height_ok) or abs(dz0) > 1e-3 or max(tilts, default=0.0) > 0.01)
    out["yaw_applied"] = bool(max(yaws, default=0.0) > 0.01)
    out.update({"max_yaw_correction_deg": max(yaws, default=0.0), "max_tilt_correction_deg": max(tilts, default=0.0),
                "z_ref": z_ref, "theta_manhattan_deg": None if theta_g is None else math.degrees(theta_g),
                "graph_cost_before": res.cost_before, "graph_cost_after": res.cost_after})
    return res.nodes, out


def _fuse_scene(frames: list[Frame], runs: list[ChunkRun], X: list[np.ndarray], owners: list[int],
                flags: list[str]) -> Scene:
    n = len(frames)
    present = [f for f in range(n) if owners[f] >= 0]
    usable = [f for f in present if runs[owners[f]].pred(f).pose_ok] or present[:1]
    keyframes = usable[::KEYFRAME_EVERY]
    key_of = {f: i for i, f in enumerate(keyframes)}
    nearest_key = {f: key_of[min(keyframes, key=lambda k: abs(k - f))] for f in present}
    if len(usable) < len(present):
        flags.append(f"frames_without_pose:{len(present) - len(usable)}")
    views: list[CameraView] = []
    frame_T = np.full((n, 4, 4), np.nan)
    pts, nrm, ws, vidx, score, wmeans = [], [], [], [], [], []
    for f in present:
        c = owners[f]
        run, p = runs[c], runs[c].pred(f)
        s, R, t = decompose_sim3(X[c])
        T_wc = _world_pose(X[c], p.T_wc)
        frame_T[f] = T_wc
        normals, ok = run.normals(f)
        w = p.weight
        wmeans.append(float(w[p.mask].mean()) if p.mask.any() else 0.0)
        sel = ok & p.mask & (w >= MIN_WEIGHT)
        grid = np.zeros_like(sel)
        grid[::FUSE_STRIDE, ::FUSE_STRIDE] = True
        sel &= grid
        pw = p.pts3d[sel] @ (s * R).T + t
        pts.append(pw)
        nrm.append(normals[sel].astype(np.float32) @ R.T)
        ws.append(w[sel])
        vidx.append(np.full(len(pw), nearest_key[f], np.int64))
        score.append(w[sel] + (0.5 if f in key_of else 0.0))
        if f in key_of:
            pm = (p.pts3d @ (s * R).T + t).astype(np.float32)
            pm[~p.mask] = 0.0
            fr = frames[f]
            views.append(CameraView(
                id=f"frame{fr.source_index:06d}", image_path=fr.path, width=int(p.image_size[0]),
                height=int(p.image_size[1]), K=p.K_image(), T_wc=T_wc, pointmap=p.uncrop(pm, 0.0),
                valid=p.uncrop(p.mask, False), conf=p.uncrop(w, 0.0), timestamp=fr.timestamp,
                meta={"frame_index": f, "chunk": c, "metric_scale": p.metric_scale, "conf_kind": "normalised"},
            ))
    centers = np.array([v.center for v in views])
    P, N, W, V = fuse_points(np.concatenate(pts), np.concatenate(nrm), np.concatenate(ws), np.concatenate(vidx),
                             centers, voxel=VOXEL, max_points=MAX_POINTS, score=np.concatenate(score))
    if len(P) < 100:
        flags.append("few_points")
    meta = {"flags": flags, "_mean_conf": float(np.mean(wmeans)) if wmeans else 0.0,
            "frames": {"source_index": [fr.source_index for fr in frames], "timestamp": [fr.timestamp for fr in frames],
                       "path": [str(fr.path) for fr in frames], "owner_chunk": owners, "T_wc": frame_T,
                       "pose_ok": [owners[f] >= 0 and runs[owners[f]].pred(f).pose_ok for f in range(n)],
                       "keyframes": keyframes}}
    return Scene(tier="video", views=views, points=P, normals=N, weights=W, view_index=V, meta=meta)
