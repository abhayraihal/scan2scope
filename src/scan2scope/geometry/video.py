"""Video tier: frames sampled from one walkthrough clip, MapAnything on overlapping chunks chained with Sim(3)
on the shared frames, then loop closure, gravity, floor and Manhattan anchoring, fused into one Scene.

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
    chain,
    optimize_pose_graph,
    robust_sim3,
    sim3_error,
)
from scan2scope.geometry.mapanything_backend import (
    MAX_VIEWS,
    MapAnythingRunner,
    OutputCacheLike,
    ViewPrediction,
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
SCALE_SIGMA_BASE = 0.05
SINGLE_RUN_SCALE_SIGMA = 0.08  # photo.SCALE_LOG_SIGMA: one MapAnything metric estimate
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


def pose_fallback(dst: list[ViewPrediction], src: list[ViewPrediction]) -> np.ndarray:
    """Sim(3) from the camera poses of shared frames when point registration fails."""
    pairs = [(d, s) for d, s in zip(dst, src) if d.pose_ok and s.pose_ok]
    if not pairs:
        return np.eye(4)
    dst, src = [d for d, _ in pairs], [s for _, s in pairs]
    Rs = [d.T_wc[:3, :3] @ s.T_wc[:3, :3].T for d, s in zip(dst, src)]
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ np.diag([1.0, 1.0, np.sign(np.linalg.det(U @ Vt))]) @ Vt
    cd = np.array([d.T_wc[:3, 3] for d in dst])
    cs = np.array([s.T_wc[:3, 3] for s in src])
    base_d = np.linalg.norm(cd - cd.mean(0), axis=1).sum()
    base_s = np.linalg.norm(cs - cs.mean(0), axis=1).sum()
    if len(dst) >= 2 and base_s > 0.05 and base_d > 0.05:
        s = base_d / base_s
    else:
        s = _median_depth(dst) / max(_median_depth(src), 1e-6)
    t = (cd - s * cs @ R.T).mean(0)
    return make_T(R, t, s)


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

    # chain chunk c onto chunk c-1 on their shared frames
    rel, chunk_info = [], [{"index": 0, "frames": list(spans[0]), "metric_scale": runs[0].preds[0].metric_scale,
                            "relative_scale": 1.0, "align_residual_m": 0.0, "align_inlier_frac": 1.0,
                            "align_method": "reference"}]
    seq_edges: list[Edge] = []
    for c in range(1, len(runs)):
        shared = list(range(runs[c].start, runs[c - 1].end))
        dst = [runs[c - 1].pred(f) for f in shared]
        src = [runs[c].pred(f) for f in shared]
        fit = register_views(dst, src, seed=c)
        method = "points"
        T = fit.T
        if not fit.ok:
            T, method = pose_fallback(dst, src), "poses"
            flags.append(f"chunk_align_fallback:{c}")
            log.warning("chunk %d: point registration failed (%d correspondences), using camera poses", c, fit.n)
        rel.append(T)
        seq_edges.append(_edge(c - 1, c, T, fit.residual_m, fit.ok))
        chunk_info.append({"index": c, "frames": [runs[c].start, runs[c].end],
                           "metric_scale": runs[c].preds[0].metric_scale,
                           "relative_scale": float(decompose_sim3(T)[0]),
                           "align_residual_m": float(fit.residual_m) if math.isfinite(fit.residual_m) else None,
                           "align_inlier_frac": float(fit.inlier_frac), "align_method": method,
                           "shared_frames": len(shared)})
    X_chain = chain(rel)
    owned = [[f for f in range(run.start, run.end) if owners[f] == c] for c, run in enumerate(runs)]
    X_base, scale, ginfo = _scale_and_gravity(runs, X_chain, owners, owned)
    floors_chain = [_chunk_floor(runs[c], X_base[c], owned[c]) for c in range(len(runs))]

    loop: dict[str, Any] = {"attempted": False, "accepted": False, "residual_m": None}
    anchor: dict[str, Any] = {"max_yaw_correction_deg": 0.0, "max_tilt_correction_deg": 0.0}
    floors_mid = floors_chain
    X = X_base
    if drift_correction:
        X = X_chain
        if len(runs) >= 2:
            X, loop = _loop_closure(frames, runs, owners, X, seq_edges, runner, cache, source_key, loop_frames)
        if loop["accepted"]:
            X, scale, ginfo = _scale_and_gravity(runs, X, owners, owned)
            floors_mid = [_chunk_floor(runs[c], X[c], owned[c]) for c in range(len(runs))]
        else:
            X = X_base
        X, anchor = _anchor_chunks(runs, X, owned, floors_mid, seq_edges + loop.pop("_edges", []))
    loop.pop("_edges", None)
    floors_after = [_chunk_floor(runs[c], X[c], owned[c]) for c in range(len(runs))]
    for c, ci in enumerate(chunk_info):
        ci["world_scale"] = float(decompose_sim3(X[c])[0])
        ci["floor_z_chain"] = None if floors_chain[c] is None else floors_chain[c]["z"]
        ci["floor_z_final"] = None if floors_after[c] is None else floors_after[c]["z"]
        ci.update(anchor.get("per_chunk", {}).get(c, {}))
    scale_log_sigma = math.sqrt(SCALE_SIGMA_BASE ** 2 + scale["spread"] ** 2 / len(runs))
    if len(runs) < 3:  # one or two chunks cannot show their spread; fall back to the single-run photo prior
        scale_log_sigma = max(scale_log_sigma, SINGLE_RUN_SCALE_SIGMA / math.sqrt(len(runs)))
    scale["sigma_log"] = scale_log_sigma

    scene = _fuse_scene(frames, runs, X, owners, flags)
    quality = {
        "n_frames": n, "duration_s": float(info.get("duration_s", frames[-1].timestamp - frames[0].timestamp)),
        "blur_drop_rate": float(info.get("blur_drop_rate", 0.0)), "mean_conf": scene.meta.pop("_mean_conf"),
        "n_chunks": len(runs), "sample_fps": float(info.get("sample_fps", 0.0)), "chunk_size": chunk_size,
        "overlap": overlap, "scale_spread_log": scale["spread"], "thin": n < 3,
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
             if runs[owners[f]].pred(f).pose_ok]
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


def _loop_closure(frames, runs, owners, X, seq_edges, runner, cache, source_key, loop_frames):
    n = len(frames)
    L = int(min(loop_frames, runs[0].end - runs[0].start, runs[-1].end - runs[-1].start, n // 2))
    loop: dict[str, Any] = {"attempted": False, "accepted": False, "residual_m": None}
    if L < 2 or runs[-1].start < L:
        loop["reason"] = "too_few_frames"
        return X, loop
    first, last = list(range(L)), list(range(n - L, n))
    imgs = [_load_rgb(frames[f].path) for f in first + last]
    key = {**source_key, "loop": True, "frames": [frames[f].source_index for f in first + last]}
    preds = runner.infer(imgs, None, key=key, cache=cache)
    loop["attempted"] = True
    fit0 = register_views([runs[0].pred(f) for f in first], preds[:L], seed=101)
    fitK = register_views([runs[-1].pred(f) for f in last], preds[L:], seed=102)
    resid = max(fit0.residual_m, fitK.residual_m)
    over = min(covisibility(preds[L:], preds[:L]), covisibility(preds[:L], preds[L:]))
    loop.update({"residual_m": float(resid) if math.isfinite(resid) else None, "overlap": over,
                 "frames": L, "inlier_frac": float(min(fit0.inlier_frac, fitK.inlier_frac))})
    if not (fit0.ok and fitK.ok):
        loop["reason"] = "registration_failed"
        return X, loop
    Z = fit0.T @ invert(fitK.T)
    err = sim3_error(Z, invert(X[0]) @ X[-1])
    centers = np.array([apply(X[owners[f]], runs[owners[f]].pred(f).T_wc[:3, 3]) for f in range(n)
                        if runs[owners[f]].pred(f).pose_ok]).reshape(-1, 3)
    path = float(np.sum(np.linalg.norm(np.diff(centers, axis=0), axis=1))) if len(centers) > 1 else 0.0
    depth = _median_depth(runs[0].preds)
    loop.update({"error_trans_m": err["trans"], "error_rot_deg": err["rot_deg"], "error_log_scale": err["log_scale"],
                 "path_length_m": path})
    checks = {
        "residual": resid <= max(0.03, 0.03 * depth),
        "overlap": over >= LOOP_MIN_OVERLAP,
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
    usable = [f for f in range(n) if runs[owners[f]].pred(f).pose_ok] or [0]
    keyframes = usable[::KEYFRAME_EVERY]
    key_of = {f: i for i, f in enumerate(keyframes)}
    nearest_key = [key_of[min(keyframes, key=lambda k: abs(k - f))] for f in range(n)]
    if len(usable) < n:
        flags.append(f"frames_without_pose:{n - len(usable)}")
    views: list[CameraView] = []
    frame_T = np.zeros((n, 4, 4))
    pts, nrm, ws, vidx, score, wmeans = [], [], [], [], [], []
    for f in range(n):
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
                       "pose_ok": [runs[owners[f]].pred(f).pose_ok for f in range(n)], "keyframes": keyframes}}
    return Scene(tier="video", views=views, points=P, normals=N, weights=W, view_index=V, meta=meta)
