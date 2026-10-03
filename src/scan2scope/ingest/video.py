"""Video ingest with PyAV: metadata probe and upright, sharp keyframes for the video tier."""

from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from scan2scope.ingest.images import intrinsics_from_35mm

log = logging.getLogger("scan2scope.ingest")

_HDR_TRC = (16, 18)  # AVCOL_TRC_SMPTE2084 (PQ) and AVCOL_TRC_ARIB_STD_B67 (HLG)
_LOCAL_MEDIAN_HALF = 7  # blur threshold uses the median of +-7 neighbouring keyframes (about 7 s at 2 fps)

_KEYS = {
    "make": ("com.apple.quicktime.make", "com.android.manufacturer", "make"),
    "model": ("com.apple.quicktime.model", "com.android.model", "model"),
    "software": ("com.apple.quicktime.software", "com.android.version", "software"),
    "lens_model": ("com.apple.quicktime.camera.lens_model",),
    "focal_35mm": ("com.apple.quicktime.camera.focal_length.35mm_equivalent",),
    "creation_time": ("com.apple.quicktime.creationdate", "creation_time"),
}


@dataclass
class VideoInfo:
    duration_s: float
    fps: float
    n_frames: int
    width: int  # upright, after rotation_deg is applied
    height: int
    rotation_deg: int  # clockwise rotation that makes decoded frames upright: 0, 90, 180 or 270
    codec: str
    pix_fmt: str | None
    is_hdr: bool
    focal_35mm: float | None = None
    lens_model: str | None = None
    creation_time: str | None = None
    make: str | None = None
    model: str | None = None
    software: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FrameRecord:
    index: int  # frame number in decode (presentation) order, from 0
    t: float  # seconds after the first frame
    path: Path
    sharpness: float  # variance of the Laplacian on the downscaled grey frame
    width: int  # of the written upright JPEG
    height: int


def rotate_upright(img: np.ndarray, rotation_deg: int) -> np.ndarray:
    """Rotate an (H, W, ...) image clockwise by rotation_deg, a multiple of 90."""
    k = round(rotation_deg / 90.0) % 4
    return np.ascontiguousarray(np.rot90(img, k=-k)) if k else img


def _clockwise_from_ccw(ccw_deg: float) -> int:
    return round(-ccw_deg / 90.0) % 4 * 90


def _frame_rotation(frame: Any, stream: Any) -> int:
    """Clockwise display rotation from the frame's display matrix, or the legacy 'rotate' stream tag."""
    ccw = float(frame.rotation)  # 0 when the frame carries no display matrix
    if ccw:
        return _clockwise_from_ccw(ccw)
    try:
        return round(float(stream.metadata.get("rotate", 0)) / 90.0) % 4 * 90
    except (TypeError, ValueError):
        return 0


def _tags(container: Any, stream: Any) -> dict[str, str]:
    tags: dict[str, str] = {}
    for md in (stream.metadata, container.metadata):  # container (QuickTime keys) wins
        for k, v in dict(md).items():
            if v not in (None, ""):
                tags[str(k).lower()] = str(v)
    return tags


def _tag(tags: dict[str, str], field: str) -> str | None:
    for k in _KEYS[field]:
        v = tags.get(k)
        if v and v.strip():
            return v.strip()
    return None


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) and f > 0 else None


def _fps(stream: Any) -> float:
    for rate in (stream.average_rate, stream.guessed_rate, stream.base_rate):
        f = _float(rate) if rate is not None else None
        if f:
            return f
    return 0.0


def probe(path: str | Path) -> VideoInfo:
    """Metadata of the first video stream. Raises ValueError when the file cannot be read as video."""
    import av

    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as exc:
        raise ValueError(f"cannot read {path} as a video: {exc}") from exc
    with container:
        if not container.streams.video:
            raise ValueError(f"{path} has no video stream")
        stream = container.streams.video[0]
        cc = stream.codec_context
        if cc is None:
            raise ValueError(f"{path}: no decoder for its video codec")
        fps = _fps(stream)
        n_frames = int(stream.frames or 0)
        duration = 0.0
        if stream.duration and stream.time_base:
            duration = float(stream.duration * stream.time_base)
        elif container.duration:
            duration = container.duration / 1_000_000
        elif n_frames and fps:
            duration = n_frames / fps
        if not n_frames and duration and fps:
            n_frames = round(duration * fps)
        width, height, trc, rotation = cc.width, cc.height, cc.color_trc, 0
        try:
            frame = next(container.decode(stream), None)
        except av.error.FFmpegError as exc:
            log.warning("could not decode the first frame of %s: %s", path, exc)
            frame = None
        if frame is not None:
            rotation = _frame_rotation(frame, stream)
            width, height = width or frame.width, height or frame.height
            trc = frame.color_trc if frame.color_trc not in (None, 2) else trc  # 2 = unspecified
        if rotation in (90, 270):
            width, height = height, width
        tags = _tags(container, stream)
        return VideoInfo(
            duration_s=round(duration, 6),
            fps=fps,
            n_frames=n_frames,
            width=int(width),
            height=int(height),
            rotation_deg=rotation,
            codec=cc.name,
            pix_fmt=cc.pix_fmt,
            is_hdr=trc in _HDR_TRC,
            focal_35mm=_float(_tag(tags, "focal_35mm")),
            lens_model=_tag(tags, "lens_model"),
            creation_time=_tag(tags, "creation_time"),
            make=_tag(tags, "make"),
            model=_tag(tags, "model"),
            software=_tag(tags, "software"),
        )


def video_intrinsics(info: VideoInfo, width: int, height: int) -> np.ndarray | None:
    """K from the QuickTime 35 mm equivalent focal length on the frame diagonal, or None when it is missing."""
    return intrinsics_from_35mm(info.focal_35mm, width, height)


def _blur_keep_mask(sharpness: np.ndarray, rel: float) -> np.ndarray:
    """Keep frames at or above rel times the median sharpness of their neighbourhood.

    A local median keeps low-texture stretches (a blank wall) that a global median would drop, which would
    leave gaps in the walkthrough. Clips of up to 15 keyframes use the global median.
    """
    n = len(sharpness)
    keep = np.ones(n, bool)
    if n < 3 or rel <= 0:
        return keep
    width = 2 * _LOCAL_MEDIAN_HALF + 1
    for i in range(n):
        lo = min(max(0, i - _LOCAL_MEDIAN_HALF), max(0, n - width))  # full-width window, shifted at the ends
        keep[i] = sharpness[i] >= rel * float(np.median(sharpness[lo:lo + width]))
    return keep


def sample_frames(
    path: str | Path,
    out_dir: str | Path,
    target_fps: float = 2.0,
    max_frames: int = 240,
    min_sharpness_rel: float = 0.35,
    *,
    jpeg_quality: int = 92,
    sharpness_px: int = 480,
) -> list[FrameRecord]:
    """Write the sharpest frame of each 1/target_fps window as out_dir/frame_<index>.jpg, upright 8-bit RGB.

    The rate drops below target_fps when needed so at most max_frames come out; frames under
    min_sharpness_rel times the local median sharpness are then dropped. A decode error part-way keeps the
    frames read so far. out_dir/frames.json records the parameters and the result.
    """
    import av
    import cv2
    from av.video.reformatter import VideoReformatter
    from PIL import Image

    if target_fps <= 0:
        raise ValueError(f"target_fps must be positive, got {target_fps}")
    info = probe(path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("frame_*.jpg"):
        old.unlink()
    duration = info.duration_s or (info.n_frames / info.fps if info.fps else 0.0)
    rate = float(target_fps)
    if duration > 0 and max_frames > 0:
        rate = min(rate, max_frames / duration)
    fallback_fps = info.fps or 30.0
    reformatter = VideoReformatter()

    def sharpness(frame: Any) -> float:
        s = sharpness_px / max(frame.width, frame.height, 1)
        w, h = max(8, round(frame.width * s)), max(8, round(frame.height * s))
        grey = reformatter.reformat(frame, width=w, height=h, format="gray").to_ndarray()
        return float(cv2.Laplacian(grey, cv2.CV_32F).var())

    def write(best: tuple[float, int, float, Any]) -> FrameRecord:
        score, index, t, frame = best
        rgb = rotate_upright(frame.to_ndarray(format="rgb24"), info.rotation_deg)
        out = out_dir / f"frame_{index:05d}.jpg"
        Image.fromarray(rgb).save(out, quality=jpeg_quality)
        return FrameRecord(index, round(t, 4), out, round(score, 3), int(rgb.shape[1]), int(rgb.shape[0]))

    records: list[FrameRecord] = []
    best: tuple[float, int, float, Any] | None = None
    window = -1
    t0: float | None = None
    index = -1
    decode_error: str | None = None
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        try:
            for frame in container.decode(stream):
                index += 1
                t_abs = frame.time if frame.time is not None else index / fallback_fps
                if t0 is None:
                    t0 = t_abs
                t = max(0.0, t_abs - t0)
                w = max(window, int(t * rate + 1e-9))
                if w != window:
                    if best is not None:
                        records.append(write(best))
                    best, window = None, w
                score = sharpness(frame)
                if best is None or score > best[0]:
                    best = (score, index, t, frame)
        except av.error.FFmpegError as exc:
            decode_error = f"{type(exc).__name__}: {exc}"
            log.warning("decoding %s stopped after frame %d: %s", path, index, exc)
    if best is not None:
        records.append(write(best))

    if max_frames > 0 and len(records) > max_frames:  # duration metadata was short or missing
        pick = set(np.unique(np.round(np.linspace(0, len(records) - 1, max_frames)).astype(int)).tolist())
        for i, r in enumerate(records):
            if i not in pick:
                r.path.unlink(missing_ok=True)
        records = [r for i, r in enumerate(records) if i in pick]

    keep = _blur_keep_mask(np.array([r.sharpness for r in records]), min_sharpness_rel)
    dropped = [r for r, k in zip(records, keep) if not k]
    for r in dropped:
        r.path.unlink(missing_ok=True)
    kept = [r for r, k in zip(records, keep) if k]
    if not kept:
        log.warning("no frames could be sampled from %s", path)
    log.info("sampled %d frames from %s (%d decoded, %.2f fps windows, %d dropped as blurred)",
             len(kept), Path(path).name, index + 1, rate, len(dropped))
    manifest = {
        "video": str(path),
        "params": {"target_fps": target_fps, "max_frames": max_frames, "min_sharpness_rel": min_sharpness_rel,
                   "jpeg_quality": jpeg_quality, "sharpness_px": sharpness_px},
        "window_fps": rate,
        "decoded_frames": index + 1,
        "decode_error": decode_error,
        "info": info.to_dict(),
        "frames": [{**asdict(r), "path": r.path.name} for r in kept],
        "dropped_blurred": [{"index": r.index, "t": r.t, "sharpness": r.sharpness} for r in dropped],
    }
    (out_dir / "frames.json").write_text(json.dumps(manifest, indent=1))
    return kept
