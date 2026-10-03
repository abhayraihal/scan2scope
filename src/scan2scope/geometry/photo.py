"""Photo tier: one MapAnything run per room folder; each room gets its own gravity-aligned metric frame.

The image loading and folder listing here are thin stand-ins for scan2scope.ingest.images; integration can
switch to the ingest versions once they land.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

from scan2scope.geometry import gravity
from scan2scope.geometry.mapanything_backend import MapAnythingRunner, OutputCacheLike
from scan2scope.geometry.pointmaps import normals_from_pointmap
from scan2scope.types import CameraView, Scene

log = logging.getLogger("scan2scope.geometry")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}
MAX_PHOTOS = 8
MAX_IMAGE_SIDE = 4096
VOXEL = 0.02
MAX_POINTS = 400_000
SCALE_LOG_SIGMA = 0.08
MIN_WEIGHT = 0.1  # drop pixels with MapAnything confidence below about 1.4
NORMAL_STEP = 2
FILM_DIAGONAL_MM = math.hypot(36.0, 24.0)
LOW_LIGHT_ISO = 1600.0
LOW_LIGHT_EXPOSURE_S = 1.0 / 15.0
MAX_REFINE_DEG = 30.0

_heif_registered = False


@dataclass
class PhotoExif:
    focal_35mm: float | None = None
    iso: float | None = None
    exposure_s: float | None = None
    make: str | None = None
    model: str | None = None

    @property
    def low_light(self) -> bool:
        return bool((self.iso is not None and self.iso > LOW_LIGHT_ISO)
                    or (self.exposure_s is not None and self.exposure_s > LOW_LIGHT_EXPOSURE_S))


def _register_heif() -> None:
    global _heif_registered
    if not _heif_registered:
        try:
            from pillow_heif import register_heif_opener

            register_heif_opener()
        except ImportError:
            log.warning("pillow-heif is not installed; HEIC photos cannot be read")
        _heif_registered = True


def _number(v: Any) -> float | None:
    if isinstance(v, (tuple, list)):
        v = v[0] if v else None
    if isinstance(v, bytes):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) else None


def read_exif(im: Image.Image) -> PhotoExif:
    try:
        exif = im.getexif()
    except Exception:
        return PhotoExif()
    try:
        sub = exif.get_ifd(0x8769)
    except Exception:
        sub = {}

    def tag(t: int) -> Any:
        return sub.get(t, exif.get(t))

    def text(t: int) -> str | None:
        v = exif.get(t)
        return str(v).strip("\x00 ") if v else None

    return PhotoExif(focal_35mm=_number(tag(0xA405)), iso=_number(tag(0x8827)), exposure_s=_number(tag(0x829A)),
                     make=text(0x010F), model=text(0x0110))


def load_photo(path: Path, max_side: int = MAX_IMAGE_SIDE) -> tuple[np.ndarray, PhotoExif, tuple[int, int]]:
    """Upright RGB uint8 array, EXIF, and the upright size of the file. Photos whose long side exceeds max_side
    are downscaled uniformly, so the array still spans the whole photo."""
    _register_heif()
    with Image.open(path) as im:
        exif = read_exif(im)
        up = ImageOps.exif_transpose(im).convert("RGB")
    size = up.size
    if max(size) > max_side:
        f = max_side / max(size)
        up = up.resize((max(1, round(size[0] * f)), max(1, round(size[1] * f))), Image.Resampling.LANCZOS)
    return np.asarray(up), exif, size


def intrinsics_from_exif(exif: PhotoExif, width: int, height: int) -> np.ndarray | None:
    """Pinhole K from the 35 mm equivalent focal length, converted on the image diagonal."""
    f35 = exif.focal_35mm
    if f35 is None or not 5.0 <= f35 <= 800.0:
        return None
    f = f35 / FILM_DIAGONAL_MM * math.hypot(width, height)
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def natural_key(name: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", name)]


def _images_in(folder: Path) -> list[Path]:
    files = [p for p in folder.iterdir()
             if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in IMAGE_SUFFIXES]
    return sorted(files, key=lambda p: natural_key(p.name))


def list_room_folders(root: Path) -> list[tuple[str, list[Path]]]:
    """Room folders in natural order with their photos. A folder of loose photos is one room; a single wrapper
    folder (as zips often add) is descended into."""
    root = Path(root)
    if root.is_file():
        return [(root.parent.name or root.stem, [root])] if root.suffix.lower() in IMAGE_SUFFIXES else []
    subdirs = sorted([d for d in root.iterdir() if d.is_dir() and not d.name.startswith((".", "__"))],
                     key=lambda d: natural_key(d.name))
    rooms = [(d.name, _images_in(d)) for d in subdirs]
    rooms = [r for r in rooms if r[1]]
    if rooms:
        return rooms
    loose = _images_in(root)
    if loose:
        return [(root.name, loose)]
    if len(subdirs) == 1:
        return list_room_folders(subdirs[0])
    return []


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def spread_indices(n: int, k: int) -> list[int]:
    if n <= k:
        return list(range(n))
    return sorted({round(x) for x in np.linspace(0, n - 1, k).tolist()})


def gravity_alignment(T_wcs: list[np.ndarray], normals: np.ndarray, weights: np.ndarray
                      ) -> tuple[np.ndarray, dict[str, Any]]:
    """Rotation taking world up to +z. The hint is minus the mean camera y axis (OpenCV y points down in an
    upright image); gravity.estimate_up refines it on floor and ceiling normals."""
    ys = np.array([np.asarray(T)[:3, 1] for T in T_wcs], float)
    down = ys.mean(0)
    strength = float(np.linalg.norm(down))
    if not np.isfinite(down).all() or strength < 1e-6:
        down = ys[0]
    hint = -down / np.linalg.norm(down)
    up, refined, angle = hint, False, 0.0
    if len(normals) >= 50:
        cand = gravity.estimate_up(np.asarray(normals, float), np.asarray(weights, float), hint)
        if np.isfinite(cand).all():
            angle = float(np.degrees(np.arccos(np.clip(cand @ hint, -1.0, 1.0))))
            if angle <= MAX_REFINE_DEG:
                up, refined = cand, True
    R = gravity.align_up_to_z(up)
    info = {"up_in_model": [float(x) for x in up], "hint": [float(x) for x in hint], "hint_strength": strength,
            "refined": refined, "refine_angle_deg": angle, "R": R.tolist()}
    return R, info


def view_samples(pts: np.ndarray, mask: np.ndarray, weight: np.ndarray, cam_center: np.ndarray, *,
                 stride: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Points, unit normals (towards the camera) and weights of usable pixels, plus their pixel mask."""
    nrm, ok = normals_from_pointmap(pts, mask, cam_center, step=NORMAL_STEP)
    sel = ok & mask & (weight >= MIN_WEIGHT) & np.isfinite(pts).all(-1)
    if stride > 1:
        grid = np.zeros_like(sel)
        grid[::stride, ::stride] = True
        sel &= grid
    return pts[sel].astype(np.float32), nrm[sel].astype(np.float32), weight[sel].astype(np.float32), sel


def fuse_points(points: np.ndarray, normals: np.ndarray, weights: np.ndarray, view_index: np.ndarray,
                cam_centers: np.ndarray, *, voxel: float = VOXEL, max_points: int = MAX_POINTS,
                score: np.ndarray | None = None, seed: int = 0
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Confidence-weighted voxel average. Each voxel keeps the view of its best-scoring point (default: highest
    weight), and its normal is re-oriented towards that view's camera."""
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32), np.zeros(0, np.float32),
             np.zeros(0, np.int64))
    sane = np.isfinite(points).all(1) & (np.abs(points) < 1e3).all(1)
    if not sane.any():
        return empty
    points, normals, weights, view_index = points[sane], normals[sane], weights[sane], view_index[sane]
    score = weights if score is None else score[sane]
    q = np.floor(points / voxel).astype(np.int64)
    q -= q.min(0)
    dims = q.max(0) + 1
    key = (q[:, 0] * dims[1] + q[:, 1]) * dims[2] + q[:, 2]
    _, inv = np.unique(key, return_inverse=True)
    inv = inv.reshape(-1)
    m = int(inv.max()) + 1
    w = weights.astype(np.float64) + 1e-6
    wsum = np.bincount(inv, weights=w, minlength=m)
    P = np.stack([np.bincount(inv, weights=w * points[:, k], minlength=m) for k in range(3)], 1) / wsum[:, None]
    N = np.stack([np.bincount(inv, weights=w * normals[:, k], minlength=m) for k in range(3)], 1)
    W = np.bincount(inv, weights=weights.astype(np.float64), minlength=m) / np.bincount(inv, minlength=m)
    order = np.lexsort((-score, inv))
    first = order[np.r_[True, inv[order][1:] != inv[order][:-1]]]
    V = view_index[first].astype(np.int64)
    nn = np.linalg.norm(N, axis=1)
    weak = nn < 0.3 * wsum
    N = N / np.maximum(nn, 1e-12)[:, None]
    N[weak] = normals[first[weak]]
    flip = ((cam_centers[V] - P) * N).sum(1) < 0
    N[flip] *= -1
    if m > max_points:
        keep = np.sort(np.random.default_rng(seed).choice(m, max_points, replace=False))
        P, N, W, V = P[keep], N[keep], W[keep], V[keep]
    return P.astype(np.float32), N.astype(np.float32), W.astype(np.float32), V


def rotate_pose(R: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    G = np.eye(4)
    G[:3, :3] = R
    return G @ T_wc


def build_room_scene(name: str, paths: list[Path], *, cache: OutputCacheLike | None = None,
                     runner: MapAnythingRunner | None = None, max_photos: int = MAX_PHOTOS) -> Scene | None:
    """Scene of one room folder, or None when no photo in it can be read."""
    flags: list[str] = []
    loaded = []
    for p in paths:
        try:
            img, exif, size = load_photo(p)
        except Exception as exc:
            log.warning("room %s: cannot read %s: %s", name, p.name, exc)
            flags.append(f"unreadable_photo:{p.name}")
            continue
        loaded.append((p, img, exif, size))
    if not loaded:
        return None
    n_total = len(loaded)
    if n_total > max_photos:
        loaded = [loaded[i] for i in spread_indices(n_total, max_photos)]
        flags.append(f"photos_capped:{n_total}->{len(loaded)}")
    if len(loaded) == 1:
        flags.append("thin")

    imgs = [im for _, im, _, _ in loaded]
    Ks = [intrinsics_from_exif(ex, im.shape[1], im.shape[0]) for _, im, ex, _ in loaded]
    runner = runner or MapAnythingRunner.get()
    key = {"tier": "photo", "room": name, "files": [file_sha256(p) for p, *_ in loaded]}
    preds = runner.infer(imgs, Ks, key=key, cache=cache)
    for pr in preds:
        flags.extend(f for f in pr.meta.get("flags", []) if f not in flags)

    weights = [pr.weight for pr in preds]
    samples = [view_samples(pr.pts3d, pr.mask, w, pr.T_wc[:3, 3]) for pr, w in zip(preds, weights)]
    nrm_all = np.concatenate([s[1] for s in samples]) if samples else np.zeros((0, 3))
    w_all = np.concatenate([s[2] for s in samples]) if samples else np.zeros(0)
    sub = slice(None, None, max(1, len(nrm_all) // 200_000))
    R, ginfo = gravity_alignment([pr.T_wc for pr in preds], nrm_all[sub], w_all[sub])

    views, pts, nrms, ws, vidx = [], [], [], [], []
    for i, ((path, img, exif, size), pr, (p, nr, w, _)) in enumerate(zip(loaded, preds, samples)):
        T_wc = rotate_pose(R, pr.T_wc)
        pm = (pr.pts3d @ R.T).astype(np.float32)
        pm[~pr.mask] = 0.0
        K_exif = intrinsics_from_exif(exif, size[0], size[1])
        views.append(CameraView(
            id=f"{name}/{path.name}", image_path=Path(path), width=int(size[0]), height=int(size[1]),
            K=pr.K_image(size[0], size[1]), T_wc=T_wc, pointmap=pr.uncrop(pm, 0.0),
            valid=pr.uncrop(pr.mask, False), conf=pr.uncrop(weights[i], 0.0), room_hint=name,
            meta={"exif": asdict(exif), "K_exif": None if K_exif is None else K_exif.tolist(),
                  "intrinsics_given": pr.intrinsics_given, "metric_scale": pr.metric_scale,
                  "conf_kind": "normalised"},
        ))
        pts.append(p @ R.T)
        nrms.append(nr @ R.T)
        ws.append(w)
        vidx.append(np.full(len(p), i, np.int64))
    centers = np.array([v.center for v in views])
    P, N, W, V = fuse_points(np.concatenate(pts), np.concatenate(nrms), np.concatenate(ws), np.concatenate(vidx),
                             centers, voxel=VOXEL, max_points=MAX_POINTS)
    if len(P) < 100:
        flags.append("few_points")
    valid_w = np.concatenate([w[pr.mask] for w, pr in zip(weights, preds)])
    n_exif = sum(K is not None for K in Ks)
    quality = {
        "n_photos": len(loaded), "n_photos_total": n_total, "exif_focal": n_exif == len(loaded),
        "n_exif_focal": n_exif, "low_light": any(ex.low_light for _, _, ex, _ in loaded),
        "n_low_light": sum(ex.low_light for _, _, ex, _ in loaded),
        "mean_conf": float(valid_w.mean()) if valid_w.size else 0.0,
        "valid_fraction": float(np.mean([pr.mask.mean() for pr in preds])), "thin": len(loaded) == 1,
    }
    meta = {"quality": quality, "metric_scale": [pr.metric_scale for pr in preds], "gravity": ginfo,
            "rotation_applied": R.tolist(), "flags": flags, "photos": [p.name for p, *_ in loaded],
            "drift": {"enabled": False, "reason": "photo rooms are single MapAnything runs"}}
    log.info("room %s: %d photos, %d points, mean conf %.2f%s", name, len(loaded), len(P), quality["mean_conf"],
             f", flags {flags}" if flags else "")
    return Scene(tier="photo", views=views, points=P, normals=N, weights=W, view_index=V,
                 scale_log_sigma=SCALE_LOG_SIGMA, room_hint=name, meta=meta)


def build_room_scenes(root: Path, work_dir: Path, *, cache: OutputCacheLike | None = None,
                      runner: MapAnythingRunner | None = None, max_photos: int = MAX_PHOTOS) -> list[Scene]:
    """One Scene per room folder, in natural folder order. Rooms with no readable photo are skipped and
    recorded as a flag on the first scene."""
    rooms = list_room_folders(Path(root))
    if not rooms:
        raise ValueError(f"no photos found under {root}; expected one folder of photos per room")
    scenes: list[Scene] = []
    skipped: list[str] = []
    for name, paths in rooms:
        scene = build_room_scene(name, paths, cache=cache, runner=runner, max_photos=max_photos)
        if scene is None:
            log.warning("room %s skipped: no readable photo", name)
            skipped.append(name)
        else:
            scenes.append(scene)
    if not scenes:
        raise ValueError(f"none of the photos under {root} could be read")
    if skipped:
        scenes[0].meta["flags"].extend(f"room_skipped:{n}" for n in skipped)
    return scenes
