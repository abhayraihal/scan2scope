"""Doorway-photo registration.

A photo taken in room A through an open door also shows part of room B. MapAnything runs on that photo plus
B's photos; the run is aligned to B's existing frame on B's own cameras, which gives the photo's pose in B's
frame, and with its pose in A that yields the A-to-B transform, restricted to yaw, translation and scale
because both room frames are gravity aligned.
"""

from __future__ import annotations

import hashlib
import inspect
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from scan2scope.geometry.se3 import decompose_sim3, umeyama
from scan2scope.stitch.doors import (
    WALL_THICKNESS_PRIOR,
    WALL_THICKNESS_RANGE,
    door_pair_transform,
    facing_yaw,
    nearest_edge,
    room_polygon,
)
from scan2scope.stitch.solver import Door, Hypothesis, StitchRoom, inv2, rigid2, snap_yaw
from scan2scope.types import CameraView

log = logging.getLogger("scan2scope.stitch")

# runner(views, cache_key, cache) -> per-view predictions (T_wc and pts3d in the run's frame), or None
RunnerFn = Callable[[list[CameraView], dict, Any], Any]

MAX_CANDIDATE_VIEWS = 7


@dataclass
class DoorwayPhoto:
    room: int
    view: CameraView
    door: Door
    score: float


@dataclass
class Pred:
    T_wc: np.ndarray
    pts3d: np.ndarray | None = None


@dataclass
class Alignment:
    """Similarity from the run's frame into the candidate room's frame: x_room = s R x_run + t."""

    s: float
    R: np.ndarray
    t: np.ndarray
    rms: float  # camera centre residual in the room's frame, metres
    rot_err_deg: float  # median camera rotation disagreement between the two runs
    n: int
    method: str


@dataclass
class DoorHit:
    door: Door | None
    factor: float
    along: float = float("nan")
    across: float = float("nan")
    angle_deg: float = float("nan")


# --- the model call ----------------------------------------------------------------------------------

def _load_rgb(path: Path) -> np.ndarray:
    try:
        from scan2scope.ingest.images import load_image
    except ImportError:
        load_image = None
    if load_image is not None:
        out = load_image(path)
        return np.asarray(out[0] if isinstance(out, tuple) else out)
    from PIL import Image, ImageOps

    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        pass
    with Image.open(path) as im:
        return np.asarray(ImageOps.exif_transpose(im).convert("RGB"))


def mapanything_runner(views: list[CameraView], key: dict, cache: Any) -> Any:
    """Default runner: MapAnything on the views' image files with their intrinsics."""
    if any(v.image_path is None or not Path(v.image_path).is_file() for v in views):
        log.info("stitch: doorway registration skipped, image files are missing")
        return None
    from scan2scope.geometry.mapanything_backend import MapAnythingRunner

    images, intrinsics = [], []
    for v in views:
        rgb = _load_rgb(Path(v.image_path))
        K = np.asarray(v.K, float).copy()
        h, w = rgb.shape[:2]
        if v.width > 0 and v.height > 0 and (w, h) != (v.width, v.height):
            K[0] *= w / v.width
            K[1] *= h / v.height
        images.append(rgb)
        intrinsics.append(K)
    runner = MapAnythingRunner.get()
    if "cache" in inspect.signature(runner.infer).parameters:
        return runner.infer(images, intrinsics, key, cache=cache)
    return runner.infer(images, intrinsics, key)


def _to_numpy(x: Any) -> np.ndarray | None:
    if x is None:
        return None
    if hasattr(x, "detach"):
        x = x.detach().float().cpu().numpy()
    try:
        return np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return None


def _field(p: Any, names: tuple[str, ...]) -> Any:
    for n in names:
        v = p.get(n) if isinstance(p, dict) else getattr(p, n, None)
        if v is not None:
            return v
    return None


def _points(x: Any) -> np.ndarray | None:
    P = _to_numpy(x)
    if P is None or P.ndim < 3 or P.shape[-1] != 3:
        return None
    return P.reshape(P.shape[-3:])


def as_predictions(out: Any, n: int) -> list[Pred] | None:
    """Normalise runner output (list of ViewPrediction-like objects or dicts, or one batched dict)."""
    if out is None:
        return None
    pose_keys = ("T_wc", "camera_poses", "camera_pose", "pose")
    if isinstance(out, dict):
        T = _to_numpy(_field(out, pose_keys))
        P = _to_numpy(_field(out, ("pts3d", "pointmaps", "pointmap")))
        if T is None or T.shape != (n, 4, 4):
            return None
        ok = P is not None and P.ndim == 4 and len(P) == n
        return [Pred(T[i], P[i] if ok else None) for i in range(n)]
    try:
        out = list(out)
    except TypeError:
        return None
    if len(out) != n:
        return None
    preds = []
    for p in out:
        T = _to_numpy(_field(p, pose_keys))
        if T is None or T.size != 16:
            return None
        preds.append(Pred(T.reshape(4, 4), _points(_field(p, ("pts3d", "pointmap", "pts3d_world")))))
    return preds


_SHA: dict[str, str] = {}


def _file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    p = Path(path)
    try:
        st = p.stat()
    except OSError:
        return None
    memo = f"{p.resolve()}:{st.st_size}:{st.st_mtime_ns}"
    if memo not in _SHA:
        h = hashlib.sha256()
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        _SHA[memo] = h.hexdigest()
    return _SHA[memo]


def run_key(views: list[CameraView]) -> dict:
    from scan2scope.config import MODELS

    items = []
    for v in views:
        sha = _file_sha256(v.image_path)
        items.append({"sha256": sha, "id": None if sha else v.id, "size": [int(v.width), int(v.height)],
                      "K": np.round(np.asarray(v.K, float), 2).tolist()})
    return {"op": "stitch.doorway_registration", "version": 1, "model": MODELS["mapanything"].revision,
            "views": items}


# --- doorway photos ----------------------------------------------------------------------------------

def _pose_ok(view: CameraView) -> bool:
    T = np.asarray(view.T_wc, float)
    K = np.asarray(view.K, float)
    return T.shape == (4, 4) and K.shape == (3, 3) and bool(np.isfinite(T).all() and np.isfinite(K).all())


def _project(view: CameraView, P: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    T = np.asarray(view.T_wc, float)
    K = np.asarray(view.K, float)
    X = (P - T[:3, 3]) @ T[:3, :3]
    z = X[:, 2]
    zs = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * X[:, 0] / zs + K[0, 1] * X[:, 1] / zs + K[0, 2]
    v = K[1, 1] * X[:, 1] / zs + K[1, 2]
    return u, v, z


def _through_factor(view: CameraView, door: Door, z0: float, h: float) -> float:
    """Share of point-map pixels in the door's image box that lie beyond the wall, i.e. seen through it."""
    if view.pointmap is None:
        return 1.0
    pm = np.asarray(view.pointmap, float)
    if pm.ndim != 3 or pm.shape[2] != 3:
        return 1.0
    half = 0.5 * door.width * door.tangent
    corners = np.array([[*(door.center + sx * half), z] for sx in (-1, 1)
                        for z in (z0 + 0.1, z0 + min(h, 2.0) - 0.1)])
    u, v, z = _project(view, corners)
    if (z <= 0.05).any():
        return 1.0
    hh, ww = pm.shape[:2]
    W, H = float(view.width), float(view.height)
    j0, j1 = ((np.clip([u.min(), u.max()], 0, W) + 0.5) * ww / W - 0.5).astype(int)
    i0, i1 = ((np.clip([v.min(), v.max()], 0, H) + 0.5) * hh / H - 0.5).astype(int)
    j0, i0 = max(j0, 0), max(i0, 0)
    if j1 <= j0 or i1 <= i0:
        return 1.0
    patch = pm[i0:i1 + 1, j0:j1 + 1].reshape(-1, 3)
    ok = np.isfinite(patch).all(1)
    if view.valid is not None and np.shape(view.valid) == (hh, ww):
        ok &= np.asarray(view.valid)[i0:i1 + 1, j0:j1 + 1].reshape(-1)
    if ok.sum() < 20:
        return 1.0
    beyond = (patch[ok, :2] - door.center) @ door.normal < -0.15
    return float(0.4 + 0.6 * beyond.mean())


def view_door_score(view: CameraView, door: Door, floor_z: float) -> float:
    """How well a view looks through a door from inside its room: 0 when it does not, up to 1."""
    if not _pose_ok(view) or view.width <= 0 or view.height <= 0:
        return 0.0
    T = np.asarray(view.T_wc, float)
    c, f = T[:3, 3], T[:3, 2]
    rel = c[:2] - door.center
    inside = float(rel @ door.normal)
    fxy = f[:2]
    nf = float(np.linalg.norm(fxy))
    if inside < 0.2 or nf < 0.3:
        return 0.0
    fxy = fxy / nf
    facing = float(-(fxy @ door.normal))
    if facing < 0.5:
        return 0.0
    z0 = float(floor_z) if np.isfinite(floor_z) else float(c[2]) - 1.4
    h = door.height or 2.0
    zm = z0 + 0.5 * min(h, 2.2)
    half = 0.5 * door.width * door.tangent
    P = np.array([[*door.center, zm], [*(door.center - half), zm], [*(door.center + half), zm]])
    u, v, z = _project(view, P)
    W, H = float(view.width), float(view.height)
    if z[0] <= 0.3 or not 0.0 <= u[0] <= W or not -0.25 * H <= v[0] <= 1.25 * H:
        return 0.0
    centrality = 1.0 - abs(u[0] - W / 2) / (W / 2)
    if (z[1:] > 0.05).all():
        lo, hi = sorted(u[1:])
        vis = float(np.clip((min(hi, W) - max(lo, 0.0)) / max(hi - lo, 1e-6), 0.0, 1.0))
    else:
        vis = 0.5
    # does the optical axis itself pass through the opening?
    hit = c[:2] + (inside / facing) * fxy
    hit_f = 1.0 if abs((hit - door.center) @ door.tangent) <= 0.5 * door.width + 0.1 else 0.6
    dist = float(np.linalg.norm(rel))
    dist_f = 1.0 if dist <= 4.0 else float(np.exp(-(dist - 4.0) / 2.0))
    through = _through_factor(view, door, z0, h)
    return float(facing * (0.5 + 0.5 * centrality) * vis * hit_f * dist_f * through)


def find_doorway_photos(room: StitchRoom, *, min_score: float = 0.3, per_door: int = 1) -> list[DoorwayPhoto]:
    """The best views of each door or passage seen from inside the room."""
    if room.scene is None:
        return []
    out = []
    for door in room.doors:
        scored = sorted(((view_door_score(v, door, room.room.floor_z), k)
                         for k, v in enumerate(room.scene.views)), key=lambda s: (-s[0], s[1]))
        for s, k in scored[:per_door]:
            if s >= min_score:
                out.append(DoorwayPhoto(room.index, room.scene.views[k], door, s))
    return out


def _posed_views(room: StitchRoom) -> list[CameraView]:
    return [v for v in room.scene.views if _pose_ok(v)] if room.scene is not None else []


def _width_ratio(a: float, b: float) -> float:
    return min(a, b) / max(a, b)


def rank_candidates(rooms: list[StitchRoom], photo: DoorwayPhoto,
                    hints: dict[tuple[int, str], int] | None = None) -> list[int]:
    """Rooms worth registering a doorway photo into: the room door matching put behind this door (hints), then
    rooms with a compatible door width, then folder neighbours and rooms with many doors."""
    a = rooms[photo.room]
    hinted = (hints or {}).get((a.index, photo.door.id))

    def prio(b: StitchRoom) -> float:
        best = max((_width_ratio(d.width, photo.door.width) for d in b.doors), default=0.0)
        compat = 2.0 if best >= 0.8 else 1.0 if best >= 0.6 else 0.0
        return 10.0 * (b.index == hinted) + compat + 1.0 / abs(a.index - b.index) + 0.1 * min(len(b.doors), 4)

    cands = [b.index for b in rooms if b.index != a.index and _posed_views(b)]
    return sorted(cands, key=lambda k: (-prio(rooms[k]), k))


def select_views(room: StitchRoom, door_a: Door) -> list[CameraView]:
    """At most 7 of the candidate's views, preferring those that see a door compatible with door_a."""
    views = _posed_views(room)
    if len(views) <= MAX_CANDIDATE_VIEWS:
        return views
    doors = [d for d in room.doors if _width_ratio(d.width, door_a.width) >= 0.6] or room.doors
    score = [max((view_door_score(v, d, room.room.floor_z) for d in doors), default=0.0) for v in views]
    keep = sorted(sorted(range(len(views)), key=lambda k: (-score[k], k))[:MAX_CANDIDATE_VIEWS])
    return [views[k] for k in keep]


# --- registration geometry ---------------------------------------------------------------------------

def _orth(M: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(M)
    if np.linalg.det(U @ Vt) < 0:
        U[:, -1] *= -1
    return U @ Vt


def _angle_deg(Ra: np.ndarray, Rb: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(Ra.T @ Rb) - 1.0) / 2.0, -1.0, 1.0))))


def align_run_to_room(T_room: list[np.ndarray], T_run: list[np.ndarray]) -> Alignment | None:
    """Similarity mapping the run's frame onto the room's frame from the same cameras' poses in both."""
    pairs = [(np.asarray(a, float), np.asarray(b, float)) for a, b in zip(T_room, T_run)]
    pairs = [(a, b) for a, b in pairs
             if a.shape == (4, 4) and b.shape == (4, 4) and np.isfinite(a).all() and np.isfinite(b).all()]
    n = len(pairs)
    if n == 0:
        return None
    Rr = [_orth(a[:3, :3]) for a, _ in pairs]
    Rq = [_orth(b[:3, :3]) for _, b in pairs]
    cr = np.array([a[:3, 3] for a, _ in pairs])
    cq = np.array([b[:3, 3] for _, b in pairs])
    R = _orth(sum(r @ q.T for r, q in zip(Rr, Rq)))
    s = t = None
    method = "rotation_mean"
    if n >= 3 and np.linalg.svd(cr - cr.mean(0), compute_uv=False)[1] / np.sqrt(n) > 0.15:
        su, Ru, tu = decompose_sim3(umeyama(cq, cr, with_scale=True))
        if su > 0 and _angle_deg(Ru, R) < 15.0:
            s, R, t, method = su, Ru, tu, "umeyama_centres"
    if s is None:
        dq, dr = cq - cq.mean(0), cr - cr.mean(0)
        den = float((dq ** 2).sum())
        if n >= 2 and den > n * 0.15 ** 2:
            s = float((dr * (dq @ R.T)).sum() / den)
        else:
            s, method = 1.0, method + "+unit_scale"
        t = cr.mean(0) - s * R @ cq.mean(0)
    if not np.isfinite(s) or s <= 0.05:
        return None
    res = cr - (s * cq @ R.T + t)
    rms = float(np.sqrt((res ** 2).sum(1).mean()))
    rot = float(np.median([_angle_deg(r, R @ q) for r, q in zip(Rr, Rq)]))
    return Alignment(float(s), R, np.asarray(t, float), rms, rot, n, method)


def _depth(P: np.ndarray, T_wc: np.ndarray) -> np.ndarray:
    return ((P - T_wc[:3, 3]) @ _orth(T_wc[:3, :3]))[:, 2]


def depth_ratio(view: CameraView, pred: Pred, grid: tuple[int, int] = (24, 32)) -> float | None:
    """Depth of the view's points in its room frame over their depth in the run, on a shared pixel grid."""
    if view.pointmap is None or pred.pts3d is None:
        return None
    pa, pr = np.asarray(view.pointmap, float), np.asarray(pred.pts3d, float)
    if pa.ndim != 3 or pr.ndim != 3 or pa.shape[2] != 3 or pr.shape[2] != 3:
        return None
    gy, gx = [(np.arange(g) + 0.5) / g for g in grid]

    def sample(P: np.ndarray) -> np.ndarray:
        ii = np.minimum((gy * P.shape[0]).astype(int), P.shape[0] - 1)
        jj = np.minimum((gx * P.shape[1]).astype(int), P.shape[1] - 1)
        return P[np.ix_(ii, jj)].reshape(-1, 3)

    A, B = sample(pa), sample(pr)
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    if view.valid is not None and np.shape(view.valid) == pa.shape[:2]:
        ok &= sample(np.repeat(np.asarray(view.valid, float)[..., None], 3, axis=2))[:, 0] > 0.5
    if ok.sum() < 30:
        return None
    za, zb = _depth(A[ok], np.asarray(view.T_wc, float)), _depth(B[ok], np.asarray(pred.T_wc, float))
    keep = (za > 0.1) & (zb > 0.1)
    if keep.sum() < 30:
        return None
    return float(np.median(za[keep] / zb[keep]))


def points_factor(pred: Pred, al: Alignment, room: StitchRoom) -> tuple[float, dict]:
    """Agreement of the doorway photo's points that land inside the candidate room with that room's points."""
    if pred.pts3d is None or room.scene is None or len(room.scene.points) < 50:
        return 1.0, {"points_check": "unavailable"}
    P = np.asarray(pred.pts3d, float).reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    if len(P) > 4000:
        P = P[np.linspace(0, len(P) - 1, 4000).astype(int)]
    P = al.s * P @ al.R.T + al.t
    poly_xy = room_polygon(room.room)
    if poly_xy is None or len(P) == 0:
        return 1.0, {"points_check": "unavailable"}
    # the room's own walls count; the far face of a shared wall is a wall thickness away and does not
    inner = Polygon(poly_xy).buffer(0.05)
    if inner.is_empty:
        return 1.0, {"points_check": "unavailable"}
    keep = shapely.contains_xy(inner, P[:, 0], P[:, 1])
    fz, cz = room.room.floor_z, room.room.ceiling_z
    if np.isfinite(fz) and np.isfinite(cz) and cz > fz:
        keep &= (P[:, 2] > fz + 0.05) & (P[:, 2] < cz - 0.05)
    Q = P[keep]
    if len(Q) < 50:
        return 1.0, {"points_check": "too_few_inside", "points_n": len(Q)}
    tree = room.aux.get("kdtree")
    if tree is None:
        pts = np.asarray(room.scene.points, float)
        pts = pts[np.isfinite(pts).all(1)]
        if len(pts) > 30000:
            pts = pts[np.linspace(0, len(pts) - 1, 30000).astype(int)]
        tree = room.aux["kdtree"] = cKDTree(pts)
    med = float(np.median(tree.query(Q, k=1)[0]))
    return float(np.clip(np.exp(-0.5 * (med / 0.2) ** 2), 0.05, 1.0)), {"points_median_m": round(med, 4),
                                                                          "points_n": len(Q)}


def match_door(room_b: StitchRoom, door_a: Door, oa_b: np.ndarray, na_b: np.ndarray) -> DoorHit:
    """Does a's door, mapped into b's frame, land across the wall from one of b's doors?"""
    lo, hi = WALL_THICKNESS_RANGE
    best: DoorHit | None = None
    for db in room_b.doors:
        ratio = _width_ratio(db.width, door_a.width)
        if ratio < 0.6:
            continue
        rel = oa_b - db.center
        across = float(-(rel @ db.normal))
        along = float(rel @ db.tangent)
        ang = float(np.degrees(np.arccos(np.clip(-(na_b @ db.normal), -1.0, 1.0))))
        dev = max(lo - 0.03 - across, across - hi - 0.05, 0.0)
        f = np.exp(-0.5 * ((along / 0.2) ** 2 + (dev / 0.15) ** 2 + (ang / 20.0) ** 2))
        f *= np.exp(-0.5 * ((1.0 - ratio) / 0.12) ** 2)
        if best is None or f > best.factor:
            best = DoorHit(db, float(f), along, across, ang)
    if best is not None and best.factor >= 0.05:
        return best
    # b's layout may have missed the door: a landing on b's boundary, facing out of b, still counts a little
    P = room_polygon(room_b.room)
    edge = nearest_edge(P, oa_b) if P is not None else None
    if edge is None:
        return DoorHit(None, 0.0)
    dist, n_in = edge
    ang = float(np.degrees(np.arccos(np.clip(-(na_b @ n_in), -1.0, 1.0))))
    return DoorHit(None, float(0.3 * np.exp(-0.5 * ((dist / 0.2) ** 2 + (ang / 20.0) ** 2))), angle_deg=ang)


def register_photo(room_a: StitchRoom, photo: DoorwayPhoto, room_b: StitchRoom, b_views: list[CameraView],
                   preds: list[Pred]) -> tuple[Hypothesis | None, dict]:
    door_a = photo.door
    rec: dict[str, Any] = {"room_a": room_a.id, "room_b": room_b.id, "view": photo.view.id,
                           "opening_a": door_a.id, "source": "doorway_photo", "n_views": len(b_views)}
    al = align_run_to_room([v.T_wc for v in b_views], [p.T_wc for p in preds[1:]])
    Td = np.asarray(preds[0].T_wc, float)
    if al is None or Td.shape != (4, 4) or not np.isfinite(Td).all():
        rec["status"] = "alignment_failed"
        return None, rec
    R_bd = al.R @ _orth(Td[:3, :3])
    c_bd = al.s * al.R @ Td[:3, 3] + al.t
    TA = np.asarray(photo.view.T_wc, float)
    R_ad, c_ad = _orth(TA[:3, :3]), TA[:3, 3]
    s_a = depth_ratio(photo.view, preds[0])
    s_ba = al.s / s_a if s_a else al.s
    R_ba = R_bd @ R_ad.T
    tilt = float(np.degrees(np.arccos(np.clip(R_ba[2, 2], -1.0, 1.0))))
    yaw_ba = float(np.arctan2(R_ba[1, 0] - R_ba[0, 1], R_ba[0, 0] + R_ba[1, 1]))
    Rz = rigid2(yaw_ba)[:2, :2]
    oa_b = c_bd[:2] + s_ba * Rz @ (door_a.center - c_ad[:2])
    na_b = Rz @ door_a.normal
    hit = match_door(room_b, door_a, oa_b, na_b)
    if hit.door is not None:
        # the door pair fixes the yaw and the along-wall position; the wall thickness is the registration's
        # estimate shrunk towards the prior by how well the run's cameras aligned
        theta, delta, snapped = snap_yaw(facing_yaw(door_a, hit.door), room_a.manhattan, room_b.manhattan)
        w = 0.05 ** 2 / (0.05 ** 2 + float(np.clip(al.rms, 0.03, 0.3)) ** 2)
        thick = float(np.clip(w * hit.across + (1.0 - w) * WALL_THICKNESS_PRIOR, *WALL_THICKNESS_RANGE))
        T_ab = door_pair_transform(door_a, hit.door, theta, thick)
    else:
        theta, delta, snapped = snap_yaw(-yaw_ba, room_a.manhattan, room_b.manhattan)
        T_ba = rigid2(-theta)
        T_ba[:2, 2] = oa_b - T_ba[:2, :2] @ door_a.center
        T_ab = inv2(T_ba)
        thick = None
    f_pts, pinfo = points_factor(preds[0], al, room_b)
    factors = {
        "align": float(np.exp(-0.5 * (al.rms / 0.15) ** 2)),
        "rotation": float(np.exp(-0.5 * (al.rot_err_deg / 6.0) ** 2)),
        "tilt": float(np.exp(-0.5 * (tilt / 6.0) ** 2)),
        "door": hit.factor,
        "manhattan": float(np.exp(-0.5 * (np.degrees(delta) / 8.0) ** 2)),
        "views": 1.0 if al.n >= 3 else 0.75 if al.n == 2 else 0.5,
        "points": f_pts,
    }
    score = 0.95 * float(np.prod(list(factors.values())))
    rec.update({
        "status": "ok", "opening_b": hit.door.id if hit.door else None, "score": round(score, 4),
        "factors": {k: round(v, 4) for k, v in factors.items()}, "align_method": al.method,
        "center_rms_m": round(al.rms, 4), "rotation_err_deg": round(al.rot_err_deg, 3),
        "tilt_deg": round(tilt, 3),
        "relative_scale": round(float(s_ba), 4), "scale_from_depth": s_a is not None,
        "door_along_m": None if not np.isfinite(hit.along) else round(hit.along, 4),
        "door_across_m": None if not np.isfinite(hit.across) else round(hit.across, 4),
        "thickness_m": None if thick is None else round(thick, 4),
        "yaw_snap_deg": round(float(np.degrees(delta)), 3),
        "snapped": bool(snapped), **pinfo,
    })
    if not np.isfinite(score) or score <= 0.0:
        rec["status"] = "rejected"
        return None, rec
    return Hypothesis(room_a.index, room_b.index, T_ab, score, "doorway_photo", door_a.id,
                      hit.door.id if hit.door else None, scale_ba=float(s_ba), evidence=rec), rec


def _run(runner: RunnerFn, view: CameraView, room_b: StitchRoom, door_a: Door, cache: Any,
         flags: list[str]) -> tuple[list[CameraView], list[Pred]] | str:
    """One registration run, or a status string: no_views, unavailable, import_error, failed, bad_output."""
    b_views = select_views(room_b, door_a)
    if not b_views:
        return "no_views"
    views = [view] + b_views
    try:
        out = runner(views, run_key(views), cache)
    except ImportError as exc:
        log.warning("stitch: doorway registration unavailable: %s", exc)
        return "import_error"
    except Exception as exc:  # one failed run must not stop the stitch
        log.warning("stitch: doorway registration run failed: %s", exc)
        flag = f"doorway_registration_failed:{type(exc).__name__}"
        if flag not in flags:
            flags.append(flag)
        return "failed"
    if out is None:
        return "unavailable"
    preds = as_predictions(out, len(views))
    if preds is None:
        log.warning("stitch: runner returned %s, expected %d view predictions", type(out).__name__,
                    len(views))
        return "bad_output"
    return b_views, preds


def register_doorways(rooms: list[StitchRoom], runner: RunnerFn, cache: Any, *, max_runs: int = 24,
                      per_photo: int = 3, hints: dict[tuple[int, str], int] | None = None,
                      settle_score: float = 0.6) -> tuple[list[Hypothesis], list[dict], list[str], dict]:
    """Doorway-photo hypotheses for every room, within a budget of model runs.

    Candidates are tried best first across all photos; a photo stops once it lands on a candidate's door with
    at least settle_score.
    """
    photos = [p for r in rooms for p in find_doorway_photos(r)]
    queue = sorted((rank, -p.score, pi, b) for pi, p in enumerate(photos)
                   for rank, b in enumerate(rank_candidates(rooms, p, hints)[:per_photo]))
    runs: dict[tuple, Any] = {}
    hyps: list[Hypothesis] = []
    records: list[dict] = []
    flags: list[str] = []
    settled: set[int] = set()
    n_runs = skipped = unavailable = failed_in_row = 0
    for _, _, pi, b in queue:
        p = photos[pi]
        if pi in settled:
            continue
        rk = (p.room, p.view.id, b)
        if rk not in runs:
            if n_runs >= max_runs:
                skipped += 1
                continue
            n_runs += 1
            runs[rk] = _run(runner, p.view, rooms[b], p.door, cache, flags)
            unavailable += runs[rk] == "unavailable"
            failed_in_row = failed_in_row + 1 if runs[rk] == "failed" else 0
        res = runs[rk]
        if isinstance(res, str):
            records.append({"room_a": rooms[p.room].id, "room_b": rooms[b].id, "view": p.view.id,
                            "opening_a": p.door.id, "source": "doorway_photo", "status": res})
            # a missing model or repeated failures will not get better on the next photo
            if res == "import_error":
                flags.append("doorway_registration_unavailable:import_error")
                break
            if failed_in_row >= 3:
                flags.append("doorway_registration_stopped:repeated_failures")
                break
            continue
        h, rec = register_photo(rooms[p.room], p, rooms[b], *res)
        records.append(rec)
        if h is not None:
            hyps.append(h)
            if h.opening_b is not None and h.score >= settle_score:
                settled.add(pi)
    if skipped:
        flags.append(f"doorway_runs_skipped:{skipped}")
    if n_runs and unavailable == n_runs:
        flags.append("doorway_registration_unavailable")
    stats = {"photos": len(photos), "runs": n_runs, "skipped_runs": skipped, "hypotheses": len(hyps),
             "settled_photos": len(settled)}
    log.info("stitch: %d doorway photos, %d registration runs, %d hypotheses", len(photos), n_runs, len(hyps))
    return hyps, records, flags, stats
