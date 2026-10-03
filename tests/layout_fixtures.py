"""Synthetic scenes for the layout tests.

A property is a union of axis-aligned free-space boxes (room interiors, door and window voids, outdoor
slabs) plus solid boxes (furniture). Pinhole cameras cast rays; a ray stops where it leaves the free
space or enters a solid, which gives realistic visibility through doors and windows and occlusion
behind furniture. Noise, point dropout, outliers and a global yaw are applied afterwards.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from scan2scope.geometry.se3 import rot_z
from scan2scope.types import CameraView, Scene


@dataclass
class Box:
    lo: tuple[float, float, float]
    hi: tuple[float, float, float]


@dataclass
class Cam:
    pos: tuple[float, float, float]
    yaw_deg: float
    pitch_deg: float = 0.0
    room: str | None = None


@dataclass
class Opening:
    room: str
    type: str  # door | window
    wall_axis: str  # "x": wall at constant x, "y": wall at constant y
    wall_coord: float  # room-side face coordinate
    u0: float  # opening extent along the wall, in plan coordinates
    u1: float
    z0: float
    z1: float
    leads_to: str | None = None


@dataclass
class Synth:
    free: list[Box]
    solids: list[Box]
    cams: list[Cam]
    height: float
    rooms: dict[str, list[tuple[float, float, float, float]]]  # name -> rectangles (x0, y0, x1, y1)
    openings: list[Opening] = field(default_factory=list)


def _slab(o: np.ndarray, d: np.ndarray, boxes: list[Box]) -> tuple[np.ndarray, ...]:
    lo = np.array([b.lo for b in boxes], float)
    hi = np.array([b.hi for b in boxes], float)
    dd = np.where(np.abs(d) < 1e-12, 1e-12, d)
    t0 = (lo[None] - o[:, None]) / dd[:, None]
    t1 = (hi[None] - o[:, None]) / dd[:, None]
    tmin, tmax = np.minimum(t0, t1), np.maximum(t0, t1)
    return tmin.max(-1), tmax.min(-1), tmin.argmax(-1), tmax.argmin(-1)


def _cam_rotation(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    y, p = np.radians(yaw_deg), np.radians(pitch_deg)
    f = np.array([np.cos(y) * np.cos(p), np.sin(y) * np.cos(p), np.sin(p)])
    r = np.array([np.sin(y), -np.cos(y), 0.0])
    return np.stack([r, np.cross(f, r), f], 1)  # columns: x right, y down, z forward


def cast(syn: Synth, cam: Cam, w: int, h: int, hfov_deg: float, max_range: float,
         rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Ray-cast one camera. Returns points, normals (towards the camera), K and T_wc."""
    fx = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    K = np.array([[fx, 0, w / 2 - 0.5], [0, fx, h / 2 - 0.5], [0, 0, 1.0]])
    R = _cam_rotation(cam.yaw_deg, cam.pitch_deg)
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, cam.pos
    u, v = np.meshgrid(np.arange(w) + rng.uniform(-0.5, 0.5), np.arange(h) + rng.uniform(-0.5, 0.5))
    dc = np.stack([(u.ravel() - K[0, 2]) / fx, (v.ravel() - K[1, 2]) / fx, np.ones(u.size)], 1)
    d = dc @ R.T
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    o = np.broadcast_to(np.asarray(cam.pos, float), d.shape)

    t_in, t_out, _, ax_out = _slab(o, d, syn.free)
    t = np.zeros(len(d))
    last = np.full(len(d), -1)
    for _ in range(12):
        cont = (t_in <= t[:, None] + 1e-7) & (t_out > t[:, None] + 1e-7)
        ok = cont.any(1)
        if not ok.any():
            break
        tt = np.where(cont, t_out, -np.inf)
        b = tt.argmax(1)
        t[ok] = tt[ok, b[ok]]
        last[ok] = b[ok]
    ax = np.where(last >= 0, ax_out[np.arange(len(d)), np.maximum(last, 0)], 2)
    if syn.solids:
        s_in, s_out, s_ax, _ = _slab(o, d, syn.solids)
        hit = (s_in > 1e-6) & (s_in < s_out)
        ts = np.where(hit, s_in, np.inf)
        j = ts.argmin(1)
        tj = ts[np.arange(len(d)), j]
        use = tj < t
        t = np.where(use, tj, t)
        ax = np.where(use, s_ax[np.arange(len(d)), j], ax)
    keep = (last >= 0) & np.isfinite(t) & (t > 0.05) & (t < max_range)
    p = o[keep] + t[keep, None] * d[keep]
    n = np.zeros_like(p)
    n[np.arange(len(p)), ax[keep]] = -np.sign(d[keep, ax[keep]])
    return p, n, K, T


def make_scene(syn: Synth, *, noise: float = 0.01, normal_noise: float = 0.03, drop: float = 0.0,
               outliers: float = 0.0, rays: tuple[int, int] = (192, 144), hfov_deg: float = 67.0,
               max_range: float = 8.0, yaw_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0),
               seed: int = 0, tier: str = "lidar", room_hint: str | None = None, depth_noise: float = 0.0,
               view_jitter: tuple[float, float] = (0.0, 0.0)) -> Scene:
    """depth_noise: extra noise along each ray as a fraction of depth; view_jitter: per-view rigid error
    (translation sigma in m, rotation sigma in degrees) like multi-view inconsistency in feed-forward models."""
    rng = np.random.default_rng(seed)
    pts, nrm, vidx, views = [], [], [], []
    for i, cam in enumerate(syn.cams):
        p, n, K, T = cast(syn, cam, rays[0], rays[1], hfov_deg, max_range, rng)
        if depth_noise > 0:
            ray = p - T[:3, 3]
            depth = np.linalg.norm(ray, axis=1, keepdims=True)
            p = p + ray / depth * rng.normal(0.0, depth_noise, (len(p), 1)) * depth
        if view_jitter[0] > 0 or view_jitter[1] > 0:
            ax = rng.normal(size=3)
            Rj = Rotation.from_rotvec(ax / np.linalg.norm(ax) * np.radians(rng.normal(0.0, view_jitter[1]))).as_matrix()
            c = T[:3, 3]
            p = (p - c) @ Rj.T + c + rng.normal(0.0, view_jitter[0], 3)
            n = n @ Rj.T
        pts.append(p)
        nrm.append(n)
        vidx.append(np.full(len(p), i))
        views.append(CameraView(id=f"v{i:03d}", image_path=None, width=rays[0], height=rays[1], K=K, T_wc=T,
                                room_hint=cam.room))
    P = np.concatenate(pts)
    N = np.concatenate(nrm)
    V = np.concatenate(vidx)
    P = P + rng.normal(0.0, noise, P.shape)
    N = N + rng.normal(0.0, normal_noise, N.shape)
    if drop > 0:
        keep = rng.random(len(P)) >= drop
        P, N, V = P[keep], N[keep], V[keep]
    if outliers > 0:
        m = round(outliers * len(P) / (1 - outliers))
        lo, hi = P.min(0) - 0.5, P.max(0) + 0.5
        P = np.concatenate([P, rng.uniform(lo, hi, (m, 3))])
        N = np.concatenate([N, rng.normal(size=(m, 3))])
        V = np.concatenate([V, rng.integers(0, len(views), m)])
    centers = np.array([v.T_wc[:3, 3] for v in views])
    N /= np.maximum(np.linalg.norm(N, axis=1, keepdims=True), 1e-12)
    flip = ((centers[V] - P) * N).sum(1) < 0
    N[flip] *= -1
    W = rng.uniform(0.6, 1.0, len(P))

    Rz = rot_z(np.radians(yaw_deg))
    t = np.array([shift[0], shift[1], 0.0])
    P = P @ Rz.T + t
    N = N @ Rz.T
    for v in views:
        v.T_wc = v.T_wc.copy()
        v.T_wc[:3, :3] = Rz @ v.T_wc[:3, :3]
        v.T_wc[:3, 3] = Rz @ v.T_wc[:3, 3] + t
    return Scene(tier=tier, views=views, points=P.astype(np.float32), normals=N.astype(np.float32),
                 weights=W.astype(np.float32), view_index=V.astype(np.int64), room_hint=room_hint)


def to_world(xy: np.ndarray, yaw_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    c, s = np.cos(np.radians(yaw_deg)), np.sin(np.radians(yaw_deg))
    xy = np.asarray(xy, float)
    return xy @ np.array([[c, s], [-s, c]]) + np.asarray(shift)


def truth_polygon(syn: Synth, room: str, yaw_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)):
    """Union of a room's rectangles as a shapely polygon in world coordinates."""
    from shapely.geometry import Polygon
    from shapely.ops import unary_union

    rects = [Polygon(to_world([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], yaw_deg, shift))
             for x0, y0, x1, y1 in syn.rooms[room]]
    return unary_union(rects)


def opening_center(op: Opening, yaw_deg: float = 0.0, shift: tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    mid = 0.5 * (op.u0 + op.u1)
    xy = (op.wall_coord, mid) if op.wall_axis == "x" else (mid, op.wall_coord)
    return to_world(np.array([xy]), yaw_deg, shift)[0]


def cyclic_match(found: list[float], truth: list[float]) -> float:
    """Largest error of found wall lengths against truth under the best cyclic shift (inf on a count mismatch)."""
    if len(found) != len(truth):
        return float("inf")
    f, t = np.asarray(found), np.asarray(truth)
    return float(min(np.abs(np.roll(f, k) - t).max() for k in range(len(f))))


def overlap_area(polys: list[np.ndarray]) -> float:
    from shapely.geometry import Polygon

    shapes = [Polygon(p) for p in polys]
    return float(sum(shapes[i].intersection(shapes[j]).area for i in range(len(shapes))
                     for j in range(i + 1, len(shapes))))


H = 2.6


def _outdoor(x0: float, y0: float, x1: float, y1: float) -> list[Box]:
    """Outdoor slabs around a property footprint, reachable only through window voids."""
    d = 3.0
    return [Box((x0 - d, y0 - d, -4), (x0, y1 + d, 8)), Box((x1, y0 - d, -4), (x1 + d, y1 + d, 8)),
            Box((x0, y0 - d, -4), (x1, y0, 8)), Box((x0, y1, -4), (x1, y1 + d, 8))]


def single_room(furniture: bool = False, occluder: bool = False) -> Synth:
    """4.2 x 3.1 x 2.6 m room, door in the y=0 wall into a corridor, window in the x=4.2 wall."""
    t = 0.12
    free = [Box((0, 0, 0), (4.2, 3.1, H)),
            Box((0.6, -t, 0), (1.5, 0.0, 2.05)),  # door void
            Box((-0.5, -t - 1.5, 0), (3.2, -t, H)),  # corridor beyond the door
            Box((4.2, 1.0, 0.9), (4.2 + t, 2.2, 2.1))]  # window void
    free += _outdoor(-0.5 - t, -t - 1.5 - t, 4.2 + t, 3.1 + t)
    solids = []
    if furniture:
        solids += [Box((1.6, 2.3, 0), (3.4, 3.1, 0.85)),  # sideboard against the y=3.1 wall
                   Box((2.0, 1.0, 0.0), (2.9, 1.8, 0.75))]  # table
    if occluder:
        solids.append(Box((0.0, 1.0, 0.0), (0.35, 2.2, 1.9)))  # bookshelf hiding part of the x=0 wall
    cams = [Cam((0.5, 0.5, 1.5), 37), Cam((3.7, 0.5, 1.5), 143), Cam((3.7, 2.6, 1.5), -143),
            Cam((0.5, 2.6, 1.5), -37), Cam((2.1, 1.55, 1.45), -124, -12), Cam((2.6, 1.55, 1.5), 0, 5),
            Cam((1.2, 2.3, 1.5), -95, -8), Cam((2.5, 1.2, 1.6), 90, 15)]
    ops = [Opening("R", "door", "y", 0.0, 0.6, 1.5, 0.0, 2.05, None),
           Opening("R", "window", "x", 4.2, 1.0, 2.2, 0.9, 2.1, None)]
    return Synth(free, solids, cams, H, {"R": [(0, 0, 4.2, 3.1)]}, ops)


def l_room() -> Synth:
    """L-shaped room: 5 x 2.5 m arm along x plus a 2.5 x 2.5 m arm along y; door into a corridor."""
    t = 0.12
    free = [Box((0, 0, 0), (5.0, 2.5, H)), Box((0, 2.5, 0), (2.5, 5.0, H)),
            Box((3.4, -t, 0), (4.3, 0.0, 2.05)), Box((2.5, -t - 1.4, 0), (5.6, -t, H))]
    cams = [Cam((0.5, 0.5, 1.5), 20), Cam((4.5, 0.5, 1.5), 160), Cam((4.5, 2.0, 1.5), -150),
            Cam((0.5, 4.5, 1.5), -60), Cam((2.0, 4.5, 1.5), -120), Cam((1.2, 1.2, 1.5), 60, 10),
            Cam((2.2, 2.2, 1.5), -45, -10), Cam((3.8, 1.6, 1.4), -90, -15), Cam((1.2, 3.5, 1.5), 180),
            Cam((1.5, 1.0, 1.5), 90)]
    ops = [Opening("R", "door", "y", 0.0, 3.4, 4.3, 0.0, 2.05, None)]
    return Synth(free, [], cams, H, {"R": [(0, 0, 5.0, 2.5), (0, 2.5, 2.5, 5.0)]}, ops)


def two_areas(passage: str) -> Synth:
    """Two 4 x 3 m areas split by a 12 cm wall at x = 4.0 with one passage through it.

    passage: "wide" (2.2 m, floor to ceiling), "narrow" (1.0 m, floor to ceiling, no header) or
    "cased" (2.0 m wide with a header at 2.1 m).
    """
    y0, y1, top = {"wide": (0.8, 3.0, H), "narrow": (1.0, 2.0, H), "cased": (0.5, 2.5, 2.1)}[passage]
    free = [Box((0, 0, 0), (4.0, 3.0, H)), Box((4.12, 0, 0), (8.12, 3.0, H)), Box((4.0, y0, 0), (4.12, y1, top))]
    cams = [Cam((0.5, 0.5, 1.5), 35), Cam((3.4, 2.5, 1.5), -140), Cam((0.6, 2.5, 1.5), -30), Cam((2.0, 1.5, 1.5), 0),
            Cam((7.6, 0.5, 1.5), 145), Cam((4.7, 2.5, 1.5), -40), Cam((7.5, 2.5, 1.5), -150), Cam((6.0, 1.5, 1.5), 180)]
    for c in cams:
        c.room = "west" if c.pos[0] < 4.0 else "east"
    rooms = {"west": [(0, 0, 4.0, 3.0)], "east": [(4.12, 0, 8.12, 3.0)]}
    ops = [Opening("west", "opening" if passage != "cased" else "door", "x", 4.0, y0, y1, 0.0, top, "east")]
    return Synth(free, [], cams, H, rooms, ops)


def hallway_three_rooms() -> Synth:
    """Hallway along y=0..1.2 with rooms A and B above it and a room C at its east end."""
    t = 0.12
    rooms = {"H": [(0, 0, 6.2, 1.2)], "A": [(0, 1.32, 3.0, 4.5)], "B": [(3.12, 1.32, 6.2, 4.5)],
             "C": [(6.32, 0, 8.5, 2.6)]}
    free = [Box((r[0], r[1], 0), (r[2], r[3], H)) for rs in rooms.values() for r in rs]
    free += [Box((1.0, 1.2, 0), (1.9, 1.32, 2.05)), Box((4.3, 1.2, 0), (5.2, 1.32, 2.05)),
             Box((6.2, 0.15, 0), (6.32, 0.95, 2.05)),
             Box((0.9, 4.5, 0.9), (2.1, 4.5 + t, 2.1)), Box((4.0, 4.5, 0.9), (5.4, 4.5 + t, 2.1)),
             Box((8.5, 0.8, 1.2), (8.5 + t, 1.8, 2.0))]
    free += _outdoor(-t, -t, 8.5 + t, 4.5 + t)
    cams = [Cam((0.6, 0.6, 1.5), 0), Cam((3.1, 0.6, 1.5), 180), Cam((3.1, 0.6, 1.5), 0),
            Cam((1.45, 0.3, 1.5), 90, -5), Cam((4.75, 0.3, 1.5), 90, -5), Cam((5.6, 0.6, 1.5), 0, -5),
            Cam((5.6, 0.6, 1.5), 180),
            Cam((0.5, 1.8, 1.5), 40), Cam((2.5, 1.8, 1.5), 130), Cam((2.5, 4.0, 1.5), -130),
            Cam((0.5, 4.0, 1.5), -40), Cam((1.45, 2.4, 1.5), -90, -10),
            Cam((3.6, 1.8, 1.5), 40), Cam((5.7, 1.8, 1.5), 130), Cam((5.7, 4.0, 1.5), -130),
            Cam((3.6, 4.0, 1.5), -40), Cam((4.75, 2.4, 1.5), -90, -10),
            Cam((6.8, 0.4, 1.5), 45), Cam((8.0, 0.4, 1.5), 135), Cam((8.0, 2.2, 1.5), -135),
            Cam((6.8, 2.2, 1.5), -45), Cam((7.4, 0.55, 1.5), 180, -10)]
    for c in cams:
        x, y = c.pos[:2]
        c.room = next(k for k, rs in rooms.items() if any(r[0] <= x <= r[2] and r[1] <= y <= r[3] for r in rs))
    ops = [Opening("H", "door", "y", 1.2, 1.0, 1.9, 0, 2.05, "A"), Opening("A", "door", "y", 1.32, 1.0, 1.9, 0, 2.05, "H"),
           Opening("H", "door", "y", 1.2, 4.3, 5.2, 0, 2.05, "B"), Opening("B", "door", "y", 1.32, 4.3, 5.2, 0, 2.05, "H"),
           Opening("H", "door", "x", 6.2, 0.15, 0.95, 0, 2.05, "C"), Opening("C", "door", "x", 6.32, 0.15, 0.95, 0, 2.05, "H"),
           Opening("A", "window", "y", 4.5, 0.9, 2.1, 0.9, 2.1), Opening("B", "window", "y", 4.5, 4.0, 5.4, 0.9, 2.1),
           Opening("C", "window", "x", 8.5, 0.8, 1.8, 1.2, 2.0)]
    return Synth(free, [], cams, H, rooms, ops)
