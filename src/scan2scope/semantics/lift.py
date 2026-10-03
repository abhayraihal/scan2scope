"""Lift image masks through a view's point map onto room surfaces and measure them there.

A mask at image resolution is reduced to the point-map grid by area coverage (point-map pixel (i, j) covers the
image pixels given by the CameraView mapping; any mask resolution that spans the whole image works the same
way). Each covered point-map pixel contributes its world point and its area vector dP/dj x dP/di, so areas are
the sum of per-pixel surface areas projected on the assigned surface.

Surface coordinates: walls use u along the wall from its start and v = z minus the room floor; the floor and
the ceiling use plan x (u) and y (v).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import shapely
from shapely.geometry import Point, Polygon

from scan2scope.types import CameraView, Plan, Room, Wall

log = logging.getLogger("scan2scope.semantics")


@dataclass
class LiftConfig:
    wall_tol: float = 0.15  # nearest wall line of the room within this distance (m)
    wall_loose_tol: float = 0.30  # vertical patches facing the nearest wall still count up to this distance
    wall_facing_cos: float = 0.9
    height_tol: float = 0.30  # horizontal patches must lie this close to the floor or the ceiling
    edge_height_tol: float = 0.10  # same, for patches whose orientation is unclear
    off_plane_tol: float = 0.08  # points farther than this from the local surface plane are not on it
    min_pixels: float = 3.0  # minimum mask coverage on the surface, in point-map pixels
    min_valid_fraction: float = 0.25  # share of the mask that needs geometry
    room_shift: float = 0.05  # median moved this far towards the camera before point-in-polygon
    room_snap: float = 0.5  # nearest room within this distance when the point is inside none
    horizontal_cos: float = 0.82  # |n_z| at or above: floor or ceiling patch (35 degrees from vertical)
    vertical_sin: float = 0.57  # |n_z| at or below: wall-like patch
    q_lo: float = 0.02
    q_hi: float = 0.98


@dataclass
class LiftedMask:
    view_id: str
    points: np.ndarray  # (n, 3) world points of covered point-map pixels that have geometry
    coverage: np.ndarray  # (n,) share of each of those pixels covered by the mask
    cross: np.ndarray  # (n, 3) area vector dP/dj x dP/di of each pixel (m2 per point-map pixel)
    normals: np.ndarray  # (n, 3) unit normals oriented towards the camera
    coverage_total: float  # mask area in point-map pixels
    coverage_valid: float  # the part of it on pixels with geometry
    cam_center: np.ndarray

    @property
    def valid_fraction(self) -> float:
        return self.coverage_valid / self.coverage_total if self.coverage_total > 0 else 0.0

    @property
    def area_weights(self) -> np.ndarray:
        return self.coverage * np.linalg.norm(self.cross, axis=1)


@dataclass
class SurfaceAssignment:
    room: Room
    surface_id: str
    kind: str  # wall | floor | ceiling
    wall: Wall | None
    normal: np.ndarray  # (3,) unit normal of the surface pointing into the room
    distance: float  # distance between the patch median and the surface (m)
    match: str  # wall | wall_loose | floor | ceiling


@dataclass
class SurfaceMeasure:
    area: float  # m2 on the surface
    width: float  # extent along u
    height: float  # extent along v
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    length: float  # extent along the principal axis (crack length)
    endpoints: np.ndarray  # (2, 2) uv ends of the principal axis extent
    pixel_m: float  # typical point-map pixel footprint on the surface
    inlier_fraction: float  # share of covered pixels with geometry that lie on the surface
    valid_fraction: float
    n_pixels: int


def wquantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    """Weighted quantile, each sample's weight centred on it, extrapolated half a sample past the ends."""
    x = np.asarray(x, float)
    w = np.asarray(w, float)
    if len(x) == 0:
        return float("nan")
    if len(x) == 1 or w.sum() <= 0:
        return float(np.median(x))
    o = np.argsort(x, kind="stable")
    x, w = x[o], w[o]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    if q <= c[0]:
        j = 1
    elif q >= c[-1]:
        j = len(x) - 1
    else:
        return float(np.interp(q, c, x))
    dc = c[j] - c[j - 1]
    v = x[j - 1] + (q - c[j - 1]) * (x[j] - x[j - 1]) / dc if dc > 0 else x[j]
    lo = x[0] - 0.5 * (x[1] - x[0])
    hi = x[-1] + 0.5 * (x[-1] - x[-2])
    return float(np.clip(v, lo, hi))


def robust_extent(x: np.ndarray, w: np.ndarray, q_lo: float, q_hi: float) -> tuple[float, float]:
    """Range from the q_lo to q_hi weighted quantiles, rescaled to the full span of a uniform spread."""
    a, b = wquantile(x, w, q_lo), wquantile(x, w, q_hi)
    half = 0.5 * (b - a) / (q_hi - q_lo)
    c = 0.5 * (a + b)
    return c - half, c + half


def pointmap_jacobian(pm: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Derivatives of a point map along columns (j) and rows (i) and the pixels where both exist.

    Central differences where both one-sided differences exist and agree within a factor of two, otherwise the
    shorter one-sided difference, so depth edges at the border of a surface do not inflate its pixel areas.
    """
    P = pm.astype(np.float64)
    V = valid & np.isfinite(P).all(-1)
    derivs = []
    for axis in (1, 0):
        n = P.shape[axis]
        fwd = np.full(P.shape, np.nan)
        bwd = np.full(P.shape, np.nan)
        if n >= 2:
            lo = [slice(None)] * 3
            hi = [slice(None)] * 3
            lo[axis], hi[axis] = slice(0, n - 1), slice(1, n)
            diff = P[tuple(hi)] - P[tuple(lo)]
            diff[~(V[tuple(hi[:2])] & V[tuple(lo[:2])])] = np.nan
            fwd[tuple(lo)] = diff
            bwd[tuple(hi)] = diff
        nf = np.linalg.norm(fwd, axis=-1)
        nb = np.linalg.norm(bwd, axis=-1)
        hf, hb = np.isfinite(nf), np.isfinite(nb)
        with np.errstate(invalid="ignore"):
            agree = hf & hb & (np.fmax(nf, nb) <= 2.0 * np.fmin(nf, nb))
            use_f = ~agree & hf & (~hb | (nf <= nb))
        use_b = ~agree & ~use_f & hb
        d = np.full(P.shape, np.nan)
        d[agree] = 0.5 * (fwd[agree] + bwd[agree])
        d[use_f] = fwd[use_f]
        d[use_b] = bwd[use_b]
        derivs.append(d)
    ok = V & np.isfinite(derivs[0]).all(-1) & np.isfinite(derivs[1]).all(-1)
    return derivs[0], derivs[1], ok


def mask_coverage(mask: np.ndarray, h: int, w: int) -> np.ndarray:
    """Share of each point-map pixel covered by an image mask spanning the same field of view."""
    m = np.asarray(mask, np.float32)
    if m.shape == (h, w):
        return m
    return cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA)


def view_valid(view: CameraView) -> np.ndarray | None:
    pm = view.pointmap
    if pm is None or pm.ndim != 3 or pm.shape[2] != 3:
        return None
    valid = np.isfinite(pm).all(-1)
    if view.valid is not None:
        if view.valid.shape != pm.shape[:2]:
            return None
        valid &= view.valid.astype(bool)
    return valid


def lift_mask(view: CameraView, mask: np.ndarray, *, jac: tuple | None = None, ring: int = 0) -> LiftedMask | None:
    """World points and pixel area vectors under a mask; with ring > 0, a band of that many pixels around it."""
    valid = view_valid(view)
    if valid is None:
        return None
    pm = view.pointmap
    h, w = pm.shape[:2]
    cov = mask_coverage(mask, h, w)
    if ring > 0:
        core = cov >= 0.5
        grown = cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=ring).astype(bool)
        cov = (grown & ~core).astype(np.float32)
    covered = cov > 1e-3
    total = float(cov[covered].sum())
    if total <= 0:
        return None
    dj, di, ok = jac if jac is not None else pointmap_jacobian(pm, valid)
    sel = covered & ok
    pts = pm[sel].astype(np.float64)
    cross = np.cross(dj[sel], di[sel])
    norm = np.linalg.norm(cross, axis=1)
    good = norm > 1e-12
    pts, cross, norm = pts[good], cross[good], norm[good]
    c = cov[sel][good].astype(np.float64)
    normals = cross / norm[:, None]
    cam = np.asarray(view.T_wc, float)[:3, 3]
    flip = ((cam - pts) * normals).sum(1) < 0
    normals[flip] *= -1
    return LiftedMask(view.id, pts, c, cross, normals, total, float(c.sum()), cam)


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(float(np.dot(ab, ab)), 1e-12), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


_POLY_CACHE: dict[int, tuple[object, list[Polygon]]] = {}


def room_polygons(plan: Plan) -> list[Polygon]:
    hit = _POLY_CACHE.get(id(plan))
    if hit is not None and hit[0] is plan:
        return hit[1]
    polys = []
    for r in plan.rooms:
        try:
            poly = Polygon(np.asarray(r.polygon, float))
            polys.append(poly if poly.is_valid else poly.buffer(0))
        except (ValueError, TypeError, shapely.errors.ShapelyError):  # degenerate polygon: never matches
            polys.append(Polygon())
    _POLY_CACHE.clear()
    _POLY_CACHE[id(plan)] = (plan, polys)
    return polys


def room_of(plan: Plan, xy: np.ndarray, cam_xy: np.ndarray | None, cfg: LiftConfig) -> Room | None:
    """Room containing xy, after moving it slightly towards the camera (wall patches sit on the boundary)."""
    if not plan.rooms:
        return None
    p = np.asarray(xy, float)
    if cam_xy is not None:
        d = np.asarray(cam_xy, float) - p
        n = float(np.linalg.norm(d))
        if n > 1e-6:
            p = p + d / n * min(cfg.room_shift, n)
    polys = room_polygons(plan)
    pt = Point(float(p[0]), float(p[1]))
    for room, poly in zip(plan.rooms, polys):
        if not poly.is_empty and poly.contains(pt):
            return room
    dists = [poly.distance(pt) if not poly.is_empty else np.inf for poly in polys]
    k = int(np.argmin(dists))
    return plan.rooms[k] if dists[k] <= cfg.room_snap else None


def _wall_normal3(wall: Wall) -> np.ndarray:
    n = np.asarray(wall.normal_in, float)
    n = n / max(float(np.linalg.norm(n)), 1e-12)
    return np.array([n[0], n[1], 0.0])


def assign_surface(lifted: LiftedMask, plan: Plan, cfg: LiftConfig) -> tuple[SurfaceAssignment | None, str]:
    """Room by point-in-polygon on the median, then the nearest wall within wall_tol, else floor or ceiling."""
    w = lifted.area_weights
    if len(w) == 0 or w.sum() <= 0:
        return None, "no_geometry"
    med = np.array([wquantile(lifted.points[:, k], w, 0.5) for k in range(3)])
    nbar = (lifted.normals * w[:, None]).sum(0)
    nn = float(np.linalg.norm(nbar))
    nbar = nbar / nn if nn > 1e-12 else np.zeros(3)
    nz = abs(float(nbar[2]))
    room = room_of(plan, med[:2], lifted.cam_center[:2], cfg)
    if room is None:
        return None, "outside_rooms"
    horizontal = nz >= cfg.horizontal_cos
    if not horizontal and room.walls:
        dists = [segment_distance(med[:2], np.asarray(wl.start, float), np.asarray(wl.end, float))
                 for wl in room.walls]
        k = int(np.argmin(dists))
        wall = room.walls[k]
        n3 = _wall_normal3(wall)
        if dists[k] <= cfg.wall_tol:
            return SurfaceAssignment(room, wall.id, "wall", wall, n3, dists[k], "wall"), ""
        if nz <= cfg.vertical_sin and dists[k] <= cfg.wall_loose_tol and float(nbar @ n3) >= cfg.wall_facing_cos:
            return SurfaceAssignment(room, wall.id, "wall", wall, n3, dists[k], "wall_loose"), ""
    dz_floor = abs(float(med[2]) - room.floor_z)
    dz_ceil = abs(float(med[2]) - room.ceiling_z)
    tol = cfg.height_tol if horizontal else cfg.edge_height_tol
    if min(dz_floor, dz_ceil) <= tol:
        if dz_floor <= dz_ceil:
            return SurfaceAssignment(room, f"{room.id}-FLOOR", "floor", None, np.array([0.0, 0.0, 1.0]),
                                     dz_floor, "floor"), ""
        return SurfaceAssignment(room, f"{room.id}-CEIL", "ceiling", None, np.array([0.0, 0.0, -1.0]),
                                 dz_ceil, "ceiling"), ""
    return None, "off_surface"


def surface_coords(points: np.ndarray, a: SurfaceAssignment) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(u, v, signed distance from the surface plane towards the room interior) for world points."""
    room = a.room
    if a.kind == "wall":
        s = np.asarray(a.wall.start, float)
        e = np.asarray(a.wall.end, float)
        t = (e - s) / max(float(np.linalg.norm(e - s)), 1e-12)
        rel = points[:, :2] - s
        return rel @ t, points[:, 2] - room.floor_z, rel @ a.normal[:2]
    if a.kind == "floor":
        return points[:, 0].copy(), points[:, 1].copy(), points[:, 2] - room.floor_z
    return points[:, 0].copy(), points[:, 1].copy(), room.ceiling_z - points[:, 2]


def _on_surface_extent(u: np.ndarray, v: np.ndarray, a: SurfaceAssignment, margin: float = 0.1) -> np.ndarray:
    room = a.room
    if a.kind == "wall":
        length = float(np.linalg.norm(np.asarray(a.wall.end, float) - np.asarray(a.wall.start, float)))
        height = room.ceiling_z - room.floor_z
        return (u >= -margin) & (u <= length + margin) & (v >= -margin) & (v <= height + margin)
    poly = room_polygons_for(room).buffer(margin)
    return shapely.contains_xy(poly, u, v)


def room_polygons_for(room: Room) -> Polygon:
    try:
        poly = Polygon(np.asarray(room.polygon, float))
    except (ValueError, TypeError, shapely.errors.ShapelyError):  # degenerate polygon: contains nothing
        return Polygon()
    return poly if poly.is_valid else poly.buffer(0)


def measure_on_surface(lifted: LiftedMask, a: SurfaceAssignment, cfg: LiftConfig) -> tuple[SurfaceMeasure | None, str]:
    """Area, uv extents and principal-axis length of the lifted mask on its assigned surface."""
    if lifted.valid_fraction < cfg.min_valid_fraction:
        return None, "low_valid_fraction"
    u, v, d = surface_coords(lifted.points, a)
    proj = np.abs(lifted.cross @ a.normal)  # pixel area projected on the surface
    d0 = wquantile(d, lifted.area_weights, 0.5)
    inl = (np.abs(d - d0) <= cfg.off_plane_tol) & _on_surface_extent(u, v, a)
    c_in = float(lifted.coverage[inl].sum())
    if c_in < cfg.min_pixels:
        return None, "too_few_surface_pixels"
    wa = lifted.coverage[inl] * proj[inl]
    if wa.sum() <= 0:
        return None, "no_surface_area"
    area = float(wa.sum()) * lifted.coverage_total / max(lifted.coverage_valid, 1e-12)
    uu, vv = u[inl], v[inl]
    u_lo, u_hi = robust_extent(uu, wa, cfg.q_lo, cfg.q_hi)
    v_lo, v_hi = robust_extent(vv, wa, cfg.q_lo, cfg.q_hi)
    uv = np.stack([uu, vv], 1)
    mean = (uv * wa[:, None]).sum(0) / wa.sum()
    cov = ((uv - mean) * wa[:, None]).T @ (uv - mean) / wa.sum()
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    t = (uv - mean) @ axis
    t_lo, t_hi = robust_extent(t, wa, cfg.q_lo, cfg.q_hi)
    pixel = float(np.sqrt(max(wquantile(proj[inl], lifted.coverage[inl], 0.5), 0.0)))
    return SurfaceMeasure(
        area=area, width=u_hi - u_lo, height=v_hi - v_lo, u_range=(u_lo, u_hi), v_range=(v_lo, v_hi),
        length=t_hi - t_lo, endpoints=np.stack([mean + axis * t_lo, mean + axis * t_hi]), pixel_m=pixel,
        inlier_fraction=c_in / max(lifted.coverage_valid, 1e-12), valid_fraction=lifted.valid_fraction,
        n_pixels=int(inl.sum()),
    ), ""


@dataclass
class ObjectPlacement:
    room_id: str | None
    xy: np.ndarray  # (2,) plan position (weighted median)
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    n_pixels: int
    normal_z: float = 0.0  # |z| of the area-weighted mean normal: near 0 in a wall, near 1 on floor or ceiling


def see_through_fraction(view: CameraView, mask: np.ndarray, ring: LiftedMask, behind_m: float = 0.15) -> float:
    """Share of a mask with no geometry or lying behind the plane of its surrounding band.

    Open doors, windows and mirrors show depth beyond the wall (or none at all); a sheet of paper on the wall
    is coplanar with its surround and scores near zero.
    """
    valid = view_valid(view)
    if valid is None or len(ring.points) < 3:
        return 0.0
    h, w = view.pointmap.shape[:2]
    inside = mask_coverage(mask, h, w) >= 0.5
    total = int(inside.sum())
    if total == 0:
        return 0.0
    wts = ring.area_weights
    c = np.array([wquantile(ring.points[:, k], wts, 0.5) for k in range(3)])
    n = (ring.normals * wts[:, None]).sum(0)
    n /= max(float(np.linalg.norm(n)), 1e-12)  # oriented towards the camera
    pts = view.pointmap[inside & valid].astype(np.float64)
    behind = int(((c - pts) @ n > behind_m).sum())
    return (int((inside & ~valid).sum()) + behind) / total


def place_object(lifted: LiftedMask, plan: Plan, cfg: LiftConfig) -> ObjectPlacement | None:
    """Plan position, plan extent (5th to 95th percentile) and height range of a lifted object mask."""
    w = lifted.area_weights
    if len(w) < 3 or w.sum() <= 0:
        return None
    p = lifted.points
    xy = np.array([wquantile(p[:, 0], w, 0.5), wquantile(p[:, 1], w, 0.5)])
    room = room_of(plan, xy, lifted.cam_center[:2], cfg)
    rng = [(wquantile(p[:, k], w, 0.05), wquantile(p[:, k], w, 0.95)) for k in range(3)]
    n = (lifted.normals * w[:, None]).sum(0)
    nz = abs(float(n[2])) / max(float(np.linalg.norm(n)), 1e-12)
    return ObjectPlacement(room.id if room else None, xy, rng[0], rng[1], rng[2], len(w), nz)
