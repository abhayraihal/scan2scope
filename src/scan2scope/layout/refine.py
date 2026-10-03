"""Room outlines from each room's own wall faces: furniture filter, per-room wall refinement, regularisation.

The cell complex puts every room edge on a wall line fitted once for the whole property. Two parallel faces of
different rooms a few centimetres apart then share one line placed between them, an outline steps between the
two faces of a wall, and furniture fronts cut notches into rooms. Per room:

1. Furniture filter: an edge whose face points never reach the band CEILING_BAND below the room's ceiling is a
   furniture front or side. A run of such edges between two walls is replaced by those walls when that adds
   floor no other room holds, at most FURNITURE_DEPTH deep. This runs on the cell lines before refinement, and
   once more on the refined outline for notches whose walls sat on the two faces of one wall body.
2. Refinement: each edge is refitted from the wall points within BAND of it that face into the room and lie
   along it: a trimmed mean of their offset, started at the densest offset and iterated, or a slanted line
   when that explains the points clearly better (the edge then sits where the line crosses its midpoint). An
   edge with fewer than MIN_POINTS such points keeps its cell-complex position and is flagged
   wall_not_refined. Corners are where consecutive refined lines meet.
3. Regularisation: an edge shorter than STEP_MAX between two parallel edges goes. Between edges that run the
   same way it is a step, and both snap to the face with more support over their joint extent and merge into
   one edge; between edges that run opposite ways it is the end of a strip that thin (a wall body, a slot),
   and the strip is cut back. It also runs once on the cell lines, so the walls around a notch line up first.

Coordinates are in the Manhattan frame. An outline is rectilinear and counter-clockwise, held as a cyclic list
of edges whose axes alternate; vertex k is where edge k - 1 meets edge k.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from shapely.geometry import Polygon

from scan2scope.layout import walls as W
from scan2scope.layout.floor_ceiling import robust_sigma

log = logging.getLogger("scan2scope.layout")

BAND = 0.15  # wall points this close to an edge are candidates for its face
STEP_MAX = 0.25  # shorter edges between parallel edges are removed
CEILING_BAND = 0.5  # a face whose points never come this close to the ceiling is furniture
MIN_POINTS = 30  # fewer face voxels than this: the edge keeps its cell-complex position
END_MARGIN = 0.03  # voxels this close to either end of an edge are left out (corner voxels mix two faces)
WIN_MIN, WIN_MAX = 0.015, 0.08  # trimming window of the face fit
MODE_RES = 0.005
TOP_QUANTILE = 0.99
COLLINEAR_TOL = 0.05  # walls this close on one line are one wall when the notch between them is filled
FURNITURE_DEPTH = 1.0  # deepest notch that is filled
FURNITURE_AREA = 0.25  # largest filled area as a share of the room
SLANT_FLAG = float(np.tan(np.radians(1.0)))


@dataclass
class WallPoints:
    """Wall voxels per face direction (walls.DIRS), sorted by the coordinate across the face."""

    v: list[np.ndarray] = field(default_factory=list)  # across the face
    t: list[np.ndarray] = field(default_factory=list)  # along the face
    z: list[np.ndarray] = field(default_factory=list)
    wa: list[np.ndarray] = field(default_factory=list)  # one vote per voxel: mode, support, heights
    wf: list[np.ndarray] = field(default_factory=list)  # fit weight

    @classmethod
    def build(cls, P: np.ndarray, N: np.ndarray, w_area: np.ndarray, w_fit: np.ndarray, z_lo: float,
              z_hi: float) -> WallPoints:
        m = np.flatnonzero((np.abs(N[:, 2]) < W.WALL_NZ) & (P[:, 2] > z_lo) & (P[:, 2] < z_hi))
        codes = W.direction_codes(N[m, :2])
        out = cls()
        for k, (axis, _) in enumerate(W.DIRS):
            idx = m[codes == k]
            idx = idx[np.argsort(P[idx, axis], kind="stable")]
            out.v.append(P[idx, axis])
            out.t.append(P[idx, 1 - axis])
            out.z.append(P[idx, 2])
            out.wa.append(w_area[idx])
            out.wf.append(w_fit[idx])
        return out

    def select(self, axis: int, sign: int, lo: float, hi: float, t_lo: float, t_hi: float, z_lo: float,
               z_hi: float) -> tuple[np.ndarray, ...]:
        """v, t, z, wa, wf of the voxels facing sign along axis with v in [lo, hi] and t, z in range."""
        k = W.DIRS.index((axis, sign))
        i0, i1 = np.searchsorted(self.v[k], [lo, hi])
        t, z = self.t[k][i0:i1], self.z[k][i0:i1]
        m = (t >= t_lo) & (t <= t_hi) & (z > z_lo) & (z < z_hi)
        return self.v[k][i0:i1][m], t[m], z[m], self.wa[k][i0:i1][m], self.wf[k][i0:i1][m]


@dataclass
class FaceFit:
    """The face an edge lies on: coord + slope * (t - t_mid) across the edge, from n inlier voxels."""

    coord: float
    slope: float = 0.0
    t_mid: float = 0.0
    n: int = 0
    rms: float = 0.0
    sigma: float = 0.0
    mass: float = 0.0
    refined: bool = False
    flags: tuple[str, ...] = ()


@dataclass
class Edge:
    axis: int  # 0: the line x = coord, 1: the line y = coord
    coord: float
    sign: int  # +1 when the room lies on the positive side of the line
    fit: FaceFit | None = None
    kind: str = "unknown"  # furniture filter: wall | furniture | unknown


@dataclass
class Outline:
    polygon: np.ndarray
    edges: list[Edge]  # edge k runs from polygon[k] to polygon[k + 1]
    filled: list[Polygon] = field(default_factory=list)  # floor added behind furniture
    steps: int = 0  # steps and strips removed


def edges_of(poly: np.ndarray) -> list[Edge] | None:
    """Edges of a counter-clockwise rectilinear polygon without collinear vertices, else None."""
    K = len(poly)
    if K < 4 or K % 2:
        return None
    out = []
    for e in range(K):
        d = np.asarray(poly[(e + 1) % K], float) - np.asarray(poly[e], float)
        axis = 1 if abs(d[0]) >= abs(d[1]) else 0
        run = d[1 - axis]
        if abs(d[axis]) > 1e-9 or abs(run) < 1e-9:
            return None
        sign = int(np.sign(run)) if axis == 1 else -int(np.sign(run))  # inward normal (-dy, dx)
        out.append(Edge(axis, float(poly[e][axis]), sign))
    if any(out[k].axis == out[k - 1].axis for k in range(K)):
        return None
    return out


def outline(edges: list[Edge]) -> np.ndarray:
    P = np.empty((len(edges), 2))
    for k, e in enumerate(edges):
        prev = edges[k - 1].coord
        P[k] = (e.coord, prev) if e.axis == 0 else (prev, e.coord)
    return P


def _span(edges: list[Edge], k: int) -> tuple[float, float]:
    """Along-line coordinates of the start and the end of edge k."""
    return edges[k - 1].coord, edges[(k + 1) % len(edges)].coord


def _length(edges: list[Edge], k: int) -> float:
    a, b = _span(edges, k)
    return abs(b - a)


def consistent(edges: list[Edge]) -> bool:
    """Axes alternate, every edge runs the way its inward side requires, and the outline is simple."""
    K = len(edges)
    if K < 4 or K % 2:
        return False
    for k, e in enumerate(edges):
        if edges[k - 1].axis == e.axis:
            return False
        a, b = _span(edges, k)
        if abs(b - a) < 1e-4:
            return False
        if (int(np.sign(b - a)) if e.axis == 1 else -int(np.sign(b - a))) != e.sign:
            return False
    poly = Polygon(outline(edges))
    return bool(poly.is_valid and poly.exterior.is_ccw)


def _quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    return float(x[o][min(int(np.searchsorted(cw, q * cw[-1])), len(x) - 1)])


def _mode(v: np.ndarray, w: np.ndarray, c0: float, sigma: float) -> float:
    """Densest offset within BAND of c0 (weighted histogram smoothed at the noise level)."""
    nb = round(2 * BAND / MODE_RES)
    i = np.clip(((v - (c0 - BAND)) / MODE_RES).astype(np.int64), 0, nb - 1)
    h = np.bincount(i, weights=w, minlength=nb).astype(float)
    sm = gaussian_filter1d(h, max(sigma, 0.01) / MODE_RES, mode="constant")
    return c0 - BAND + (int(np.argmax(sm)) + 0.5) * MODE_RES


def _trim(v: np.ndarray, t: np.ndarray, w: np.ndarray, a: float, b: float, tm: float,
          sigma: float) -> tuple[float, float, np.ndarray]:
    """Trimmed mean offset of the line a + b (t - tm): window 2.5 robust sigmas, iterated."""
    s = max(sigma, WIN_MIN / 2.5)
    for _ in range(6):
        r = v - a - b * (t - tm)
        m = np.abs(r) < float(np.clip(2.5 * s, WIN_MIN, WIN_MAX))
        if m.sum() < 3 or w[m].sum() <= 0:
            break
        da = float(np.average(r[m], weights=w[m]))
        a += da
        s = robust_sigma(r[m] - da, w[m])
    r = v - a - b * (t - tm)
    return a, s, np.abs(r) < float(np.clip(2.5 * s, WIN_MIN, WIN_MAX))


@dataclass
class _Room:
    """The wall points within one room's height range and the capture's wall noise."""

    pts: WallPoints
    z0: float
    z1: float
    sigma: float

    def band(self, e: Edge, t0: float, t1: float, center: float | None = None,
             half: float = BAND) -> tuple[np.ndarray, ...]:
        lo, hi = min(t0, t1), max(t0, t1)
        m = min(END_MARGIN, 0.25 * (hi - lo))
        c = e.coord if center is None else center
        return self.pts.select(e.axis, e.sign, c - half, c + half, lo + m, hi - m, self.z0, self.z1)

    def fit(self, e: Edge, t0: float, t1: float, start: float | None = None) -> FaceFit:
        """Face of edge e over [t0, t1], from the densest offset within BAND or else from `start`."""
        tm = 0.5 * (t0 + t1)
        c0 = e.coord if start is None else start
        v, t, _, wa, wf = self.band(e, t0, t1, c0)
        if len(v) < MIN_POINTS:
            sparse = "wall_face_sparse" if len(v) else "wall_face_missing"
            return FaceFit(e.coord, t_mid=tm, n=len(v), sigma=self.sigma, flags=(sparse, "wall_not_refined"))
        c = _mode(v, wa, c0, self.sigma) if start is None else c0
        a, s, inl = _trim(v, t, wf, c, 0.0, tm, self.sigma)
        slope = 0.0
        if inl.sum() >= 20 and (len(v) > 1.15 * inl.sum() or s > 1.5 * self.sigma):
            # a wall a few degrees off the Manhattan axis spreads over many offsets: try one rotated line
            a2, b2, tm2 = W._slant_fit(t, v, wf, a, max(2.5 * self.sigma, 0.02))
            a2 += b2 * (tm - tm2)
            r2 = v - a2 - b2 * (t - tm)
            inl2 = np.abs(r2) < max(float(np.clip(2.5 * s, WIN_MIN, WIN_MAX)), 2.5 * self.sigma)
            s2 = robust_sigma(r2[inl2], wf[inl2]) if inl2.sum() > 10 else s
            if W.SLANT_MIN < abs(b2) <= W.SLANT_MAX and (inl2.sum() >= 1.15 * inl.sum() or s2 < 0.7 * s):
                a, s, inl = _trim(v, t, wf, a2, b2, tm, self.sigma)
                slope = b2
        n = int(inl.sum())
        if n < MIN_POINTS:
            return FaceFit(e.coord, t_mid=tm, n=n, sigma=self.sigma,
                           flags=("wall_face_sparse", "wall_not_refined"))
        r = v[inl] - a - slope * (t[inl] - tm)
        rms = float(np.sqrt(np.average(r ** 2, weights=wf[inl])))
        flags = (f"wall_slanted:{np.degrees(np.arctan(slope)):.1f}deg",) if abs(slope) > SLANT_FLAG else ()
        return FaceFit(float(a), float(slope), float(tm), n, rms, float(s), float(wa[inl].sum()), True, flags)

    def support(self, e: Edge, t0: float, t1: float) -> float:
        """Voxels on the face of e (its fitted line, else its coordinate) over [t0, t1]."""
        f = e.fit
        if f is not None and f.refined:
            a, b, tm, s = f.coord, f.slope, f.t_mid, f.sigma
        else:
            a, b, tm, s = e.coord, 0.0, 0.5 * (t0 + t1), self.sigma
        win = float(np.clip(2.5 * s, WIN_MIN, WIN_MAX))
        v, t, _, wa, _ = self.band(e, t0, t1, a, win + abs(b) * abs(t1 - t0))
        return float(wa[np.abs(v - a - b * (t - tm)) < win].sum())

    def kind(self, e: Edge, t0: float, t1: float, ceil_z: float) -> str:
        """furniture when the face points never reach the band CEILING_BAND below the ceiling."""
        _, _, z, wa, _ = self.band(e, t0, t1)
        if len(z) < MIN_POINTS:
            return "unknown"
        return "furniture" if _quantile(z, wa, TOP_QUANTILE) < ceil_z - CEILING_BAND else "wall"


def _furniture_runs(edges: list[Edge]) -> list[tuple[int, int, list[int]]]:
    """(wall before, wall after, edges between) for each run between walls that holds a furniture edge."""
    K = len(edges)
    walls = [k for k, e in enumerate(edges) if e.kind == "wall"]
    out = []
    for i, a in enumerate(walls):
        b = walls[(i + 1) % len(walls)]
        run = [(a + j) % K for j in range(1, (b - a) % K or K)]
        if run and b != a and any(edges[k].kind == "furniture" for k in run):
            out.append((a, b, run))
    return out


def _bridge(edges: list[Edge], a: int, b: int, run: list[int], room: _Room) -> list[Edge] | None:
    """The outline without `run`: wall a continues straight into wall b, or meets it at a corner."""
    ea, eb = edges[a], edges[b]
    drop = set(run)
    if ea.axis == eb.axis:
        if ea.sign != eb.sign or abs(ea.coord - eb.coord) > COLLINEAR_TOL:
            return None
        drop.add(b)
        t0, t1 = _span(edges, a)[0], _span(edges, b)[1]
        if abs(ea.coord - eb.coord) > 1e-9:
            # one wall on two nearby lines: keep the line with more face points along the joint extent
            best = max((ea, eb), key=lambda q: room.support(q, t0, t1))
            ea = Edge(ea.axis, best.coord, ea.sign, kind="wall")
    out = [ea if k == a else e for k, e in enumerate(edges) if k not in drop]
    if len(out) < 4 or not consistent(out):
        return None
    # a wall left shorter than a step was the notch's own side, and it reaches the ceiling band: no furniture
    if any((e is ea or e is eb) and _length(out, k) < STEP_MAX for k, e in enumerate(out)):
        return None
    return out


def _thin(region: Polygon, width: float) -> bool:
    return all(min(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1]) <= width + 1e-9
               for g in getattr(region, "geoms", [region]) if g.area > 1e-9)


def _added(edges: list[Edge], new: list[Edge]) -> Polygon | None:
    """The floor a fill adds, at most FURNITURE_AREA of the room and FURNITURE_DEPTH deep; it may give up
    only slivers where the two walls it joins were not exactly on one line."""
    p0, p1 = Polygon(outline(edges)), Polygon(outline(new))
    if p1.area <= p0.area + 1e-6 or p1.area - p0.area > FURNITURE_AREA * p0.area:
        return None
    if not _thin(p0.difference(p1), COLLINEAR_TOL):
        return None
    region = p1.difference(p0)
    return region if _thin(region, FURNITURE_DEPTH) else None


def _fill(edges: list[Edge], room: _Room, ceil_z: float,
          claimable: Callable[[Polygon], bool]) -> tuple[list[Edge], list[Polygon]]:
    """Replace runs of furniture edges between two walls by the walls behind them."""
    for k, e in enumerate(edges):
        e.kind = room.kind(e, *_span(edges, k), ceil_z)
    filled: list[Polygon] = []
    while len(edges) > 4:
        for a, b, run in _furniture_runs(edges):
            new = _bridge(edges, a, b, run, room)
            region = None if new is None else _added(edges, new)
            if region is not None and claimable(region):
                log.debug("furniture notch filled: %.2f m2", region.area)
                filled.append(region)
                edges = new
                break
        else:
            break
    return edges, filled


def _merge_step(edges: list[Edge], e: int, room: _Room, refit: bool) -> list[Edge]:
    """Remove the step edge e: its neighbours snap to the better-supported face and become one edge,
    refitted over the joint extent when `refit`."""
    K = len(edges)
    ia, ib = (e - 1) % K, (e + 1) % K
    a, b = edges[ia], edges[ib]
    t0, t1 = edges[(e - 2) % K].coord, edges[(e + 2) % K].coord
    sa, sb = room.support(a, t0, t1), room.support(b, t0, t1)
    best = a if sa >= sb else b
    merged = Edge(a.axis, best.coord, a.sign, kind=a.kind)
    if refit:
        merged.fit = room.fit(merged, t0, t1, start=best.coord)
        if merged.fit.refined:
            merged.coord = merged.fit.coord
    log.debug("step %.3f m removed: faces %.3f (support %.0f) and %.3f (%.0f) -> %.3f",
              abs(b.coord - a.coord), a.coord, sa, b.coord, sb, merged.coord)
    return [merged if k == ia else q for k, q in enumerate(edges) if k not in (e, ib)]


def _cut_strip(edges: list[Edge], e: int) -> list[Edge]:
    """Remove a strip narrower than STEP_MAX whose end is edge e (its sides e - 1 and e + 1 run opposite
    ways): the end and the shorter side go, and the edge beyond the shorter side extends to the longer
    side."""
    K = len(edges)
    la = abs(edges[e].coord - edges[(e - 2) % K].coord)
    lb = abs(edges[(e + 2) % K].coord - edges[e].coord)
    drop = {(e - 1) % K, e} if la <= lb else {e, (e + 1) % K}
    log.debug("strip %.3f m wide, %.3f m long removed", abs(edges[(e + 1) % K].coord - edges[e - 1].coord),
              min(la, lb))
    return [q for k, q in enumerate(edges) if k not in drop]


def _regularise(edges: list[Edge], room: _Room, refit: bool) -> tuple[list[Edge], int]:
    """Remove edges shorter than STEP_MAX between parallel edges, shortest first: a step between edges that
    run the same way, or the end of a thin strip between edges that run opposite ways."""
    n = 0
    while len(edges) > 4:
        K = len(edges)
        step, e = min((abs(edges[(k + 1) % K].coord - edges[k - 1].coord), k) for k in range(K))
        if step >= STEP_MAX:
            break
        if edges[e - 1].sign == edges[(e + 1) % K].sign:
            edges = _merge_step(edges, e, room, refit)
        else:
            edges = _cut_strip(edges, e)
        n += 1
    return edges, n


def _refit_all(edges: list[Edge], room: _Room, keep_face: bool) -> None:
    """Fit every edge over its current extent, then move each onto its face (all at once)."""
    fits = [room.fit(e, *_span(edges, k), start=e.coord if keep_face else None) for k, e in enumerate(edges)]
    for e, f in zip(edges, fits, strict=True):
        e.fit = f
        if f.refined:
            e.coord = f.coord


def room_outline(poly: np.ndarray, pts: WallPoints, floor_z: float, ceil_z: float, sigma: float,
                 claimable: Callable[[Polygon], bool]) -> Outline | None:
    """The room's outline from its cell polygon. None when the polygon is not rectilinear or the refined
    outline is not a simple counter-clockwise polygon; the caller then keeps the cell polygon.

    claimable(region) is False for plan regions that hold cells of another room.
    """
    edges = edges_of(poly)
    if edges is None:
        return None
    room = _Room(pts, floor_z + 0.05, ceil_z - 0.03, sigma)
    edges, steps = _regularise(edges, room, refit=False)
    edges, filled = _fill(edges, room, ceil_z, claimable)
    _refit_all(edges, room, keep_face=False)
    edges, n = _regularise(edges, room, refit=True)
    steps += n
    # a notch between walls on the two faces of one wall body lines up only once the walls are refitted
    edges, late = _fill(edges, room, ceil_z, claimable)
    if late:
        edges, n = _regularise(edges, room, refit=True)
        steps += n
    _refit_all(edges, room, keep_face=True)
    if not consistent(edges):
        log.debug("refined outline is not simple; keeping the cell polygon")
        return None
    P = outline(edges)
    start = int(np.lexsort((P[:, 0], P[:, 1]))[0])
    return Outline(np.roll(P, -start, axis=0), edges[start:] + edges[:start], filled + late, steps)
