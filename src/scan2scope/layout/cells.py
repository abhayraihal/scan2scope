"""Cell complex from wall lines, 2-D free-space evidence, inside test and room segmentation.

Coordinates are in the Manhattan frame. Rooms are separated by wall bodies (cells with no free space) and
by boundaries with wall evidence; a gap with a wall above it (a door or window head) counts as wall, so
doors never merge rooms. Header-less gaps of door width are closed too unless the line is only crossing
a corridor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from shapely.geometry import Polygon
from shapely.geometry import box as sbox
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

from scan2scope.layout.walls import TGrid, WallLine

log = logging.getLogger("scan2scope.layout")

FREE_RES = 0.025
COORD_MERGE = 0.01
SMALL_GAP = 0.5
DOOR_MAX = 1.4
OPEN_RUN = 0.3
INSIDE_COVERAGE = 0.5
MIN_ROOM_AREA = 0.7
MIN_ROOM_WIDTH = 0.5
NO_CAMERA_COVERAGE = 0.75
NO_CAMERA_SUPPORT = 0.6
NO_CAMERA_FLOOR = 0.3


@dataclass
class Grid2D:
    x0: float
    y0: float
    nx: int
    ny: int
    res: float = FREE_RES


@dataclass
class Occupancy:
    """Occupied 3-D voxels; a ray that enters one has hit a surface, whatever lies beyond it."""

    origin: np.ndarray
    res: float
    grid: np.ndarray  # (nx, ny, nz) bool


def occupancy(P: np.ndarray, lo: np.ndarray, hi: np.ndarray, res: float = 0.05, min_count: int = 2) -> Occupancy:
    dims = np.maximum(np.ceil((hi - lo) / res).astype(np.int64), 1)
    idx = np.floor((P - lo) / res).astype(np.int64)
    ok = ((idx >= 0) & (idx < dims)).all(1)
    keys, cnt = np.unique(np.ravel_multi_index(idx[ok].T, dims), return_counts=True)
    grid = np.zeros(int(np.prod(dims)), bool)
    grid[keys[cnt >= min_count]] = True
    return Occupancy(np.asarray(lo, float), res, grid.reshape(tuple(dims)))


def free_space(o3: np.ndarray, e3: np.ndarray, n2: np.ndarray, grid: Grid2D, margin: float,
               rng: np.random.Generator, occ: Occupancy | None = None, max_samples: int = 24_000_000,
               chunk: int = 1_000_000) -> np.ndarray:
    """Ray samples per plan cell along camera-to-point segments.

    A segment stops `margin` short of its point, measured along the point's horizontal normal so rays
    grazing a wall do not mark the wall body, and stops at the first occupied voxel it enters, so rays to
    outliers behind a wall do not mark the space behind it.
    """
    d3 = e3 - o3
    L2 = np.linalg.norm(d3[:, :2], axis=1)
    nh = np.linalg.norm(n2, axis=1)
    cos = np.abs((d3[:, :2] * n2).sum(1)) / np.maximum(L2 * nh, 1e-12)
    back = np.where(nh > 0.7, margin / np.maximum(cos, 0.2), margin)
    frac = 1.0 - back / np.maximum(L2, 1e-12)
    ok = (L2 > 1e-6) & (frac > 0)
    L3 = np.linalg.norm(d3, axis=1)
    o3, u3, Lf = o3[ok], d3[ok] / L3[ok, None], (L3 * frac)[ok]
    counts = np.zeros(grid.nx * grid.ny)
    if len(Lf) == 0:
        return counts.reshape(grid.nx, grid.ny)
    step = grid.res
    ns = np.ceil(Lf / step).astype(np.int64) + 1
    if ns.sum() > max_samples:
        keep = rng.random(len(ns)) < max_samples / ns.sum()
        o3, u3, Lf, ns = o3[keep], u3[keep], Lf[keep], ns[keep]
    cs = np.cumsum(ns)
    start = 0
    while start < len(ns):
        base = cs[start - 1] if start else 0
        end = max(int(np.searchsorted(cs, base + chunk, side="right")), start + 1)
        nn = ns[start:end]
        rid = np.repeat(np.arange(end - start), nn)
        k = np.arange(len(rid)) - np.repeat(np.cumsum(nn) - nn, nn)
        s = np.minimum(k * step, Lf[start:end][rid])
        p = o3[start:end][rid] + u3[start:end][rid] * s[:, None]
        keep = np.ones(len(rid), bool)
        if occ is not None:
            j = np.floor((p - occ.origin) / occ.res).astype(np.int64)
            inb = ((j >= 0) & (j < occ.grid.shape)).all(1)
            hit = np.zeros(len(rid), bool)
            hit[inb] = occ.grid[j[inb, 0], j[inb, 1], j[inb, 2]]
            hit &= s > 0.1  # ignore a camera's own neighbourhood
            first = np.full(end - start, np.iinfo(np.int64).max)
            np.minimum.at(first, rid[hit], k[hit])
            keep = k < first[rid]
        ix = np.floor((p[:, 0] - grid.x0) / grid.res).astype(np.int64)
        iy = np.floor((p[:, 1] - grid.y0) / grid.res).astype(np.int64)
        m = keep & (ix >= 0) & (ix < grid.nx) & (iy >= 0) & (iy < grid.ny)
        counts += np.bincount(ix[m] * grid.ny + iy[m], minlength=grid.nx * grid.ny)
        start = end
    return counts.reshape(grid.nx, grid.ny)


@dataclass
class LineGroup:
    """Wall lines of one axis whose coordinates agree within 1 cm."""

    axis: int
    coord: float
    members: list[int]
    solid: np.ndarray
    upper: np.ndarray
    closed: np.ndarray | None = None
    synthetic: bool = False


def group_lines(lines: list[WallLine], axis: int) -> list[LineGroup]:
    idx = sorted((i for i, q in enumerate(lines) if q.axis == axis), key=lambda i: lines[i].coord)
    groups: list[LineGroup] = []
    last = -np.inf
    for i in idx:
        q = lines[i]
        if groups and q.coord - last < COORD_MERGE:
            g = groups[-1]
            g.members.append(i)
            g.solid = g.solid | q.solid
            g.upper = g.upper | q.upper
            g.synthetic = g.synthetic and q.synthetic
        else:
            groups.append(LineGroup(axis, q.coord, [i], q.solid.copy(), q.upper.copy(), synthetic=q.synthetic))
        last = q.coord
    for g in groups:
        w = np.array([max(lines[i].area, 1e-6) for i in g.members])
        g.coord = float(np.average([lines[i].coord for i in g.members], weights=w))
    return groups


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Half-open index ranges of consecutive True values."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.astype(np.int8), [0]]))
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _corridor(g: LineGroup, others: list[LineGroup], ogrid: TGrid, ta: float, tb: float) -> bool:
    """True when a perpendicular wall runs through either end of the gap, i.e. the line only crosses a corridor."""
    for te in (ta, tb):
        for og in others:
            if og.synthetic or abs(og.coord - te) > 0.2:
                continue
            a0, a1 = ogrid.span(g.coord - 0.4, g.coord - 0.1)
            b0, b1 = ogrid.span(g.coord + 0.1, g.coord + 0.4)
            if a1 > a0 and b1 > b0 and og.solid[a0:a1].mean() >= 0.5 and og.solid[b0:b1].mean() >= 0.5:
                return True
    return False


def close_groups(gx: list[LineGroup], gy: list[LineGroup], grids: tuple[TGrid, TGrid]) -> None:
    """Wall evidence per profile bin, with door and window heads and short data gaps counted as wall."""
    for groups, others, axis in ((gx, gy, 0), (gy, gx, 1)):
        grid, ogrid = grids[axis], grids[1 - axis]
        for g in groups:
            closed = g.solid.copy()
            for a, b in runs(g.upper & ~g.solid):
                if (b - a) * grid.res <= DOOR_MAX + 1e-9:
                    closed[a:b] = True
            for a, b in runs(~closed):
                if a == 0 or b == len(closed):
                    continue
                length = (b - a) * grid.res
                door_sized = (length <= DOOR_MAX + 1e-9 and g.solid[max(a - 2, 0):a].any() and g.solid[b:b + 2].any()
                              and not _corridor(g, others, ogrid, grid.t0 + a * grid.res, grid.t0 + b * grid.res))
                if length < SMALL_GAP - 1e-9 or door_sized:
                    closed[a:b] = True
            g.closed = closed


def _sat(mask: np.ndarray) -> np.ndarray:
    S = np.zeros((mask.shape[0] + 1, mask.shape[1] + 1))
    S[1:, 1:] = mask.astype(float).cumsum(0).cumsum(1)
    return S


def _integral(S: np.ndarray, grid: Grid2D, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Integral of the mask over [x0, x] x [y0, y]; bilinear in the SAT is exact for a piecewise-constant mask."""
    gx = np.clip((x - grid.x0) / grid.res, 0, grid.nx)
    gy = np.clip((y - grid.y0) / grid.res, 0, grid.ny)
    i = np.minimum(np.floor(gx).astype(np.int64), grid.nx - 1)
    j = np.minimum(np.floor(gy).astype(np.int64), grid.ny - 1)
    fx, fy = gx - i, gy - j
    v = (S[i, j] * (1 - fx) * (1 - fy) + S[i + 1, j] * fx * (1 - fy) + S[i, j + 1] * (1 - fx) * fy
         + S[i + 1, j + 1] * fx * fy)
    return v * grid.res ** 2


@dataclass
class Complex:
    xs: np.ndarray
    ys: np.ndarray
    gx: list[LineGroup]
    gy: list[LineGroup]
    grids: tuple[TGrid, TGrid]
    coverage: np.ndarray  # (nx, ny) free-space share of each cell
    floor_cov: np.ndarray  # (nx, ny) share of each cell with floor points
    thin: np.ndarray
    inside: np.ndarray
    open_v: np.ndarray  # (nx - 1, ny): passage across x = xs[i + 1] between cells (i, j) and (i + 1, j)
    open_h: np.ndarray  # (nx, ny - 1): passage across y = ys[j + 1]
    flags: list[str] = field(default_factory=list)

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.xs) - 1, len(self.ys) - 1

    def segment(self, axis: int, k: int, a: float, b: float, which: str = "closed") -> np.ndarray:
        """Profile bins (closed or upper) of the boundary on line group k of `axis` between t = a and t = b."""
        g = (self.gx if axis == 0 else self.gy)[k]
        prof = g.closed if which == "closed" else g.upper
        grid = self.grids[axis]
        i0, i1 = grid.span(a, b)
        if i1 <= i0:
            i = int(np.clip(grid.index(0.5 * (a + b)), 0, grid.n - 1))
            return prof[i:i + 1]
        return prof[i0:i1]


def _is_open(closed: np.ndarray, length: float, res: float) -> bool:
    free = runs(~closed)
    if not free:
        return False
    longest = max(b - a for a, b in free) * res
    return longest >= min(OPEN_RUN, 0.5 * length) - 1e-9


def _cell_share(mask: np.ndarray, grid: Grid2D, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    S = _sat(mask)
    X0, Y0 = np.meshgrid(xs[:-1], ys[:-1], indexing="ij")
    X1, Y1 = np.meshgrid(xs[1:], ys[1:], indexing="ij")
    integ = (_integral(S, grid, X1, Y1) - _integral(S, grid, X0, Y1) - _integral(S, grid, X1, Y0)
             + _integral(S, grid, X0, Y0))
    return np.clip(integ / np.maximum((X1 - X0) * (Y1 - Y0), 1e-12), 0, 1)


def build_complex(gx: list[LineGroup], gy: list[LineGroup], grids: tuple[TGrid, TGrid], free: np.ndarray,
                  floor: np.ndarray, grid: Grid2D, t_thin: float) -> Complex:
    xs = np.array([g.coord for g in gx])
    ys = np.array([g.coord for g in gy])
    nx, ny = len(xs) - 1, len(ys) - 1
    coverage = _cell_share(free, grid, xs, ys)
    thin = np.minimum(np.diff(xs)[:, None], np.diff(ys)[None, :]) < t_thin
    inside = coverage >= INSIDE_COVERAGE
    cx = Complex(xs, ys, gx, gy, grids, coverage, _cell_share(floor, grid, xs, ys), thin, inside,
                 np.zeros((max(nx - 1, 0), ny), bool), np.zeros((nx, max(ny - 1, 0)), bool))
    wide_v = np.zeros_like(cx.open_v)
    wide_h = np.zeros_like(cx.open_h)
    for i in range(nx - 1):
        for j in range(ny):
            c = cx.segment(0, i + 1, ys[j], ys[j + 1])
            cx.open_v[i, j] = _is_open(c, ys[j + 1] - ys[j], grids[0].res)
            wide_v[i, j] = cx.open_v[i, j] and c.mean() <= 0.5
    for i in range(nx):
        for j in range(ny - 1):
            c = cx.segment(1, j + 1, xs[i], xs[i + 1])
            cx.open_h[i, j] = _is_open(c, xs[i + 1] - xs[i], grids[1].res)
            wide_h[i, j] = cx.open_h[i, j] and c.mean() <= 0.5
    # thin cells that lost their free space to the ray margin inherit it across boundaries that are mostly
    # open (furniture slivers, niches, corridor strips); a wall body with a single gap is not one of them
    for _ in range(10):
        grow = np.zeros_like(inside)
        grow[:-1] |= inside[1:] & wide_v
        grow[1:] |= inside[:-1] & wide_v
        grow[:, :-1] |= inside[:, 1:] & wide_h
        grow[:, 1:] |= inside[:, :-1] & wide_h
        new = grow & thin & ~inside
        if not new.any():
            break
        inside |= new
    cx.inside = inside
    return cx


@dataclass
class Region:
    cells: list[tuple[int, int]]
    area: float
    bbox: tuple[float, float, float, float]
    n_cams: int
    coverage: float
    support: float  # share of the region boundary with wall evidence
    floor_cov: float = 0.0
    first_cam: int = 1 << 30
    neighbors: dict[int, float] = field(default_factory=dict)  # region index -> shared boundary length
    neighbors_tall: dict[int, float] = field(default_factory=dict)  # same, where wall reaches door-head height


def label_regions(cx: Complex) -> np.ndarray:
    nx, ny = cx.shape
    idx = np.arange(nx * ny).reshape(nx, ny)
    ins = cx.inside
    v = cx.open_v & ins[:-1, :] & ins[1:, :]
    h = cx.open_h & ins[:, :-1] & ins[:, 1:]
    rows = np.concatenate([idx[:-1][v], idx[:, :-1][h]])
    cols = np.concatenate([idx[1:][v], idx[:, 1:][h]])
    g = coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(nx * ny, nx * ny))
    _, lab = connected_components(g, directed=False)
    lab = lab.reshape(nx, ny)
    out = np.full((nx, ny), -1)
    for k, u in enumerate(np.unique(lab[ins])):
        out[(lab == u) & ins] = k
    return out


def region_stats(cx: Complex, lab: np.ndarray, cam_xy: np.ndarray) -> list[Region]:
    nx, ny = cx.shape
    xs, ys = cx.xs, cx.ys
    nreg = int(lab.max()) + 1 if lab.size else 0
    cam_i = np.searchsorted(xs, cam_xy[:, 0]) - 1 if len(cam_xy) else np.zeros(0, int)
    cam_j = np.searchsorted(ys, cam_xy[:, 1]) - 1 if len(cam_xy) else np.zeros(0, int)
    regs: list[Region] = []
    for r in range(nreg):
        ii, jj = np.nonzero(lab == r)
        a = (xs[ii + 1] - xs[ii]) * (ys[jj + 1] - ys[jj])
        cams = [k for k in range(len(cam_xy)) if 0 <= cam_i[k] < nx and 0 <= cam_j[k] < ny
                and lab[cam_i[k], cam_j[k]] == r]
        sup_len = tot_len = 0.0
        nb: dict[int, float] = {}
        nb_tall: dict[int, float] = {}
        for i, j in zip(ii.tolist(), jj.tolist()):
            for di, dj in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                i2, j2 = i + di, j + dj
                other = lab[i2, j2] if (0 <= i2 < nx and 0 <= j2 < ny) else -1
                if other == r:
                    continue
                if di:
                    k, a0, a1, axis = (i if di < 0 else i + 1), ys[j], ys[j + 1], 0
                else:
                    k, a0, a1, axis = (j if dj < 0 else j + 1), xs[i], xs[i + 1], 1
                length = a1 - a0
                tot_len += length
                sup_len += length * float(cx.segment(axis, k, a0, a1).mean())
                if other >= 0:
                    nb[int(other)] = nb.get(int(other), 0.0) + length
                    tall = length * float(cx.segment(axis, k, a0, a1, "upper").mean())
                    nb_tall[int(other)] = nb_tall.get(int(other), 0.0) + tall
        regs.append(Region(list(zip(ii.tolist(), jj.tolist())), float(a.sum()),
                           (float(xs[ii].min()), float(ys[jj].min()), float(xs[ii + 1].max()), float(ys[jj + 1].max())),
                           len(cams), float(np.average(cx.coverage[ii, jj], weights=a)),
                           sup_len / max(tot_len, 1e-9), float(np.average(cx.floor_cov[ii, jj], weights=a)),
                           min(cams) if cams else 1 << 30, nb, nb_tall))
    return regs


def select_rooms(regs: list[Region], t_thin: float, single_room: bool) -> tuple[list[list[int]], list[str]]:
    """Group regions into rooms. Returns lists of region indices (the first is the room's main region) and flags."""
    flags: list[str] = []
    kind = []
    for r in regs:
        w, h = r.bbox[2] - r.bbox[0], r.bbox[3] - r.bbox[1]
        if min(w, h) < max(t_thin, 0.25):
            kind.append("thin")
        elif r.area < MIN_ROOM_AREA or min(w, h) < MIN_ROOM_WIDTH:
            kind.append("small")
        elif r.n_cams > 0:
            kind.append("room")
        elif r.coverage >= NO_CAMERA_COVERAGE and r.support >= NO_CAMERA_SUPPORT and r.floor_cov >= NO_CAMERA_FLOOR:
            kind.append("room")
            flags.append("room_without_cameras")
        else:
            kind.append("reject")
    for k, (r, c) in enumerate(zip(regs, kind)):
        log.debug("region %d %s: area %.2f bbox %s cams %d coverage %.2f support %.2f floor %.2f", k, c, r.area,
                  tuple(round(v, 2) for v in r.bbox), r.n_cams, r.coverage, r.support, r.floor_cov)
    rooms = {k: [k] for k, c in enumerate(kind) if c == "room"}
    for k, c in enumerate(kind):
        if c != "small":
            continue
        # merge only across furniture-height boundaries (wardrobe or shelf fronts), never through a full wall
        adj = [n for n, length in regs[k].neighbors.items() if kind[n] == "room"
               and regs[k].neighbors_tall.get(n, 0.0) < 0.5 * length]
        if len(adj) == 1:
            rooms[adj[0]].append(k)
    if not rooms and regs:
        k = max(range(len(regs)), key=lambda q: (kind[q] != "thin", regs[q].n_cams, regs[q].area))
        rooms = {k: [k]}
        flags.append("room_fallback_largest_region")
    if single_room and len(rooms) > 1:
        k = max(rooms, key=lambda q: (regs[q].n_cams, regs[q].area))
        flags.append(f"single_room_dropped:{len(rooms) - 1}")
        rooms = {k: rooms[k]}
    order = sorted(rooms, key=lambda q: (regs[q].first_cam, -regs[q].area))
    return [rooms[q] for q in order], flags


def cells_polygon(cx: Complex, cells: list[tuple[int, int]]) -> tuple[np.ndarray, float]:
    """Counter-clockwise outline of a union of cells with collinear vertices removed; returns it and the hole area."""
    xs, ys = cx.xs, cx.ys
    u = unary_union([sbox(xs[i], ys[j], xs[i + 1], ys[j + 1]) for i, j in cells])
    if u.geom_type != "Polygon":
        u = max(getattr(u, "geoms", [u]), key=lambda g: g.area)
    hole = float(Polygon(u.exterior).area - u.area)
    ring = orient(Polygon(u.exterior), 1.0)
    pts = drop_collinear(np.asarray(ring.exterior.coords)[:-1])
    start = int(np.lexsort((pts[:, 0], np.round(pts[:, 1], 6)))[0])
    return np.roll(pts, -start, axis=0), hole


def drop_collinear(pts: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    keep = list(range(len(pts)))
    changed = True
    while changed and len(keep) > 3:
        changed = False
        for k in range(len(keep)):
            a, b, c = pts[keep[k - 1]], pts[keep[k]], pts[keep[(k + 1) % len(keep)]]
            u, v = b - a, c - b
            if np.linalg.norm(u) < 1e-9 or abs(u[0] * v[1] - u[1] * v[0]) < tol:
                del keep[k]
                changed = True
                break
    return pts[keep]
