"""build_plan: a gravity-aligned Scene to a Plan of rooms, walls and openings (values and evidence only)."""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Polygon

from scan2scope.layout import cells as C
from scan2scope.layout import floor_ceiling as FC
from scan2scope.layout import openings as OP
from scan2scope.layout import refine as R
from scan2scope.layout import walls as W
from scan2scope.types import Adjacency, Measurement, Opening, Plan, Room, Scene, Wall

log = logging.getLogger("scan2scope.layout")

VOXEL = 0.02
MAX_RAYS = 300_000
MAX_FREE_RAYS = 100_000
FAR_LIMIT = 60.0
MAX_REACH = 10.0  # plan domain: camera bounding box plus this margin
MAX_EXTENT = 60.0


@dataclass
class _Data:
    P: np.ndarray  # voxel centroids, Manhattan frame
    N: np.ndarray
    wp: np.ndarray  # mean confidence per voxel (peak finding, occupancy)
    wf: np.ndarray  # summed confidence per voxel (fits)
    wn: np.ndarray  # length of the confidence-weighted normal sum; projected on a wall normal: face mass
    O: np.ndarray  # ray origins (raw point subsample), Manhattan frame
    E: np.ndarray  # ray ends
    cams: np.ndarray  # camera centres, Manhattan frame
    cam_ok: np.ndarray


def _rot(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _xy(A: np.ndarray, R: np.ndarray) -> np.ndarray:
    out = np.array(A, dtype=float, copy=True)
    out[..., :2] = A[..., :2] @ R.T
    return out


def _center(T: object) -> np.ndarray:
    try:
        return np.asarray(T, float).reshape(4, 4)[:3, 3]
    except (TypeError, ValueError):
        return np.full(3, np.nan)


def _sanitize(scene: Scene) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                     list[str]]:
    flags: list[str] = []
    P = np.asarray(scene.points if scene.points is not None else np.zeros((0, 3)), float).reshape(-1, 3)
    n = len(P)
    N = np.asarray(scene.normals, float) if scene.normals is not None else np.zeros((0, 3))
    if N.shape != (n, 3):
        flags.append("normals_missing")
        N = np.zeros((n, 3))
    Wt = np.asarray(scene.weights, float).reshape(-1) if scene.weights is not None else np.zeros(0)
    if Wt.shape != (n,):
        flags.append("weights_missing")
        Wt = np.ones(n)
    V = np.asarray(scene.view_index).reshape(-1) if scene.view_index is not None else np.zeros(0)
    if V.shape != (n,):
        flags.append("view_index_missing")
        V = np.full(n, -1)
    V = np.where(np.isfinite(V.astype(float)), V, -1).astype(np.int64)
    cams = np.array([_center(v.T_wc) for v in scene.views]).reshape(-1, 3)
    cam_ok = np.isfinite(cams).all(1)
    if len(cams) and not cam_ok.all():
        flags.append(f"bad_view_poses:{int((~cam_ok).sum())}")
    Wt = np.clip(np.nan_to_num(Wt, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    nn = np.linalg.norm(N, axis=1)
    ok = np.isfinite(P).all(1) & np.isfinite(N).all(1) & (nn > 1e-6) & (Wt > 0)
    if n and ok.sum() < n:
        flags.append(f"invalid_points_dropped:{int(n - ok.sum())}")
    if ok.any():
        med = np.median(P[ok], axis=0)
        far = ok & (np.abs(P - med) > FAR_LIMIT).any(1)
        if far.any():
            flags.append(f"far_points_dropped:{int(far.sum())}")
            ok &= ~far
    P, N, Wt, V = P[ok], N[ok] / nn[ok, None], Wt[ok], V[ok]
    V = np.where((V >= 0) & (V < len(cams)), V, -1)
    if len(cams):
        V = np.where(V >= 0, np.where(cam_ok[np.maximum(V, 0)], V, -1), -1)
    else:
        flags.append("no_views")
    return P, N, Wt, V, cams, cam_ok, flags


def _voxelize(P: np.ndarray, N: np.ndarray, Wt: np.ndarray, voxel: float) -> tuple[np.ndarray, ...]:
    k = np.floor(P / voxel).astype(np.int64)
    k -= k.min(0)
    dims = k.max(0) + 1
    key = (k[:, 0] * dims[1] + k[:, 1]) * dims[2] + k[:, 2]
    _, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    inv = inv.reshape(-1)
    m = len(cnt)
    Pv = np.stack([np.bincount(inv, P[:, i], m) for i in range(3)], 1) / cnt[:, None]
    Nv = np.stack([np.bincount(inv, N[:, i] * Wt, m) for i in range(3)], 1)
    nn = np.linalg.norm(Nv, axis=1)
    keep = nn > 1e-9
    Nv = Nv / np.maximum(nn, 1e-12)[:, None]
    ws = np.bincount(inv, Wt, m)
    return Pv[keep], Nv[keep], (ws / cnt)[keep], ws[keep], nn[keep]


def _label(hint: str | None) -> str | None:
    if not hint:
        return None
    s = re.sub(r"^\s*\d+\s*[-_.]*\s*", "", str(hint)).replace("_", " ").strip()
    return s or None


def _m(value: float, kind: str = "length", unit: str = "m", **evidence) -> Measurement:
    return Measurement(value=float(value), unit=unit, kind=kind, evidence=evidence)


def _empty_plan(flags: list[str], meta: dict) -> Plan:
    return Plan(rooms=[], adjacency=[], footprint_area=_m(0.0, "area", "m2"), extent_x=_m(0.0),
                extent_y=_m(0.0), flags=flags + ["no_rooms"], meta={"layout": meta})


def _grids(P: np.ndarray, cams: np.ndarray, pad: float = 1.0) -> tuple[tuple[W.TGrid, W.TGrid], tuple]:
    pts = np.concatenate([P[:, :2], cams[:, :2]]) if len(cams) else P[:, :2]
    lo = np.percentile(pts, 0.1, axis=0) - pad
    hi = np.percentile(pts, 99.9, axis=0) + pad
    if len(cams):
        # rooms lie within a few metres of some camera; distant outdoor surfaces must not blow up the grids
        lo = np.maximum(lo, cams[:, :2].min(0) - MAX_REACH)
        hi = np.minimum(hi, cams[:, :2].max(0) + MAX_REACH)
    hi = np.minimum(hi, lo + MAX_EXTENT)
    ty = W.TGrid(float(lo[1]), max(int(np.ceil((hi[1] - lo[1]) / W.T_RES)), 1))
    tx = W.TGrid(float(lo[0]), max(int(np.ceil((hi[0] - lo[0]) / W.T_RES)), 1))
    return (ty, tx), (lo, hi)


def build_plan(scene: Scene, *, single_room: bool = False) -> Plan:
    t_start = time.perf_counter()
    P_raw, N_raw, W_raw, V_raw, cams_w, cam_ok, flags = _sanitize(scene)
    meta: dict = {"n_points_in": int(len(scene.points) if scene.points is not None else 0)}
    if len(P_raw) < 200:
        log.warning("layout: only %d usable points", len(P_raw))
        return _empty_plan(flags + ["too_few_points"], meta)
    rng = np.random.default_rng(0)
    Pv, Nv, wp, wf, wn = _voxelize(P_raw, N_raw, W_raw, VOXEL)
    ridx = rng.permutation(np.flatnonzero(V_raw >= 0))[:MAX_RAYS]
    if len(ridx) == 0:
        flags.append("no_camera_rays")
    O_w, E_w, En_w = cams_w[V_raw[ridx]], P_raw[ridx], N_raw[ridx]
    cams_ok = cams_w[cam_ok]

    fc = FC.estimate(Pv[:, 2], Nv[:, 2], wp, wf, Pv[:, :2], cams_ok[:, 2] if len(cams_ok) else None)
    flags += fc.flags
    if fc.floor.observed and fc.floor.tilt > 0.01:
        flags.append(f"floor_tilted:{fc.floor.tilt:.3f}")
    floor_z, ceil_z = fc.floor.z, fc.ceiling.z
    wall_m = (np.abs(Nv[:, 2]) < W.WALL_NZ) & (Pv[:, 2] > floor_z + 0.15) & (Pv[:, 2] < ceil_z - 0.15)
    theta, conc = W.manhattan_frame(Nv[wall_m, :2], wp[wall_m])
    if wall_m.sum() < 100 or conc < 0.15:
        flags.append("manhattan_weak")
    Rm = _rot(-theta)
    d = _Data(_xy(Pv, Rm), _xy(Nv, Rm), wp, wf, wn, _xy(O_w, Rm), _xy(E_w, Rm), _xy(cams_w, Rm), cam_ok)
    En = _xy(En_w, Rm)
    codes = W.direction_codes(d.N[wall_m, :2])
    off_axis = float((wp[wall_m][codes < 0]).sum() / max(wp[wall_m].sum(), 1e-9))
    if off_axis > 0.25:
        flags.append(f"non_manhattan_walls:{off_axis:.2f}")

    wl = (np.abs(d.N[:, 2]) < W.WALL_NZ) & (d.P[:, 2] > floor_z + 0.05) & (d.P[:, 2] < ceil_z - 0.05)
    grids, (lo, hi) = _grids(d.P[wl] if wl.sum() > 10 else d.P, d.cams[cam_ok])
    sigma0 = float(np.clip(fc.floor.sigma, 0.006, 0.08)) if fc.floor.observed else 0.02
    lines, sigma = W.detect_lines(d.P[wl], d.N[wl], wp[wl], wf[wl], floor_z, ceil_z, sigma0, grids)
    if lines and (sigma > 1.5 * sigma0 or sigma < 0.6 * sigma0):
        sigma0 = float(np.clip(sigma, 0.006, 0.08))
        lines, sigma = W.detect_lines(d.P[wl], d.N[wl], wp[wl], wf[wl], floor_z, ceil_z, sigma0, grids)
    sigma = float(np.clip(sigma, 0.004, 0.10))

    margin = max(0.015, 1.5 * sigma)
    g2 = C.Grid2D(float(lo[0]), float(lo[1]), int(np.ceil((hi[0] - lo[0]) / C.FREE_RES)),
                  int(np.ceil((hi[1] - lo[1]) / C.FREE_RES)))
    nf = min(len(d.O), MAX_FREE_RAYS)
    occ = C.occupancy(d.P, np.array([lo[0], lo[1], floor_z - 0.3]), np.array([hi[0], hi[1], ceil_z + 0.3]))
    counts = (C.free_space(d.O[:nf], d.E[:nf], En[:nf, :2], g2, margin, rng, occ) if nf
              else np.zeros((g2.nx, g2.ny)))
    pos = counts[counts > 0]
    n_min = max(2.0, 0.03 * float(np.median(pos))) if len(pos) else 2.0
    free = counts >= n_min
    floor = _floor_mask(d, fc, sigma, g2)
    if free.sum() * C.FREE_RES ** 2 < 0.5 and floor.any():
        free |= floor
        flags.append("free_space_from_floor")
    lines += _closure_lines(lines, free, g2, grids, flags)
    gx, gy = C.group_lines(lines, 0), C.group_lines(lines, 1)
    t_thin = max(0.2, 2 * margin + 0.05)
    meta.update({"manhattan_angle_deg": round(float(np.degrees(theta)), 3),
                 "manhattan_concentration": round(conc, 3),
                 "noise_sigma": round(sigma, 4), "ray_margin": round(margin, 4), "floor_z": round(floor_z, 4),
                 "ceiling_z": round(ceil_z, 4), "n_voxels": len(Pv), "n_rays": len(d.O),
                 "n_lines": len([q for q in lines if not q.synthetic]), "free_min_count": round(n_min, 2)})
    groups: list[list[int]] = []
    if len(gx) >= 2 and len(gy) >= 2:
        C.close_groups(gx, gy, grids)
        cx = C.build_complex(gx, gy, grids, free, floor, g2, t_thin)
        lab = C.label_regions(cx)
        regs = C.region_stats(cx, lab, d.cams[:, :2][cam_ok])
        groups, sflags = C.select_rooms(regs, t_thin, single_room)
        flags += sflags
        meta["n_cells"] = int(lab.size)
    if not groups:
        # no enclosed free space: one room over the robust extent of what was seen, so later stages still run
        rect = _fallback_rect(d, fc, sigma)
        if rect is None:
            return _empty_plan(flags + ["walls_not_found"], meta)
        flags.append("room_fallback_extent")
        lines = [q for q in lines if not q.synthetic] + [
            W.synthetic_line(0, rect[0], grids[0]), W.synthetic_line(0, rect[2], grids[0]),
            W.synthetic_line(1, rect[1], grids[1]), W.synthetic_line(1, rect[3], grids[1])]
        gx = C.group_lines([q for q in lines if q.synthetic], 0)
        gy = C.group_lines([q for q in lines if q.synthetic], 1)
        C.close_groups(gx, gy, grids)
        cx = C.build_complex(gx, gy, grids, free, floor, g2, t_thin)
        cx.inside[:] = True
        lab = C.label_regions(cx)
        regs = C.region_stats(cx, lab, d.cams[:, :2][cam_ok])
        groups = [[0]]

    room_of_cell = np.full(cx.shape, -1)
    for k, g in enumerate(groups):
        for r in g:
            for i, j in regs[r].cells:
                room_of_cell[i, j] = k
    plan = _assemble(scene, d, lines, cx, room_of_cell, groups, regs, fc, sigma, theta, flags, meta)
    meta["time_s"] = round(time.perf_counter() - t_start, 3)
    log.info("layout: %d rooms, %d walls, %d openings in %.1fs", len(plan.rooms),
             sum(len(r.walls) for r in plan.rooms), sum(len(r.openings) for r in plan.rooms), meta["time_s"])
    return plan


def _fallback_rect(d: _Data, fc: FC.FloorCeiling, sigma: float) -> tuple[float, float, float, float] | None:
    """Robust plan extent (2nd to 98th percentile) of floor points, else of all points; None if degenerate."""
    nz = np.abs(d.N[:, 2]) if "normals_unoriented" in fc.flags else d.N[:, 2]
    fl = (nz > FC.HORIZONTAL_NZ) & (np.abs(d.P[:, 2] - fc.floor.z) < max(0.05, 3 * sigma))
    xy = d.P[fl, :2] if fl.sum() >= 100 else d.P[:, :2]
    if len(xy) < 10:
        return None
    lo, hi = np.percentile(xy, 2, axis=0), np.percentile(xy, 98, axis=0)
    if (hi - lo).min() < 0.5:
        return None
    return float(lo[0]), float(lo[1]), float(hi[0]), float(hi[1])


def _floor_mask(d: _Data, fc: FC.FloorCeiling, sigma: float, g2: C.Grid2D) -> np.ndarray:
    """Plan cells with floor points near the global floor height, dilated by 5 cm."""
    from scipy.ndimage import maximum_filter

    nz = np.abs(d.N[:, 2]) if "normals_unoriented" in fc.flags else d.N[:, 2]
    m = (nz > FC.HORIZONTAL_NZ) & (np.abs(d.P[:, 2] - fc.floor.z) < max(0.05, 3 * sigma))
    ix = np.floor((d.P[m, 0] - g2.x0) / g2.res).astype(np.int64)
    iy = np.floor((d.P[m, 1] - g2.y0) / g2.res).astype(np.int64)
    ok = (ix >= 0) & (ix < g2.nx) & (iy >= 0) & (iy < g2.ny)
    grid = np.zeros((g2.nx, g2.ny), bool)
    grid[ix[ok], iy[ok]] = True
    return maximum_filter(grid, size=5)


def _closure_lines(lines: list[W.WallLine], free: np.ndarray, g2: C.Grid2D, grids: tuple[W.TGrid, W.TGrid],
                   flags: list[str]) -> list[W.WallLine]:
    """Synthetic boundary lines where free space runs more than 25 cm past the outermost observed wall."""
    if not free.any():
        return []
    fi, fj = np.nonzero(free)
    fx = g2.x0 + (np.percentile(fi, [0.5, 99.5]) + 0.5) * g2.res
    fy = g2.y0 + (np.percentile(fj, [0.5, 99.5]) + 0.5) * g2.res
    out = []
    for axis, (f_lo, f_hi) in ((0, fx), (1, fy)):
        cs = [q.coord for q in lines if q.axis == axis]
        if not cs or f_lo < min(cs) - 0.25:
            out.append(W.synthetic_line(axis, f_lo, grids[axis]))
        if not cs or f_hi > max(cs) + 0.25:
            out.append(W.synthetic_line(axis, f_hi, grids[axis]))
    if out:
        flags.append(f"closure_lines:{len(out)}")
    return out


@dataclass
class _Face:
    n: int = 0
    rms: float = 0.0
    sigma: float = 0.0
    flags: tuple[str, ...] = ()
    slope: float = 0.0
    t_mid: float = 0.0


def _face_evidence(lines: list[W.WallLine], axis: int, coord: float, n_sign: int, ta: float,
                   tb: float) -> _Face:
    """Inlier statistics of the observed wall face that a polygon edge lies on, restricted to the edge."""
    cand = [q for q in lines if q.axis == axis and not q.synthetic and abs(q.coord - coord) <= 0.015]
    match = [q for q in cand if q.sign == n_sign]
    if not match:
        if any(q.axis == axis and q.synthetic and abs(q.coord - coord) <= 0.015 for q in lines):
            return _Face(flags=("wall_unobserved",))
        return _Face(flags=("wall_face_mismatch",) if cand else ("wall_face_missing",))
    q = min(match, key=lambda q: abs(q.coord - coord))
    m = (q.t_pts >= ta) & (q.t_pts <= tb)
    if m.sum() < 5:
        return _Face(int(m.sum()), q.rms, q.sigma, ("wall_face_sparse",), q.slope, q.t_mid)
    r = q.r_pts[m] + (q.coord - coord)
    w = q.w_pts[m]
    rms = float(np.sqrt(np.average(r ** 2, weights=w)))
    slanted = abs(q.slope) > np.tan(np.radians(1.0))
    flags = (f"wall_slanted:{np.degrees(np.arctan(q.slope)):.1f}deg",) if slanted else ()
    return _Face(int(m.sum()), rms, q.sigma, flags, q.slope, q.t_mid)


def _claimable(cx: C.Complex, room_of_cell: np.ndarray, k: int) -> Callable[[Polygon], bool]:
    """Test for a region of the plan: True when no room other than room k holds any of its cells."""

    def ok(region: Polygon) -> bool:
        return bool(np.isin(room_of_cell[_cells_in(cx, region)], (-1, k)).all())

    return ok


def _cells_in(cx: C.Complex, region: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Indices of the cells whose centres lie inside the region."""
    x0, y0, x1, y1 = region.bounds
    i0, i1 = max(int(np.searchsorted(cx.xs, x0, "right")) - 1, 0), int(np.searchsorted(cx.xs, x1))
    j0, j1 = max(int(np.searchsorted(cx.ys, y0, "right")) - 1, 0), int(np.searchsorted(cx.ys, y1))
    ii, jj = np.meshgrid(np.arange(i0, min(i1, cx.shape[0])), np.arange(j0, min(j1, cx.shape[1])),
                         indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    inside = shapely.contains_xy(region, 0.5 * (cx.xs[ii] + cx.xs[ii + 1]), 0.5 * (cx.ys[jj] + cx.ys[jj + 1]))
    return ii[inside], jj[inside]


def _cell_of(cx: C.Complex, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny = cx.shape
    i = np.searchsorted(cx.xs, xy[:, 0]) - 1
    j = np.searchsorted(cx.ys, xy[:, 1]) - 1
    ok = (i >= 0) & (i < nx) & (j >= 0) & (j < ny)
    return np.where(ok, i, 0), np.where(ok, j, 0), ok


def _room_at(cx: C.Complex, room_of_cell: np.ndarray, xy: np.ndarray) -> np.ndarray:
    i, j, ok = _cell_of(cx, np.atleast_2d(xy))
    return np.where(ok, room_of_cell[i, j], -1)


def _assemble(scene: Scene, d: _Data, lines: list[W.WallLine], cx: C.Complex, room_of_cell: np.ndarray,
              groups: list[list[int]], regs: list[C.Region], fc: FC.FloorCeiling, sigma: float, theta: float,
              flags: list[str], meta: dict) -> Plan:
    Rw = _rot(theta)
    vroom = _room_at(cx, room_of_cell, d.P[:, :2])
    if "normals_unoriented" in fc.flags:
        up = down = np.abs(d.N[:, 2]) > FC.HORIZONTAL_NZ
    else:
        up, down = d.N[:, 2] > FC.HORIZONTAL_NZ, d.N[:, 2] < -FC.HORIZONTAL_NZ
    cam_room = np.where(d.cam_ok, _room_at(cx, room_of_cell, d.cams[:, :2]), -1) if len(d.cams) else []
    pts = R.WallPoints.build(d.P, d.N, d.wp, d.wf, fc.floor.z - 0.3, fc.ceiling.z + 0.3)
    rooms: list[Room] = []
    polys_m: list[np.ndarray] = []
    n_filled = n_steps = 0
    for k, g in enumerate(groups):
        rid = f"R{k + 1}"
        rflags: list[str] = []
        poly_c, hole = C.cells_polygon(cx, [c for r in g for c in regs[r].cells])
        if hole > 1e-6:
            rflags.append(f"holes_filled:{hole:.2f}")
        fm, cm = (vroom == k) & up, (vroom == k) & down
        fl = FC.room_level(d.P[fm, 2], d.wp[fm], d.wf[fm], d.P[fm, :2], fc.floor.z, "low", sigma)
        ce = FC.room_level(d.P[cm, 2], d.wp[cm], d.wf[cm], d.P[cm, :2], fc.ceiling.z, "high", sigma)
        if fl is None:
            fl = fc.floor
            rflags.append("floor_from_global")
        if ce is None:
            ce = fc.ceiling
            rflags.append("ceiling_from_global" if fc.ceiling.observed else "ceiling_assumed")
        if ce.z - fl.z < FC.MIN_CEILING_HEIGHT:
            fl, ce = fc.floor, fc.ceiling
            rflags.append("levels_from_global")
        hgt = ce.z - fl.z
        ceil_obs = _ceiling_coverage(cx, room_of_cell, k, d.P[cm & (np.abs(d.P[:, 2] - ce.z) < 0.1), :2])
        lev_ev = {"n_points": int(fl.n + ce.n), "fit_rms": float(np.hypot(fl.rms, ce.rms)),
                  "floor_rms": fl.rms, "ceiling_rms": ce.rms, "floor_n": fl.n, "ceiling_n": ce.n,
                  "floor_tilt": fl.tilt, "ceiling_tilt": ce.tilt, "observed_fraction": ceil_obs,
                  "ceiling_observed": bool(ce.observed), "noise_sigma": sigma}

        # the room's own faces: furniture notches filled, edges refitted, short steps removed
        out = R.room_outline(poly_c, pts, fl.z, ce.z, sigma, _claimable(cx, room_of_cell, k))
        if out is None:
            poly_m, fits = poly_c, None
            rflags.append("outline_not_refined")
        else:
            poly_m, fits = out.polygon, [e.fit for e in out.edges]
            for region in out.filled:
                room_of_cell[_cells_in(cx, region)] = k
            if out.filled:
                rflags.append(f"furniture_filled:{len(out.filled)}")
            n_filled += len(out.filled)
            n_steps += out.steps

        K = len(poly_m)
        edges = []
        for e in range(K):
            p, q = poly_m[e], poly_m[(e + 1) % K]
            dx, dy = q - p
            axis, coord, t0, t1 = (1, p[1], p[0], q[0]) if abs(dx) >= abs(dy) else (0, p[0], p[1], q[1])
            nin = np.array([-dy, dx]) / max(float(np.hypot(dx, dy)), 1e-12)
            n_sign = int(np.sign(nin[axis])) or 1
            if fits is None:
                face = _face_evidence(lines, axis, coord, n_sign, min(t0, t1), max(t0, t1))
            else:
                fit = fits[e]
                face = _Face(fit.n, fit.rms, fit.sigma, fit.flags, fit.slope, fit.t_mid)
            frame = OP.WallFrame(axis, float(coord), n_sign, float(t0), 1 if t1 > t0 else -1,
                                 float(abs(t1 - t0)), fl.z, ce.z, face.slope, face.t_mid)
            wa = OP.analyze_wall(frame, d.P, d.N, d.wp, d.wn, d.O, d.E, face.sigma if face.n >= 30 else sigma)
            if face.n < 30 and wa.openings:
                # without the room's own wall face there is no gap to measure, only rays into the unknown
                wa.openings = []
                wa.flags.append("openings_not_checked")
            edges.append((p, q, nin, face, frame, wa))

        walls: list[Wall] = []
        openings: list[Opening] = []
        for e, (p, q, nin, face, frame, wa) in enumerate(edges):
            wid = f"{rid}-W{e + 1}"
            prev_f, next_f = edges[e - 1][3], edges[(e + 1) % K][3]
            length = _m(frame.length, "length", n_points=face.n, fit_rms=face.rms,
                        observed_fraction=wa.observed_fraction, face_sigma=face.sigma,
                        end_fit_rms=[prev_f.rms, next_f.rms], end_n_points=[prev_f.n, next_f.n],
                        noise_sigma=sigma)
            walls.append(Wall(wid, rid, Rw @ p, Rw @ q, length, _m(hgt, "height", **lev_ev), Rw @ nin,
                              float(np.clip(wa.observed_fraction, 0, 1)),
                              {"n_points": face.n, "fit_rms": face.rms, "face_sigma": face.sigma,
                               "n_plane_points": wa.n_points, "refined": bool(fits and fits[e].refined)},
                              list(face.flags) + wa.flags))
            for f in wa.openings:
                t = frame.t_start + frame.t_dir * 0.5 * (f.u0 + f.u1)
                c_m = np.array([frame.coord, t]) if frame.axis == 0 else np.array([t, frame.coord])
                ev = {"n_points": f.n_points, "edge_rms": f.edge_rms, "edge_rms_lr": list(f.edge_rms_lr),
                      "observed_fraction": f.see_fraction, "see_through_ratio": f.see_ratio,
                      "fit_rms": f.edge_rms, "edges_observed": list(f.edges_observed), "noise_sigma": sigma}
                is_win = f.type == "window"
                op = Opening(f"{rid}-O{len(openings) + 1}", rid, wid, f.type, _m(f.u0, "offset", **ev),
                             _m(f.u1 - f.u0, "width", **ev),
                             _m((f.z1 - f.z0) if is_win else (f.z1 - fl.z), "height", **ev),
                             _m(f.z0 - fl.z, "height", **ev) if is_win else None, Rw @ c_m, None,
                             f.confidence,
                             {"header": f.header, "u0": f.u0, "u1": f.u1, "z0": f.z0, "z1": f.z1,
                              "_probe": (c_m, nin)}, list(f.flags))
                openings.append(op)

        area = _shoelace(poly_m)
        wl = np.array([w.length.value for w in walls])
        seen = np.array([w.length.evidence["n_points"] >= 30 for w in walls])
        if wl[seen].sum() < 0.5 * wl.sum():
            rflags.append("walls_mostly_unobserved")
        rms = float(np.average([w.length.evidence["fit_rms"] for w in walls], weights=wl))
        obs = float(np.average([w.observed_fraction for w in walls], weights=wl))
        area_ev = {"n_walls": K, "fit_rms": rms, "observed_fraction": obs,
                   "n_points": int(sum(w.length.evidence["n_points"] for w in walls)), "noise_sigma": sigma}
        view_ids = [v.id for v, r in zip(scene.views, cam_room) if r == k]
        label, source = _room_label(scene, view_ids, len(groups), k)
        rooms.append(Room(rid, label, poly_m @ Rw.T, walls, openings, float(fl.z), float(ce.z),
                          _m(hgt, "height", **lev_ev), _m(area, "area", "m2", **area_ev),
                          _m(float(wl.sum()), "length", **area_ev), view_ids, source, rflags,
                          {"n_cells": sum(len(regs[r].cells) for r in g), "coverage": regs[g[0]].coverage,
                           "boundary_support": regs[g[0]].support,
                           "n_cameras": int(sum(regs[r].n_cams for r in g)),
                           "polygon_manhattan": poly_m.tolist(), "polygon_cells": poly_c.tolist()}))
        polys_m.append(poly_m)
    meta["outline"] = {"furniture_filled": n_filled, "steps_removed": n_steps}

    adjacency = _connect(rooms, cx, room_of_cell)
    allp = np.concatenate(polys_m)
    ext = allp.max(0) - allp.min(0)
    tot_ev = {"n_rooms": len(rooms), "frame": "manhattan", "manhattan_angle_deg": float(np.degrees(theta)),
              "n_points": int(sum(r.floor_area.evidence["n_points"] for r in rooms)),
              "fit_rms": float(np.mean([r.floor_area.evidence["fit_rms"] for r in rooms])),
              "observed_fraction": float(np.mean([r.floor_area.evidence["observed_fraction"]
                                                  for r in rooms])),
              "noise_sigma": sigma}
    footprint = _m(sum(r.floor_area.value for r in rooms), "area", "m2", **tot_ev)
    return Plan(rooms, adjacency, footprint, _m(ext[0], "length", **tot_ev), _m(ext[1], "length", **tot_ev),
                flags, {"layout": meta})


def _shoelace(p: np.ndarray) -> float:
    x, y = p[:, 0], p[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ceiling_coverage(cx: C.Complex, room_of_cell: np.ndarray, k: int, xy: np.ndarray,
                      res: float = 0.25) -> float:
    """Share of the room's area (on a 25 cm grid) with ceiling inliers."""
    ii, jj = np.nonzero(room_of_cell == k)
    if len(ii) == 0:
        return 0.0
    x0, x1, y0, y1 = cx.xs[ii].min(), cx.xs[ii + 1].max(), cx.ys[jj].min(), cx.ys[jj + 1].max()
    gx, gy = np.arange(x0 + res / 2, x1, res), np.arange(y0 + res / 2, y1, res)
    if len(gx) == 0 or len(gy) == 0:
        return 0.0
    G = np.stack(np.meshgrid(gx, gy, indexing="ij"), -1).reshape(-1, 2)
    inside = _room_at(cx, room_of_cell, G) == k
    if not inside.any():
        return 0.0
    hit = np.zeros(len(G), bool)
    if len(xy):
        ix = np.clip(((xy[:, 0] - x0) / res).astype(int), 0, len(gx) - 1)
        iy = np.clip(((xy[:, 1] - y0) / res).astype(int), 0, len(gy) - 1)
        hit[ix * len(gy) + iy] = True
    return float((hit & inside).sum() / inside.sum())


def _room_label(scene: Scene, view_ids: list[str], n_rooms: int, k: int) -> tuple[str, str | None]:
    if scene.room_hint and n_rooms == 1:
        return _label(scene.room_hint) or f"Room {k + 1}", scene.room_hint
    ids = set(view_ids)
    hints = [v.room_hint for v in scene.views if v.id in ids and v.room_hint]
    if hints:
        best = max(sorted(set(hints)), key=hints.count)
        return _label(best) or f"Room {k + 1}", best
    return f"Room {k + 1}", None


def _connect(rooms: list[Room], cx: C.Complex, room_of_cell: np.ndarray) -> list[Adjacency]:
    """connects_to for openings with another room across the gap; adjacency from matched opening pairs."""
    index = {r.id: k for k, r in enumerate(rooms)}
    found: list[tuple[int, Opening, np.ndarray]] = []
    for k, r in enumerate(rooms):
        for op in r.openings:
            c_m, nin = op.evidence.pop("_probe")
            for dist in (0.2, 0.35, 0.5, 0.7):
                other = int(_room_at(cx, room_of_cell, c_m - dist * nin)[0])
                if other >= 0 and other != k:
                    op.connects_to = rooms[other].id
                    break
            found.append((k, op, c_m))
    adj: list[Adjacency] = []
    used: set[str] = set()
    for k, op, c in found:
        if op.connects_to is None or op.id in used:
            continue
        k2 = index[op.connects_to]
        best, bd = None, 0.8
        for k3, op2, c2 in found:
            if k3 == k2 and op2.connects_to == rooms[k].id and op2.id not in used:
                dist = float(np.linalg.norm(c2 - c))
                if dist < bd:
                    best, bd = op2, dist
        used.add(op.id)
        if best is not None:
            used.add(best.id)
        first, second = (op, best) if k < k2 else (best, op)
        conf = op.confidence if best is None else 0.5 * (op.confidence + best.confidence)
        adj.append(Adjacency(rooms[min(k, k2)].id, rooms[max(k, k2)].id, first.id if first else None,
                             second.id if second else None, float(np.clip(conf, 0, 1)), "shared_frame"))
    return adj
