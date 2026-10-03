"""Manhattan frame and wall lines: per normal direction, 1-D density peaks of wall points.

All coordinates here are in the Manhattan frame (plan rotated so walls run along x and y).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from scan2scope.geometry.gravity import manhattan_angle
from scan2scope.layout.floor_ceiling import hist_peaks, robust_sigma, weighted_median

log = logging.getLogger("scan2scope.layout")

WALL_NZ = 0.3
DIRS = ((0, 1), (0, -1), (1, 1), (1, -1))  # (axis, sign): face normals +x, -x, +y, -y
DIR_TOL = float(np.cos(np.radians(25.0)))
T_RES = 0.05  # support profile bins along a line
Z_RES = 0.10
MIN_LINE_AREA = 0.2  # m2 of observed face
MIN_LINE_COLS = 5  # contiguous profile bins (25 cm) with at least 40 cm of observed height
MAX_PEAKS_PER_DIR = 80
SLANT_REACH = 0.25  # points within this of a peak are tested for one slightly rotated wall
SLANT_MIN, SLANT_MAX = float(np.tan(np.radians(0.3))), float(np.tan(np.radians(8.0)))


@dataclass
class TGrid:
    """1-D grid shared by all lines of one axis: x = c lines run along y, y = c lines along x."""

    t0: float
    n: int
    res: float = T_RES

    def centers(self) -> np.ndarray:
        return self.t0 + (np.arange(self.n) + 0.5) * self.res

    def index(self, t: np.ndarray | float) -> np.ndarray:
        return np.floor((np.asarray(t) - self.t0) / self.res).astype(np.int64)

    def span(self, a: float, b: float) -> tuple[int, int]:
        """Bins whose centres lie in [a, b], clipped to the grid."""
        i0 = int(np.ceil((a - self.t0) / self.res - 0.5))
        i1 = int(np.floor((b - self.t0) / self.res - 0.5)) + 1
        return max(i0, 0), min(i1, self.n)


@dataclass
class WallLine:
    axis: int  # 0: the line x = coord; 1: the line y = coord
    sign: int  # face normal direction along the axis (+1/-1), 0 for synthetic closure lines
    coord: float
    sigma: float = 0.0  # robust spread of inlier offsets
    rms: float = 0.0
    n: int = 0
    window: float = 0.0
    area: float = 0.0  # observed face area, m2
    solid: np.ndarray | None = None  # per T-grid bin: wall present at mid height (0.5 to 2.0 m)
    upper: np.ndarray | None = None  # per T-grid bin: wall present above door-head height
    t_pts: np.ndarray | None = None  # inlier along-line coordinates
    r_pts: np.ndarray | None = None  # inlier offsets from coord
    w_pts: np.ndarray | None = None
    synthetic: bool = False
    slope: float = 0.0  # d(offset)/d(t) of a wall a few degrees off the Manhattan axis
    t_mid: float = 0.0  # along-line position where the line sits exactly at coord
    half_span: float = 0.0


def manhattan_frame(nxy: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Dominant wall direction in [-pi/4, pi/4) and the concentration of normals around it (0 to 1)."""
    if len(nxy) < 10 or w.sum() <= 0:
        return 0.0, 0.0
    th = manhattan_angle(nxy, w)
    n = nxy / np.maximum(np.linalg.norm(nxy, axis=1, keepdims=True), 1e-12)
    ang = np.arctan2(n[:, 1], n[:, 0])
    # the circular mean is pulled by walls a few degrees off square; re-estimate from normals near the mode
    for tol in (np.radians(8.0), np.radians(4.0)):
        dev = np.angle(np.exp(4j * (ang - th))) / 4.0
        near = np.abs(dev) < tol
        if near.sum() < 10:
            break
        th = manhattan_angle(nxy[near], w[near])
    if th >= np.pi / 4:
        th -= np.pi / 2
    conc = float(np.abs((w * np.exp(4j * ang)).sum()) / w.sum())
    return float(th), conc


def direction_codes(nxy: np.ndarray) -> np.ndarray:
    """Index into DIRS of the axis direction each horizontal normal is within 25 degrees of, else -1."""
    n = nxy / np.maximum(np.linalg.norm(nxy, axis=1, keepdims=True), 1e-12)
    dots = np.stack([n[:, 0], -n[:, 0], n[:, 1], -n[:, 1]], 1)
    return np.where(dots.max(1) >= DIR_TOL, dots.argmax(1), -1)


def band_profiles(t: np.ndarray, z: np.ndarray, grid: TGrid, floor_z: float,
                  ceil_z: float) -> tuple[np.ndarray, np.ndarray, float, int]:
    """Occupancy of a wall face on (t, z) cells.

    Returns mid-height and upper-band support per bin, the observed area, and the longest run of tall columns
    (one-bin holes bridged), which keeps door jambs and other wall-thickness surfaces from becoming lines.
    """
    zb0, zb1 = floor_z + 0.05, ceil_z - 0.05
    nz = max(1, int(np.ceil((zb1 - zb0) / Z_RES)))
    it = grid.index(t)
    iz = np.floor((z - zb0) / Z_RES).astype(np.int64)
    ok = (it >= 0) & (it < grid.n) & (iz >= 0) & (iz < nz)
    cnt = np.bincount(it[ok] * nz + iz[ok], minlength=grid.n * nz).reshape(grid.n, nz)
    pos = cnt[cnt > 0]
    k = max(2.0, 0.2 * float(np.median(pos))) if len(pos) else 2.0
    occ = cnt >= k
    zc = zb0 + (np.arange(nz) + 0.5) * Z_RES
    mid = (zc >= floor_z + 0.5) & (zc <= min(floor_z + 2.0, ceil_z - 0.1))
    up = zc >= floor_z + 2.1
    solid = occ[:, mid].mean(1) >= 0.4 if mid.any() else occ.any(1)
    upper = occ[:, up].mean(1) >= 0.25 if (up.any() and ceil_z - floor_z >= 2.25) else np.zeros(grid.n, bool)
    tall = occ.sum(1) >= 4
    tall[1:-1] |= tall[:-2] & tall[2:]
    d = np.diff(np.concatenate([[0], tall.astype(np.int8), [0]]))
    longest = int((np.flatnonzero(d == -1) - np.flatnonzero(d == 1)).max()) if tall.any() else 0
    return solid, upper, float(occ.sum() * T_RES * Z_RES), longest


def _refine_sorted(v: np.ndarray, w: np.ndarray, c0: float, win0: float, win_min: float,
                   win_max: float, iters: int = 4) -> tuple[float, float, float, int, int, float]:
    """refine() on presorted values using index windows. Returns c, sigma, rms, i0, i1, window."""
    c, win, s = float(c0), float(win0), float(win0) / 2.5
    for _ in range(iters):
        i0, i1 = np.searchsorted(v, [c - win, c + win])
        if i1 - i0 < 3 or w[i0:i1].sum() <= 0:
            break
        vv, ww = v[i0:i1], w[i0:i1]
        c = float(np.average(vv, weights=ww))
        s = robust_sigma(vv - c, ww)
        win = float(np.clip(2.5 * s, win_min, win_max))
    i0, i1 = np.searchsorted(v, [c - win, c + win])
    if i1 - i0 == 0 or w[i0:i1].sum() <= 0:
        return c, s, 0.0, int(i0), int(i1), win
    rms = float(np.sqrt(np.average((v[i0:i1] - c) ** 2, weights=w[i0:i1])))
    return c, s, rms, int(i0), int(i1), win


def _slant_fit(t: np.ndarray, v: np.ndarray, w: np.ndarray, c0: float, scale: float,
               iters: int = 8) -> tuple[float, float, float]:
    """Tukey-weighted line v = a + b (t - tm) started from the flat line at c0. Returns a, b, tm.

    The scale starts wide so a wall several degrees off the flat start line is captured, then shrinks.
    """
    tm = float(np.median(t))
    a, b = float(c0), 0.0
    A = np.column_stack([np.ones_like(t), t - tm])
    for it in range(iters):
        sc = max(scale, 0.10 * 0.6 ** it)
        u = (v - a - b * (t - tm)) / (4.685 * sc)
        wt = w * np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        if (wt > 0).sum() < 20:
            break
        sw = np.sqrt(wt)
        (a, b), *_ = np.linalg.lstsq(A * sw[:, None], v * sw, rcond=None)
    return float(a), float(b), tm


def _near(x: float, kept: list[WallLine], tol: float) -> bool:
    """True when x falls on an accepted line of the same direction, along its whole slanted extent."""
    return _near_range(x, x, kept, tol)


def _near_range(lo: float, hi: float, kept: list[WallLine], tol: float) -> bool:
    """True when the offsets [lo, hi] spanned by a line overlap an accepted line's span (plus tol)."""
    return any(lo < q.coord + tol + abs(q.slope) * q.half_span
               and hi > q.coord - tol - abs(q.slope) * q.half_span for q in kept)


def detect_lines(P: np.ndarray, N: np.ndarray, w_peak: np.ndarray, w_fit: np.ndarray, floor_z: float,
                 ceil_z: float, sigma0: float, grids: tuple[TGrid, TGrid]) -> tuple[list[WallLine], float]:
    """Wall lines for the four face directions, and the area-weighted wall noise estimate."""
    codes = direction_codes(N[:, :2])
    merge_tol = max(0.04, 2.0 * sigma0)
    lines: list[WallLine] = []
    for k, (axis, sign) in enumerate(DIRS):
        sel = np.flatnonzero(codes == k)
        if len(sel) < 50:
            continue
        order = np.argsort(P[sel, axis], kind="stable")
        sel = sel[order]
        v, t, z = P[sel, axis], P[sel, 1 - axis], P[sel, 2]
        wp, wf = w_peak[sel], w_fit[sel]
        lo, hi = float(v[int(0.0005 * len(v))]) - 0.05, float(v[int(0.9995 * (len(v) - 1))]) + 0.05
        peaks = hist_peaks(v, wp, lo, hi, 0.01, max(0.01, 0.7 * sigma0), max(0.02, 1.5 * sigma0))
        if not peaks:
            continue
        best = max(p.support for p in peaks)
        peaks = sorted((p for p in peaks if p.support >= max(20.0, 0.005 * best)), key=lambda p: -p.support)
        kept: list[WallLine] = []
        used = np.zeros(len(v), bool)  # points already explained by an accepted line of this direction
        for p in peaks[:MAX_PEAKS_PER_DIR]:
            if _near(p.center, kept, merge_tol):
                continue
            c, s, rms, i0, i1, win = _refine_sorted(v, wf, p.center, max(0.03, 3 * sigma0), 0.015, 0.12)
            idx = np.arange(i0, i1)[~used[i0:i1]]
            if len(idx) < 30 or _near(c, kept, merge_tol):
                continue
            slope, tm = 0.0, float(np.median(t[idx]))
            j0, j1 = (int(x) for x in np.searchsorted(v, [c - SLANT_REACH, c + SLANT_REACH]))
            free = np.arange(j0, j1)[~used[j0:j1]]
            if len(free) > 1.15 * len(idx) or s > 1.5 * sigma0:
                # a wall a few degrees off axis spreads over many histogram bins: fit one rotated line
                a_, b_, tm_ = _slant_fit(t[free], v[free], wf[free], c, max(2.5 * sigma0, 0.02))
                r_all = v[free] - (a_ + b_ * (t[free] - tm_))
                inl = np.abs(r_all) < max(win, 2.5 * sigma0)
                hs = 0.5 * float(np.ptp(t[free][inl])) if inl.any() else 0.0
                log.debug("slant fit at %.3f: slope %.2f deg, %d inliers vs %d flat", c,
                          np.degrees(np.arctan(b_)), int(inl.sum()), len(idx))
                s_slant = robust_sigma(r_all[inl], wf[free][inl]) if inl.sum() > 10 else s
                better = inl.sum() >= 1.15 * len(idx) or s_slant < 0.7 * s
                if (SLANT_MIN < abs(b_) <= SLANT_MAX and better
                        and not _near_range(a_ - abs(b_) * hs, a_ + abs(b_) * hs, kept, merge_tol)):
                    idx = free[inl]
                    c, slope, tm, s = a_, b_, tm_, s_slant
                    rms = float(np.sqrt(np.average((v[idx] - c) ** 2, weights=wf[idx])))
            solid, upper, area, cols = band_profiles(t[idx], z[idx], grids[axis], floor_z, ceil_z)
            if area < MIN_LINE_AREA or cols < MIN_LINE_COLS:
                continue
            kept.append(WallLine(axis, sign, c, s, rms, len(idx), win, area, solid, upper, t[idx].copy(),
                                 v[idx] - c, wf[idx].copy(), slope=slope, t_mid=tm,
                                 half_span=0.5 * float(t[idx].max() - t[idx].min())))
            used[idx] = True
        lines += kept
    if not lines:
        return lines, sigma0
    sig = weighted_median(np.array([q.sigma for q in lines]), np.array([q.area for q in lines]))
    return lines, float(sig)


def synthetic_line(axis: int, coord: float, grid: TGrid) -> WallLine:
    """Closure line where free space runs past the outermost observed wall."""
    z = np.zeros(grid.n, bool)
    return WallLine(axis, 0, float(coord), solid=z, upper=z.copy(), t_pts=np.zeros(0), r_pts=np.zeros(0),
                    w_pts=np.zeros(0), synthetic=True)
