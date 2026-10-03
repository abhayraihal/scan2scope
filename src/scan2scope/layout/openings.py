"""Openings per wall from grids in (along-wall u, height z) at 2 cm.

An opening is an empty region of the wall face that rays pass through (see-through evidence). Empty
regions without rays through them are unobserved wall, not openings; they only lower observed_fraction.
Each edge is found by walking out of the gap to half the wall level of the face-mass profile and is then
moved to that profile's steepest rise.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d, label, uniform_filter, uniform_filter1d

log = logging.getLogger("scan2scope.layout")

RES = 0.02
DOOR_MIN_HEIGHT = 1.8
DOOR_WIDTH = (0.5, 1.6)
NEAR_FLOOR = 0.15
WINDOW_MIN_SILL = 0.3
MIN_SIZE = 0.25  # smallest window width or height
SEE_MASK = 0.15  # see-through density relative to the wall's hit density, per cell
SEE_ACCEPT = 0.25  # same, averaged over a candidate
EMPTY = 0.15  # occupancy below this share of the wall level counts as empty
SEE_DEPTH = 0.25  # a ray end must lie this far past the wall face to count as seen through it


@dataclass
class WallFrame:
    axis: int  # 0: wall on x = coord, u runs along y; 1: wall on y = coord, u runs along x
    coord: float
    n_sign: int  # +1 when the room interior is on the positive side of the axis
    t_start: float
    t_dir: int  # +1 when u grows with the along-line coordinate
    length: float
    floor_z: float
    ceil_z: float
    slope: float = 0.0  # the face sits at coord + slope * (t - t_mid) for walls slightly off axis
    t_mid: float = 0.0

    def offset(self, X: np.ndarray) -> np.ndarray:
        """Signed distance from the wall face, positive into the room."""
        a, b = self.axis, 1 - self.axis
        return (X[:, a] - self.coord - self.slope * (X[:, b] - self.t_mid)) * self.n_sign


@dataclass
class OpeningFit:
    type: str
    u0: float
    u1: float
    z0: float
    z1: float
    header: bool
    see_ratio: float
    edge_rms: float
    edge_rms_lr: tuple[float, float]
    edges_observed: tuple[bool, bool, bool]  # left, right, top
    n_points: int
    see_fraction: float
    confidence: float
    flags: list[str] = field(default_factory=list)


@dataclass
class WallAnalysis:
    observed_fraction: float
    n_points: int
    openings: list[OpeningFit]
    flags: list[str] = field(default_factory=list)


def _cross(prof: np.ndarray, start: int, step: int, half: float, max_bins: int) -> float | None:
    """Walk from bin `start` in direction `step` to the first bin >= half; interpolated position in bin units."""
    n = len(prof)
    j = start
    for _ in range(8):  # step back into the gap if the seed overshot onto the wall
        if 0 <= j < n and prof[j] >= half:
            j -= step
    if not (0 <= j < n) or prof[j] >= half:
        return None
    for _ in range(max_bins):
        k = j + step
        if not (0 <= k < n):
            return None
        if prof[k] >= half:
            f = (half - prof[j]) / max(prof[k] - prof[j], 1e-12)
            return (j + 0.5) + step * float(np.clip(f, 0.0, 1.0))
        j = k
    return None


def _steepest(prof: np.ndarray, e: float, step: int, sigma_bins: float) -> float:
    """Move an edge estimate to the steepest rise of the smoothed profile towards the wall, parabola-refined.

    The half-level crossing is biased when wall density changes near the edge (a camera's frame edge, for
    one); the gradient peak of a blurred step is not.
    """
    sm = gaussian_filter1d(prof.astype(float), max(sigma_bins, 0.7), mode="nearest")
    g = np.diff(sm) * step  # g[k] sits on the boundary between bins k and k + 1
    if len(g) == 0:
        return e
    search = max(2, round(3 * max(sigma_bins, 0.7)))
    c = round(e) - 1
    lo, hi = max(c - search, 0), min(c + search + 1, len(g))
    if hi <= lo:
        return e
    k = lo + int(np.argmax(g[lo:hi]))
    if g[k] <= 0:
        return e
    off = 0.0
    if 0 < k < len(g) - 1:
        den = g[k - 1] - 2 * g[k] + g[k + 1]
        if den < 0:
            off = float(np.clip(0.5 * (g[k - 1] - g[k + 1]) / den, -0.5, 0.5))
    return k + 1 + off


def _edge(prof: np.ndarray, start: int, step: int, level_hint: float, max_bins: int,
          sigma_bins: float) -> float | None:
    """Edge position in bin units: walk to half the wall level, then snap to the steepest rise."""
    if level_hint <= 0:
        return None
    e = _cross(uniform_filter1d(prof, 3, mode="nearest"), start, step, 0.5 * level_hint, max_bins)
    return None if e is None else _steepest(prof, e, step, sigma_bins)


def analyze_wall(wf: WallFrame, P: np.ndarray, N: np.ndarray, w_occ: np.ndarray, w_nsum: np.ndarray,
                 O: np.ndarray, E: np.ndarray, sigma: float) -> WallAnalysis:
    a, b = wf.axis, 1 - wf.axis
    L, fz, cz = wf.length, wf.floor_z, wf.ceil_z
    win = float(np.clip(2.5 * sigma, 0.03, 0.10))
    nu, nz = max(int(np.ceil(L / RES)), 1), max(int(np.ceil((cz - fz) / RES)), 1)

    d = wf.offset(P)
    u = (P[:, b] - wf.t_start) * wf.t_dir
    sel = ((np.abs(d) < win) & (u >= 0) & (u < L) & (P[:, 2] > fz + 0.03) & (P[:, 2] < cz - 0.03)
           & (np.abs(N[:, a]) >= 0.3) & (np.abs(N[:, 2]) < 0.9))
    iu = np.minimum((u[sel] / RES).astype(np.int64), nu - 1)
    iz = np.clip(((P[sel, 2] - fz) / RES).astype(np.int64), 0, nz - 1)
    occ = np.bincount(iu * nz + iz, weights=w_occ[sel], minlength=nu * nz).reshape(nu, nz)
    # face mass per voxel: a corner voxel that is mostly jamb or sill contributes only its face share
    face_mass = np.abs(N[sel, a]) * w_nsum[sel]
    cnt = np.bincount(iu * nz + iz, weights=face_mass, minlength=nu * nz).reshape(nu, nz)
    n_pts = int(sel.sum())

    dO, dE = wf.offset(O), wf.offset(E)
    uE = (E[:, b] - wf.t_start) * wf.t_dir
    hit = (np.abs(dE) < win) & (uE >= 0) & (uE < L) & (E[:, 2] > fz) & (E[:, 2] < cz)
    hits = np.bincount(np.minimum((uE[hit] / RES).astype(np.int64), nu - 1) * nz
                       + np.clip(((E[hit, 2] - fz) / RES).astype(np.int64), 0, nz - 1), minlength=nu * nz)
    # see-through needs the point well past the face: wall points pushed behind it by depth noise or a
    # misregistered view would otherwise look like a hole
    deep = max(SEE_DEPTH, 3 * sigma)
    cross = ((dO > 0.05) & (dE < -deep)) | ((dO < -0.05) & (dE > deep))
    lam = dO[cross] / (dO[cross] - dE[cross])
    Q = O[cross] + lam[:, None] * (E[cross] - O[cross])
    uQ = (Q[:, b] - wf.t_start) * wf.t_dir
    ok = (uQ >= 0) & (uQ < L) & (Q[:, 2] > fz) & (Q[:, 2] < cz)
    see = np.bincount(np.minimum((uQ[ok] / RES).astype(np.int64), nu - 1) * nz
                      + np.clip(((Q[ok, 2] - fz) / RES).astype(np.int64), 0, nz - 1), minlength=nu * nz)
    hits, see = hits.reshape(nu, nz).astype(float), see.reshape(nu, nz).astype(float)

    occ_s = uniform_filter(occ, 3, mode="constant")
    hits_s = uniform_filter(hits, 5, mode="constant")
    see_s = uniform_filter(see, 5, mode="constant")
    pos = occ_s[occ_s > 0]
    level_occ = float(np.median(pos)) if len(pos) else 0.0
    posh = hits_s[hits_s > 0]
    level_hit = float(np.median(posh)) if len(posh) else 0.0
    flags: list[str] = []
    observed = occ_s >= EMPTY * level_occ if level_occ > 0 else np.zeros_like(occ, bool)
    if level_hit <= 0 or n_pts < 30:
        if n_pts < 30:
            flags.append("wall_unobserved")
        return WallAnalysis(float(observed.mean()), n_pts, [], flags)

    ratio = see_s / level_hit
    empty = ~observed
    cand, ncand = label(empty & (ratio >= SEE_MASK), structure=np.ones((3, 3)))
    fits: list[OpeningFit] = []
    taken = np.zeros_like(empty)
    for k in range(1, ncand + 1):
        m = cand == k
        if m.sum() * RES * RES < 0.04 or (taken & m).any():
            continue
        fit = _fit_candidate(m, cnt, hits_s, see_s, empty, level_hit, wf, sigma)
        if fit is None:
            continue
        i0, i1 = int(fit.u0 / RES), int(np.ceil(fit.u1 / RES))
        j0, j1 = int((fit.z0 - fz) / RES), int(np.ceil((fit.z1 - fz) / RES))
        taken[max(i0, 0):i1, max(j0, 0):j1] = True
        fits.append(fit)
    fits = _dedupe(fits)
    obs = observed | taken
    return WallAnalysis(float(obs.mean()), n_pts, sorted(fits, key=lambda f: f.u0), flags)


def _fit_candidate(m: np.ndarray, cnt: np.ndarray, hits_s: np.ndarray, see_s: np.ndarray,
                   empty: np.ndarray, level_hit: float, wf: WallFrame, sigma: float) -> OpeningFit | None:
    nu, nz = m.shape
    fz, cz = wf.floor_z, wf.ceil_z
    ii, jj = np.nonzero(m)
    i0, i1, j0, j1 = int(ii.min()), int(ii.max()) + 1, int(jj.min()), int(jj.max()) + 1
    flags: list[str] = []
    # jambs from the column profile over the seed's central rows, then head and sill from the row profile
    trim = max(1, (j1 - j0) // 6)
    rows = np.arange(j0 + trim, max(j1 - trim, j0 + trim + 1))
    rows = rows[rows < nz]
    colprof = cnt[:, rows].sum(1)
    wall_cols = (~empty[:, rows]).mean(1) >= 0.5
    level = float(np.median(colprof[wall_cols & (colprof > 0)])) if (wall_cols & (colprof > 0)).any() else 0.0
    # walks start at the emptiest column/row of the seed, so a seed that spills onto sparse wall still works
    max_walk = int(1.8 / RES)
    sb = sigma / RES
    start = i0 + int(np.argmin(uniform_filter1d(colprof, 5, mode="nearest")[i0:i1]))
    el = _edge(colprof, start, -1, level, max_walk, sb)
    er = _edge(colprof, start, 1, level, max_walk, sb)
    left_obs, right_obs = el is not None, er is not None
    u0 = el * RES if el is not None else (0.0 if i0 <= 2 else i0 * RES)
    u1 = er * RES if er is not None else (wf.length if i1 >= nu - 2 else i1 * RES)
    if not left_obs and i0 > 2:
        flags.append("edge_unobserved:left")
    if not right_obs and i1 < nu - 2:
        flags.append("edge_unobserved:right")
    log.debug("gap seed u=[%.2f, %.2f] z=[%.2f, %.2f] level %.1f -> jambs %s %s", i0 * RES, i1 * RES,
              j0 * RES, j1 * RES, level, None if el is None else round(el * RES, 3),
              None if er is None else round(er * RES, 3))
    if u1 - u0 < MIN_SIZE:
        return None

    c0 = int(np.clip(np.floor(u0 / RES + 0.1 * (u1 - u0) / RES), 0, nu - 1))
    c1 = int(np.clip(np.ceil(u1 / RES - 0.1 * (u1 - u0) / RES), c0 + 1, nu))
    rowprof = cnt[c0:c1].sum(0)
    # wall level for the head/sill walk from the jamb-side wall density; inside a floor-to-ceiling gap the
    # row profile holds only noise
    rlevel = level * (c1 - c0) / max(len(rows), 1)
    if rlevel <= 0 and (rowprof > 0).any():
        rlevel = float(np.median(rowprof[rowprof > 0]))
    zstart = j0 + int(np.argmin(uniform_filter1d(rowprof, 5, mode="nearest")[j0:j1]))
    et = _edge(rowprof, zstart, 1, rlevel, nz, sb)
    header = et is not None
    z1 = fz + et * RES if et is not None else cz
    above = empty[c0:c1, j1:]
    if et is None and above.size and above.mean() <= 0.7:
        flags.append("edge_unobserved:top")
    eb = _edge(rowprof, zstart, -1, rlevel, nz, sb)
    z0 = fz + eb * RES if eb is not None else fz
    near_floor = eb is None or z0 - fz <= NEAR_FLOOR
    if near_floor:
        z0 = fz

    # the seed must sit inside a real gap: mostly empty over the refined box, with rays through it
    bi0, bi1 = int(np.clip(u0 / RES, 0, nu - 1)), int(np.clip(np.ceil(u1 / RES), 1, nu))
    bj0, bj1 = int(np.clip((z0 - fz) / RES, 0, nz - 1)), int(np.clip(np.ceil((z1 - fz) / RES), 1, nz))
    box_empty = float(empty[bi0:bi1, bj0:bj1].mean())
    pad = int(0.5 / RES)
    ring = hits_s[max(bi0 - pad, 0):min(bi1 + pad, nu), bj0:bj1]
    ring = ring[ring > 0]
    local_hit = float(np.median(ring)) if len(ring) else level_hit
    see_ratio = float(see_s[bi0:bi1, bj0:bj1].mean() / max(local_hit, 1e-9))
    see_frac = float((see_s[bi0:bi1, bj0:bj1] > SEE_MASK * local_hit).mean())
    if box_empty < 0.6 or see_ratio < SEE_ACCEPT:
        log.debug("rejected gap u=[%.2f, %.2f] z=[%.2f, %.2f]: empty %.2f see %.2f", u0, u1, z0, z1,
                  box_empty, see_ratio)
        return None

    width, top = u1 - u0, z1 - fz
    if near_floor:
        if top < DOOR_MIN_HEIGHT:
            return None
        typ = "door" if (DOOR_WIDTH[0] <= width <= DOOR_WIDTH[1] and header) else "opening"
        if width < DOOR_WIDTH[0]:
            return None
    else:
        if z0 - fz < WINDOW_MIN_SILL and top >= DOOR_MIN_HEIGHT and DOOR_WIDTH[0] <= width <= DOOR_WIDTH[1]:
            typ = "door"
            flags.append("raised_threshold")
        elif z1 - z0 >= MIN_SIZE:
            typ = "window"
        else:
            return None

    er_rows = _central_rows(nz, z0, z1, fz)
    n_edge = int(sum((cnt[max(int(e / RES) - 5, 0):int(e / RES) + 5][:, er_rows] > 0).sum() for e in (u0, u1)))
    rms_l, rms_r = _edge_rms(cnt, er_rows, u0, u1, sigma)
    conf = float(np.clip(min(see_ratio, 1.0) * (0.6 + 0.4 * box_empty), 0, 1))
    if not (left_obs and right_obs):
        conf *= 0.7
    erms = float(np.sqrt(0.5 * (rms_l ** 2 + rms_r ** 2)))
    return OpeningFit(typ, float(u0), float(u1), float(z0), float(z1), header, see_ratio, erms, (rms_l, rms_r),
                      (left_obs, right_obs, header), n_edge, see_frac, conf, flags)


def _central_rows(nz: int, z0: float, z1: float, fz: float) -> np.ndarray:
    """Row indices of the central part of the opening, where both jambs are wall."""
    a = int(np.clip((z0 - fz) / RES, 0, nz - 1))
    b = int(np.clip((z1 - fz) / RES, a + 1, nz))
    m = max(1, (b - a) // 8)
    return np.arange(a + m, max(b - m, a + m + 1))


def _edge_rms(cnt: np.ndarray, rows: np.ndarray, u0: float, u1: float, sigma: float) -> tuple[float, float]:
    """Spread of the jamb positions measured in 10 cm row bands."""
    band = int(0.10 / RES)
    est_l, est_r = [], []
    for s in range(0, len(rows) - band + 1, band):
        r = rows[s:s + band]
        prof = cnt[:, r].sum(1)
        pos = prof[prof > 0]
        if len(pos) < 4:
            continue
        lvl = float(np.median(pos))
        i0 = int(np.clip(u0 / RES + 1, 0, len(prof) - 1))
        i1 = int(np.clip(u1 / RES - 1, 0, len(prof) - 1))
        el = _edge(prof, i0, -1, lvl, 15, sigma / RES)
        er = _edge(prof, i1, 1, lvl, 15, sigma / RES)
        if el is not None:
            est_l.append(el * RES)
        if er is not None:
            est_r.append(er * RES)
    fallback = max(sigma, RES / np.sqrt(12))
    def spread(v: list[float]) -> float:
        return float(max(1.4826 * np.median(np.abs(np.asarray(v) - np.median(v))), RES / np.sqrt(12)))

    rl = spread(est_l) if len(est_l) >= 3 else fallback
    rr = spread(est_r) if len(est_r) >= 3 else fallback
    return rl, rr


def _dedupe(fits: list[OpeningFit]) -> list[OpeningFit]:
    """Drop openings whose box mostly overlaps a more confident one (split seeds of the same gap)."""
    out: list[OpeningFit] = []
    for f in sorted(fits, key=lambda q: -q.confidence):
        dup = False
        for g in out:
            du = min(f.u1, g.u1) - max(f.u0, g.u0)
            dz = min(f.z1, g.z1) - max(f.z0, g.z0)
            if du > 0 and dz > 0 and du * dz > 0.5 * min((f.u1 - f.u0) * (f.z1 - f.z0), (g.u1 - g.u0) * (g.z1 - g.z0)):
                dup = True
                break
        if not dup:
            out.append(f)
    return out
