"""Floor and ceiling heights from weighted height histograms of near-horizontal surfaces.

Also holds the small 1-D histogram and robust-statistics helpers the wall stage reuses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

log = logging.getLogger("scan2scope.layout")

HORIZONTAL_NZ = 0.85
MIN_CEILING_HEIGHT = 1.8
MAX_CEILING_HEIGHT = 6.0
DEFAULT_CEILING_HEIGHT = 2.5
MIN_LEVEL_SUPPORT = 150.0  # voxel weight, about 0.1 m2 of 2 cm voxels


@dataclass
class Peak:
    center: float
    support: float


@dataclass
class Level:
    """A refined horizontal plane: z is the inlier mean (global) or median (per room)."""

    z: float
    rms: float = 0.0
    sigma: float = 0.0
    n: int = 0
    window: float = 0.0
    tilt: float = 0.0  # slope magnitude of a least-squares plane through the inliers
    observed: bool = True


@dataclass
class FloorCeiling:
    floor: Level
    ceiling: Level
    flags: list[str] = field(default_factory=list)


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    if len(x) == 0:
        return float("nan")
    o = np.argsort(x)
    cw = np.cumsum(w[o])
    if cw[-1] <= 0:
        return float(np.median(x))
    return float(x[o][min(np.searchsorted(cw, 0.5 * cw[-1]), len(x) - 1)])


def robust_sigma(r: np.ndarray, w: np.ndarray) -> float:
    return 1.4826 * weighted_median(np.abs(r), w) if len(r) else 0.0


def hist_peaks(v: np.ndarray, w: np.ndarray, lo: float, hi: float, bin_size: float, smooth: float,
               halfwidth: float) -> list[Peak]:
    """Local maxima of a smoothed weighted histogram, each with the weight within +-halfwidth."""
    if len(v) == 0 or hi <= lo:
        return []
    nb = max(3, int(np.ceil((hi - lo) / bin_size)))
    idx = np.clip(((v - lo) / bin_size).astype(np.int64), 0, nb - 1)
    h = np.bincount(idx, weights=w, minlength=nb).astype(float)
    sm = gaussian_filter1d(h, max(smooth / bin_size, 0.5), mode="constant")
    pk, _ = find_peaks(np.concatenate([[0.0], sm, [0.0]]))
    pk = pk - 1
    cs = np.concatenate([[0.0], np.cumsum(h)])
    k = max(1, round(halfwidth / bin_size))
    out = []
    for i in pk:
        a, b = max(0, i - k), min(nb, i + k + 1)
        out.append(Peak(lo + (i + 0.5) * bin_size, float(cs[b] - cs[a])))
    return out


def refine(v: np.ndarray, w: np.ndarray, c0: float, win0: float, win_min: float = 0.015,
           win_max: float = 0.12, iters: int = 4) -> tuple[float, float, float, np.ndarray, float]:
    """Weighted mean of inliers in a window that follows the robust spread: c, sigma, rms, mask, window."""
    c, win, s = float(c0), float(win0), float(win0) / 2.5
    m = np.abs(v - c) < win
    for _ in range(iters):
        if m.sum() < 3 or w[m].sum() <= 0:
            break
        c = float(np.average(v[m], weights=w[m]))
        s = robust_sigma(v[m] - c, w[m])
        win = float(np.clip(2.5 * s, win_min, win_max))
        m = np.abs(v - c) < win
    if m.sum() == 0 or w[m].sum() <= 0:
        return c, s, 0.0, m, win
    rms = float(np.sqrt(np.average((v[m] - c) ** 2, weights=w[m])))
    return c, s, rms, m, win


def _tilt(xy: np.ndarray, z: np.ndarray, w: np.ndarray) -> tuple[float, float]:
    """Slope magnitude and residual rms of a weighted plane z = a x + b y + c."""
    if len(z) < 10:
        return 0.0, 0.0
    A = np.column_stack([xy - xy.mean(0), np.ones(len(z))])
    sw = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(A * sw[:, None], z * sw, rcond=None)
    res = z - A @ coef
    return float(np.hypot(coef[0], coef[1])), float(np.sqrt(np.average(res ** 2, weights=w)))


def _level(z: np.ndarray, w_fit: np.ndarray, peaks: list[Peak], pick: str, sigma0: float) -> Level | None:
    if not peaks:
        return None
    best = max(p.support for p in peaks)
    strong = [p for p in peaks if p.support >= max(0.2 * best, MIN_LEVEL_SUPPORT)]
    if not strong:
        return None
    c0 = min(p.center for p in strong) if pick == "low" else max(p.center for p in strong)
    c, s, rms, m, win = refine(z, w_fit, c0, max(0.04, 3 * sigma0), 0.015, 0.10)
    return Level(z=c, rms=rms, sigma=s, n=int(m.sum()), window=win)


def estimate(z: np.ndarray, nz: np.ndarray, w_peak: np.ndarray, w_fit: np.ndarray, xy: np.ndarray,
             cam_z: np.ndarray | None = None) -> FloorCeiling:
    """Global floor (lowest strong up-facing peak) and ceiling (highest strong down-facing peak, 1.8 m up)."""
    flags: list[str] = []
    up, down = nz > HORIZONTAL_NZ, nz < -HORIZONTAL_NZ
    if (up.sum() < 30) != (down.sum() < 30) and (np.abs(nz) > HORIZONTAL_NZ).sum() >= 30:
        flags.append("normals_unoriented")
        up = down = np.abs(nz) > HORIZONTAL_NZ
    horiz = up | down
    if horiz.sum() < 30:
        flags += ["floor_not_observed", "ceiling_not_observed"]
        f = float(np.percentile(z, 1)) if len(z) else 0.0
        if cam_z is not None and len(cam_z):
            f = min(f, float(np.median(cam_z)) - 1.4) if len(z) else float(np.median(cam_z)) - 1.4
        top = float(np.percentile(z, 99)) if len(z) else f + DEFAULT_CEILING_HEIGHT
        c = top if top - f >= MIN_CEILING_HEIGHT else f + DEFAULT_CEILING_HEIGHT
        return FloorCeiling(Level(f, observed=False), Level(c, observed=False), flags)

    lo, hi = np.percentile(z[horiz], [0.1, 99.9])
    lo, hi = float(lo) - 0.1, float(hi) + 0.1
    sigma0 = 0.02
    peaks_up = hist_peaks(z[up], w_peak[up], lo, hi, 0.01, 0.02, 0.03)
    if cam_z is not None and len(cam_z):
        cz = float(np.median(cam_z))
        plausible = [p for p in peaks_up if 0.85 <= cz - p.center <= 2.2]
        if plausible and max(p.support for p in plausible) >= 0.1 * max(p.support for p in peaks_up):
            peaks_up = plausible
        else:
            flags.append("floor_camera_height_unusual")
    floor = _level(z[up], w_fit[up], peaks_up, "low", sigma0)
    if floor is None:
        flags.append("floor_not_observed")
        floor = Level(float(np.percentile(z[horiz], 2)), observed=False)
    else:
        sel = up & (np.abs(z - floor.z) < floor.window)
        floor.tilt, _ = _tilt(xy[sel], z[sel], w_fit[sel])

    sigma = max(floor.sigma, 0.005) if floor.observed else sigma0
    cand = [p for p in hist_peaks(z[down], w_peak[down], lo, hi, 0.01, 0.02, 0.03)
            if MIN_CEILING_HEIGHT <= p.center - floor.z <= MAX_CEILING_HEIGHT]
    if cam_z is not None and len(cam_z):
        cand = [p for p in cand if p.center > float(np.median(cam_z)) + 0.05] or cand
    ceiling = _level(z[down], w_fit[down], cand, "high", sigma)
    if ceiling is None:
        flags.append("ceiling_not_observed")
        vert = np.abs(nz) < 0.3  # walls stop at the ceiling; all points would include outdoor geometry
        top = float(np.percentile(z[vert], 99.5)) if vert.sum() >= 30 else floor.z + DEFAULT_CEILING_HEIGHT
        plausible = MIN_CEILING_HEIGHT <= top - floor.z <= MAX_CEILING_HEIGHT
        c = top if plausible else floor.z + DEFAULT_CEILING_HEIGHT
        ceiling = Level(c, observed=False)
    else:
        sel = down & (np.abs(z - ceiling.z) < ceiling.window)
        ceiling.tilt, _ = _tilt(xy[sel], z[sel], w_fit[sel])
    log.debug("floor %.3f (rms %.3f, n %d), ceiling %.3f (rms %.3f, n %d)", floor.z, floor.rms, floor.n,
              ceiling.z, ceiling.rms, ceiling.n)
    return FloorCeiling(floor, ceiling, flags)


def room_level(z: np.ndarray, w_peak: np.ndarray, w_fit: np.ndarray, xy: np.ndarray, z_global: float,
               pick: str, sigma: float, search: float = 0.25) -> Level | None:
    """Per-room level: lowest or highest strong peak within +-search of the global level, then the weighted
    median of its inliers."""
    m = np.abs(z - z_global) < search
    if m.sum() < 30:
        return None
    peaks = hist_peaks(z[m], w_peak[m], z_global - search, z_global + search, 0.01, 0.02, 0.03)
    if not peaks:
        return None
    best = max(p.support for p in peaks)
    strong = [p for p in peaks if p.support >= 0.3 * best]
    c0 = min(p.center for p in strong) if pick == "low" else max(p.center for p in strong)
    zz, wf = z[m], w_fit[m]
    _, s, _, inl, win = refine(zz, wf, c0, max(0.04, 3 * sigma), 0.015, 0.10)
    if inl.sum() < 10:
        return None
    med = weighted_median(zz[inl], wf[inl])
    tilt, _ = _tilt(xy[m][inl], zz[inl], wf[inl])
    rms = float(np.sqrt(np.average((zz[inl] - med) ** 2, weights=wf[inl])))
    return Level(z=med, rms=rms, sigma=s, n=int(inl.sum()), window=win, tilt=tilt)
