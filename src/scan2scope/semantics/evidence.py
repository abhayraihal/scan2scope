"""Image evidence that checks a detector label: mask shape, thin dark lines and how an object differs from its wall.

A crack is a thin line darker than the surface on both sides of it, so a crack box must contain one: the
black-hat transform (closing minus image) keeps dark structures narrower than its kernel, and the components
that are long and thin are the line. Holes, stains and peeling paint are areas, so their masks are compact.
Glass at night, mirrors and doors are much darker than a lit wall, while a sheet of paper taped to it is not.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MaskShape:
    area_px: float
    length_px: float  # extent along the principal axis
    width_px: float  # area / length: the mean thickness
    elongation: float  # length / width
    fill: float  # mask area over its box area


@dataclass
class LineEvidence:
    mask: np.ndarray  # (h, w) bool at image size: the thin dark line pixels
    length_px: float  # summed principal-axis length of the line components
    width_px: float  # mean thickness of those components
    contrast: float  # median black-hat response on the line, in L* units (0..100)
    span: float  # extent of the line along its principal axis over the box diagonal
    n_components: int
    wander: float = 0.0  # spread across the line over its length (0 for a ruler-straight line)


@dataclass
class LineConfig:
    max_width_px: float = 7.0  # thicker dark structures are not hairline or drawn cracks at the working size
    min_contrast: float = 6.0  # L* units; also at least noise_k robust sigmas of the black-hat map
    noise_k: float = 4.0
    weak_frac: float = 0.5  # hysteresis: pixels above this share of the threshold extend a line
    bridge_px: int = 2  # gaps up to about twice this along a line are joined
    min_elongation: float = 4.0
    min_length_frac: float = 0.15  # of the box diagonal, per component
    keep_contrast_frac: float = 0.5  # components fainter than this share of the strongest line are dropped
    min_span: float = 0.5  # the kept line must cross at least this share of the box diagonal
    # a shorter line is a dot, a screw head or a switch edge at the working size, not something to call a crack
    min_line_px: float = 30.0
    min_aspect: float = 2.5  # major over minor spread of the line pixels: a ring or an outline is not a crack
    # a crack wanders: lines straighter than this (spread across the line over its length) are joints, gaps
    # between cabinet or door panels, frame edges, grout and corner lines
    min_wander: float = 0.01
    margin_px: int = 12  # context around the box for the closing


def lightness(rgb: np.ndarray) -> np.ndarray:
    """CIE L* in 0..100 of an RGB uint8 image."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2LAB)
    return lab[..., 0].astype(np.float32) * (100.0 / 255.0)


def _axis_stats(ys: np.ndarray, xs: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Principal-axis length (uniform-bar estimate sqrt(12 var)), the axis and the centre of pixel coordinates."""
    pts = np.stack([xs, ys], 1).astype(np.float64)
    c = pts.mean(0)
    if len(pts) < 2:
        return 1.0, np.array([1.0, 0.0]), c
    cov = np.cov((pts - c).T)
    evals, evecs = np.linalg.eigh(cov)
    axis = evecs[:, int(np.argmax(evals))]
    t = (pts - c) @ axis
    length = max(float(t.max() - t.min()) + 1.0, float(np.sqrt(12.0 * max(evals.max(), 0.0))))
    return length, axis, c


def mask_shape(mask: np.ndarray) -> MaskShape | None:
    """Shape of a boolean mask: principal-axis length, mean width, elongation and box fill."""
    ys, xs = np.nonzero(mask)
    n = len(xs)
    if n == 0:
        return None
    length, _, _ = _axis_stats(ys, xs)
    width = n / max(length, 1.0)
    box_area = float((xs.max() - xs.min() + 1) * (ys.max() - ys.min() + 1))
    return MaskShape(float(n), length, width, length / max(width, 1e-6), n / box_area)


def thin_dark_line(L: np.ndarray, box: np.ndarray, cfg: LineConfig | None = None) -> LineEvidence | None:
    """Thin dark line inside a box of a lightness image, or None when the box holds no such line."""
    cfg = cfg or LineConfig()
    h, w = L.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    bx0, by0 = max(0, int(np.floor(x0))), max(0, int(np.floor(y0)))
    bx1, by1 = min(w, int(np.ceil(x1))), min(h, int(np.ceil(y1)))
    if bx1 - bx0 < 3 or by1 - by0 < 3:
        return None
    m = cfg.margin_px
    cx0, cy0, cx1, cy1 = max(0, bx0 - m), max(0, by0 - m), min(w, bx1 + m), min(h, by1 + m)
    crop = np.ascontiguousarray(L[cy0:cy1, cx0:cx1], dtype=np.float32)
    k = 2 * int(np.ceil(cfg.max_width_px)) + 1
    se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    bh = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, se)
    inner = bh[by0 - cy0:by1 - cy0, bx0 - cx0:bx1 - cx0]
    med = float(np.median(inner))
    sigma = 1.4826 * float(np.median(np.abs(inner - med)))
    thr = max(cfg.min_contrast, med + cfg.noise_k * sigma)
    strong = inner > thr
    weak = inner > cfg.weak_frac * thr
    # hysteresis: weak pixels count when they connect to strong ones; small gaps along a line are bridged
    bridged = cv2.dilate(weak.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=cfg.bridge_px)
    n, lab = cv2.connectedComponents(bridged, connectivity=8)
    diag = float(np.hypot(bx1 - bx0, by1 - by0))
    comps = []
    for i in range(1, n):
        sel = (lab == i) & weak
        if not (sel & strong).any():
            continue
        ys, xs = np.nonzero(sel)
        if len(xs) < 3:
            continue
        length, _, _ = _axis_stats(ys, xs)
        width = len(xs) / max(length, 1.0)
        if length < cfg.min_length_frac * diag or length / max(width, 1e-6) < cfg.min_elongation:
            continue
        if width > cfg.max_width_px:
            continue
        comps.append((float(np.median(inner[ys, xs])), length, width, ys, xs))
    if not comps:
        return None
    best = max(c[0] for c in comps)
    comps = [c for c in comps if c[0] >= cfg.keep_contrast_frac * best]
    ys = np.concatenate([c[3] for c in comps])
    xs = np.concatenate([c[4] for c in comps])
    _, axis, centre = _axis_stats(ys, xs)
    pts = np.stack([xs, ys], 1) - centre
    t = pts @ axis
    extent = float(t.max() - t.min()) + 1.0
    span = extent / max(diag, 1.0)
    if span < cfg.min_span or extent < cfg.min_line_px:
        return None
    # shape test on the strong pixels: weak ones bridged in from texture (ruled paper, plaster grain) would
    # widen a straight stroke into a blob
    core = strong[ys, xs]
    sp = pts[core] if core.sum() >= 3 else pts
    _, s_axis, _ = _axis_stats(sp[:, 1], sp[:, 0])
    along = (sp - sp.mean(0)) @ s_axis
    across = (sp - sp.mean(0)) @ np.array([-s_axis[1], s_axis[0]])
    sd_across = float(np.std(across))
    if float(np.std(along)) < cfg.min_aspect * max(sd_across, 0.5):
        return None
    wander = sd_across / max(float(along.max() - along.min()) + 1.0, 1.0)
    if wander < cfg.min_wander:
        return None
    out = np.zeros((h, w), bool)
    out[ys + by0, xs + bx0] = True
    length = float(sum(c[1] for c in comps))
    return LineEvidence(out, length, float(len(xs)) / max(length, 1.0), float(np.median(inner[ys, xs])), span,
                        len(comps), wander)


def ring_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Band of px pixels around a boolean mask."""
    core = mask.astype(np.uint8)
    grown = cv2.dilate(core, np.ones((3, 3), np.uint8), iterations=max(1, int(px)))
    return grown.astype(bool) & ~mask.astype(bool)


def darker_than_surround(L: np.ndarray, mask: np.ndarray, ring_px: int = 8) -> float:
    """Median L* of the band around a mask minus the median L* inside it (positive: the inside is darker)."""
    inside = mask.astype(bool)
    if not inside.any():
        return 0.0
    ring = ring_mask(inside, ring_px)
    if not ring.any():
        return 0.0
    return float(np.median(L[ring]) - np.median(L[inside]))
