"""Image evidence behind damage classes: mask shape, thin dark lines, and how much darker an object is than its wall."""

from __future__ import annotations

import itertools

import cv2
import numpy as np
import pytest

from scan2scope.semantics.evidence import (
    LineConfig,
    darker_than_surround,
    lightness,
    mask_shape,
    ring_mask,
    thin_dark_line,
)


def wall(h=300, w=400, level=200, noise=2.0, seed=0):
    rng = np.random.default_rng(seed)
    img = np.clip(rng.normal(level, noise, (h, w)), 0, 255)
    return np.repeat(img[..., None], 3, axis=2).astype(np.uint8)


def stroke(img, a, b, color, thickness, bow=0.08):
    """A hand-drawn line from a to b that bows sideways by `bow` of its length (cracks and pen strokes wander)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    t = np.linspace(0, 1, 40)[:, None]
    normal = np.array([-(b - a)[1], (b - a)[0]])
    pts = a + t * (b - a) + np.sin(np.pi * t) * bow * normal
    cv2.polylines(img, [np.round(pts).astype(np.int32)], False, color, thickness)


def test_mask_shape_tells_lines_from_blobs():
    line = np.zeros((200, 200), np.uint8)
    cv2.line(line, (20, 30), (170, 160), 1, 3)
    disk = np.zeros((200, 200), np.uint8)
    cv2.circle(disk, (100, 100), 40, 1, -1)
    s_line, s_disk = mask_shape(line.astype(bool)), mask_shape(disk.astype(bool))
    assert s_line.length_px == pytest.approx(np.hypot(150, 130), rel=0.08)
    assert s_line.width_px < 6 and s_line.elongation > 30
    assert s_disk.elongation < 2.5 and s_disk.fill > 0.7
    assert mask_shape(np.zeros((5, 5), bool)) is None


def test_thin_dark_line_finds_a_drawn_line_and_its_extent():
    img = wall()
    stroke(img, (150, 80), (190, 200), (40, 40, 40), 3)  # a pen stroke on a pale wall
    ev = thin_dark_line(lightness(img), np.array([135, 70, 205, 210.0]))
    assert ev is not None and ev.n_components == 1
    assert ev.length_px == pytest.approx(np.hypot(40, 120), rel=0.15)
    assert ev.width_px < 6 and ev.contrast > 30 and ev.span > 0.8
    ys, xs = np.nonzero(ev.mask)
    assert xs.min() >= 135 and xs.max() <= 205 and ys.min() >= 70 and ys.max() <= 210


def test_thin_dark_line_joins_a_faint_broken_hairline():
    img = wall(noise=1.5)
    pts = np.array([[20, 150], [80, 132], [140, 160], [200, 138], [260, 162], [320, 134], [380, 148]])
    for (x0, y0), (x1, y1) in itertools.pairwise(pts):
        cv2.line(img, (int(x0), int(y0)), (int(x1), int(y1)), (180, 180, 180), 2)
    img[:, 100:104] = 200  # gaps along the crack
    img[:, 250:253] = 200
    ev = thin_dark_line(lightness(img), np.array([10, 120, 390, 175.0]))
    assert ev is not None and ev.span > 0.9 and ev.width_px < 6


@pytest.mark.parametrize("case", ["blank", "blob", "bright_line", "step", "ring", "dot", "gap"])
def test_thin_dark_line_rejects_what_is_not_a_line(case):
    img = wall()
    if case == "gap":
        cv2.line(img, (170, 72), (172, 208), (50, 50, 50), 2)  # the ruler-straight gap between two cabinet doors
    elif case == "blob":
        cv2.circle(img, (170, 140), 25, (60, 60, 60), -1)  # a hole or a dark stain is an area
    elif case == "bright_line":
        img[:] = 120
        cv2.line(img, (150, 80), (190, 200), (240, 240, 240), 3)
    elif case == "step":
        img[:, 170:] = 120  # the edge of a darker paint band is a step, not a valley
    elif case == "ring":
        cv2.circle(img, (170, 140), 30, (60, 60, 60), 2)  # the outline of a fitting or a patched hole
    elif case == "dot":
        cv2.line(img, (165, 135), (175, 150), (40, 40, 40), 2)  # a mark too short to be told from a crack
    assert thin_dark_line(lightness(img), np.array([135, 70, 205, 210.0]), LineConfig()) is None


def test_darker_than_surround_separates_dark_glass_from_paper():
    img = wall(level=210)
    mask = np.zeros(img.shape[:2], bool)
    mask[100:200, 150:250] = True
    night = img.copy()
    night[mask] = 25  # black glass
    paper = img.copy()
    paper[mask] = 240  # a white sheet on a cream wall
    assert darker_than_surround(lightness(night), mask) > 50
    assert darker_than_surround(lightness(paper), mask) < 0
    ring = ring_mask(mask, 8)
    assert not (ring & mask).any() and ring.sum() > 0
