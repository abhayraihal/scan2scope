"""Checks that a damage box has to pass before it is reported: look-alikes, distractors it lies on, the thin-line test
for cracks, class plausibility per surface, and the dark-glass and floor tests for windows and mirrors."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from PIL import Image
from semantics_fixtures import box_view, make_plan, project, rect_on_wall_y, rect_room

from scan2scope.semantics import SemanticsConfig, analyze, drop_lookalikes, resolve_class, surface_rule
from scan2scope.semantics.detector import DAMAGE_PROMPTS, Detection
from scan2scope.semantics.segmenter import pack_masks, unpack_masks
from scan2scope.types import Scene

LO, HI = np.array([0.0, 0.0, 0.0]), np.array([4.0, 3.0, 2.5])
S = 1024 / 640  # view images are 640 x 480, the detector sees 1024 x 768


class Scripted:
    """Fixed detections per view (view k is told apart by the grey code k in its top-left corner)."""

    def __init__(self, script):
        self.script = script

    def cache_key(self, sha, size, prompt):
        return {"sha": sha, "prompt": prompt.text}

    def predict(self, image, prompt):
        h, w = image.shape[:2]
        rows = self.script[f"v{(int(image[0, 0, 0]) - 50) // 10}"].get(prompt.kind, [])
        boxes = np.array([np.asarray(b, float) / [w, h, w, h] for b, _, _ in rows], np.float32).reshape(-1, 4)
        ps = np.full((len(rows), len(prompt.phrases)), 0.05, np.float32)
        for k, (_, phrase, score) in enumerate(rows):
            ps[k, prompt.phrases.index(phrase)] = score
        return {"boxes": boxes, "phrase_scores": ps}


class BoxSegmenter:
    """Masks equal to the prompt boxes."""

    def cache_key(self, sha, size, boxes):
        return {"sha": sha, "boxes": np.round(boxes, 1).tolist()}

    def predict(self, image, boxes):
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), bool)
        for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes)):
            masks[k, round(y0):round(y1) + 1, round(x0):round(x1) + 1] = True
        return {"masks": pack_masks(masks), "shape": np.array([h, w]), "iou": np.full(len(boxes), 0.9, np.float32)}

    def masks(self, raw, boxes):
        m = unpack_masks(raw["masks"], tuple(raw["shape"]))
        return [(m[k], 0.9, False) for k in range(len(boxes))]


def scene(tmp_path, cams, target_y=3.0, draw=None, target_z=None):
    """Views of the 4 x 3 x 2.5 m box room on pale wall images; draw(img, view) paints content in image pixels."""
    views = []
    for k, cam in enumerate(cams):
        tz = cam[2] if target_z is None else target_z
        v = box_view(f"v{k}", cam, (cam[0], target_y, tz), LO, HI, image_path=tmp_path / f"v{k}.png")
        img = np.full((480, 640, 3), 205, np.uint8)
        if draw is not None:
            draw(img, v)
        img[:6, :6] = 50 + 10 * k  # the view code the scripted detector reads
        Image.fromarray(img).save(v.image_path)
        views.append(v)
    return Scene(tier="photo", views=views, points=np.zeros((0, 3)), normals=np.zeros((0, 3)), weights=np.zeros(0),
                 view_index=np.zeros(0, int))


def wbox(quad, view, pad=0.0):
    """Working-image box around world points (pad in working pixels)."""
    p = project(np.asarray(quad, float), view.T_wc, view.K)
    lo, hi = (p.min(0) + 0.5) * S - 0.5, (p.max(0) + 0.5) * S - 0.5
    return [lo[0] - pad, lo[1] - pad, hi[0] + pad, hi[1] + pad]


CRACK = (np.array([1.85, 3.0, 0.9]), np.array([2.15, 3.0, 1.5]))  # a drawn line 0.67 m long on the north wall
CAMS = [(2.0, 1.0, 1.2), (1.8, 0.9, 1.25)]


def draw_crack(img, view):
    """The crack drawn by hand: it bows 3 cm off the straight line between its ends."""
    t = np.linspace(0, 1, 40)[:, None]
    a, b = CRACK
    side = np.array([0.6, 0.0, -0.3]) / np.hypot(0.6, 0.3)  # in the wall plane, across the crack
    pts = a + t * (b - a) + np.sin(np.pi * t) * 0.03 * side
    p = project(pts, view.T_wc, view.K)
    cv2.polylines(img, [np.round(p).astype(np.int32)], False, (35, 35, 35), 2)


def test_a_crack_is_measured_along_its_line(tmp_path):
    plan = make_plan(rect_room())
    sc = scene(tmp_path, CAMS, draw=draw_crack)
    script = {v.id: {"damage": [(wbox(np.stack(CRACK), v, pad=12), "crack", 0.3)]} for v in sc.views}
    res = analyze([sc], plan, None, detector=Scripted(script), segmenter=BoxSegmenter())
    assert [(d.cls, d.surface_id) for d in res.damage] == [("crack", "R1-W3")]
    d = res.damage[0]
    assert d.length.value == pytest.approx(np.hypot(0.3, 0.6), rel=0.12)
    # W3 runs from x=4 to x=0, so u = 4 - x
    assert d.u_range[0] == pytest.approx(1.85, abs=0.06) and d.u_range[1] == pytest.approx(2.15, abs=0.06)
    assert d.v_range[0] == pytest.approx(0.9, abs=0.06) and d.v_range[1] == pytest.approx(1.5, abs=0.06)
    assert d.area.value < 0.03  # the line, not the box around it
    assert d.evidence["n_views"] == 2 and d.score == pytest.approx(1 - 0.7 * 0.7)


def test_a_crack_box_on_a_blank_wall_is_dropped(tmp_path):
    plan = make_plan(rect_room())
    sc = scene(tmp_path, CAMS)
    script = {v.id: {"damage": [(wbox(np.stack(CRACK), v, pad=12), "crack", 0.45)]} for v in sc.views}
    res = analyze([sc], plan, None, detector=Scripted(script), segmenter=BoxSegmenter())
    assert res.damage == []
    assert {r["reason"] for r in res.dropped if r.get("kind") == "damage"} == {"crack_without_thin_line"}


def test_an_area_class_on_a_thin_line_becomes_a_crack():
    L = np.full((300, 400), 80.0, np.float32)
    line = np.zeros((300, 400), np.uint8)
    t = np.linspace(0, 1, 40)[:, None]
    pts = np.array([100, 60]) + t * np.array([30, 180]) + np.sin(np.pi * t) * np.array([12, -2])
    cv2.polylines(line, [np.round(pts).astype(np.int32)], False, 1, 2)
    L[line > 0] = 30.0
    d = Detection("hole", "damage", np.array([90, 50, 140, 250.0]), 0.42, class_scores={"hole": 0.42, "crack": 0.2})
    cls, scores, mask, info = resolve_class(d, line.astype(bool), lambda: L, DAMAGE_PROMPTS, SemanticsConfig())
    assert cls == "crack" and scores["crack"] == pytest.approx(0.42) and scores["hole"] == pytest.approx(0.2)
    assert info["relabel"] == "hole->crack:thin_line" and mask.sum() < 0.5 * line.size
    blob = np.zeros((300, 400), bool)
    blob[120:170, 90:140] = True
    dark = L.copy()
    dark[blob] = 45.0  # a real hole is a dark area, even with the line running into it
    cls2, _, mask2, info2 = resolve_class(d, blob, lambda: dark, DAMAGE_PROMPTS, SemanticsConfig())
    assert cls2 == "hole" and mask2 is blob and info2["darker_by"] > 30
    # a pale area crossed by a dark line (the paper with a drawn crack) is the line
    cls3, _, _, info3 = resolve_class(d, blob, lambda: L, DAMAGE_PROMPTS, SemanticsConfig())
    assert cls3 == "crack" and info3["relabel"] == "hole->crack:thin_line"
    weak = Detection("hole", "damage", np.array([300, 20, 340, 60.0]), 0.25, class_scores={"hole": 0.25})
    blank = np.zeros((300, 400), bool)
    blank[25:55, 305:335] = True
    cls4, _, _, info4 = resolve_class(weak, blank, lambda: dark, DAMAGE_PROMPTS, SemanticsConfig())
    assert cls4 is None and info4["reason"] == "hole_below_0.35"  # no line, and too weak for a hole


def test_lookalikes_and_distractors_drop_damage_boxes():
    hole = Detection("hole", "damage", np.array([100, 100, 130, 128.0]), 0.39)
    lamp = Detection("smoke_detector", "distractor", np.array([98, 99, 131, 130.0]), 0.33)
    far = Detection("light_switch", "distractor", np.array([300, 300, 320, 330.0]), 0.9)
    kept, gone = drop_lookalikes([hole], [lamp, far], 0.5, 0.1)
    assert kept == [] and gone[0][1] == "lookalike_smoke_detector"
    weak = Detection("smoke_detector", "distractor", np.array([98, 99, 131, 130.0]), 0.25)
    kept2, _ = drop_lookalikes([hole], [weak], 0.5, 0.1)  # explains the box much worse than "hole" does
    assert kept2 == [hole]


def test_damage_on_a_curtain_and_peeling_paint_on_the_floor_are_dropped(tmp_path):
    plan = make_plan(rect_room())
    sc = scene(tmp_path, [(2.0, 0.6, 1.3), (1.8, 0.5, 1.35)], target_z=0.7)
    patch = rect_on_wall_y(3.0, 1.7, 2.1, 1.4, 1.8)
    floor = np.array([[1.6, 2.3, 0.0], [2.2, 2.3, 0.0], [2.2, 2.7, 0.0], [1.6, 2.7, 0.0]])  # a mat by the wall
    script = {}
    for v in sc.views:
        script[v.id] = {"damage": [(wbox(patch, v), "peeling paint", 0.55), (wbox(floor, v), "peeling paint", 0.6)],
                        "distractor": [(wbox(patch, v, pad=80), "curtain", 0.7)]}
    res = analyze([sc], plan, None, detector=Scripted(script), segmenter=BoxSegmenter())
    assert res.damage == []
    reasons = {r["reason"] for r in res.dropped if r.get("kind") == "damage"}
    assert reasons == {"on_curtain", "peeling_paint_implausible_on_floor"}


def test_floor_stains_need_two_views():
    allowed, need = surface_rule(SemanticsConfig().surface_evidence, "water_stain", "floor")
    assert allowed and need == (0.6, 2)
    assert surface_rule(SemanticsConfig().surface_evidence, "water_stain", "wall") == (True, None)
    assert surface_rule(SemanticsConfig().surface_evidence, "mold", "floor") == (False, None)
    for cls in ("crack", "hole"):  # joints and dark gaps: three views and a strong combined score
        assert surface_rule(SemanticsConfig().surface_evidence, cls, "floor") == (True, (0.75, 3))


def test_dark_glass_counts_as_a_window_and_a_window_reaching_the_floor_does_not(tmp_path):
    plan = make_plan(rect_room())
    win = rect_on_wall_y(3.0, 1.6, 2.8, 1.0, 2.0)  # coplanar in the point map: night glass, no depth behind
    door = rect_on_wall_y(3.0, 0.7, 1.3, 0.0, 2.0)  # a glazed door, dark too, but it stands on the floor

    def draw(img, view):
        for quad, level in ((win, 25), (door, 30)):
            p = np.clip(project(quad, view.T_wc, view.K), 0, [639, 479]).astype(int)
            img[p[:, 1].min():p[:, 1].max(), p[:, 0].min():p[:, 0].max()] = level

    sc = scene(tmp_path, [(2.0, 0.5, 1.2), (1.9, 0.6, 1.25)], draw=draw, target_z=0.8)
    stain = rect_on_wall_y(3.0, 1.8, 2.1, 1.3, 1.6)  # inside the window: a reflection, not the wall
    script = {v.id: {"damage": [(wbox(stain, v), "water stain", 0.6)],
                     "object": [(wbox(win, v), "window", 0.6), (wbox(door, v), "window", 0.5)]} for v in sc.views}
    res = analyze([sc], plan, None, detector=Scripted(script), segmenter=BoxSegmenter())
    assert res.damage == []
    assert [o.cls for o in res.objects] == ["window"] and res.objects[0].z_range[0] > 0.8
    reasons = [r["reason"] for r in res.dropped if r.get("kind") == "object"]
    assert "reaches_floor" in reasons
    kept = [r for r in res.dropped if r.get("kind") == "damage"]
    assert kept and all(r["reason"] == "inside_window" for r in kept)
