"""Lifting masks onto room surfaces, cross-view merging, detector decoding and analyze() with fake models."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from scan2scope.config import ModelSpec
from scan2scope.semantics import SemanticsConfig, analyze
from scan2scope.semantics.detector import (DAMAGE_PROMPTS, OBJECT_PROMPTS, Detection, DetectorConfig,
                                           GroundingDinoDetector, decode, nms)
from scan2scope.semantics.lift import (LiftConfig, LiftedMask, assign_surface, lift_mask, measure_on_surface,
                                       robust_extent, wquantile)
from scan2scope.semantics.merge import (DamageObservation, MergeConfig, ObjectObservation, combine_scores,
                                        merge_damage, merge_objects)
from scan2scope.semantics.segmenter import SegmenterConfig, pack_masks, postprocess, unpack_masks
from scan2scope.semantics import suppress_inside_objects
from scan2scope.types import Scene

from semantics_fixtures import box_view, make_plan, polygon_mask, project, rect_on_wall_y, rect_room

LO, HI = np.array([0.0, 0.0, 0.0]), np.array([4.0, 3.0, 2.5])
CFG = LiftConfig()


def _lift(view, world_quad, scale=1.0):
    mask = polygon_mask(view.width, view.height, project(world_quad, view.T_wc, view.K), scale)
    return lift_mask(view, mask)


def _measure(view, world_quad, plan, scale=1.0):
    lifted = _lift(view, world_quad, scale)
    a, reason = assign_surface(lifted, plan, CFG)
    assert a is not None, reason
    m, reason = measure_on_surface(lifted, a, CFG)
    assert m is not None, reason
    return a, m


def test_weighted_quantile_and_extent_recover_uniform_span():
    x = np.linspace(0.005, 0.995, 100)  # centres of 100 cells covering [0, 1]
    w = np.ones_like(x)
    assert wquantile(x, w, 0.5) == pytest.approx(0.5, abs=1e-9)
    lo, hi = robust_extent(x, w, 0.02, 0.98)
    assert (lo, hi) == pytest.approx((0.0, 1.0), abs=0.01)
    lo3, hi3 = robust_extent(np.array([0.5, 1.5, 2.5]), np.ones(3), 0.02, 0.98)  # three unit cells
    assert hi3 - lo3 == pytest.approx(3.0, abs=0.05)


def test_area_on_fronto_parallel_wall():
    plan = make_plan(rect_room())
    view = box_view("v1", (2.0, 1.0, 1.2), (2.0, 3.0, 1.2), LO, HI)
    a, m = _measure(view, rect_on_wall_y(3.0, 1.6, 2.4, 0.9, 1.5), plan)
    assert a.surface_id == "R1-W3" and a.kind == "wall"
    assert m.area == pytest.approx(0.48, rel=0.03)
    assert m.width == pytest.approx(0.8, rel=0.03)
    assert m.height == pytest.approx(0.6, rel=0.03)
    # W3 runs from (4, 3) to (0, 3), so u = 4 - x; v is height above the floor
    assert m.u_range == pytest.approx((1.6, 2.4), abs=0.02)
    assert m.v_range == pytest.approx((0.9, 1.5), abs=0.02)


def test_area_on_oblique_wall_uses_projected_jacobian():
    plan = make_plan(rect_room())
    view = box_view("v1", (0.6, 0.8, 1.6), (2.2, 3.0, 1.1), LO, HI)
    _, m = _measure(view, rect_on_wall_y(3.0, 1.6, 2.4, 0.9, 1.5), plan)
    assert m.area == pytest.approx(0.48, rel=0.04)
    assert m.width == pytest.approx(0.8, rel=0.04)


def test_area_with_depth_noise_and_mask_at_other_resolution():
    plan = make_plan(rect_room())
    view = box_view("v1", (2.0, 1.0, 1.2), (2.0, 3.0, 1.2), LO, HI)
    rng = np.random.default_rng(0)
    c = view.T_wc[:3, 3]
    ray = view.pointmap - c
    depth = np.linalg.norm(ray, axis=-1, keepdims=True)
    view.pointmap = (c + ray * (1 + rng.normal(0, 0.01, depth.shape) / depth)).astype(np.float32)
    _, m = _measure(view, rect_on_wall_y(3.0, 1.6, 2.4, 0.9, 1.5), plan, scale=0.6)
    assert m.area == pytest.approx(0.48, rel=0.06)


def test_invalid_pixels_are_extrapolated_and_reported():
    plan = make_plan(rect_room())
    view = box_view("v1", (2.0, 1.0, 1.2), (2.0, 3.0, 1.2), LO, HI)
    quad = rect_on_wall_y(3.0, 1.6, 2.4, 0.9, 1.5)
    full = _lift(view, quad)
    view.valid[:, :78] = False  # left part of the mask has no geometry
    lifted = _lift(view, quad)
    assert 0.3 < lifted.valid_fraction < 0.9
    a, _ = assign_surface(lifted, plan, CFG)
    m, _ = measure_on_surface(lifted, a, CFG)
    assert m.valid_fraction == pytest.approx(lifted.valid_fraction)
    assert m.area == pytest.approx(0.48, rel=0.05)
    assert full.valid_fraction == pytest.approx(1.0)


def test_surface_assignment_picks_the_right_wall_at_a_corner():
    plan = make_plan(rect_room())
    view = box_view("v1", (2.2, 1.2, 1.3), (4.0, 3.0, 1.2), LO, HI)
    east = np.array([[4.0, 2.55, 1.0], [4.0, 2.9, 1.0], [4.0, 2.9, 1.4], [4.0, 2.55, 1.4]])
    north = rect_on_wall_y(3.0, 3.55, 3.9, 1.0, 1.4)
    a_e, m_e = _measure(view, east, plan)
    a_n, m_n = _measure(view, north, plan)
    assert a_e.surface_id == "R1-W2"
    assert a_n.surface_id == "R1-W3"
    assert m_e.area == pytest.approx(0.35 * 0.4, rel=0.06)
    assert m_n.u_range == pytest.approx((0.1, 0.45), abs=0.02)


def test_surface_assignment_between_two_parallel_walls_of_a_hallway():
    hall_hi = np.array([1.2, 5.0, 2.5])
    plan = make_plan(rect_room(w=1.2, d=5.0))
    view = box_view("v1", (0.9, 1.0, 1.4), (0.0, 3.0, 1.0), LO, hall_hi)
    west = np.array([[0.0, 2.6, 0.8], [0.0, 3.2, 0.8], [0.0, 3.2, 1.2], [0.0, 2.6, 1.2]])
    a, m = _measure(view, west, plan)
    assert a.surface_id == "R1-W4"  # west wall, not the east wall 1.2 m away
    assert m.area == pytest.approx(0.24, rel=0.06)


def test_floor_and_ceiling_assignment_by_orientation_and_height():
    plan = make_plan(rect_room())
    down = box_view("v1", (1.0, 0.8, 1.5), (1.8, 1.8, 0.0), LO, HI)
    floor = np.array([[1.5, 1.5, 0.0], [2.1, 1.5, 0.0], [2.1, 2.0, 0.0], [1.5, 2.0, 0.0]])
    a, m = _measure(down, floor, plan)
    assert a.surface_id == "R1-FLOOR"
    assert m.area == pytest.approx(0.30, rel=0.04)
    assert m.u_range == pytest.approx((1.5, 2.1), abs=0.02)  # plan x
    up = box_view("v2", (1.0, 0.8, 1.2), (1.8, 1.8, 2.5), LO, HI)
    ceil = np.array([[1.5, 1.5, 2.5], [2.1, 1.5, 2.5], [2.1, 2.0, 2.5], [1.5, 2.0, 2.5]])
    a2, m2 = _measure(up, ceil, plan)
    assert a2.surface_id == "R1-CEIL"
    assert m2.area == pytest.approx(0.30, rel=0.04)


def test_shared_wall_damage_goes_to_the_room_that_sees_it():
    r1 = rect_room("R1", 0.0, 0.0, 4.0, 3.0)
    r2 = rect_room("R2", 4.1, 0.0, 3.0, 3.0)
    plan = make_plan(r1, r2)
    view = box_view("v1", (2.0, 1.5, 1.3), (4.0, 1.5, 1.2), LO, HI)
    quad = np.array([[4.0, 1.2, 1.0], [4.0, 1.8, 1.0], [4.0, 1.8, 1.4], [4.0, 1.2, 1.4]])
    a, _ = _measure(view, quad, plan)
    assert a.room.id == "R1" and a.surface_id == "R1-W2"


def test_patch_at_table_height_is_off_surface():
    plan = make_plan(rect_room())
    n = 50
    pts = np.column_stack([np.linspace(1.0, 1.5, n), np.full(n, 1.5), np.full(n, 0.75)])
    lifted = LiftedMask("v1", pts, np.ones(n), np.tile([0.0, 0.0, 1e-4], (n, 1)), np.tile([0.0, 0.0, 1.0], (n, 1)),
                        float(n), float(n), np.array([1.0, 0.5, 1.5]))
    a, reason = assign_surface(lifted, plan, CFG)
    assert a is None and reason == "off_surface"


def test_patch_outside_every_room_is_dropped():
    plan = make_plan(rect_room())
    n = 20
    pts = np.column_stack([np.full(n, 6.0), np.linspace(1.0, 1.4, n), np.linspace(1.0, 1.4, n)])
    lifted = LiftedMask("v1", pts, np.ones(n), np.tile([1e-4, 0.0, 0.0], (n, 1)), np.tile([-1.0, 0.0, 0.0], (n, 1)),
                        float(n), float(n), np.array([6.5, 1.2, 1.2]))
    a, reason = assign_surface(lifted, plan, CFG)
    assert a is None and reason == "outside_rooms"


def test_crack_length_from_principal_axis():
    plan = make_plan(rect_room())
    view = box_view("v1", (2.0, 0.9, 1.0), (2.0, 3.0, 1.0), LO, HI, width=1280, height=960, f=1000.0,
                    pm_w=320, pm_h=240)
    # 1.0 m diagonal strip, 2 cm wide, from (x=2.6, z=0.5) to (x=2.0, z=1.3)
    a0, a1 = np.array([2.6, 3.0, 0.5]), np.array([2.0, 3.0, 1.3])
    t = (a1 - a0) / np.linalg.norm(a1 - a0)
    nrm = np.array([t[2], 0.0, -t[0]]) * 0.01
    quad = np.array([a0 - nrm, a1 - nrm, a1 + nrm, a0 + nrm])
    _, m = _measure(view, quad, plan)
    assert m.length == pytest.approx(1.0, rel=0.05)
    ends = sorted(map(tuple, np.round(m.endpoints, 2)))
    assert ends[0] == pytest.approx((1.4, 0.5), abs=0.05) and ends[1] == pytest.approx((2.0, 1.3), abs=0.05)


def _obs(view, cls="water_stain", surface="R1-W3", u=(1.6, 2.4), v=(0.9, 1.5), area=0.48, score=0.4):
    return DamageObservation(view_id=view, cls=cls, score=score, room_id="R1", surface_id=surface, kind="wall",
                             area=area, width=u[1] - u[0], height=v[1] - v[0], length=0.0, u_range=u, v_range=v,
                             endpoints=np.array([[u[0], v[0]], [u[1], v[1]]]))


def test_merge_two_views_of_one_stain():
    merged, dropped = merge_damage([_obs("v1", area=0.48, score=0.40),
                                    _obs("v2", u=(1.65, 2.45), v=(0.92, 1.52), area=0.50, score=0.42)])
    assert len(merged) == 1 and not dropped
    m = merged[0]
    assert m.area == pytest.approx(0.49)
    assert m.u_range == pytest.approx((1.6, 2.45)) and m.v_range == pytest.approx((0.9, 1.52))
    assert m.score == pytest.approx(1 - 0.6 * 0.58)
    assert sorted(m.view_ids) == ["v1", "v2"] and m.evidence["n_views"] == 2 and not m.evidence["single_view"]


def test_merge_needs_matching_class_and_surface_and_overlap():
    obs = [_obs("v1", score=0.6), _obs("v2", cls="mold", score=0.6), _obs("v3", surface="R1-W2", score=0.6),
           _obs("v4", u=(3.0, 3.5), score=0.6)]
    merged, _ = merge_damage(obs)
    assert len(merged) == 4


def test_merge_by_centre_distance_without_overlap():
    a = _obs("v1", u=(1.0, 1.1), v=(1.0, 1.1), area=0.01, score=0.4)
    b = _obs("v2", u=(1.2, 1.3), v=(1.05, 1.15), area=0.01, score=0.4)
    merged, _ = merge_damage([a, b])
    assert len(merged) == 1 and len(merged[0].view_ids) == 2


def test_single_view_detections_below_half_are_dropped_and_reported():
    merged, dropped = merge_damage([_obs("v1", score=0.45), _obs("v2", u=(3.0, 3.6), score=0.55)])
    assert len(merged) == 1 and merged[0].evidence["single_view"] is True
    assert len(dropped) == 1 and dropped[0]["reason"].startswith("single_view_score_below")


def test_detections_from_the_same_view_are_not_merged():
    merged, _ = merge_damage([_obs("v1", score=0.6), _obs("v1", u=(1.7, 2.5), score=0.55)])
    assert len(merged) == 2


def test_combine_scores_uses_top_k():
    assert combine_scores([0.5]) == pytest.approx(0.5)
    assert combine_scores([0.5, 0.5, 0.5, 0.9], top_k=3) == pytest.approx(1 - 0.1 * 0.5 * 0.5)


def test_merge_objects_by_plan_distance():
    def o(view, xy, score=0.6):
        return ObjectObservation(view, "sink", score, "R1", np.array(xy), (xy[0] - 0.2, xy[0] + 0.2),
                                 (xy[1] - 0.2, xy[1] + 0.2), (0.8, 0.95))

    merged, dropped = merge_objects([o("v1", (1.0, 2.8)), o("v2", (1.2, 2.75)), o("v3", (3.5, 0.5), 0.4)])
    assert len(merged) == 1 and len(dropped) == 1
    assert merged[0].xy == pytest.approx((1.1, 2.775)) and merged[0].z_range == pytest.approx((0.8, 0.95))


def test_decode_thresholds_nms_and_class_mapping():
    raw = {"boxes": np.array([[0.1, 0.1, 0.3, 0.3], [0.11, 0.1, 0.31, 0.3], [0.5, 0.5, 0.6, 0.6],
                              [0.0, 0.0, 1.0, 1.0], [0.12, 0.12, 0.2, 0.2], [0.7, 0.1, 0.8, 0.2]], np.float32),
           "phrase_scores": np.array([[0.50, 0.1, 0.1, 0.1, 0.1], [0.45, 0.1, 0.1, 0.1, 0.1],
                                      [0.1, 0.1, 0.1, 0.1, 0.40], [0.9, 0.1, 0.1, 0.1, 0.1],
                                      [0.48, 0.1, 0.1, 0.1, 0.1], [0.30, 0.1, 0.1, 0.1, 0.1]], np.float32)}
    dets = decode(raw, DAMAGE_PROMPTS, 1000, 800, DetectorConfig())
    assert [(d.cls, round(d.score, 2)) for d in dets] == [("water_stain", 0.5), ("peeling_paint", 0.4)]
    assert dets[0].box == pytest.approx([100, 80, 300, 240])
    objs = decode({"boxes": raw["boxes"][:1], "phrase_scores": np.array([[0.1] * 9 + [0.33]], np.float32)},
                  OBJECT_PROMPTS, 1000, 800, DetectorConfig())
    assert objs[0].cls == "washing_machine" and objs[0].kind == "object"
    assert decode({"boxes": np.zeros((0, 4)), "phrase_scores": np.zeros((0, 5))}, DAMAGE_PROMPTS, 10, 10,
                  DetectorConfig()) == []


def test_nms_drops_contained_boxes():
    boxes = np.array([[0, 0, 100, 100], [10, 10, 30, 30], [200, 200, 220, 220]], float)
    keep = nms(boxes, np.array([0.9, 0.8, 0.5]), 0.5, 0.85)
    assert keep.tolist() == [0, 2]


def test_prompt_text_format():
    assert DAMAGE_PROMPTS.text == "water stain. mold. crack. hole. peeling paint."
    assert OBJECT_PROMPTS.text.endswith("refrigerator. washing machine.")
    assert DAMAGE_PROMPTS.box_threshold == 0.35 and OBJECT_PROMPTS.box_threshold == 0.30


def test_damage_inside_window_box_is_suppressed():
    dets = [Detection("water_stain", "damage", np.array([110, 110, 150, 150.0]), 0.5),
            Detection("water_stain", "damage", np.array([300, 300, 350, 350.0]), 0.5),
            Detection("window", "object", np.array([100, 100, 200, 200.0]), 0.6)]
    kept, dropped = suppress_inside_objects(dets, ("door", "window", "mirror"), 0.6)
    assert len(kept) == 2 and dropped[0][1] == "inside_window"


def test_segmenter_postprocess_clips_and_falls_back_to_box():
    m = np.zeros((100, 100), bool)
    m[:, :] = True
    out, fb = postprocess(m, np.array([20, 20, 40, 40.0]), SegmenterConfig())
    assert not fb and out.sum() == pytest.approx(22 * 22, abs=30)
    out2, fb2 = postprocess(np.zeros((100, 100), bool), np.array([20, 20, 40, 40.0]), SegmenterConfig())
    assert fb2 and out2.sum() == 400
    packed = pack_masks(np.stack([out, out2]))
    assert (unpack_masks(packed, (100, 100)) == np.stack([out, out2])).all()


class FakeDetector:
    """Projected stain as a damage box and a fixed 'sink' box; the view is told apart by its grey level."""

    def __init__(self, boxes_by_view):
        self.boxes_by_view = boxes_by_view
        self.calls = 0

    def cache_key(self, sha, size, prompt):
        return {"sha": sha, "size": list(size), "prompt": prompt.text}

    def predict(self, image, prompt):
        self.calls += 1
        h, w = image.shape[:2]
        dmg, obj = self.boxes_by_view[f"v{(int(image[0, 0, 0]) - 50) // 10}"]
        b = dmg if prompt.kind == "damage" else obj
        ps = np.full((1, len(prompt.phrases)), 0.05, np.float32)
        ps[0, 0 if prompt.kind == "damage" else 3] = 0.6  # water stain / sink
        return {"boxes": (np.asarray(b, float) / [w, h, w, h])[None].astype(np.float32), "phrase_scores": ps}


class FakeSegmenter:
    """Box-shaped masks; exact for fronto-parallel views of a rectangle."""

    def cache_key(self, sha, size, boxes):
        return {"sha": sha, "boxes": np.round(boxes, 1).tolist()}

    def predict(self, image, boxes):
        h, w = image.shape[:2]
        masks = np.zeros((len(boxes), h, w), bool)
        for k, (x0, y0, x1, y1) in enumerate(np.asarray(boxes)):
            masks[k, int(round(y0)):int(round(y1)) + 1, int(round(x0)):int(round(x1)) + 1] = True
        return {"masks": pack_masks(masks), "shape": np.array([h, w]), "iou": np.full(len(boxes), 0.9, np.float32)}

    def masks(self, raw, boxes):
        m = unpack_masks(raw["masks"], tuple(raw["shape"]))
        return [(m[k], 0.9, False) for k in range(len(boxes))]


class DictCache:
    def __init__(self):
        self.store = {}

    def compute(self, key, fn):
        k = json.dumps(key, sort_keys=True)
        if k not in self.store:
            self.store[k] = fn()
        return self.store[k]


def _scene_with_images(tmp_path, cams, target_y=3.0):
    views = []
    for k, cam in enumerate(cams):
        path = tmp_path / f"v{k}.png"
        Image.new("RGB", (640, 480), (50 + 10 * k,) * 3).save(path)
        views.append(box_view(f"v{k}", cam, (cam[0], target_y, cam[2]), LO, HI, image_path=path))
    return Scene(tier="photo", views=views, points=np.zeros((0, 3)), normals=np.zeros((0, 3)), weights=np.zeros(0),
                 view_index=np.zeros(0, int))


def _working_box(quad, view, s=1024 / 640):
    p = project(quad, view.T_wc, view.K)
    return [*((p.min(0) + 0.5) * s - 0.5), *((p.max(0) + 0.5) * s - 0.5)]


def test_analyze_end_to_end_with_fake_models(tmp_path):
    plan = make_plan(rect_room())
    scene = _scene_with_images(tmp_path, [(2.0, 1.0, 1.2), (1.8, 0.9, 1.2)])
    stain = rect_on_wall_y(3.0, 1.6, 2.4, 0.9, 1.5)
    sink = rect_on_wall_y(3.0, 2.6, 3.0, 0.8, 1.0)  # in view of both cameras
    det = FakeDetector({v.id: (_working_box(stain, v), _working_box(sink, v)) for v in scene.views})
    cache = DictCache()
    res = analyze([scene], plan, tmp_path / "work", cache=cache, detector=det, segmenter=FakeSegmenter())
    assert len(res.damage) == 1
    d = res.damage[0]
    assert (d.id, d.cls, d.surface_id, d.room_id) == ("D1", "water_stain", "R1-W3", "R1")
    assert d.area.value == pytest.approx(0.48, rel=0.06) and d.area.unit == "m2"
    assert d.width.value == pytest.approx(0.8, rel=0.06) and d.height.value == pytest.approx(0.6, rel=0.06)
    assert d.score == pytest.approx(0.84, abs=1e-3) and sorted(d.view_ids) == ["v0", "v1"]
    assert d.length is None and d.evidence["n_views"] == 2
    assert d.area.evidence["n_views"] == 2 and len(d.area.evidence["per_view"]) == 2
    assert d.area.lo is None and d.area.hi is None  # intervals belong to the uncertainty stage
    assert len(res.objects) == 1
    o = res.objects[0]
    assert (o.id, o.cls, o.room_id) == ("O1", "sink", "R1")
    assert o.xy == pytest.approx((2.8, 3.0), abs=0.03) and o.z_range[0] == pytest.approx(0.8, abs=0.05)
    debug = json.loads((tmp_path / "work" / "semantics" / "detections.json").read_text())
    assert len(debug["detections"]) == 4 and debug["damage"][0]["id"] == "D1"
    calls = det.calls
    again = analyze([scene], plan, None, cache=cache, detector=det, segmenter=FakeSegmenter())
    assert det.calls == calls  # second run is served from the cache
    assert again.damage[0].area.value == d.area.value


def test_analyze_without_usable_views_needs_no_models(tmp_path):
    plan = make_plan(rect_room())
    scene = _scene_with_images(tmp_path, [(2.0, 1.0, 1.2)])
    scene.views[0].pointmap = None
    res = analyze([scene], plan, None, detector=object(), segmenter=object())
    assert res.damage == [] and res.objects == []
    assert "semantics_no_views" in res.flags and "semantics_views_no_pointmap:1" in res.flags


def test_analyze_missing_weights_raises_runtime_error(tmp_path):
    plan = make_plan(rect_room())
    scene = _scene_with_images(tmp_path, [(2.0, 1.0, 1.2)])
    det = GroundingDinoDetector()
    det.spec = ModelSpec("missing", "nobody/no-such-model", "0", "none", ())
    with pytest.raises(RuntimeError, match="fetch-weights"):
        analyze([scene], plan, None, detector=det, segmenter=FakeSegmenter())


def test_analyze_with_empty_plan_flags_and_returns():
    res = analyze([], make_plan(), None)
    assert res.flags == ["semantics_no_rooms"]


def test_semantics_config_defaults_match_contract():
    cfg = SemanticsConfig()
    assert cfg.detector.long_side == 1024 and cfg.lift.wall_tol == 0.15
    assert cfg.merge.min_iou == 0.2 and cfg.merge.max_center_dist == 0.3 and cfg.merge.min_single_view_score == 0.5
    assert MergeConfig().top_k == 3
