"""Error model (uncertainty.annotate) and split-conformal calibration (uncertainty.calibrate)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml
from _output_plan import (
    M,
    make_damage,
    make_plan,
    poly_room,
    rect_room,
)

from scan2scope.types import TIERS, Measurement, Plan
from scan2scope.uncertainty import annotate
from scan2scope.uncertainty.calibrate import (
    as_record,
    conformal_level,
    fit_q,
    loro_coverage,
    room_quantile,
    write_calibration,
)
from scan2scope.uncertainty.model import load_priors

Z = 1.645
NO_CAL: dict = {"tiers": {}}  # q = 1 for every tier, independent of the shipped calibration.yaml


def measurements(plan, damage) -> dict[str, Measurement]:
    out = {"footprint": plan.footprint_area, "extent_x": plan.extent_x, "extent_y": plan.extent_y}
    for r in plan.rooms:
        out[f"{r.id}.ceiling"] = r.ceiling_height
        out[f"{r.id}.area"] = r.floor_area
        out[f"{r.id}.perimeter"] = r.perimeter
        for w in r.walls:
            out[f"{w.id}.length"] = w.length
            out[f"{w.id}.height"] = w.height
        for o in r.openings:
            for k in ("offset", "width", "height", "sill"):
                if getattr(o, k) is not None:
                    out[f"{o.id}.{k}"] = getattr(o, k)
    for d in damage:
        for k in ("area", "width", "height", "length"):
            if getattr(d, k) is not None:
                out[f"{d.id}.{k}"] = getattr(d, k)
    return out


def run(tier: str, quality: dict | None = None, calibration: dict | None = None, plan=None, damage=None):
    plan = plan or make_plan()
    damage = damage if damage is not None else make_damage()
    quality = quality if quality is not None else {"tier": tier, "scale_log_sigma": 0.0, "scenes": [{}]}
    rec = annotate(plan, damage, tier=tier, quality=quality, calibration=calibration or NO_CAL)
    return plan, damage, rec


def half(m: Measurement) -> float:
    return (m.hi - m.lo) / 2


@pytest.mark.parametrize("tier", TIERS)
def test_every_measurement_gets_an_interval(tier):
    plan, damage, rec = run(tier)
    ms = measurements(plan, damage)
    assert len(ms) == 3 + 2 * 3 + 8 * 2 + 3 * 3 + 1 + 3 * 3 + 1
    for name, m in ms.items():
        assert m.lo is not None and m.hi is not None, name
        assert 0.0 <= m.lo <= m.value <= m.hi, name
        assert m.evidence["sigma"] > 0 and m.evidence["q"] == 1.0, name
    assert plan.meta["uncertainty"] is rec
    assert rec["status"] == "prior" and rec["tier"] == tier


def test_widths_ordered_photo_video_lidar():
    widths = {t: {k: m.hi - m.lo for k, m in measurements(*run(t)[:2]).items()} for t in TIERS}
    for name in widths["lidar"]:
        assert widths["photo"][name] >= widths["video"][name] > widths["lidar"][name], name
        if not name.startswith("D"):  # damage has one additive term for every tier and the same scale floor
            assert widths["photo"][name] > widths["video"][name], name


def test_length_formula_with_residual_in_quadrature():
    plan, _, _ = run("lidar")
    w = plan.rooms[1].walls[0]  # 3 m, observed 0.9, n_points 5000, residual 0.004
    a = math.hypot(0.008, 0.004)
    sigma = math.hypot(3.0 * 0.003, a)
    assert w.length.evidence["sigma"] == pytest.approx(sigma)
    assert w.length.lo == pytest.approx(3.0 - Z * sigma)
    assert w.length.hi == pytest.approx(3.0 + Z * sigma)


def test_opening_and_ceiling_formulas():
    plan, _, _ = run("video")
    door = plan.rooms[0].openings[0]
    assert door.width.evidence["sigma"] == pytest.approx(math.hypot(0.8 * 0.08, 0.03))
    # heights add the vertical term on top of the shared scale
    ceiling = math.sqrt((2.5 * 0.08) ** 2 + (2.5 * 0.1) ** 2 + 0.02 ** 2)
    assert plan.rooms[0].ceiling_height.evidence["sigma"] == pytest.approx(ceiling)
    plan, _, _ = run("lidar")
    assert plan.rooms[0].ceiling_height.evidence["sigma"] == pytest.approx(math.hypot(2.5 * 0.003, 0.006))


def test_scale_term_uses_floor_or_capture_estimate():
    for scale, expect in ((0.01, 0.08), (0.12, 0.12)):
        plan, _, rec = run("video", quality={"scale_log_sigma": scale, "scenes": []})
        assert rec["scale_sigma"] == pytest.approx(expect)
        assert plan.rooms[1].walls[0].length.evidence["sigma_parts"]["scale"] == pytest.approx(3.0 * expect)


def test_observed_fraction_and_few_points_inflate_additive_term():
    plan, _, _ = run("photo")
    r1 = plan.rooms[0]
    add = {w.id: w.length.evidence["sigma_parts"]["additive"] for w in r1.walls}
    assert add["R1-W3"] == pytest.approx(math.hypot(0.04 * 2.0, 0.004))  # observed 0.2
    assert add["R1-W1"] == pytest.approx(math.hypot(0.04, 0.004))  # observed 0.9

    plan = make_plan()
    plan.rooms[0].walls[0].observed_fraction = 0.5
    plan.rooms[0].walls[1].length.evidence["n_points"] = 20
    plan, _, _ = run("photo", plan=plan)
    w1, w2 = plan.rooms[0].walls[:2]
    assert w1.length.evidence["sigma_parts"]["additive"] == pytest.approx(math.hypot(0.04 * 1.4, 0.004))
    assert w2.length.evidence["sigma_parts"]["additive"] == pytest.approx(math.hypot(0.04 * 1.6, 0.004))


def test_area_perimeter_and_footprint_formulas():
    plan, _, _ = run("video")
    s = 0.08
    r1, r2 = plan.rooms
    a = math.hypot(0.025, 0.004)
    t2 = 3.0 * a * 4
    assert r2.floor_area.evidence["sigma"] == pytest.approx(math.hypot(2 * 9.0 * s, t2))
    assert r2.perimeter.evidence["sigma"] == pytest.approx(math.hypot(12.0 * s, 4 * a))
    a_w3 = math.hypot(0.05, 0.004)
    t1 = 4.0 * a + 3.0 * a + 4.0 * a_w3 + 3.0 * a
    assert r1.floor_area.evidence["sigma"] == pytest.approx(math.hypot(2 * 12.0 * s, t1))
    fp = plan.footprint_area.evidence["sigma"]
    assert fp == pytest.approx(math.hypot(2 * 21.0 * s, math.hypot(t1, t2)))
    # The scale term is correlated, so the footprint is wider than independent room areas would give.
    assert fp > math.hypot(r1.floor_area.evidence["sigma"], r2.floor_area.evidence["sigma"])
    assert plan.extent_x.evidence["sigma"] == pytest.approx(math.hypot(7.1 * s, 0.025 * math.sqrt(2)))


def test_placement_uncertain_inflates_footprint_and_extents_only():
    base, _, _ = run("photo")
    plan = make_plan()
    plan.flags.append("placement_uncertain:R2")
    plan, _, rec = run("photo", plan=plan)
    assert rec["placement_uncertain"] == ["R2"]
    for name in ("footprint_area", "extent_x", "extent_y"):
        assert half(getattr(plan, name)) == pytest.approx(1.5 * half(getattr(base, name)))
    assert half(plan.rooms[1].walls[0].length) == pytest.approx(half(base.rooms[1].walls[0].length))

    plan = make_plan()
    plan.rooms[0].flags.append("placement_uncertain")
    plan, _, rec = run("photo", plan=plan)
    assert rec["placement_uncertain"] == ["R1"]


def test_room_context_factors_from_scene_quality():
    quality = {"scale_log_sigma": 0.0, "scenes": [
        {"room_hint": "02 kitchen", "n_photos": 7, "intrinsics_source": "exif"},
        {"room_hint": "01 living", "n_photos": 3, "low_light": True, "intrinsics_source": "default"},
    ]}
    plan, _, rec = run("photo", quality=quality)
    assert rec["room_factors"]["R1"]["factor"] == pytest.approx(1.3 * 1.5 * 1.2)
    assert rec["room_factors"]["R1"]["reasons"] == ["low_light", "few_photos", "missing_exif_focal"]
    assert rec["room_factors"]["R2"]["factor"] == 1.0
    w = plan.rooms[0].walls[0]
    assert w.length.evidence["sigma_parts"]["additive"] == pytest.approx(math.hypot(0.04 * 2.34, 0.004))


def test_low_light_flag_on_the_room_and_index_mapping():
    plan = make_plan()
    plan.rooms[1].flags.append("low_light")
    quality = {"scale_log_sigma": 0.0, "scenes": [{"n_images": 8}, {"n_images": 2}]}
    plan, _, rec = run("photo", plan=plan, quality=quality)
    assert rec["room_factors"]["R1"]["factor"] == 1.0
    assert rec["room_factors"]["R2"]["factor"] == pytest.approx(1.3 * 1.5)


def test_scene_named_for_one_room_does_not_apply_to_another():
    quality = {"scale_log_sigma": 0.0, "scenes": [{"room_hint": "01 living", "low_light": True}]}
    _, _, rec = run("photo", quality=quality)
    assert rec["room_factors"]["R1"]["reasons"] == ["low_light"]
    assert rec["room_factors"]["R2"]["reasons"] == []
    _, _, rec = run("video", quality={"scale_log_sigma": 0.0, "scenes": [{"low_light": True}]})
    assert rec["room_factors"]["R1"]["reasons"] == rec["room_factors"]["R2"]["reasons"] == ["low_light"]


def test_photo_count_falls_back_to_room_views_and_capture_flags_merge():
    plan = make_plan()
    plan.rooms[0].view_ids = ["a", "b", "c"]
    plan.rooms[1].view_ids = ["d", "e", "f", "g", "h"]
    quality = {"scale_log_sigma": 0.0, "flags": ["low_light"], "scenes": [{"flags": ["x"]}, {"flags": ["y"]}]}
    _, _, rec = run("photo", plan=plan, quality=quality)
    assert rec["room_factors"]["R1"]["reasons"] == ["low_light", "few_photos"]
    assert rec["room_factors"]["R2"]["reasons"] == ["low_light"]


def test_missing_footprint_does_not_crash():
    plan = make_plan()
    plan.footprint_area = None
    annotate(plan, [], tier="lidar", quality={}, calibration=NO_CAL)
    assert plan.extent_x.lo < 7.1 < plan.extent_x.hi


def test_lo_is_clipped_at_zero():
    _, damage, _ = run("photo")
    crack = damage[2]
    assert crack.area.value - Z * crack.area.evidence["sigma"] < 0
    assert crack.area.lo == 0.0 and crack.area.hi > crack.area.value


def test_q_scales_every_interval():
    base = measurements(*run("video")[:2])
    plan, damage, rec = run("video", calibration={"tiers": {"video": {"q": 2.0, "status": "calibrated"}}})
    assert rec["q"] == 2.0 and rec["status"] == "calibrated"
    for name, m in measurements(plan, damage).items():
        assert m.hi - m.value == pytest.approx(2.0 * (base[name].hi - base[name].value)), name
        assert m.evidence["q"] == 2.0


def test_reads_calibration_file(tmp_path):
    path = tmp_path / "calibration.yaml"
    write_calibration({"level": 0.9, "tiers": {"lidar": {"q": 1.5, "status": "calibrated", "n_rooms": 12}}}, path)
    rec = annotate(make_plan(), make_damage(), tier="lidar", quality={}, calibration=path)
    assert rec["q"] == 1.5 and rec["status"] == "calibrated" and rec["calibration"]["n_rooms"] == 12


def test_missing_calibration_file_means_prior(tmp_path):
    rec = annotate(make_plan(), [], tier="video", quality=None, calibration=tmp_path / "absent.yaml")
    assert rec["q"] == 1.0 and rec["status"] == "prior"


def test_odd_input_degrades_without_crashing():
    plan = make_plan()
    plan.rooms[0].walls[0].length.value = float("nan")
    plan.rooms[0].walls[1].observed_fraction = None
    plan.rooms[1].walls = []  # room with a polygon but no walls
    rec = annotate(plan, make_damage(), tier="drone", quality={"scale_log_sigma": "n/a", "scenes": [None, 3]},
                   calibration=NO_CAL)
    assert rec["model_tier"] == "photo"
    assert "uncertainty_unknown_tier:drone" in plan.flags
    w = plan.rooms[0].walls[0]
    assert w.length.lo is None and w.length.evidence["sigma"] is None
    assert plan.rooms[1].floor_area.lo < 9.0 < plan.rooms[1].floor_area.hi


def test_shipped_files_match_the_spec():
    pri = load_priors()
    assert pri["tiers"]["lidar"]["additive"] == {"length": 0.008, "height": 0.006, "opening": 0.012}
    assert pri["tiers"]["video"]["additive"] == {"length": 0.025, "height": 0.02, "opening": 0.03}
    assert pri["tiers"]["photo"]["additive"] == {"length": 0.04, "height": 0.035, "opening": 0.04}
    # one MapAnything metric estimate is good to about 0.08 in log scale, video chunks share that error
    assert [pri["tiers"][t]["scale_floor"] for t in ("lidar", "video", "photo")] == [0.003, 0.08, 0.08]
    assert [pri["tiers"][t]["vertical"] for t in ("lidar", "video", "photo")] == [0.0, 0.1, 0.1]
    st = pri["structure"]
    keys = ("min_face_points", "unobserved_end", "short_wall_m", "step_wall_m", "step_observed")
    assert [st[k] for k in keys] == [30, 0.15, 0.5, 1.0, 0.6]


# evidence ---------------------------------------------------------------------------------------------------


def one_room(room) -> Plan:
    return Plan(rooms=[room], adjacency=[], footprint_area=M(room.floor_area.value, "area", "m2"),
                extent_x=M(4.0), extent_y=M(3.0))


def notched_room(depth: float = 0.2):
    """4 x 3 m room whose bottom wall steps in by depth between x = 1.5 and 2.5, as a counter face would."""
    poly = np.array([[0, 0], [1.5, 0], [1.5, depth], [2.5, depth], [2.5, 0], [4, 0], [4, 3], [0, 3]], float)
    return poly_room("R1", "living", poly)


def symmetric(m: Measurement) -> bool:
    return m.hi - m.value == pytest.approx(m.value - m.lo) and "upper" not in m.evidence["sigma_parts"]


def unobserve(wall) -> None:
    wall.flags = ["wall_unobserved"]
    wall.length.evidence["n_points"] = 0
    wall.observed_fraction = 0.0


@pytest.mark.parametrize("tier", TIERS)
def test_clean_rectangle_keeps_symmetric_intervals(tier):
    plan = one_room(rect_room("R1", "living", 0, 0, 4, 3))
    rec = annotate(plan, [], tier=tier, quality={}, calibration=NO_CAL)
    room = plan.rooms[0]
    for m in [w.length for w in room.walls] + [room.floor_area, room.ceiling_height, room.perimeter]:
        assert symmetric(m)
        assert "widened" not in m.evidence
    assert rec["room_factors"]["R1"]["structure"] == [] and rec["capture_reasons"] == []


def test_wall_ending_at_a_step_may_span_the_room():
    plan = one_room(notched_room())
    annotate(plan, [], tier="lidar", quality={}, calibration=NO_CAL)
    walls = {w.id: w.length for w in plan.rooms[0].walls}
    # W1 (1.5 m) ends at a 0.2 m step: if the notch is an artefact, W1, W3 and W5 are one 4 m wall
    w1 = walls["R1-W1"]
    assert w1.hi > 4.0 and w1.value - w1.lo == pytest.approx(Z * w1.evidence["sigma"])
    assert w1.hi - 4.0 < 0.05  # the far end keeps the LiDAR precision
    assert {"wall_end_step", "wall_fragment"} <= set(w1.evidence["widened"])
    assert walls["R1-W3"].hi > 4.0 and walls["R1-W5"].hi > 4.0
    # a step that is an artefact vanishes rather than grows, and the walls away from the notch are untouched
    assert symmetric(walls["R1-W2"]) and walls["R1-W2"].hi < 0.25
    for wid in ("R1-W6", "R1-W7", "R1-W8"):
        assert symmetric(walls[wid]), wid
    area = plan.rooms[0].floor_area
    assert area.value == pytest.approx(11.8) and area.hi > 12.0
    assert "room_fragment" in area.evidence["widened"]


def test_low_short_face_counts_as_a_step():
    room = notched_room(depth=0.8)  # 0.8 m returns: a full-height face this short is taken as a real wall
    plan = one_room(room)
    annotate(plan, [], tier="lidar", quality={}, calibration=NO_CAL)
    assert symmetric(room.walls[0].length)
    for k in (1, 3):  # ... a face covering little of its floor-to-ceiling plane is a counter or wardrobe side
        room.walls[k].observed_fraction = 0.35
    annotate(plan, [], tier="lidar", quality={}, calibration=NO_CAL)
    assert {"wall_end_step", "wall_fragment"} <= set(room.walls[0].length.evidence["widened"])


def test_unobserved_end_wall_adds_a_share_of_the_extent():
    room = rect_room("R1", "living", 0, 0, 4, 3)
    unobserve(room.walls[1])  # the right wall, 3 m
    plan = one_room(room)
    annotate(plan, [], tier="video", quality={}, calibration=NO_CAL)
    bottom, right, top, left = (w.length for w in room.walls)
    for m in (bottom, top):  # both end at the unobserved wall
        assert m.evidence["sigma_parts"]["end_position"] == pytest.approx(0.15 * 4.0)
        assert "wall_end_unobserved" in m.evidence["widened"]
        assert symmetric(m)  # they already span the room, so nothing one-sided
    assert "end_position" not in left.evidence["sigma_parts"]
    assert "wall_unobserved" in right.evidence["widened"]
    # the area carries the unobserved wall's position: its 3 m length times 0.15 of the 4 m extent across it
    assert room.floor_area.evidence["sigma_parts"]["unobserved_walls"] == pytest.approx(3.0 * 0.15 * 4.0)


def test_fragment_with_an_unobserved_end_reaches_the_room_extent():
    poly = np.array([[0, 0], [1.5, 0], [1.5, 0.6], [4, 0.6], [4, 3], [0, 3]], float)  # an L with a 0.6 m step
    room = poly_room("R1", "living", poly)
    unobserve(room.walls[1])
    plan = one_room(room)
    annotate(plan, [], tier="video", quality={}, calibration=NO_CAL)
    w1 = room.walls[0].length  # 1.5 m, ends at the unobserved step; the room is 4 m wide
    assert w1.hi > 4.0 and {"wall_end_unobserved", "wall_fragment"} <= set(w1.evidence["widened"])


def test_end_wall_evidence_inflates_the_length():
    a = 0.025
    room = rect_room("R1", "living", 0, 0, 4, 3, observed=(0.9, 0.2, 0.9, 0.9))
    plan = one_room(room)
    annotate(plan, [], tier="video", quality={}, calibration=NO_CAL)
    bottom, left = room.walls[0].length, room.walls[3].length
    expect = math.sqrt(0.5 * a * a * (2.0 ** 2 - 1))  # the right wall's 0.2 observed share doubles its term
    assert bottom.evidence["sigma_parts"]["end_noise"] == pytest.approx(expect)
    assert "end_noise" not in left.evidence["sigma_parts"]
    # a face spread well beyond the capture's surface noise (two faces fitted as one) adds its excess
    room = rect_room("R1", "living", 0, 0, 4, 3)
    room.walls[1].length.evidence.update(fit_rms=0.05, noise_sigma=0.02)
    plan = one_room(room)
    annotate(plan, [], tier="lidar", quality={}, calibration=NO_CAL)
    assert room.walls[0].length.evidence["sigma_parts"]["end_noise"] == pytest.approx(0.03)


def test_chunk_scale_spread_widens_the_shared_scale():
    scales = (0.8, 1.0, 1.0, 1.02)
    plan = make_plan()
    plan.meta["drift"] = {"chunks": [{"world_scale": w, "align_method": m}
                                     for w, m in zip(scales, ("reference", "poses", "points", "points"))]}
    rec = annotate(plan, [], tier="video", quality={"scale_log_sigma": 0.05}, calibration=NO_CAL)
    logs = np.log(scales)
    spread = float(np.sqrt(np.mean((logs - np.median(logs)) ** 2)))
    assert rec["scale_sigma"] == pytest.approx(math.hypot(0.08, spread))
    assert rec["capture_reasons"] == ["chunk_scale_spread", "chunk_align_fallback"]
    w = plan.rooms[1].walls[0].length
    assert w.evidence["sigma_parts"]["scale"] == pytest.approx(3.0 * math.hypot(0.08, spread))
    # chunks that agree leave the floor alone
    plan = make_plan()
    plan.meta["drift"] = {"chunks": [{"world_scale": 1.0}, {"world_scale": 1.01}, {"world_scale": 0.99}]}
    rec = annotate(plan, [], tier="video", quality={}, calibration=NO_CAL)
    assert rec["capture_reasons"] == [] and rec["scale_sigma"] < 1.1 * 0.08


LOOP = {"attempted": True, "accepted": False, "overlap": 0.5, "inlier_frac": 0.8, "error_rot_deg": 2.7,
        "error_log_scale": 0.27, "error_trans_m": 0.31}


@pytest.mark.parametrize("change, used", [({}, True), ({"overlap": 0.0}, False),
                                          ({"error_rot_deg": 65.0}, False), ({"accepted": True}, False)])
def test_rejected_loop_closure_counts_only_when_it_registered_well(change, used):
    plan = make_plan()
    plan.meta["drift"] = {"loop_closure": {**LOOP, **change}}
    rec = annotate(plan, [], tier="video", quality={}, calibration=NO_CAL)
    w = plan.rooms[1].walls[0].length
    if used:
        assert rec["scale_sigma"] == pytest.approx(math.hypot(0.08, 0.135))
        assert w.evidence["sigma_parts"]["drift"] == pytest.approx(0.155)
        assert rec["capture_reasons"] == ["loop_closure_rejected"]
    else:
        assert rec["scale_sigma"] == pytest.approx(0.08) and "drift" not in w.evidence["sigma_parts"]
        assert rec["capture_reasons"] == []


def test_lidar_drift_record_leaves_the_scale_alone():
    plan = make_plan()
    plan.meta["drift"] = {"enabled": True, "segments": 11, "loop_closures_accepted": 2}
    rec = annotate(plan, [], tier="lidar", quality={"scale_log_sigma": 0.003}, calibration=NO_CAL)
    assert rec["scale_sigma"] == pytest.approx(0.003) and rec["capture_reasons"] == []




# calibration ------------------------------------------------------------------------------------------------


def synthetic_records(n_rooms: int, ratio: float, q_reported: float = 1.0, seed: int = 0, per_room: int = 4):
    """Errors with sigma = ratio * model sigma, reported half-widths z * q * model sigma."""
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n_rooms):
        for k in range(per_room):
            sigma = rng.uniform(0.01, 0.06)
            recs.append({"tier": "video", "kind": ("wall_length", "ceiling_height")[k % 2], "room": f"home/r{i}",
                         "err": rng.normal(0.0, ratio * sigma), "half_width": Z * q_reported * sigma})
    return recs


def test_conformal_level():
    assert conformal_level(9) == 1.0
    assert conformal_level(8) > 1.0
    assert conformal_level(19) == pytest.approx(18 / 19)
    assert conformal_level(0) == math.inf


def test_room_quantile_weights_rooms_equally():
    recs = [as_record({"tier": "t", "kind": "k", "room": f"r{i}", "err": float(i + 1), "half_width": 1.0})
            for i in range(19)]
    assert room_quantile(recs, conformal_level(19)) == 18.0
    # One room with many records must not outweigh the others.
    heavy = recs + [as_record({"tier": "t", "kind": "k", "room": "r0", "err": 100.0, "half_width": 1.0})] * 50
    assert room_quantile(heavy, 0.5) < 15.0


def test_fit_q_recovers_the_multiplier(tmp_path):
    path = tmp_path / "calibration.yaml"
    table = fit_q(synthetic_records(300, ratio=1.7), path=path)
    entry = table["tiers"]["video"]
    assert entry["status"] == "calibrated" and entry["n_rooms"] == 300 and entry["n_records"] == 1200
    assert entry["q"] == pytest.approx(1.7, rel=0.06)
    on_disk = yaml.safe_load(path.read_text())
    assert on_disk["tiers"]["video"]["q"] == entry["q"]
    assert on_disk["tiers"]["photo"] == {"q": 1.0, "status": "prior"}


def test_fit_q_multiplies_the_q_in_force(tmp_path):
    path = tmp_path / "calibration.yaml"
    write_calibration({"level": 0.9, "tiers": {"video": {"q": 2.0, "status": "prior"}}}, path)
    table = fit_q(synthetic_records(300, ratio=1.7, q_reported=2.0, seed=1), path=path)
    assert table["tiers"]["video"]["q"] == pytest.approx(1.7, rel=0.06)
    assert table["tiers"]["video"]["conformal_quantile"] == pytest.approx(0.85, rel=0.06)


def test_fit_q_with_few_rooms_keeps_the_prior(tmp_path):
    path = tmp_path / "calibration.yaml"
    table = fit_q(synthetic_records(5, ratio=1.7), path=path)
    entry = table["tiers"]["video"]
    assert entry["status"] == "prior" and entry["q"] == 1.0 and entry["n_rooms"] == 5
    assert entry["conformal_quantile"] is None and entry["empirical_quantile"] > 0
    assert set(entry["by_kind"]) == {"wall_length", "ceiling_height"}


def test_fit_q_without_writing(tmp_path):
    path = tmp_path / "calibration.yaml"
    fit_q(synthetic_records(20, ratio=1.0), path=path, write=False)
    assert not path.exists()


def test_loro_coverage():
    report = loro_coverage(synthetic_records(60, ratio=1.7, seed=2))["video"]
    assert report["mode"] == "loro_conformal" and report["n_rooms"] == 60
    assert report["ci"][0] <= 0.9 <= report["ci"][1] and report["contains_nominal"]
    assert 0.85 < report["coverage"] < 0.97
    small = loro_coverage(synthetic_records(6, ratio=1.7, seed=3))["video"]
    assert small["mode"] == "as_reported"
    assert small["coverage"] < 0.85  # the prior intervals are too narrow for errors 1.7x the model sigma
    assert set(small["by_kind"]) == {"wall_length", "ceiling_height"}


def test_one_sided_interval_is_scored_on_the_side_of_the_truth():
    base = {"tier": "video", "kind": "wall_length", "room": "home/r1", "half_width": 0.55, "pred": 1.0,
            "lo": 0.9, "hi": 2.0}  # widened upward only: the wall may be a fragment
    above = as_record({**base, "err": -0.5})  # truth 1.5, inside the upper side
    below = as_record({**base, "err": 0.2})  # truth 0.8, past the lower side
    assert above.score == pytest.approx(0.5) and below.score == pytest.approx(2.0)
    no_bounds = {k: v for k, v in base.items() if k not in ("lo", "hi")} | {"err": 0.2}
    assert as_record(no_bounds).score == pytest.approx(0.2 / 0.55)
    # the miss below is not confident garbage by the benchmark's rule, 0.2 against a half-width of 0.55
    report = loro_coverage([above, below])["video"]
    assert report["covered"] == 1 and report["confident_garbage"] == 0
