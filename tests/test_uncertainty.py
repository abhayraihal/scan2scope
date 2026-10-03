"""Error model (uncertainty.annotate) and split-conformal calibration (uncertainty.calibrate)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import yaml
from _output_plan import make_damage, make_plan

from scan2scope.types import TIERS, Measurement
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
        assert widths["photo"][name] > widths["video"][name] > widths["lidar"][name], name


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
    assert door.width.evidence["sigma"] == pytest.approx(math.hypot(0.8 * 0.03, 0.03))
    assert plan.rooms[0].ceiling_height.evidence["sigma"] == pytest.approx(math.hypot(2.5 * 0.03, 0.02))


def test_scale_term_uses_floor_or_capture_estimate():
    for scale, expect in ((0.01, 0.03), (0.08, 0.08)):
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
    s = 0.03
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
    assert [pri["tiers"][t]["scale_floor"] for t in ("lidar", "video", "photo")] == [0.003, 0.03, 0.05]


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
