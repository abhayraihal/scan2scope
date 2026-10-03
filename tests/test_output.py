"""result.json writer, schema validation, drawings and the console summary."""

from __future__ import annotations

import copy
import io
import json

import numpy as np
import pytest
from _output_plan import (
    PROVENANCE,
    TIMING,
    M,
    make_damage,
    make_flags,
    make_info,
    make_plan,
    make_scope,
    poly_room,
)
from PIL import Image

from scan2scope.output import console, render, schema, writer
from scan2scope.types import TIERS, Adjacency, DamageRegion, LineItem, Measurement, Opening, Plan
from scan2scope.uncertainty import annotate


def build(tier: str = "video", plan: Plan | None = None, damage: list | None = None, scope: list | None = None,
          annotate_first: bool = True) -> dict:
    plan = plan or make_plan()
    damage = make_damage() if damage is None else damage
    if annotate_first:
        annotate(plan, damage, tier=tier, quality={"scale_log_sigma": 0.0, "scenes": [{}]}, calibration={"tiers": {}})
    return writer.build_result(make_info(tier), plan, damage, make_flags(), make_scope() if scope is None else scope,
                               TIMING, PROVENANCE)


@pytest.mark.parametrize("tier", TIERS)
def test_build_result_validates(tier):
    res = build(tier)
    schema.validate(res)
    assert schema.problems(res) == []
    assert json.loads(json.dumps(res, allow_nan=False)) == res
    assert res["property"]["flags"] == []
    assert res["capture"]["tier"] == tier


def test_result_content():
    res = build("video")
    r1, r2 = res["rooms"]
    assert [w["id"] for w in r1["walls"]] == ["R1-W1", "R1-W2", "R1-W3", "R1-W4"]
    assert r1["polygon"] == [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]
    assert r1["floor_area"]["unit"] == "m2" and r1["walls"][0]["length"]["unit"] == "m"
    assert r1["source_hint"] == "01 living" and r2["label"] == "kitchen"
    door, window = r1["openings"]
    assert door["sill"] is None and door["connects_to"] == "R2" and door["center"] == [4.0, 1.4]
    assert window["type"] == "window" and window["sill"]["value"] == 0.9

    surfaces = {s["id"]: s for s in r1["surfaces"]}
    assert set(surfaces) == {"R1-W1", "R1-W2", "R1-W3", "R1-W4", "R1-FLOOR", "R1-CEIL"}
    assert surfaces["R1-W2"]["area"]["value"] == pytest.approx(3.0 * 2.5 - 0.8 * 2.0)
    assert surfaces["R1-W3"]["area"]["value"] == pytest.approx(4.0 * 2.5 - 1.2 * 1.1)
    w3 = surfaces["R1-W3"]["area"]
    length, height = r1["walls"][2]["length"], r1["walls"][2]["height"]
    w, h = window["width"], window["height"]
    assert w3["lo"] == pytest.approx(length["lo"] * height["lo"] - w["hi"] * h["hi"], abs=1e-3)
    assert w3["hi"] == pytest.approx(length["hi"] * height["hi"] - w["lo"] * h["lo"], abs=1e-3)
    assert surfaces["R1-FLOOR"]["area"] == r1["floor_area"] and surfaces["R1-CEIL"]["kind"] == "ceiling"
    assert surfaces["R1-W1"]["wall_id"] == "R1-W1" and surfaces["R1-FLOOR"]["wall_id"] is None

    prop = res["property"]
    assert prop["footprint_area"]["value"] == 21.0 and prop["footprint_area"]["unit"] == "m2"
    assert prop["adjacency"] == [{"room_a": "R1", "room_b": "R2", "opening_a": "R1-O1", "opening_b": "R2-O1",
                                  "confidence": 0.9, "source": "shared_frame"}]
    assert prop["drift_correction"]["enabled"] is True and prop["stitch"] is None
    assert [d["class"] for d in res["damage"]] == ["water_stain", "water_stain", "crack"]
    assert res["damage"][2]["length"]["value"] == 0.55 and res["damage"][0]["length"] is None
    assert res["damage"][1]["u_range"] == [5.0, 5.6]
    assert res["concealed_damage_flags"][0]["inputs"] == {"stain_area_m2": 0.25}
    assert res["scope"][1]["quantity"] == {"value": 3.0, "lo": 2.4, "hi": 3.7, "unit": "SF"}
    interval = res["conventions"]["interval"]
    assert interval["level"] == 0.9 and interval["calibration"] == "prior" and interval["q"] == 1.0
    assert "q = 1.00, prior value" in interval["method"] and "s = 0.030" in interval["method"]
    defs = res["conventions"]["definitions"]
    assert "face to face at 1 m height" in defs["wall_length"]
    assert defs["footprint"].startswith("sum of the room floor areas")
    assert res["capture"]["input_stats"] == {"n_frames": 240, "fps": 30.0}
    assert res["timing"] == TIMING and res["provenance"]["cache_mode"] == "live"


def test_lo_above_value_is_rejected():
    res = build()
    res["rooms"][0]["walls"][0]["length"]["lo"] = 4.5
    with pytest.raises(ValueError, match=r"\$\.rooms\[0\]\.walls\[0\]\.length: lo 4\.5 > value 4\.0"):
        schema.validate(res)


def test_every_problem_is_listed():
    res = build()
    res["rooms"][1]["walls"][0]["id"] = "R1-W1"
    res["rooms"][0]["openings"][0]["wall_id"] = "R1-W9"
    res["damage"][0]["surface_id"] = "R2-W9"
    res["scope"][0]["surface_id"] = "R7-FLOOR"
    res["scope"][1]["flag_ids"] = ["F9"]
    res["property"]["adjacency"][0]["room_b"] = "R9"
    res["rooms"][0]["ceiling_height"]["hi"] = 1.0
    res["rooms"][0]["floor_area"]["value"] = float("nan")
    res["timing"]["total_s"] = -1
    res["damage"][2]["id"] = "D1"
    del res["rooms"][1]["walls"][1]["observed_fraction"]
    with pytest.raises(ValueError) as exc:
        schema.validate(res)
    msg = str(exc.value)
    for needle in ("duplicate wall id 'R1-W1'", "'R1-W9' is not a wall of room 'R1'", "unknown surface 'R2-W9'",
                   "unknown surface 'R7-FLOOR'", "unknown flag 'F9'", "room_b: unknown room 'R9'",
                   "value 2.5 > hi 1.0", "$.timing.total_s", "non-finite number nan", "duplicate damage id 'D1'",
                   "'observed_fraction' is a required property"):
        assert needle in msg, needle
    assert msg.startswith("result failed validation with 11 problem(s)")


def test_schema_file_is_found_from_the_repo():
    assert schema.schema_path().name == "scan2scope.schema.json"


def test_unannotated_plan_is_flagged_not_rejected():
    res = build(annotate_first=False)
    schema.validate(res)
    assert any(f.startswith("writer:no_interval:") for f in res["property"]["flags"])
    assert res["conventions"]["interval"]["method"].startswith("No error model")
    assert "q" not in res["conventions"]["interval"]


def test_writer_repairs_odd_input_and_stays_valid():
    plan, damage = make_plan(), make_damage()
    annotate(plan, damage, tier="photo", quality={}, calibration={"tiers": {}})
    plan.rooms[0].walls[0].length.value = float("nan")
    plan.rooms[0].walls[1].height.hi = float("inf")
    plan.rooms[0].openings[1].type = "Skylight"
    plan.rooms[1].openings[0].connects_to = "R9"
    plan.rooms[1].openings.append(Opening("R2-O2", "R2", "R2-W9", "door", M(0.1), M(0.8), M(2.0)))
    plan.rooms[1].polygon = np.zeros((0, 2))
    plan.adjacency.append(Adjacency("R1", "R5", None, None, 0.4, "door_match"))
    plan.meta["drift"] = False
    damage.append(DamageRegion("D4", "R2", "R2-W7", "mold", 0.3, M(0.1, unit="m2"), M(0.3), M(0.3), (0, 0.3), (0, 0.3)))
    damage.append(DamageRegion("D5", "R2", "R1-FLOOR", "Peeling Paint", 1.4, M(0.1, unit="m2"), M(0.3), M(0.3),
                               (1.0, 1.3), (1.0, 1.3)))
    scope = make_scope() + [
        LineItem("L3", "R2", "R2-W1", "DRY", "1/2", "?", "unknown activity", Measurement(1.0, 0.8, 1.2), "SF"),
        LineItem("L4", "R9", "R2-W1", "PNT", "S", "+", "Paint", Measurement(1.0, 0.8, 1.2), "sq ft", damage_ids=["D4"]),
    ]
    res = writer.build_result(make_info("photo"), plan, damage, make_flags(), scope,
                              {"total_s": float("inf"), "stages": {"a": -1, "b": "x"}},
                              {"cache_mode": "weird", "models": [{"name": "m"}]})
    schema.validate(res)
    flags = res["property"]["flags"]
    for needle in ("writer:nonfinite_value:R1-W1.length", "writer:nonfinite_hi:R1-W2.height",
                   "writer:opening_type:R1-O2:Skylight->opening", "writer:opening_connects_to:R2-O1:unknown_room:R9",
                   "writer:dropped_opening:R2-O2:unknown_wall:R2-W9", "writer:degenerate_polygon:R2",
                   "writer:dropped_adjacency:R1-R5:unknown_room", "writer:dropped_damage:D4:unknown_surface:R2-W7",
                   "writer:damage_room:D5:R2->R1", "writer:dropped_scope:L3:activity:?", "writer:room:L4:R9->R2",
                   "writer:dropped_refs:L4.damage_ids:D4", "writer:no_interval:3"):
        assert needle in flags, needle
    r1, r2 = res["rooms"]
    assert r1["walls"][0]["length"]["lo"] <= r1["walls"][0]["length"]["value"] <= r1["walls"][0]["length"]["hi"]
    assert r2["polygon"] == [[4.1, 0.0], [7.1, 0.0], [7.1, 3.0], [4.1, 3.0]]
    d5 = res["damage"][-1]
    assert d5["class"] == "peeling_paint" and d5["score"] == 1.0 and d5["room_id"] == "R1"
    assert res["scope"][-1]["unit"] == "SF" and res["scope"][-1]["damage_ids"] == []
    assert res["property"]["drift_correction"] == {"enabled": False}
    assert res["timing"] == {"total_s": 0.0, "stages": {"a": 0.0, "b": 0.0}}
    prov = res["provenance"]
    assert prov["cache_mode"] == "none" and prov["models"] == [{"name": "m", "revision": "unknown", "license": "unknown"}]
    assert prov["device"] == "unknown" and prov["git_commit"] is None


def test_polygon_with_a_z_column_keeps_plan_coordinates():
    plan = make_plan()
    plan.rooms[0].polygon = np.c_[plan.rooms[0].polygon, np.full(4, 0.05)]
    res = build(plan=plan)
    assert res["rooms"][0]["polygon"] == [[0.0, 0.0], [4.0, 0.0], [4.0, 3.0], [0.0, 3.0]]


def test_duplicate_ids_are_renamed():
    plan, damage = make_plan(), make_damage()
    damage[1].id = "D1"
    res = build(plan=plan, damage=damage)
    schema.validate(res)
    assert [d["id"] for d in res["damage"]] == ["D1", "D1~2", "D3"]
    assert "writer:duplicate_damage_id:D1->D1~2" in res["property"]["flags"]


def test_render_writes_plan_and_room_sheets(tmp_path):
    res = build()
    info = render.render_all(res, tmp_path)
    assert info["errors"] == []
    for name in ("plan.svg", "plan.png", "rooms/R1.svg", "rooms/R2.svg"):
        assert (tmp_path / name).stat().st_size > 1000, name
    svg = (tmp_path / "plan.svg").read_text()
    for text in ("W1 4.00 m ±0.20", "R1 living", "R2 kitchen", "12.00 m² [10.60, 13.40]", "ceiling 2.50 m ±0.13",
                 "D1 water stain", "D2 water stain (ceiling)", "O1 0.80", "1 m", "tier video",
                 "footprint 21.00 [18.74, 23.26] m²", "drift correction: on"):
        assert text in svg, text
    with Image.open(tmp_path / "plan.png") as im:
        assert im.format == "PNG" and im.info["dpi"][0] == pytest.approx(150, abs=1)
    sheet = (tmp_path / "rooms" / "R1.svg").read_text()
    assert "R1-W3" in sheet and "R2 kitchen" not in sheet


def test_console_ids_match_the_render(tmp_path):
    res = build()
    render.render_all(res, tmp_path)
    svg = (tmp_path / "plan.svg").read_text()
    out = console.format_summary(res)
    for room in res["rooms"]:
        assert f"{room['id']} {room['label']}" in svg and f"ROOM {room['id']}  {room['label']}" in out
        for w in room["walls"]:
            assert f"{w['id'].split('-')[1]} {w['length']['value']:.2f} m" in svg
            lv = w["length"]
            assert f"{w['id']}  {lv['value']:.3f} [{lv['lo']:.3f}, {lv['hi']:.3f}]" in out
        for o in room["openings"]:
            assert f"{o['id'].split('-')[1]} {o['width']['value']:.2f}" in svg and o["id"] in out
    for d in res["damage"]:
        assert f"{d['id']} {d['class'].replace('_', ' ')}" in svg and f"{d['id']}  {d['room_id']}" in out


def test_console_summary_sections():
    res = build()
    buf = io.StringIO()
    console.print_summary(res, file=buf)
    out = buf.getvalue()
    for needle in ("footprint 21.00 [18.74, 23.26] m2", "ceiling height 2.500 [2.372, 2.628] m", "R1-O2    window",
                   "F1  ceiling_stain_moisture_above  medium", "  DRY", "  PNT", "L2  R2    R2-CEIL  +", "total        12.50",
                   "drift correction: on (loop closure, plane anchoring", "R1 - R2  R1-O1 / R2-O1  shared_frame"):
        assert needle in out, needle
    assert out.isascii()


def test_render_odd_results(tmp_path):
    l_room = poly_room("R1", "hall", np.array([[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], float))
    c, s = np.cos(0.4), np.sin(0.4)
    rot = np.array([[c, -s], [s, c]])
    tilted = poly_room("R2", "study", (np.array([[0, 0], [3, 0], [3, 2.5], [0, 2.5]], float) @ rot.T) + [6.0, 0.0])
    tilted.openings = [Opening("R2-O1", "R2", "R2-W1", "door", M(0.5), M(0.9), M(2.0))]
    plan = Plan([l_room, tilted], [], M(19.5, unit="m2"), M(9.3), M(4.2), flags=["placement_uncertain:R2"])
    res = build("photo", plan=plan, damage=[], scope=[])
    res["concealed_damage_flags"] = []
    schema.validate(res)
    info = render.render_all(res, tmp_path / "odd")
    assert info["errors"] == []
    assert "placement uncertain" in (tmp_path / "odd" / "plan.svg").read_text()

    empty = copy.deepcopy(res)
    empty["rooms"] = []
    info = render.render_all(empty, tmp_path / "empty")
    assert info["errors"] == [] and (tmp_path / "empty" / "plan.png").stat().st_size > 0
    assert "no rooms in this result" in (tmp_path / "empty" / "plan.svg").read_text()

    broken = copy.deepcopy(res)
    broken["rooms"][0]["polygon"] = [[0, 0], [1, 1]]
    broken["rooms"][0]["walls"][0]["start"] = None
    broken["rooms"][1]["openings"][0]["width"] = None
    info = render.render_all(broken, tmp_path / "broken")
    assert (tmp_path / "broken" / "plan.svg").stat().st_size > 0 and (tmp_path / "broken" / "rooms" / "R1.svg").exists()
    assert console.format_summary(broken)
