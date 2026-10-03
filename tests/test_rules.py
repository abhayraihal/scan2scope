"""Concealed-damage rules: each rule fires on a synthetic plan and damage, and stays quiet just outside it."""

from __future__ import annotations

import json

import numpy as np
import pytest

from scan2scope.rules import evaluate, load_rules
from scan2scope.semantics import SceneObject

from semantics_fixtures import add_opening, make_plan, rect_room, region


@pytest.fixture
def plan():
    room = rect_room()  # 4 x 3 m, 2.5 m high; W1 south, W2 east, W3 north (runs from x=4 to x=0), W4 west
    add_opening(room, 1, "door", offset=1.0, width=0.9, height=2.0)  # R1-O1 on W1, u 1.0 .. 1.9
    add_opening(room, 3, "window", offset=1.0, width=1.2, height=1.2, sill=0.9)  # R1-O2 on W3, u 1.0 .. 2.2
    return make_plan(room)


def obj(oid, cls, xy, z=(0.0, 0.9), half=0.25):
    xy = np.asarray(xy, float)
    return SceneObject(id=oid, cls=cls, room_id="R1", xy=xy, z_range=z, score=0.7,
                       x_range=(xy[0] - half, xy[0] + half), y_range=(xy[1] - half, xy[1] + half))


def fired(flags, rule_id):
    return [f for f in flags if f.rule_id == rule_id]


def test_rule_file_shape():
    rules = load_rules()
    assert 8 <= len(rules) <= 12
    ids = [r["id"] for r in rules]
    assert len(set(ids)) == len(ids) and all(i.startswith("R-") for i in ids)
    for r in rules:
        assert r["severity"] in ("low", "medium", "high")
        assert r["title"] and r["recommendation"] and r["when"].get("classes")
        assert any(s in r["basis"] for s in ("EPA-HOME", "EPA-SCHOOLS", "S500", "S520", "Project heuristic"))


def test_no_damage_no_flags(plan):
    assert evaluate(plan, [], []) == []


def test_ceiling_stain(plan):
    d = region("D1", "water_stain", "R1-CEIL", (1.0, 1.4), (1.0, 1.3))
    f = fired(evaluate(plan, [d], []), "R-CEIL-STAIN")
    assert len(f) == 1 and f[0].surface_ids == ["R1-CEIL"] and f[0].damage_ids == ["D1"]
    assert f[0].inputs["surface_kind"] == "ceiling" and f[0].inputs["area_m2"] == pytest.approx(0.12)
    assert not fired(evaluate(plan, [region("D1", "water_stain", "R1-W2", (1.0, 1.4), (1.0, 1.3))], []),
                     "R-CEIL-STAIN")
    assert not fired(evaluate(plan, [region("D1", "mold", "R1-CEIL", (1.0, 1.4), (1.0, 1.3))], []), "R-CEIL-STAIN")


def test_wall_stain_at_floor_line(plan):
    f = fired(evaluate(plan, [region("D1", "water_stain", "R1-W2", (0.5, 1.0), (0.05, 0.4))], []), "R-WALL-WICK")
    assert len(f) == 1 and f[0].inputs["bottom_above_floor_m"] == pytest.approx(0.05)
    assert f[0].severity == "high"
    assert not fired(evaluate(plan, [region("D1", "water_stain", "R1-W2", (0.5, 1.0), (0.4, 0.8))], []),
                     "R-WALL-WICK")


def test_stain_near_wet_fixture(plan):
    d = region("D1", "water_stain", "R1-W2", (1.0, 1.3), (0.2, 0.5))  # east wall x = 4, y 1.0 .. 1.3
    sink = obj("O1", "sink", (3.6, 1.2), z=(0.8, 0.95))
    f = fired(evaluate(plan, [d], [sink]), "R-FIXTURE-LEAK")
    assert len(f) == 1
    assert f[0].inputs["fixture"] == "O1" and f[0].inputs["fixture_class"] == "sink"
    assert f[0].inputs["fixture_distance_m"] == pytest.approx(np.hypot(0.15, 0.3), abs=1e-3)
    far = obj("O1", "sink", (1.0, 1.2), z=(0.8, 0.95))
    assert not fired(evaluate(plan, [d], [far]), "R-FIXTURE-LEAK")
    stove = obj("O1", "stove", (3.6, 1.2))
    assert not fired(evaluate(plan, [d], [stove]), "R-FIXTURE-LEAK")


def test_mold_containment_thresholds(plan):
    small = [region("D1", "mold", "R1-W1", (0.0, 0.5), (0.5, 1.5))]  # 0.5 m2 = 5.4 sq ft
    medium = small + [region("D2", "mold", "R1-W2", (0.0, 0.6), (0.5, 1.5))]  # 1.1 m2 = 11.8 sq ft
    large = [region("D1", "mold", "R1-W1", (0.0, 4.0), (0.0, 2.5))]  # 10 m2 = 107.6 sq ft
    flags = evaluate(plan, small, [])
    assert not fired(flags, "R-MOLD-LIMITED") and not fired(flags, "R-MOLD-FULL")
    f = fired(evaluate(plan, medium, []), "R-MOLD-LIMITED")
    assert len(f) == 1 and f[0].damage_ids == ["D1", "D2"] and f[0].surface_ids == ["R1-W1", "R1-W2"]
    assert f[0].inputs["total_area_sqft"] == pytest.approx(1.1 * 10.7639, abs=0.01)
    flags = evaluate(plan, large, [])
    assert fired(flags, "R-MOLD-FULL") and not fired(flags, "R-MOLD-LIMITED")


def test_any_mold_asks_for_hidden_growth_check_per_surface(plan):
    ds = [region("D1", "mold", "R1-W1", (0.0, 0.2), (0.5, 0.7)), region("D2", "mold", "R1-W1", (2.0, 2.2), (0.5, 0.7)),
          region("D3", "mold", "R1-CEIL", (1.0, 1.2), (1.0, 1.2))]
    f = fired(evaluate(plan, ds, []), "R-MOLD-HIDDEN")
    assert [x.damage_ids for x in f] == [["D1", "D2"], ["D3"]]
    assert not fired(evaluate(plan, [region("D1", "crack", "R1-W1", (0.0, 0.2), (0.5, 0.7))], []), "R-MOLD-HIDDEN")


def test_crack_from_opening_corner(plan):
    near = region("D1", "crack", "R1-W1", (1.9, 2.3), (2.05, 2.4), length=0.5,
                  evidence={"endpoints_uv_all": [[[1.95, 2.05], [2.3, 2.4]]]})
    f = fired(evaluate(plan, [near], []), "R-CRACK-OPENING")
    assert len(f) == 1 and f[0].inputs["opening"] == "R1-O1" and f[0].inputs["corner"] == "top_end"
    assert f[0].inputs["corner_distance_m"] == pytest.approx(np.hypot(0.05, 0.05), abs=1e-3)
    far = region("D1", "crack", "R1-W1", (3.0, 3.5), (1.0, 1.4), length=0.6,
                 evidence={"endpoints_uv_all": [[[3.0, 1.0], [3.5, 1.4]]]})
    assert not fired(evaluate(plan, [far], []), "R-CRACK-OPENING")
    other_wall = region("D1", "crack", "R1-W2", (1.9, 2.3), (2.05, 2.4), length=0.5)
    assert not fired(evaluate(plan, [other_wall], []), "R-CRACK-OPENING")


def test_crack_corner_rule_falls_back_to_the_uv_box(plan):
    d = region("D1", "crack", "R1-W3", (2.25, 2.6), (2.15, 2.4), length=0.4)  # window top corner at (2.2, 2.1)
    f = fired(evaluate(plan, [d], []), "R-CRACK-OPENING")
    assert len(f) == 1 and f[0].inputs["opening"] == "R1-O2"


def test_long_crack(plan):
    f = fired(evaluate(plan, [region("D1", "crack", "R1-W2", (0.5, 1.5), (0.5, 1.2), length=1.3)], []), "R-CRACK-LONG")
    assert len(f) == 1 and f[0].inputs["length_m"] == pytest.approx(1.3)
    assert not fired(evaluate(plan, [region("D1", "crack", "R1-W2", (0.5, 1.0), (0.5, 0.8), length=0.6)], []),
                     "R-CRACK-LONG")


def test_peeling_paint_and_hole(plan):
    flags = evaluate(plan, [region("D1", "peeling_paint", "R1-W2", (0.5, 1.0), (1.0, 1.5)),
                            region("D2", "hole", "R1-W4", (1.0, 1.1), (0.4, 0.5))], [])
    assert [f.damage_ids for f in fired(flags, "R-PEEL-MOISTURE")] == [["D1"]]
    assert [f.damage_ids for f in fired(flags, "R-HOLE-CAVITY")] == [["D2"]]
    quiet = evaluate(plan, [region("D1", "water_stain", "R1-W2", (0.5, 1.0), (1.0, 1.5))], [])
    assert not fired(quiet, "R-PEEL-MOISTURE") and not fired(quiet, "R-HOLE-CAVITY")


def test_stain_near_window(plan):
    below = region("D1", "water_stain", "R1-W3", (1.2, 1.6), (0.3, 0.6))  # window sill at 0.9
    f = fired(evaluate(plan, [below], []), "R-WINDOW-LEAK")
    assert len(f) == 1 and f[0].inputs["window"] == "R1-O2" and f[0].inputs["window_source"] == "plan_opening"
    assert f[0].inputs["window_distance_m"] == pytest.approx(0.3, abs=1e-3)
    away = region("D1", "water_stain", "R1-W3", (3.0, 3.4), (0.3, 0.6))
    assert not fired(evaluate(plan, [away], []), "R-WINDOW-LEAK")
    win = obj("O7", "window", (0.4, 3.0), z=(0.9, 2.1), half=0.05)  # a window the layout missed
    f2 = fired(evaluate(plan, [away], [win]), "R-WINDOW-LEAK")
    assert len(f2) == 1 and f2[0].inputs["window"] == "O7" and f2[0].inputs["window_source"] == "detected_object"


def test_flags_are_numbered_ordered_and_serialisable(plan):
    room2 = rect_room("R2", 4.1, 0.0, 3.0, 3.0)
    plan.rooms.append(room2)
    ds = [region("D1", "hole", "R2-W1", (1.0, 1.1), (0.4, 0.5), room_id="R2"),
          region("D2", "water_stain", "R1-CEIL", (1.0, 1.4), (1.0, 1.3)),
          region("D3", "water_stain", "R1-W2", (0.5, 1.0), (0.05, 0.4))]
    flags = evaluate(plan, ds, [])
    assert [f.id for f in flags] == [f"F{i}" for i in range(1, len(flags) + 1)]
    assert [f.room_id for f in flags] == sorted((f.room_id for f in flags), key=lambda r: r != "R1")
    for f in flags:
        json.dumps(f.inputs)
        assert f.basis and "\n" not in f.basis and f.recommendation


def test_unknown_surface_does_not_crash(plan):
    d = region("D1", "water_stain", "R9-W1", (0.0, 0.3), (0.0, 0.3), room_id="R9")
    flags = evaluate(plan, [d], [obj("O1", "sink", (0.1, 0.1))])
    assert fired(flags, "R-WALL-WICK") and not fired(flags, "R-FIXTURE-LEAK")
