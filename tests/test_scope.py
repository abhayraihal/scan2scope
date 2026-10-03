"""Scope line items: quantity formulas, unit conversion and intervals from annotated measurements."""

from __future__ import annotations

import pytest
from semantics_fixtures import add_opening, make_plan, rect_room, region

from scan2scope.rules import evaluate
from scan2scope.scope import generate, load_catalog
from scan2scope.scope.generator import M2_TO_SF, M_TO_LF
from scan2scope.semantics import SceneObject
from scan2scope.types import ConcealedFlag

SF, LF = 10.7639, 3.28084


@pytest.fixture
def plan():
    room = rect_room(rel=0.05)  # W1 4.0 [3.8, 4.2], W2 3.0 [2.85, 3.15], height 2.5 [2.375, 2.625]
    add_opening(room, 1, "door", offset=1.0, width=0.9, height=2.0, rel=0.05)
    return make_plan(room)


def items_by(items, catalog_selector):
    return [i for i in items if (i.category, i.selector) == catalog_selector]


def q(item):
    return item.quantity.value, item.quantity.lo, item.quantity.hi


def test_conversion_constants():
    assert M2_TO_SF == SF and M_TO_LF == LF
    cat = load_catalog()
    assert cat["conversions"] == {"m2_to_sf": SF, "m_to_lf": LF}
    codes = {e["category"] for e in cat["damage"] + cat["flags"]}
    assert codes == {"DRY", "PNT", "WTR", "CLN", "HMR", "INS", "DMO"}


def test_stain_seal_area_plus_margin_and_whole_wall_repaint(plan):
    d = region("D1", "water_stain", "R1-W2", (1.0, 1.5), (1.0, 1.4), rel=0.1)
    items = generate(plan, [d], [])
    (seal,) = items_by(items, ("PNT", "SEAL"))
    assert seal.unit == "SF" and seal.quantity.unit == "SF" and seal.activity == "+"
    assert q(seal) == pytest.approx((0.8 * 0.7 * SF, 0.75 * 0.66 * SF, 0.85 * 0.74 * SF))
    (paint,) = items_by(items, ("PNT", "P"))
    assert q(paint) == pytest.approx((7.5 * SF, 2.85 * 2.375 * SF, 3.15 * 2.625 * SF))
    assert seal.surface_id == paint.surface_id == "R1-W2" and seal.damage_ids == ["D1"] and seal.rule_id is None


def test_repaint_subtracts_openings_with_flipped_bounds(plan):
    items = generate(plan, [region("D1", "peeling_paint", "R1-W1", (2.5, 3.0), (1.0, 1.4), rel=0.1)], [])
    (paint,) = items_by(items, ("PNT", "P"))
    value = 4.0 * 2.5 - 0.9 * 2.0
    lo = 3.8 * 2.375 - 0.945 * 2.1
    hi = 4.2 * 2.625 - 0.855 * 1.9
    assert q(paint) == pytest.approx((value * SF, lo * SF, hi * SF))
    assert items_by(items, ("PNT", "PREP"))


def test_crack_length_in_linear_feet(plan):
    d = region("D1", "crack", "R1-W2", (0.5, 1.0), (0.5, 1.1), length=0.8, rel=0.1)
    (tape,) = items_by(generate(plan, [d], []), ("DRY", "TAPE"))
    assert tape.unit == "LF"
    assert q(tape) == pytest.approx(((0.8 + 0.15) * LF, (0.72 + 0.15) * LF, (0.88 + 0.15) * LF))


def test_holes_small_each_and_large_by_area(plan):
    small = region("D1", "hole", "R1-W2", (1.0, 1.1), (0.4, 0.5))
    large = region("D2", "hole", "R1-W4", (1.0, 1.4), (0.4, 0.8))
    items = generate(plan, [small, large], [])
    (patch,) = items_by(items, ("DRY", "PATCH"))
    assert patch.unit == "EA" and q(patch) == (1.0, 1.0, 1.0) and patch.surface_id == "R1-W2"
    (repl,) = [i for i in items_by(items, ("DRY", "1/2")) if i.surface_id == "R1-W4"]
    assert repl.activity == "&" and repl.quantity.value == pytest.approx(0.7 * 0.7 * SF)


def test_two_stains_on_one_wall_union_and_single_repaint(plan):
    a = region("D1", "water_stain", "R1-W2", (1.0, 1.5), (1.0, 1.4))
    b = region("D2", "water_stain", "R1-W2", (1.3, 1.8), (1.1, 1.5))
    items = generate(plan, [a, b], [])
    (seal,) = items_by(items, ("PNT", "SEAL"))
    assert seal.quantity.value == pytest.approx((0.56 + 0.56 - 0.5 * 0.6) * SF)
    assert seal.damage_ids == ["D1", "D2"]
    assert len(items_by(items, ("PNT", "P"))) == 1


def test_margin_is_clipped_to_the_surface(plan):
    d = region("D1", "water_stain", "R1-W2", (0.0, 0.3), (0.0, 0.2))
    (seal,) = items_by(generate(plan, [d], []), ("PNT", "SEAL"))
    assert seal.quantity.value == pytest.approx(0.45 * 0.35 * SF)


def test_missing_intervals_fall_back_to_the_value(plan):
    d = region("D1", "water_stain", "R1-W2", (1.0, 1.5), (1.0, 1.4))
    d.width.lo = d.width.hi = d.height.lo = d.height.hi = None
    (seal,) = items_by(generate(plan, [d], []), ("PNT", "SEAL"))
    assert seal.quantity.lo == seal.quantity.hi == pytest.approx(seal.quantity.value)


def test_wicking_flag_gives_flood_cut_insulation_and_drywall(plan):
    d = region("D1", "water_stain", "R1-W2", (1.0, 1.5), (0.05, 0.4), rel=0.1)
    flags = evaluate(plan, [d], [])
    items = generate(plan, [d], flags)
    (cut,) = items_by(items, ("WTR", "FCC"))
    assert cut.unit == "LF" and cut.activity == "-" and cut.rule_id == "R-WALL-WICK"
    assert q(cut) == pytest.approx((1.7 * LF, 1.65 * LF, 1.75 * LF))
    (ins,) = items_by(items, ("INS", "BATT"))
    assert q(ins) == pytest.approx((1.7 * 0.6 * SF, 1.65 * 0.6 * SF, 1.75 * 0.6 * SF))
    wick = next(f for f in flags if f.rule_id == "R-WALL-WICK")
    assert cut.flag_ids == [wick.id] and cut.damage_ids == ["D1"] and cut.surface_id == "R1-W2"


def test_flood_cut_is_clipped_to_the_wall(plan):
    d = region("D1", "water_stain", "R1-W2", (2.6, 3.0), (0.0, 0.3))
    (cut,) = items_by(generate(plan, [d], evaluate(plan, [d], [])), ("WTR", "FCC"))
    assert cut.quantity.value == pytest.approx((3.0 - (2.8 - 0.8)) * LF)


def test_moisture_mapping_once_per_surface_across_rules(plan):
    d = region("D1", "water_stain", "R1-W3", (1.0, 1.4), (1.0, 1.3))
    flags = [ConcealedFlag("F1", "R-FIXTURE-LEAK", "t", "b", "R1", ["R1-W3"], ["D1"], "high", "r"),
             ConcealedFlag("F2", "R-WINDOW-LEAK", "t", "b", "R1", ["R1-W3"], ["D1"], "medium", "r")]
    (moist,) = items_by(generate(plan, [d], flags), ("WTR", "MOIST"))
    assert q(moist) == (1.0, 1.0, 1.0) and moist.unit == "EA"
    assert moist.flag_ids == ["F1", "F2"] and moist.rule_id == "R-FIXTURE-LEAK,R-WINDOW-LEAK"


def test_mold_room_flag_gives_one_containment_item(plan):
    ds = [region("D1", "mold", "R1-W2", (0.0, 1.0), (0.5, 1.5)), region("D2", "mold", "R1-W4", (0.0, 0.6), (0.5, 1.0))]
    flags = evaluate(plan, ds, [])
    items = generate(plan, ds, flags)
    (cont,) = items_by(items, ("HMR", "CONT"))
    assert q(cont) == (1.0, 1.0, 1.0) and cont.surface_id == "R1-W2" and cont.rule_id == "R-MOLD-LIMITED"
    assert cont.damage_ids == ["D1", "D2"]  # the containment covers all mold in the room
    assert len(items_by(items, ("DMO", "INSP"))) == 2  # one inspection opening per affected surface
    assert {i.surface_id for i in items_by(items, ("HMR", "AMA"))} == {"R1-W2", "R1-W4"}


def test_items_are_numbered_in_room_surface_catalog_order(plan):
    ds = [region("D1", "water_stain", "R1-CEIL", (1.0, 1.4), (1.0, 1.3)),
          region("D2", "crack", "R1-W2", (0.5, 1.0), (0.5, 1.1), length=0.8),
          region("D3", "water_stain", "R1-W1", (2.5, 3.0), (0.05, 0.3))]
    sink = SceneObject("O1", "sink", "R1", __import__("numpy").array([2.8, 0.3]), (0.8, 0.9), 0.7)
    items = generate(plan, ds, evaluate(plan, ds, [sink]))
    assert [i.id for i in items] == [f"L{k}" for k in range(1, len(items) + 1)]
    surfaces = [i.surface_id for i in items]
    order = {"R1-W1": 0, "R1-W2": 1, "R1-W3": 2, "R1-W4": 3, "R1-FLOOR": 4, "R1-CEIL": 5}
    assert surfaces == sorted(surfaces, key=order.get)
    for i in items:
        assert i.quantity.lo <= i.quantity.value <= i.quantity.hi
        assert i.unit in ("SF", "LF", "EA") and i.activity in ("&", "-", "+", "R", "I")
        assert i.room_id == "R1" and (i.damage_ids or i.flag_ids)
    assert any(i.rule_id == "R-FIXTURE-LEAK" for i in items)


def test_unknown_surface_skips_whole_surface_items(plan):
    d = region("D1", "water_stain", "R9-W1", (1.0, 1.5), (1.0, 1.4), room_id="R9")
    items = generate(plan, [d], [])
    assert items_by(items, ("PNT", "SEAL")) and not items_by(items, ("PNT", "P"))


def test_outputs_fit_the_result_schema(plan):
    import json
    from pathlib import Path

    import jsonschema
    import numpy as np

    schema = json.loads((Path(__file__).resolve().parents[1] / "schema" / "scan2scope.schema.json").read_text())

    def check(name, instance):
        sub = {"$schema": schema["$schema"], "$defs": schema["$defs"], **schema["properties"][name]}
        jsonschema.Draft202012Validator(sub).validate(json.loads(json.dumps(instance)))

    ds = [region("D1", "water_stain", "R1-W2", (0.5, 1.0), (0.05, 0.4), rel=0.1),
          region("D2", "crack", "R1-W1", (1.9, 2.3), (2.05, 2.4), length=0.5, rel=0.1,
                 evidence={"endpoints_uv_all": [[[1.95, 2.05], [2.3, 2.4]]]}),
          region("D3", "mold", "R1-CEIL", (1.0, 2.2), (1.0, 2.0), rel=0.1),
          region("D4", "hole", "R1-W3", (1.0, 1.1), (0.4, 0.5), rel=0.1)]
    sink = SceneObject("O1", "sink", "R1", np.array([3.6, 0.6]), (0.8, 0.9), 0.7)
    flags = evaluate(plan, ds, [sink])
    items = generate(plan, ds, flags)
    assert len({f.rule_id for f in flags}) >= 6 and len(items) >= 10
    check("damage", [{"id": d.id, "room_id": d.room_id, "surface_id": d.surface_id, "class": d.cls, "score": d.score,
                      "area": d.area.to_dict(), "width": d.width.to_dict(), "height": d.height.to_dict(),
                      "length": d.length.to_dict() if d.length else None, "u_range": list(d.u_range),
                      "v_range": list(d.v_range), "view_ids": d.view_ids} for d in ds])
    check("concealed_damage_flags", [{"id": f.id, "rule_id": f.rule_id, "title": f.title, "basis": f.basis,
                                      "room_id": f.room_id, "surface_ids": f.surface_ids, "damage_ids": f.damage_ids,
                                      "severity": f.severity, "recommendation": f.recommendation,
                                      "inputs": f.inputs} for f in flags])
    check("scope", [{"id": i.id, "room_id": i.room_id, "surface_id": i.surface_id, "category": i.category,
                     "selector": i.selector, "activity": i.activity, "description": i.description,
                     "quantity": i.quantity.to_dict(), "unit": i.unit, "damage_ids": i.damage_ids,
                     "flag_ids": i.flag_ids, "rule_id": i.rule_id} for i in items])
