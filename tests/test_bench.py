"""Benchmark harness on a hand-built two-room property and hand-built result dicts with known errors.

Plan (metres, x east, y north): hallway [0, 1.2] x [0, 4], kitchen [1.3, 4.3] x [0, 4], a 0.9 m door between
them at y 1.0-1.9, the front door in the hallway's south wall at x 0.2-1.0 and a 1.2 m kitchen window in the
east wall at y 1.5-2.7. GT walls follow the protocol (entry wall first, then clockwise); offsets run from the
wall's left end seen from inside. Predicted rooms are counter-clockwise with offsets from the wall start.
"""

import copy
import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from scan2scope.bench import gates as gates_mod
from scan2scope.bench.ablation import drift_ablation
from scan2scope.bench.groundtruth import load_ground_truth, rectilinear_polygon, shoelace
from scan2scope.bench.h2h import head_to_head, parse_quantity, parse_statistics
from scan2scope.bench.match import GTRoom, align_walls, match_capture, pred_rooms
from scan2scope.bench.metrics import capture_metrics, repeatability
from scan2scope.bench.report import write_report
from scan2scope.bench.runner import run_benchmark, score_runs

ROOT = Path(__file__).resolve().parents[1]

GT_YAML = """
property: home
measured_by: tester
instrument: "laser"
date: "2026-10-04"
rooms:
  - id: "01 hallway"
    walls:
      - {id: W1, length: 1.200}
      - {id: W2, length: 4.000}
      - {id: W3, length: 1.200}
      - {id: W4, length: 4.000}
    diagonal: null
    ceiling_height: [2.500, 2.510, 2.500]
    openings:
      - {id: D1, type: door, wall: W1, offset: 0.200, width: 0.800, height: 2.000, leads_to: outside}
      - {id: D2, type: door, wall: W4, offset: 2.100, width: 0.900, height: 2.000, leads_to: "02 kitchen"}
    damage: []
  - id: "02 kitchen"
    walls:
      - {id: W1, length: 4.000}
      - {id: W2, length: 3.000}
      - {id: W3, length: 4.000}
      - {id: W4, length: 3.000}
    diagonal: null
    ceiling_height: [2.480, 2.490, 2.490]
    openings:
      - {id: D1, type: door, wall: W1, offset: 1.000, width: 0.900, height: 2.000, leads_to: "01 hallway"}
      - {id: N1, type: window, wall: W3, offset: 1.300, width: 1.200, height: 1.200, sill: 0.900}
    damage:
      - {id: X1, class: water_stain, surface: W2, offset: 0.700, bottom: 1.200, width: 0.400, height: 0.300}
captures:
  - {id: photo_1, tier: photo, path: raw/photo_1}
  - {id: photo_2, tier: photo, path: raw/photo_2, rooms: ["02 kitchen"]}
  - {id: video_1, tier: video, path: raw/video_1.mov}
"""

HALL = [(0.0, 0.0), (1.2, 0.0), (1.2, 4.0), (0.0, 4.0)]  # counter-clockwise, W1 = south
KITCHEN = [(4.3, 4.0), (1.3, 4.0), (1.3, 0.0), (4.3, 0.0)]  # counter-clockwise, W1 = north


def M(v, hw, unit="m"):
    return {"value": v, "lo": v - hw, "hi": v + hw, "unit": unit}


def room(rid, poly, *, hint=None, openings=(), lengths=None, ceiling=2.5, rel=0.03, area=None):
    P = np.asarray(poly, float)
    n = len(P)
    walls = []
    for k in range(n):
        a, b = P[k], P[(k + 1) % n]
        L = float(np.linalg.norm(b - a)) if lengths is None or lengths[k] is None else lengths[k]
        walls.append({"id": f"{rid}-W{k + 1}", "start": a.tolist(), "end": b.tolist(), "length": M(L, rel * L),
                      "height": M(ceiling, 0.05), "observed_fraction": 0.8, "flags": []})
    ops = []
    for j, (k, off, w, kind) in enumerate(openings):
        ops.append({"id": f"{rid}-O{j + 1}", "type": kind, "wall_id": f"{rid}-W{k + 1}", "offset": M(off, 0.05),
                    "width": M(w, 0.03), "height": M(2.0 if kind != "window" else 1.2, 0.05),
                    "sill": None if kind != "window" else M(0.9, 0.05), "center": None, "connects_to": None,
                    "confidence": 0.9, "flags": []})
    a = abs(shoelace(P)) if area is None else area
    perim = sum(w["length"]["value"] for w in walls)
    return {"id": rid, "label": rid, "source_hint": hint, "polygon": P.tolist(), "floor_area": M(a, 0.06 * a, "m2"),
            "perimeter": M(perim, 0.03 * perim), "ceiling_height": M(ceiling, 0.05), "walls": walls,
            "openings": ops, "surfaces": [], "flags": []}


def result(rooms, *, tier="photo", adjacency=(("R1", "R2"),), footprint=None, damage=(), drift=None, cid="cap"):
    fp = sum(r["floor_area"]["value"] for r in rooms) if footprint is None else footprint
    return {
        "schema_version": "1.0.0",
        "capture": {"id": cid, "tier": tier, "path": "x", "device": {}, "input_stats": {}, "flags": []},
        "conventions": {"units": {}, "interval": {"level": 0.9, "method": "test", "q": 1.0, "calibration": "prior"},
                        "definitions": {}},
        "property": {"footprint_area": M(fp, 0.06 * fp, "m2"), "extent_x": M(4.3, 0.1), "extent_y": M(4.0, 0.1),
                     "adjacency": [{"room_a": a, "room_b": b, "opening_a": None, "opening_b": None, "confidence": 0.9,
                                    "source": "door_match"} for a, b in adjacency],
                     "drift_correction": drift, "stitch": None, "flags": []},
        "rooms": rooms, "damage": list(damage), "concealed_damage_flags": [], "scope": [],
        "timing": {"total_s": 12.5, "stages": {"ingest": 0.5, "geometry": 9.0, "layout": 3.0}},
        "provenance": {"pipeline_version": "0.1.0", "git_commit": "0123456789abcdef", "models": [],
                       "cache_mode": "replay", "device": "cpu"},
    }


def good_rooms(tier="photo", ids=("R1", "R2")):
    hint = tier == "photo"
    hall = room(ids[0], HALL, hint="01 hallway" if hint else None, ceiling=2.505,
                openings=[(0, 0.2, 0.8, "door"), (1, 1.0, 0.9, "door")])
    kitchen = room(ids[1], KITCHEN, hint="02 kitchen" if hint else None, ceiling=2.49,
                   openings=[(1, 2.1, 0.9, "door"), (3, 1.5, 1.2, "window")])
    return [hall, kitchen]


def stain(rid="R2", wall="R2-W1", u=(1.9, 2.3), v=(1.2, 1.5), cls="water_stain"):
    w, h = u[1] - u[0], v[1] - v[0]
    return {"id": "D1", "room_id": rid, "surface_id": wall, "class": cls, "score": 0.6, "area": M(w * h * 0.7, 0.02, "m2"),
            "width": M(w, 0.05), "height": M(h, 0.05), "length": None, "u_range": list(u), "v_range": list(v),
            "view_ids": []}


@pytest.fixture
def gt_path(tmp_path):
    p = tmp_path / "data" / "home" / "ground_truth.yaml"
    p.parent.mkdir(parents=True)
    p.write_text(GT_YAML)
    return p


@pytest.fixture
def gt(gt_path):
    return load_ground_truth(gt_path)


def test_ground_truth_loader(gt, gt_path):
    assert [r.id for r in gt.rooms] == ["01 hallway", "02 kitchen"]
    hall, kitchen = gt.rooms
    assert hall.ceiling_height == pytest.approx(2.50) and kitchen.ceiling_height == pytest.approx(2.49)
    assert hall.floor_area == pytest.approx(4.8) and hall.area_method == "opposite_walls"
    assert gt.footprint() == pytest.approx(16.8)
    assert gt.footprint(gt.capture("photo_2")) == pytest.approx(12.0)
    assert gt.adjacency_pairs() == {frozenset(("01 hallway", "02 kitchen"))}
    assert gt.adjacency[0]["openings"] == ("D2", "D1")
    assert gt.capture("photo_2").rooms == ["02 kitchen"]
    assert gt.capture("video_1").path == gt_path.parent / "raw" / "video_1.mov"
    assert not gt.synthetic and kitchen.label == "kitchen"


def test_template_placeholders_load_as_missing():
    gt = load_ground_truth(ROOT / "bench/templates/ground_truth.yaml")
    hall = gt.room("01 hallway")
    assert all(w.length is None for w in hall.walls) and hall.floor_area is None and hall.ceiling_height is None
    assert any("missing:W1.length" in f for f in gt.flags)
    assert len(gt.captures) == 4


def test_floor_area_from_polygon_and_rectilinear_walls(tmp_path):
    # L-shape 4 x 3 with a 2 x 1.5 notch, walls clockwise from the W1 start
    lengths = [4.0, 3.0, 2.0, 1.5, 2.0, 1.5]
    corners, closure = rectilinear_polygon(lengths)
    assert abs(shoelace(corners)) == pytest.approx(9.0) and closure == pytest.approx(0.0)
    assert shoelace(corners) < 0  # clockwise
    noisy = rectilinear_polygon([4.004, 3.0, 2.0, 1.5, 2.0, 1.502])
    assert noisy[1] == pytest.approx(math.hypot(0.004, 0.002), abs=1e-9)
    doc = {"property": "p", "measured_by": "synthetic", "rooms": [
        {"id": "L", "walls": [{"id": f"W{k + 1}", "length": v} for k, v in enumerate(lengths)],
         "ceiling_height": [2.4], "openings": [], "damage": []},
        {"id": "P", "walls": [{"id": f"W{k + 1}", "length": v} for k, v in enumerate([2.0, 3.0, 2.0, 3.0])],
         "ceiling_height": 2.4, "polygon": [[0, 0], [0, 3], [2, 3], [2, 0]], "openings": [], "damage": []}],
        "captures": [{"id": "lidar_1", "tier": "lidar", "path": "raw/lidar_1"}]}
    p = tmp_path / "gt.yaml"
    p.write_text(yaml.safe_dump(doc))
    gt = load_ground_truth(p)
    assert gt.synthetic
    assert gt.room("L").floor_area == pytest.approx(9.0) and gt.room("L").area_method == "rectilinear"
    assert gt.room("P").floor_area == pytest.approx(6.0) and gt.room("P").area_method == "polygon"


ODD_GT = """
property: odd
rooms:
  - id: "01 hall"
    walls: [2.0, 3.0, 2.0, 3.0]
    ceiling_height: 2.45
    openings:
      - {id: D1, type: door, wall: W1, offset: 0.2, width: 0.8, height: 2.0}
      - {id: D1, type: doorway, wall: W9, offset: 0.2, width: 0.0, height: 2.0}
  - id: "02 bay"
    walls: [3.0, 1.0, 1.0, 1.0, 3.0]
    ceiling_height: ["2,40", 0]
    openings: [junk]
    damage: [{id: X1, class: Mold, surface: Ceiling, width: 0.2, height: 0.1}, {id: X1, class: crack, surface: W2}]
  - {walls: [1, 1, 1, 1]}
captures:
  - {id: a, tier: photo, path: raw/a, rooms: ["02 bay", "nope"]}
  - {id: b, tier: thermal, path: raw/b}
  - {id: c, tier: video}
"""


def test_odd_ground_truth_loads_with_flags(tmp_path):
    p = tmp_path / "gt.yaml"
    p.write_text(ODD_GT)
    gt = load_ground_truth(p)
    hall, bay = gt.rooms
    assert hall.ceiling_height == pytest.approx(2.45) and bay.ceiling_height == pytest.approx(2.40)
    assert [o.id for o in hall.openings] == ["D1", "D1~2"] and hall.openings[1].type == "opening"
    assert bay.floor_area is None and "floor_area_unavailable" in bay.flags
    assert [d.id for d in bay.damage] == ["X1", "X1~2"] and bay.damage[0].surface == "ceiling"
    assert bay.damage[0].cls == "mold"
    assert [c.id for c in gt.captures] == ["a"] and gt.captures[0].rooms == ["02 bay"]
    assert gt.footprint() is None
    for f in ("room_without_id:2", "capture_tier:b:thermal", "bad_capture:2", "capture_unknown_rooms:a:nope",
              "01 hall:opening_unknown_wall:D1~2:W9", "02 bay:bad_opening:0"):
        assert f in gt.flags, f
    m = capture_metrics(gt, gt.captures[0], result([room("R1", [(0, 0), (3, 0), (3, 1), (2, 2), (0, 2)],
                                                         hint="02 bay")], adjacency=()))
    assert m["status"] == "ok" and m["rooms"]["matched"] == 1


def _l_gt_room():
    from scan2scope.bench.groundtruth import GTOpening, GTWall

    # L-shape listed clockwise from the W1 left end at (4, 0): (4,0) (0,0) (0,1.5) (2,1.5) (2,3) (4,3)
    walls = [GTWall(f"W{k + 1}", v) for k, v in enumerate([4.0, 1.5, 2.0, 1.5, 2.0, 3.0])]
    # door in W1 (south wall, left end seen from inside is the east end) at x 2.6-3.5
    door = GTOpening("D1", "door", "W1", 0.5, 0.9, 2.0)
    return GTRoom("L", "L", walls, [2.5], 2.5, [door], [], floor_area=9.0)


L_CCW = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (2.0, 3.0), (2.0, 1.5), (0.0, 1.5)]


@pytest.mark.parametrize("start", range(6))
def test_wall_matching_survives_any_start_vertex(start):
    poly = L_CCW[start:] + L_CCW[:start]
    south = (6 - start) % 6  # index of the predicted wall (0,0)->(4,0)
    pred = pred_rooms(result([room("R1", poly, openings=[(south, 2.6, 0.9, "door")])]))[0]
    walls, ops = align_walls(_l_gt_room(), pred)
    got = {gi: np.round(np.asarray(pred.walls[pj]["start"]), 3).tolist() for gi, pj in walls.pairs}
    # GT W1 is the south wall, which starts at (0, 0) in our counter-clockwise order; W6 is the east wall
    assert got[0] == [0.0, 0.0] and got[5] == [4.0, 0.0]
    assert walls.orientation == "reversed" and not walls.missed and not walls.extra
    assert [o.status for o in ops] == ["matched"]


def test_wall_matching_on_random_rectilinear_rooms():
    from scan2scope.bench.groundtruth import GTOpening, GTWall

    rng = np.random.default_rng(7)
    for _ in range(60):
        W, H = rng.uniform(2.0, 6.0, size=2)
        a, b = rng.uniform(0.3, 0.7, size=2) * (W, H)
        P = np.array([[(0, 0), (W, 0), (W, H), (0, H)], [(0, 0), (W, 0), (W, H), (a, H), (a, b), (0, b)]][
            int(rng.integers(2))], float)
        n = len(P)
        cw = np.roll(P[::-1], -int(rng.integers(n)), axis=0)  # GT order: clockwise from a random entry wall
        gt_walls = [GTWall(f"W{k + 1}", float(np.linalg.norm(cw[(k + 1) % n] - cw[k]))) for k in range(n)]
        L1 = gt_walls[0].length
        w = min(0.9, 0.6 * L1)
        off = float(rng.uniform(0.0, L1 - w))
        g = GTRoom("X", "X", gt_walls, [2.5], 2.5, [GTOpening("D1", "door", "W1", off, w, 2.0)], [])
        Q = np.roll(P, -int(rng.integers(n)), axis=0)  # ours: counter-clockwise from a random corner
        lengths = [float(np.linalg.norm(Q[(k + 1) % n] - Q[k])) * (1 + rng.normal(0, 0.02)) for k in range(n)]
        k_door = next(k for k in range(n) if np.allclose(Q[k], cw[1]))  # our W1 starts at the GT W1's right end
        pred = pred_rooms(result([room("R1", Q.tolist(), lengths=lengths,
                                       openings=[(k_door, L1 - off - w, w, "door")])], adjacency=()))[0]
        walls, ops = align_walls(g, pred)
        got = dict(walls.pairs)
        assert len(got) == n and [o.status for o in ops] == ["matched"]
        for gi, pj in got.items():
            assert np.allclose(pred.walls[pj]["start"], cw[(gi + 1) % n])


def test_gt_listed_counter_clockwise_matches_in_same_orientation():
    from scan2scope.bench.groundtruth import GTOpening, GTWall

    g = _l_gt_room()
    rev = [g.walls[0]] + g.walls[1:][::-1]  # same walls listed the other way round from W1
    g.walls = [GTWall(f"W{k + 1}", w.length) for k, w in enumerate(rev)]
    # a consistent mirror: offsets from the right end, which is our wall start (x 2.6 from the west end)
    g.openings = [GTOpening("D1", "door", "W1", 2.6, 0.9, 2.0)]
    pred = pred_rooms(result([room("R1", L_CCW, openings=[(0, 2.6, 0.9, "door")])]))[0]
    walls, ops = align_walls(g, pred)
    assert walls.orientation == "same" and not walls.mirrored
    assert dict(walls.pairs)[0] == 0 and dict(walls.pairs)[1] == 1
    assert [o.status for o in ops] == ["matched"]


def test_square_room_without_openings_is_ambiguous_but_rectangle_with_door_is_not(gt):
    from scan2scope.bench.groundtruth import GTWall

    square = GTRoom("S", "S", [GTWall(f"W{k + 1}", 3.0) for k in range(4)], [2.5], 2.5, [], [])
    pred = pred_rooms(result([room("R1", [(0, 0), (3, 0), (3, 3), (0, 3)])], adjacency=()))[0]
    walls, _ = align_walls(square, pred)
    assert walls.ambiguous and walls.ties == 8
    m = capture_metrics(gt, gt.capture("photo_1"), result(good_rooms()))
    assert not any("ambiguous" in f for f in m["flags"])


def test_offsets_from_the_wrong_end_are_flagged_not_missed():
    g = _l_gt_room()
    g.openings[0].offset = 2.6  # measured from the west (right) end by mistake
    pred = pred_rooms(result([room("R1", L_CCW, openings=[(0, 2.6, 0.9, "door")])]))[0]
    walls, ops = align_walls(g, pred)
    assert walls.mirrored and [o.status for o in ops] == ["matched"]


def test_extra_predicted_wall_is_reported_and_others_still_match(gt):
    # kitchen with its south-east corner chamfered by 0.3 m: five walls
    poly = [(4.3, 4.0), (1.3, 4.0), (1.3, 0.0), (4.0, 0.0), (4.3, 0.3)]
    hall, _ = good_rooms()
    kitchen = room("R2", poly, hint="02 kitchen", openings=[(1, 2.1, 0.9, "door")])
    m = capture_metrics(gt, gt.capture("photo_1"), result([hall, kitchen]))
    rm = next(r for r in m["match"]["rooms"] if r["gt_room"] == "02 kitchen")
    assert len(rm["walls"]["pairs"]) == 4 and rm["walls"]["extra"] == [3]
    walls = {r["item"]: r for r in m["records"] if r["kind"] == "wall_length"}
    assert walls["02 kitchen/W2"]["err"] == pytest.approx(0.0)
    assert walls["02 kitchen/W1"]["err"] == pytest.approx(0.0)
    assert m["walls"]["extra"] == 1


def test_missed_and_phantom_opening(gt):
    hall, kitchen = good_rooms()
    kitchen["openings"] = [kitchen["openings"][0]]  # window not found
    kitchen["openings"].append({**copy.deepcopy(kitchen["openings"][0]), "id": "R2-O9", "wall_id": "R2-W3"})
    m = capture_metrics(gt, gt.capture("photo_1"), result([hall, kitchen]))
    assert m["openings"]["matched"] == 3 and m["openings"]["missed"] == 1 and m["openings"]["phantom"] == 1
    assert any(x["item"] == "02 kitchen/N1" and x["reason"] == "opening_missed" for x in m["missing"])
    rows = gates_mod.evaluate([m], None, None, None, gates_mod.load_gates())["rows"]
    op = next(r for r in rows if r["gate"] == "opening_width" and r["tier"] == "photo")
    assert op["n"] == 5 and op["measured"] == pytest.approx(3 / 5) and op["status"] == "fail"


def test_metrics_records_errors_and_coverage(gt):
    hall, kitchen = good_rooms()
    kitchen["walls"][3]["length"] = M(4.4, 0.05)  # east wall (GT W3 = 4.0) reported 10% long, interval misses
    m = capture_metrics(gt, gt.capture("photo_1"), result([hall, kitchen], damage=[stain()]))
    rec = {(r["kind"], r["item"]): r for r in m["records"]}
    east = rec["wall_length", "02 kitchen/W3"]
    assert east["err"] == pytest.approx(0.4) and east["rel_err"] == pytest.approx(0.1)
    assert not east["covered"] and east["half_width"] == pytest.approx(0.05)
    assert east["room"] == "home/02 kitchen" and east["tier"] == "photo"
    assert rec["ceiling_height", "01 hallway"]["err"] == pytest.approx(0.005)
    assert rec["floor_area", "02 kitchen"]["gt"] == pytest.approx(12.0)
    assert rec["footprint", "footprint"]["gt"] == pytest.approx(16.8)
    assert rec["opening_width", "02 kitchen/N1"]["err"] == pytest.approx(0.0)
    dmg = rec["damage_area", "02 kitchen/X1"]
    assert dmg["gt"] == pytest.approx(0.12) and dmg["basis"] == "bbox" and dmg["class_ok"]
    assert m["adjacency"]["exact"] and m["overlap"]["max_m2"] == pytest.approx(0.0)
    assert m["damage"]["matched"] == 1 and m["damage"]["phantom"] == 0
    assert not m["missing"]


def test_video_rooms_match_by_shape(gt):
    hall, kitchen = good_rooms("video", ids=("R2", "R1"))
    res = result([kitchen, hall], tier="video", drift={"enabled": True})
    match = match_capture(gt, gt.capture("video_1"), res)
    assert match.room_map == {"R2": "01 hallway", "R1": "02 kitchen"}
    assert all(r.method == "hungarian" for r in match.rooms)


THREE_ROOMS = """
property: flat
rooms:
  - {id: H, walls: [1.2, 6.0, 1.2, 6.0], ceiling_height: [2.5],
     openings: [{id: D1, type: door, wall: W2, offset: 1.0, width: 0.9, height: 2.0, leads_to: A}]}
  - {id: A, walls: [3.0, 4.0, 3.0, 4.0], ceiling_height: [2.5],
     openings: [{id: D1, type: door, wall: W1, offset: 1.0, width: 0.9, height: 2.0, leads_to: H},
                {id: O1, type: opening, wall: W2, offset: 1.0, width: 1.2, height: 2.1, leads_to: B}]}
  - {id: B, walls: [3.1, 4.0, 3.1, 4.0], ceiling_height: [2.5],
     openings: [{id: O1, type: opening, wall: W1, offset: 1.0, width: 1.2, height: 2.1, leads_to: A}]}
captures:
  - {id: video_1, tier: video, path: raw/v.mov}
"""


def test_adjacency_decides_between_rooms_of_equal_shape(tmp_path):
    p = tmp_path / "gt.yaml"
    p.write_text(THREE_ROOMS)
    gt = load_ground_truth(p)
    assert gt.adjacency_pairs() == {frozenset(("A", "H")), frozenset(("A", "B"))}
    # P1 is really A (next to the hallway and to B) but its area is closer to B's; P2 is really B
    rooms = [room("P0", [(0, 0), (1.2, 0), (1.2, 6), (0, 6)]),
             room("P1", [(1.3, 0), (4.3, 0), (4.3, 4.1), (1.3, 4.1)]),
             room("P2", [(1.3, 4.2), (4.33, 4.2), (4.33, 8.2), (1.3, 8.2)])]
    unary_only = match_capture(gt, gt.capture("video_1"), result(rooms, tier="video", adjacency=()))
    assert unary_only.room_map["P1"] == "B"
    res = result(rooms, tier="video", adjacency=(("P0", "P1"), ("P1", "P2")))
    assert match_capture(gt, gt.capture("video_1"), res).room_map == {"P0": "H", "P1": "A", "P2": "B"}


# Written by scan2scope.synth (apartment seed 1001, template single_end): polygon plus adjacency fields.
SYNTH_YAML = """
property: synth_1
measured_by: synthetic
rooms:
  - id: "01 hallway"
    walls: [{id: W1, length: 1.064}, {id: W2, length: 6.446}, {id: W3, length: 1.064}, {id: W4, length: 6.446}]
    diagonal: null
    ceiling_height: [2.501, 2.501, 2.501]
    openings:
      - {id: D1, type: door, wall: W2, offset: 1.516, width: 0.744, height: 2.026, leads_to: "02 bedroom"}
      - {id: D2, type: door, wall: W2, offset: 5.348, width: 0.856, height: 2.100, leads_to: "03 bathroom"}
      - {id: D3, type: door, wall: W3, offset: 0.150, width: 0.764, height: 2.044, leads_to: "04 living"}
    damage: []
    polygon: [[0.000, 0.000], [0.000, 1.064], [6.446, 1.064], [6.446, 0.000]]
  - id: "02 bedroom"
    walls: [{id: W1, length: 3.054}, {id: W2, length: 4.272}, {id: W3, length: 6.446}, {id: W4, length: 1.263},
            {id: W5, length: 3.392}, {id: W6, length: 3.009}]
    diagonal: 5.455
    ceiling_height: [2.501, 2.501, 2.501]
    openings:
      - {id: D1, type: door, wall: W1, offset: 0.794, width: 0.744, height: 2.026, leads_to: "01 hallway"}
      - {id: N1, type: window, wall: W2, offset: 0.520, width: 0.886, height: 1.226, sill: 0.866}
    damage: []
    polygon: [[3.054, 1.189], [0.000, 1.189], [0.000, 5.461], [6.446, 5.461], [6.446, 4.198], [3.054, 4.198]]
  - id: "03 bathroom"
    walls: [{id: W1, length: 3.274}, {id: W2, length: 2.892}, {id: W3, length: 3.274}, {id: W4, length: 2.892}]
    diagonal: null
    ceiling_height: [2.501, 2.501, 2.501]
    openings:
      - {id: D1, type: door, wall: W1, offset: 0.242, width: 0.856, height: 2.100, leads_to: "01 hallway"}
    damage: []
    polygon: [[6.446, 1.189], [3.172, 1.189], [3.172, 4.081], [6.446, 4.081]]
  - id: "04 living"
    walls: [{id: W1, length: 5.461}, {id: W2, length: 5.231}, {id: W3, length: 5.461}, {id: W4, length: 5.231}]
    diagonal: null
    ceiling_height: [2.501, 2.501, 2.501]
    openings:
      - {id: D1, type: door, wall: W1, offset: 0.150, width: 0.764, height: 2.044, leads_to: "01 hallway"}
      - {id: N1, type: window, wall: W2, offset: 3.640, width: 1.120, height: 1.118, sill: 0.966}
      - {id: N2, type: window, wall: W3, offset: 0.770, width: 1.111, height: 1.101, sill: 0.871}
      - {id: N3, type: window, wall: W3, offset: 2.845, width: 0.856, height: 1.153, sill: 0.875}
      - {id: N4, type: window, wall: W4, offset: 3.912, width: 0.899, height: 1.390, sill: 0.854}
    damage: []
    polygon: [[6.558, 0.000], [6.558, 5.461], [11.789, 5.461], [11.789, 0.000]]
adjacency:
  - {rooms: ["01 hallway", "02 bedroom"], type: door, openings: [D1, D1]}
  - {rooms: ["01 hallway", "03 bathroom"], type: door, openings: [D2, D1]}
  - {rooms: ["01 hallway", "04 living"], type: door, openings: [D3, D1]}
captures:
  - {id: lidar_1, tier: lidar, path: raw/lidar_1}
"""


def _perfect_room(rid, g, shift):
    """Our counter-clockwise room from a GT room: polygon reversed, offsets from our wall start."""
    P = np.asarray(g.polygon, float)
    n = len(P)
    ccw = np.roll(P[::-1], -shift, axis=0)
    ops = []
    for o in g.openings:
        k = g.wall_index(o.wall)
        a, b = P[k], P[(k + 1) % n]  # GT wall k runs from its left end a to b; our edge runs b -> a
        j = next(j for j in range(n) if np.allclose(ccw[j], b) and np.allclose(ccw[(j + 1) % n], a))
        ops.append((j, g.walls[k].length - o.offset - o.width, o.width, o.type))
    r = room(rid, ccw.tolist(), openings=ops, ceiling=g.ceiling_height)
    for op, o in zip(r["openings"], g.openings):
        op["height"] = M(o.height, 0.05)
    return r


def test_synthetic_ground_truth_and_a_perfect_lidar_result(tmp_path):
    p = tmp_path / "synth_1" / "ground_truth.yaml"
    p.parent.mkdir()
    p.write_text(SYNTH_YAML)
    gt = load_ground_truth(p)
    assert gt.synthetic and not gt.flags
    bedroom = gt.room("02 bedroom")
    assert bedroom.area_method == "polygon" and bedroom.floor_area == pytest.approx(17.331, abs=1e-3)
    assert shoelace(bedroom.polygon) < 0 and bedroom.closure_error == pytest.approx(0.0, abs=1e-3)
    rng = np.random.default_rng(3)
    ids = ["R3", "R1", "R4", "R2"]
    rooms = [_perfect_room(ids[k], g, int(rng.integers(len(g.walls)))) for k, g in enumerate(gt.rooms)]
    adj = [(ids[0], ids[1]), (ids[0], ids[2]), (ids[0], ids[3])]
    res = result(rooms[::-1], tier="lidar", adjacency=adj, drift={"enabled": True})
    m = capture_metrics(gt, gt.capture("lidar_1"), res)
    assert {r["pred_room"]: r["gt_room"] for r in m["match"]["rooms"]} == dict(zip(ids, [r.id for r in gt.rooms]))
    assert not m["missing"] and m["openings"]["matched"] == 11 and m["openings"]["phantom"] == 0
    assert max(abs(r["err"]) for r in m["records"]) == pytest.approx(0.0, abs=1e-9)
    assert m["adjacency"]["exact"]


def test_gates_pass_on_a_good_capture_and_rank_failures(gt):
    cfg = gates_mod.load_gates()
    good = capture_metrics(gt, gt.capture("photo_1"), result(good_rooms(), damage=[stain()]))
    out = gates_mod.evaluate([good], None, None, None, cfg)
    rows = {(r["tier"], r["gate"]): r for r in out["rows"]}
    for gate in ("wall_length", "ceiling_height", "opening_width", "floor_area", "footprint", "stitch",
                 "calibration", "confident_garbage", "result_produced"):
        assert rows["photo", gate]["status"] == "pass", gate
    assert rows["photo", "repeatability"]["status"] == "n.a."
    assert rows["lidar", "wall_length"]["status"] == "n.a."
    assert rows["photo", "wall_length"]["assumed"] is False and rows["photo", "floor_area"]["assumed"] is True
    assert out["ceiling_mode"]["photo"] == "none"

    hall, kitchen = good_rooms()
    kitchen["walls"][3]["length"] = M(4.6, 0.05)  # 15% long: 1.875x the 8% allowance, a confident miss
    bad = capture_metrics(gt, gt.capture("photo_1"), result([hall, kitchen]))
    out = gates_mod.evaluate([bad], None, None, None, cfg)
    rows = {(r["tier"], r["gate"]): r for r in out["rows"]}
    wall = rows["photo", "wall_length"]
    assert wall["status"] == "fail" and wall["measured"] == pytest.approx(0.15)
    assert wall["pass_share"] == pytest.approx(7 / 8) and wall["worst"][0]["item"] == "02 kitchen/W3"
    assert wall["score"] == pytest.approx(0.15 / 0.08 - 1.0)
    assert rows["photo", "confident_garbage"]["status"] == "fail"
    ranked = out["ranked_failures"]
    assert ranked[0]["gate"] == "confident_garbage" and {r["gate"] for r in ranked} >= {"wall_length"}
    assert [r["score"] for r in ranked] == sorted((r["score"] for r in ranked), reverse=True)


def test_odd_results_still_score(gt):
    hall, kitchen = good_rooms()
    del kitchen["walls"][2]["length"]  # matched by its start and end, but scored as missing
    kitchen["openings"].append({"id": "R2-O7", "type": "skylight", "wall_id": "R9-W1"})
    kitchen["openings"].append({"id": "R2-O8", "type": "door", "wall_id": "R2-W1", "offset": None, "width": "wide"})
    empty = {"id": "R3", "polygon": [], "walls": [], "openings": None, "floor_area": None}
    odd_damage = {"id": "D9", "room_id": "R2", "surface_id": "R2-W9", "u_range": None}
    res = result([hall, kitchen, empty, "junk"], damage=[odd_damage, stain()], footprint=16.8)
    del res["property"]["footprint_area"]
    res["property"]["adjacency"].append({"room_a": "R1", "room_b": "R7"})
    res["timing"] = None
    m = capture_metrics(gt, gt.capture("photo_1"), res)
    assert m["status"] == "ok", m["error"]
    assert m["rooms"]["extra"] == ["R3"] and m["rooms"]["matched"] == 2
    assert any(x["kind"] == "footprint" and x["reason"] == "no_value" for x in m["missing"])
    assert m["adjacency"]["unmatched_rooms"] == [["R1", "R7"]] and m["adjacency"]["exact"] is False
    assert m["openings"]["phantom"] == 2 and m["openings"]["matched"] == 4
    assert m["damage"]["matched"] == 1 and m["damage"]["phantom"] == 1
    assert {"02 kitchen/W4"} == {x["item"] for x in m["missing"] if x["kind"] == "wall_length"}
    assert m["walls"]["matched"] == 8


def test_failed_capture_counts_every_item_as_missing(gt):
    m = capture_metrics(gt, gt.capture("photo_1"), None, status="failed", error="RuntimeError: boom")
    assert m["status"] == "failed" and not m["records"]
    assert sum(x["kind"] == "wall_length" for x in m["missing"]) == 8
    rows = {(r["tier"], r["gate"]): r for r in gates_mod.evaluate([m], None, None, None, gates_mod.load_gates())["rows"]}
    assert rows["photo", "result_produced"]["status"] == "fail"
    assert rows["photo", "wall_length"]["status"] == "fail" and rows["photo", "wall_length"]["pass_share"] == 0
    assert rows["photo", "opening_width"]["measured"] == 0.0
    assert rows["photo", "calibration"]["status"] == "n.a."


def test_clopper_pearson_and_calibration_gate():
    from scipy.stats import binomtest

    for k, n in ((9, 10), (0, 12), (20, 20), (45, 60)):
        ref = binomtest(k, n).proportion_ci(confidence_level=0.95, method="exact")
        assert gates_mod.clopper_pearson(k, n) == pytest.approx((ref.low, ref.high), abs=1e-9)
    recs = [{"tier": "video", "kind": "wall_length", "room": f"p/r{i % 4}", "item": f"w{i}", "capture": "c",
             "property": "p", "gt": 3.0, "pred": 3.0 + e, "lo": 3.0 + e - 0.05, "hi": 3.0 + e + 0.05, "err": e,
             "covered": abs(e) <= 0.05, "half_width": 0.05}
            for i, e in enumerate([0.01] * 12 + [0.08] * 7 + [0.2])]
    cal = gates_mod.calibration(recs)["video"]
    assert cal["coverage"] == pytest.approx(12 / 20) and not cal["contains_nominal"]
    assert cal["confident_garbage"] == 1 and cal["rooms"] == 4
    rows = gates_mod.calibration_rows("video", cal, {"ci": 0.95, "confident_garbage_ratio": 2.0}, 0.9)
    assert [r["status"] for r in rows] == ["fail", "fail"]
    assert rows[0]["score"] > 0 and rows[1]["measured"] == 1


def test_repeatability_walls_and_ceiling_spread(gt):
    cfg = gates_mod.load_gates()
    a = capture_metrics(gt, gt.capture("photo_1"), result(good_rooms()))
    _, kitchen = good_rooms()
    kitchen["walls"][0]["length"] = M(3.012, 0.09)  # north (GT W2 = 3.0): 12 mm, inside max(1 cm, 1.5 cm)
    kitchen["walls"][1]["length"] = M(4.03, 0.12)  # west (GT W1 = 4.0): 3 cm, outside max(1 cm, 2 cm)
    kitchen["ceiling_height"] = M(2.505, 0.05)
    b = capture_metrics(gt, gt.capture("photo_2"), result([kitchen], adjacency=()))
    rep = repeatability([a, b], cfg)
    walls = {w["wall"]: w for w in rep["walls"]}
    assert set(walls) == {"W1", "W2", "W3", "W4"}
    assert walls["W2"]["pass"] and walls["W2"]["allowed"] == pytest.approx(0.015)
    assert not walls["W2"]["strict_pass"] and walls["W2"]["strict_allowed"] == pytest.approx(0.01)
    assert not walls["W1"]["pass"] and walls["W1"]["delta"] == pytest.approx(0.03)
    assert rep["ceiling"][0]["spread"] == pytest.approx(0.015)
    assert rep["structure"][0]["same_walls"] and rep["structure"][0]["same_openings"]
    rows = {(r["tier"], r["gate"]): r for r in gates_mod.evaluate([a, b], rep, None, None, cfg)["rows"]}
    assert rows["photo", "repeatability"]["status"] == "fail"
    assert rows["photo", "repeatability"]["pass_share"] == pytest.approx(3 / 4)
    assert rows["photo", "ceiling_spread"]["status"] == "fail"
    assert rows["photo", "repeat_structure"]["status"] == "pass"
    kitchen["openings"] = kitchen["openings"][:1]
    c = capture_metrics(gt, gt.capture("photo_2"), result([kitchen], adjacency=()))
    rep = repeatability([a, c], cfg)
    rows = {(r["tier"], r["gate"]): r for r in gates_mod.evaluate([a, c], rep, None, None, cfg)["rows"]}
    assert rows["photo", "repeat_structure"]["status"] == "fail"


def test_drift_ablation_and_gate(gt):
    cfg = gates_mod.load_gates()
    on = capture_metrics(gt, gt.capture("video_1"), result(good_rooms("video"), tier="video", drift={"enabled": True}))
    hall, kitchen = good_rooms("video")
    kitchen["walls"][0]["length"] = M(3.3, 0.1)
    off = capture_metrics(gt, gt.capture("video_1"),
                          result([hall, kitchen], tier="video", footprint=18.0, drift={"enabled": False}))
    abl = drift_ablation([on], [off], cfg)
    e = abl["entries"][0]
    assert e["on"]["footprint_rel_err"] == pytest.approx(0.0)
    assert e["off"]["footprint_rel_err"] == pytest.approx(1.2 / 16.8)
    assert e["footprint_gain"] == pytest.approx(1.2 / 16.8) and e["wall_max_gain"] == pytest.approx(0.3)
    rows = {(r["tier"], r["gate"]): r for r in gates_mod.evaluate([on], None, abl, None, cfg)["rows"]}
    assert rows["video", "drift_ablation"]["status"] == "pass"
    rows = {(r["tier"], r["gate"]): r for r in gates_mod.evaluate([on], None, None, None, cfg)["rows"]}
    assert rows["video", "drift_ablation"]["status"] == "fail"


STATS_CSV = (
    "Project;Home\n\n"
    "Floor;Room;Area (m²);Perimeter (m);Wall Height (m)\n"
    "Ground floor;Kitchen;12,30;14,10;2,52\n"
    "Ground floor;Hall way;4,70;10,30;2,47\n"
    "Ground floor;Total;17,00;;\n"
)


def test_head_to_head_with_fake_magicplan_export(gt):
    folder = gt.root / "magicplan"
    folder.mkdir()
    (folder / "statistics.csv").write_text(STATS_CSV)
    (folder / "dimensions.yaml").write_text(yaml.safe_dump({
        "app": "magicplan", "version": "9.4.1", "mode": "AR camera, no LiDAR",
        "rooms": {"02 kitchen": {"name": "Kitchen", "walls": {"W1": 4.05, "W2": 2.99, "W3": "4.00 m", "W4": 3.10},
                                 "openings": {"D1": 0.85, "N1": "120 cm"}},
                  "01 hallway": {"name": "Hall way", "walls": {"W2": 4.002}}}}))
    stats = parse_statistics(folder / "statistics.csv")
    assert stats["columns"] == ["Floor", "Room", "Area (m²)", "Perimeter (m)", "Wall Height (m)"]
    assert stats["mapping"]["name"] == "Room" and stats["mapping"]["floor_area"] == "Area (m²)"
    assert [r["name"] for r in stats["rows"]] == ["Kitchen", "Hall way"]
    assert stats["rows"][0]["floor_area"] == pytest.approx(12.3)

    hall, kitchen = good_rooms()
    kitchen["walls"][1]["length"] = M(4.02, 0.1)  # GT W1 4.0: ours 2 cm, theirs 5 cm -> beat
    kitchen["walls"][0]["length"] = M(3.012, 0.1)  # GT W2 3.0: ours 12 mm, theirs 10 mm -> tie within 3 mm
    kitchen["walls"][3]["length"] = M(4.01, 0.1)  # GT W3 4.0: ours 1 cm, theirs 0 -> lose
    kitchen["floor_area"] = M(12.4, 0.5, "m2")  # ours 3.3%, theirs 2.5% -> lose (more than 0.2 pp worse)
    m = capture_metrics(gt, gt.capture("photo_1"), result([hall, kitchen]))
    h = head_to_head(gt, [m])
    assert h["version"] == "9.4.1" and "Room" in h["columns_found"]
    comp = h["comparisons"][0]
    dims = {(d["room"], d["dimension"]): d for d in comp["dimensions"]}
    assert dims["02 kitchen", "W1 length"]["beat_or_tie"]
    assert dims["02 kitchen", "W2 length"]["beat_or_tie"]
    assert not dims["02 kitchen", "W3 length"]["beat_or_tie"]
    assert not dims["02 kitchen", "floor area"]["beat_or_tie"]
    assert dims["02 kitchen", "N1 width"]["theirs"] == pytest.approx(1.2)
    assert dims["02 kitchen", "ceiling height"]["theirs"] == pytest.approx(2.52)
    assert dims["01 hallway", "W2 length"]["beat_or_tie"]
    assert comp["share"] == pytest.approx(comp["beat_or_tie"] / comp["n"])
    row = gates_mod.h2h_gate("photo", {"comparisons": h["comparisons"]}, {})
    assert row["measured"] == pytest.approx(comp["share"])
    kitchen["openings"] = kitchen["openings"][:1]  # our window is missing: a loss, not a skipped dimension
    comp2 = head_to_head(gt, [capture_metrics(gt, gt.capture("photo_1"), result([hall, kitchen]))])["comparisons"][0]
    n1 = next(d for d in comp2["dimensions"] if d["dimension"] == "N1 width")
    assert n1["ours"] is None and not n1["beat_or_tie"] and comp2["n"] == comp["n"]


def test_magicplan_template_placeholders_are_ignored():
    from scan2scope.bench.h2h import load_dimensions

    tmpl = load_dimensions(ROOT / "bench/templates/magicplan_dimensions.yaml")
    assert set(tmpl["rooms"]) == {"01 hallway", "02 kitchen"} and not tmpl["flags"]
    assert tmpl["rooms"]["02 kitchen"]["name"] == "Kitchen" and not tmpl["rooms"]["02 kitchen"]["walls"]


def test_parse_quantity_units():
    assert parse_quantity("12,5 m²", "area") == pytest.approx(12.5)
    assert parse_quantity("100 sq ft", "area") == pytest.approx(9.290304)
    assert parse_quantity("10' 6\"", "length") == pytest.approx(3.2004)
    assert parse_quantity("250 cm", "length") == pytest.approx(2.5)
    assert parse_quantity("1,234.5", "length") == pytest.approx(1234.5)
    assert parse_quantity("n/a", "length") is None


def test_report_generation(gt, tmp_path):
    cfg = gates_mod.load_gates()
    runs = [
        {"property": "home", "capture": "photo_1", "tier": "photo", "variant": "main", "status": "ok",
         "error": None, "run_s": 14.2, "result": result(good_rooms(), damage=[stain()])},
        {"property": "home", "capture": "photo_2", "tier": "photo", "variant": "main", "status": "ok",
         "error": None, "run_s": 6.0, "result": result([good_rooms()[1]], adjacency=())},
        {"property": "home", "capture": "video_1", "tier": "video", "variant": "main", "status": "failed",
         "error": "RuntimeError: decoder", "run_s": 1.0, "result": None},
    ]
    gt.synthetic = True
    bench = score_runs([gt], runs, cfg)
    bench["data_root"] = str(gt.root.parent)
    paths = write_report(tmp_path / "out", bench)
    text = paths["report"].read_text()
    for heading in ("## Gate summary", "## Failing gates ranked for the fix loop", "## Repeatability",
                    "## Calibration", "## Rooms, openings and damage", "## Drift ablation",
                    "## Head-to-head against magicplan", "## Timing", "## Failed or missing captures"):
        assert heading in text, heading
    assert "photo (synthetic)" in text and "Synthetic captures: home" in text
    assert "RuntimeError: decoder" in text
    metrics = json.loads(paths["metrics"].read_text())
    gates = json.loads(paths["gates"].read_text())
    assert len(metrics["metrics"]) == 3 and gates["rows"]
    assert any(r["gate"] == "result_produced" and r["tier"] == "video" and r["status"] == "fail" for r in gates["rows"])


def test_run_benchmark_with_fake_pipeline(gt, tmp_path):
    calls = []

    def fake_run(path, out_dir, *, tier, cache_mode, drift_correction, semantics, quiet):
        calls.append((Path(out_dir).name, drift_correction, semantics))
        if Path(path).name == "photo_2":
            raise RuntimeError("no room folders")
        res = result(good_rooms(tier), tier=tier, drift={"enabled": drift_correction})
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        (Path(out_dir) / "result.json").write_text(json.dumps(res))
        return res

    raw = gt.root / "raw"
    (raw / "photo_1").mkdir(parents=True)
    (raw / "photo_2").mkdir()
    (raw / "video_1.mov").write_bytes(b"")
    broken = gt.root.parent / "broken" / "ground_truth.yaml"
    broken.parent.mkdir()
    broken.write_text("rooms: [unclosed\n")
    out = tmp_path / "runs"
    bench = run_benchmark(gt.root.parent, out, run_fn=fake_run)
    assert len(bench["load_errors"]) == 1 and "could not be read" in (out / "benchmark_report.md").read_text()
    assert ("video_1__nodrift", False, False) in calls and ("video_1", True, True) in calls
    status = {m["capture"]: m["status"] for m in bench["metrics"]}
    assert status == {"photo_1": "ok", "photo_2": "failed", "video_1": "ok"}
    assert (out / "home" / "photo_2" / "error.txt").is_file()
    assert bench["ablation"]["entries"][0]["capture"] == "video_1"
    assert (out / "benchmark_report.md").is_file() and (out / "gates.json").is_file()

    again = run_benchmark(gt.root.parent, out, skip_run=True, only=["photo_1", "home/video_1"])
    assert {m["capture"]: m["status"] for m in again["metrics"]} == {"photo_1": "ok", "video_1": "ok"}
    assert len(again["nodrift_metrics"]) == 1
    rescored = {m["capture"]: m for m in run_benchmark(gt.root.parent, out, skip_run=True)["metrics"]}
    assert rescored["photo_2"]["status"] == "failed" and "no room folders" in rescored["photo_2"]["error"]
    assert rescored["photo_1"]["timing"]["run_s"] is not None


def test_compare_runs_before_after_table(gt, tmp_path, capsys):
    import importlib.util

    spec = importlib.util.spec_from_file_location("compare_runs", ROOT / "scripts" / "compare_runs.py")
    compare_runs = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compare_runs)
    cfg = gates_mod.load_gates()
    hall, kitchen = good_rooms()
    kitchen["walls"][3]["length"] = M(4.6, 0.05)
    for side, rooms in (("before", [hall, kitchen]), ("after", good_rooms())):
        run = {"property": "home", "capture": "photo_1", "tier": "photo", "variant": "main", "status": "ok",
               "error": None, "run_s": 1.0, "result": result(rooms)}
        bench = score_runs([gt], [run], cfg)
        write_report(tmp_path / side, bench)
    out = tmp_path / "diff.md"
    assert compare_runs.main([str(tmp_path / "before"), str(tmp_path / "after"), "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "wall_length" in printed and "fixed" in printed
    text = out.read_text()
    assert "## Worst gate before the fix" in text and "photo confident_garbage" in text
    assert "| photo | wall_length | fail | pass | fixed |" in text


def test_hand_built_result_is_schema_valid():
    import jsonschema

    schema = json.loads((ROOT / "schema/scan2scope.schema.json").read_text())
    jsonschema.validate(result(good_rooms(), damage=[stain()]), schema)
