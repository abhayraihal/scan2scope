"""Layout on an L-shaped room and on a hallway with three rooms in one frame (synthetic, ray-cast)."""

from __future__ import annotations

import layout_fixtures as lf
import numpy as np
import pytest
from shapely.geometry import Polygon

from scan2scope.layout import build_plan

NOISY = {"noise": 0.03, "normal_noise": 0.1, "drop": 0.3, "outliers": 0.05, "seed": 5}
YAW, SHIFT = -28.0, (3.0, 1.0)


@pytest.fixture(scope="module")
def l_syn():
    return lf.l_room()


@pytest.fixture(scope="module")
def house():
    return lf.hallway_three_rooms()


@pytest.fixture(scope="module")
def house_clean(house):
    return build_plan(lf.make_scene(house, noise=0.01, yaw_deg=YAW, shift=SHIFT))


@pytest.fixture(scope="module")
def house_noisy(house):
    return build_plan(lf.make_scene(house, yaw_deg=YAW, shift=SHIFT, **NOISY))


@pytest.mark.parametrize("kw,tol", [({"noise": 0.01}, 0.01), (NOISY, 0.04)])
def test_l_shaped_room(l_syn, kw, tol):
    plan = build_plan(lf.make_scene(l_syn, yaw_deg=12.0, **kw))
    assert len(plan.rooms) == 1
    room = plan.rooms[0]
    assert len(room.walls) == 6
    assert Polygon(room.polygon).exterior.is_ccw
    assert lf.cyclic_match([w.length.value for w in room.walls], [5.0, 2.5, 2.5, 2.5, 2.5, 5.0]) < tol
    truth = lf.truth_polygon(l_syn, "R", 12.0)
    assert Polygon(room.polygon).symmetric_difference(truth).area < tol * 20.0
    assert abs(room.floor_area.value - 18.75) < tol * 20.0
    assert [o.type for o in room.openings] == ["door"]
    assert abs(room.openings[0].width.value - 0.9) < max(2 * tol, 0.02)


@pytest.mark.parametrize("noisy", [False, True])
@pytest.mark.parametrize("passage,n_rooms", [("wide", 1), ("cased", 1), ("narrow", 2)])
def test_passage_width_decides_open_plan(passage, n_rooms, noisy):
    syn = lf.two_areas(passage)
    kw = NOISY if noisy else {"noise": 0.01}
    plan = build_plan(lf.make_scene(syn, **kw))
    assert len(plan.rooms) == n_rooms
    if n_rooms == 1:
        assert plan.rooms[0].floor_area.value == pytest.approx(24.36, abs=0.3)
        assert not plan.adjacency
        return
    assert sorted(r.label for r in plan.rooms) == ["east", "west"]
    for r in plan.rooms:
        assert lf.cyclic_match([w.length.value for w in r.walls], [4.0, 3.0, 4.0, 3.0]) < 0.04
        assert [o.type for o in r.openings] == ["opening"]
        o = r.openings[0]
        assert abs(o.width.value - 1.0) < 0.03
        assert o.connects_to == next(q.id for q in plan.rooms if q is not r)
    assert len(plan.adjacency) == 1 and plan.adjacency[0].opening_b is not None


@pytest.mark.parametrize("noisy", [False, True])
def test_corridor_with_aligned_walls_and_facing_doors_is_one_room(noisy):
    syn = lf.corridor_aligned()
    plan = build_plan(lf.make_scene(syn, **(NOISY if noisy else {"noise": 0.01})))
    assert sorted(r.label for r in plan.rooms) == ["A", "B", "C", "D", "K"]
    corridor = _room_by_label(plan, "K")
    assert lf.cyclic_match([w.length.value for w in corridor.walls], [6.12, 1.0, 6.12, 1.0]) < 0.04
    assert sorted(o.type for o in corridor.openings) == ["door"] * 4
    pairs = {frozenset((a.room_a, a.room_b)) for a in plan.adjacency}
    assert pairs == {frozenset((corridor.id, _room_by_label(plan, k).id)) for k in "ABCD"}
    assert lf.overlap_area([r.polygon for r in plan.rooms]) < 1e-6


def _room_by_label(plan, label):
    return next(r for r in plan.rooms if r.label == label)


@pytest.mark.parametrize("which,tol", [("clean", 0.01), ("noisy", 0.04)])
def test_hallway_and_three_rooms(request, house, which, tol):
    plan = request.getfixturevalue(f"house_{which}")
    assert len(plan.rooms) == 4
    assert [r.id for r in plan.rooms] == ["R1", "R2", "R3", "R4"]
    assert sorted(r.label for r in plan.rooms) == ["A", "B", "C", "H"]
    assert plan.rooms[0].label == "H"  # walking order: the capture starts in the hallway
    for name in "HABC":
        room = _room_by_label(plan, name)
        truth = lf.truth_polygon(house, name, YAW, SHIFT)
        assert Polygon(room.polygon).exterior.is_ccw
        assert len(room.walls) == 4
        x0, y0, x1, y1 = house.rooms[name][0]
        assert lf.cyclic_match([w.length.value for w in room.walls], [x1 - x0, y1 - y0] * 2) < tol, name
        assert Polygon(room.polygon).symmetric_difference(truth).area < tol * 2 * (x1 - x0 + y1 - y0), name
        assert abs(room.ceiling_height.value - 2.6) < 0.02
        assert room.view_ids and all(v.startswith("v") for v in room.view_ids)
    assert lf.overlap_area([r.polygon for r in plan.rooms]) < 1e-6
    assert plan.footprint_area.value == pytest.approx(sum(r.floor_area.value for r in plan.rooms))


@pytest.mark.parametrize("which", ["clean", "noisy"])
def test_adjacency_through_doors(request, which):
    plan = request.getfixturevalue(f"house_{which}")
    ids = {r.label: r.id for r in plan.rooms}
    pairs = {frozenset((a.room_a, a.room_b)) for a in plan.adjacency}
    assert pairs == {frozenset((ids["H"], ids[k])) for k in "ABC"}
    by_id = {o.id: o for r in plan.rooms for o in r.openings}
    for a in plan.adjacency:
        assert a.source == "shared_frame"
        assert a.opening_a and a.opening_b
        oa, ob = by_id[a.opening_a], by_id[a.opening_b]
        assert oa.type == ob.type == "door"
        assert oa.room_id == a.room_a and ob.room_id == a.room_b
        assert oa.connects_to == a.room_b and ob.connects_to == a.room_a
        assert np.linalg.norm(oa.center - ob.center) < 0.3
        assert abs(oa.width.value - ob.width.value) < 0.06


def test_doors_and_windows_match_truth(house_clean, house):
    labels = {r.label: r for r in house_clean.rooms}
    for t in house.openings:
        room = labels[t.room]
        c = lf.opening_center(t, YAW, SHIFT)
        o = min(room.openings, key=lambda q: np.linalg.norm(q.center - c))
        assert np.linalg.norm(o.center - c) < 0.5 * (t.u1 - t.u0), (t.room, t.type)
        assert o.type == t.type
        assert abs(o.width.value - (t.u1 - t.u0)) < 0.02, (t.room, t.type, o.width.value)
        if t.type == "window":
            assert abs(o.sill.value - t.z0) < 0.03 and o.connects_to is None
        else:
            assert abs(o.height.value - t.z1) < 0.02
            assert o.connects_to == labels[t.leads_to].id
    n_true = {k: sum(1 for t in house.openings if t.room == k) for k in "HABC"}
    assert {k: len(labels[k].openings) for k in "HABC"} == n_true


def test_extents_follow_the_wall_directions(house_clean):
    assert house_clean.extent_x.value == pytest.approx(8.5, abs=0.02)
    assert house_clean.extent_y.value == pytest.approx(4.5, abs=0.02)
    assert house_clean.extent_x.evidence["frame"] == "manhattan"
    assert house_clean.meta["layout"]["manhattan_angle_deg"] == pytest.approx(YAW, abs=0.5)
