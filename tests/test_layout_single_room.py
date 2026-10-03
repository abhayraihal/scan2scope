"""Layout on one rectangular room 4.2 x 3.1 x 2.6 m with a door and a window (synthetic, ray-cast)."""

from __future__ import annotations

import time

import layout_fixtures as lf
import numpy as np
import pytest
from shapely.geometry import Polygon

from scan2scope.layout import build_plan
from scan2scope.types import CameraView, Scene

YAW, SHIFT = 17.0, (1.5, -0.7)
NOISY = {"noise": 0.03, "normal_noise": 0.1, "drop": 0.3, "outliers": 0.05, "seed": 3}


@pytest.fixture(scope="module")
def syn():
    return lf.single_room()


@pytest.fixture(scope="module")
def clean(syn):
    return build_plan(lf.make_scene(syn, noise=0.01, yaw_deg=YAW, shift=SHIFT))


@pytest.fixture(scope="module")
def noisy(syn):
    return build_plan(lf.make_scene(syn, yaw_deg=YAW, shift=SHIFT, **NOISY))


def _symdiff(room, syn) -> float:
    return Polygon(room.polygon).symmetric_difference(lf.truth_polygon(syn, "R", YAW, SHIFT)).area


def _match_openings(room, syn, tol_frac=0.5, yaw=YAW, shift=SHIFT):
    """Pair each true opening with the detected one whose centre is within half its width."""
    pairs = []
    for t in syn.openings:
        c = lf.opening_center(t, yaw, shift)
        best = min(room.openings, key=lambda o: np.linalg.norm(o.center - c), default=None)
        ok = best is not None and np.linalg.norm(best.center - c) < tol_frac * (t.u1 - t.u0)
        pairs.append((t, best if ok else None))
    return pairs


def test_one_room_ccw_with_four_walls(clean):
    assert len(clean.rooms) == 1
    room = clean.rooms[0]
    assert room.id == "R1"
    assert len(room.walls) == 4
    assert [w.id for w in room.walls] == ["R1-W1", "R1-W2", "R1-W3", "R1-W4"]
    assert Polygon(room.polygon).exterior.is_ccw
    for k, w in enumerate(room.walls):
        assert np.allclose(w.start, room.polygon[k]) and np.allclose(w.end, room.polygon[(k + 1) % 4])
        d = (w.end - w.start) / np.linalg.norm(w.end - w.start)
        assert np.allclose(w.normal_in, [-d[1], d[0]], atol=1e-9)


def test_walls_within_1cm_at_lidar_noise(clean, syn):
    room = clean.rooms[0]
    assert lf.cyclic_match([w.length.value for w in room.walls], [4.2, 3.1, 4.2, 3.1]) < 0.01
    assert _symdiff(room, syn) < 0.01 * 2 * (4.2 + 3.1)
    assert abs(room.floor_area.value - 4.2 * 3.1) < 0.01 * 2 * (4.2 + 3.1)
    assert abs(room.perimeter.value - 2 * (4.2 + 3.1)) < 0.04
    assert abs(clean.footprint_area.value - room.floor_area.value) < 1e-9
    assert sorted([clean.extent_x.value, clean.extent_y.value]) == pytest.approx([3.1, 4.2], abs=0.01)


def test_ceiling_within_half_cm(clean):
    room = clean.rooms[0]
    assert abs(room.ceiling_height.value - 2.6) < 0.005
    assert all(abs(w.height.value - 2.6) < 0.005 for w in room.walls)


def test_door_and_window(clean, syn):
    room = clean.rooms[0]
    assert sorted(o.type for o in room.openings) == ["door", "window"]
    for t, o in _match_openings(room, syn):
        assert o is not None, f"missed {t.type}"
        assert o.type == t.type
        assert abs(o.width.value - (t.u1 - t.u0)) < 0.02
        if t.type == "door":
            assert abs(o.height.value - t.z1) < 0.02
            assert o.sill is None
        else:
            assert abs(o.sill.value - t.z0) < 0.03
            assert abs(o.height.value - (t.z1 - t.z0)) < 0.03
        wall = next(w for w in room.walls if w.id == o.wall_id)
        u = np.dot(o.center - wall.start, (wall.end - wall.start) / wall.length.value)
        assert abs(u - (o.offset.value + 0.5 * o.width.value)) < 1e-6
        assert o.id.startswith("R1-O") and o.connects_to is None
        assert 0.5 < o.confidence <= 1.0


def test_measurements_carry_evidence_without_intervals(clean):
    room = clean.rooms[0]
    ms = [room.ceiling_height, room.floor_area, room.perimeter, clean.footprint_area, clean.extent_x]
    ms += [w.length for w in room.walls] + [w.height for w in room.walls]
    ms += [m for o in room.openings for m in (o.offset, o.width, o.height, o.sill) if m is not None]
    for m in ms:
        assert m.lo is None and m.hi is None
        assert {"n_points", "fit_rms", "observed_fraction"} <= set(m.evidence)
        assert "sigma" not in m.evidence  # reserved for the uncertainty stage
    for o in room.openings:
        assert "edge_rms" in o.width.evidence
    assert {"floor_rms", "ceiling_rms"} <= set(room.ceiling_height.evidence)
    assert room.floor_area.unit == "m2" and room.floor_area.kind == "area"
    for w in room.walls:
        assert w.length.evidence["n_points"] > 1000 and 0.8 < w.observed_fraction <= 1.0


def test_noisy_partial_capture_walls_within_4cm(noisy, syn):
    assert len(noisy.rooms) == 1
    room = noisy.rooms[0]
    assert lf.cyclic_match([w.length.value for w in room.walls], [4.2, 3.1, 4.2, 3.1]) < 0.04
    assert _symdiff(room, syn) < 0.04 * 2 * (4.2 + 3.1)
    assert abs(room.ceiling_height.value - 2.6) < 0.02


def test_noisy_capture_has_no_phantom_openings(noisy, syn):
    room = noisy.rooms[0]
    pairs = _match_openings(room, syn)
    matched = {id(o) for _, o in pairs if o is not None}
    assert all(o is not None for _, o in pairs)
    assert all(id(o) in matched for o in room.openings), [(o.type, o.offset.value) for o in room.openings]


@pytest.mark.parametrize("seed", [0, 1])
def test_photo_like_views_with_rigid_misalignment(syn, seed):
    """Per-view rigid errors (3 cm, 0.7 deg) and 1% depth noise, like feed-forward multi-view point maps."""
    sc = lf.make_scene(syn, noise=0.02, depth_noise=0.01, view_jitter=(0.03, 0.7), drop=0.2, outliers=0.02,
                       seed=seed, tier="photo", room_hint="01 living")
    plan = build_plan(sc, single_room=True)
    room = plan.rooms[0]
    assert lf.cyclic_match([w.length.value for w in room.walls], [4.2, 3.1, 4.2, 3.1]) < 0.05
    assert abs(room.ceiling_height.value - 2.6) < 0.04
    pairs = _match_openings(room, syn, 0.5, 0.0, (0.0, 0.0))
    assert all(o is not None and o.type == t.type for t, o in pairs)
    assert len(room.openings) == 2
    door = next(o for o in room.openings if o.type == "door")
    assert abs(door.width.value - 0.9) < 0.05


@pytest.mark.parametrize("deg", [3.0, 6.0])
def test_out_of_square_wall_stays_one_wall(syn, deg):
    sc = lf.shear_wall_x(lf.make_scene(syn, noise=0.01), 3.95, 4.45, 1.55, deg)
    plan = build_plan(sc)
    room = plan.rooms[0]
    assert len(room.walls) == 4
    assert lf.cyclic_match([w.length.value for w in room.walls], [4.2, 3.1, 4.2, 3.1]) < 0.03
    assert sorted(o.type for o in room.openings) == ["door", "window"]
    slanted = [f for w in room.walls for f in w.flags if f.startswith("wall_slanted")]
    assert len(slanted) == 1 and abs(float(slanted[0].split(":")[1].rstrip("deg")) - deg) < 1.0


def test_single_room_mode_and_label_from_folder_name(syn):
    sc = lf.make_scene(syn, noise=0.01, room_hint="02 kitchen", tier="photo")
    plan = build_plan(sc, single_room=True)
    assert len(plan.rooms) == 1
    room = plan.rooms[0]
    assert room.label == "kitchen" and room.source_hint == "02 kitchen"
    assert set(room.view_ids) == {v.id for v in sc.views}


def test_furniture_occlusion_is_not_an_opening():
    syn = lf.single_room(furniture=True, occluder=True)
    for kw in ({"noise": 0.01}, NOISY):
        plan = build_plan(lf.make_scene(syn, **kw))
        assert len(plan.rooms) == 1
        room = plan.rooms[0]
        assert lf.cyclic_match([w.length.value for w in room.walls], [4.2, 3.1, 4.2, 3.1]) < 0.04
        assert sorted(o.type for o in room.openings) == ["door", "window"]
        hidden = next(w for w in room.walls if abs(w.start[0]) < 0.05 and abs(w.end[0]) < 0.05)
        assert not [o for o in room.openings if o.wall_id == hidden.id]
        assert hidden.observed_fraction < 0.85


def test_two_million_points_under_30s(syn):
    base = lf.make_scene(syn, noise=0.0)
    rng = np.random.default_rng(7)
    reps = int(np.ceil(2_000_000 / len(base.points)))
    P = np.concatenate([base.points + rng.normal(0, 0.01, base.points.shape) for _ in range(reps)])
    P = P[:2_000_000]
    k = len(P)
    sc = Scene("lidar", base.views, P.astype(np.float32), np.tile(base.normals, (reps, 1))[:k],
               np.tile(base.weights, reps)[:k], np.tile(base.view_index, reps)[:k])
    t0 = time.perf_counter()
    plan = build_plan(sc)
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, elapsed
    assert len(plan.rooms) == 1
    assert lf.cyclic_match([w.length.value for w in plan.rooms[0].walls], [4.2, 3.1, 4.2, 3.1]) < 0.01


@pytest.mark.parametrize("case", ["no_view_index", "unoriented_normals", "far_outliers", "missing_wall"])
def test_degraded_scene_still_gives_the_room(syn, case):
    base = lf.make_scene(syn, noise=0.01)
    P, N, V = base.points.copy(), base.normals.copy(), base.view_index.copy()
    rng = np.random.default_rng(11)
    if case == "no_view_index":
        V[:] = -1
    elif case == "unoriented_normals":
        N[rng.random(len(N)) < 0.5] *= -1
    elif case == "far_outliers":
        far = rng.uniform(-45, 45, (20000, 3)).astype(np.float32)
        P = np.concatenate([P, far])
        N = np.concatenate([N, np.tile(np.float32([0, 0, 1]), (len(far), 1))])
        V = np.concatenate([V, rng.integers(0, len(base.views), len(far))])
    elif case == "missing_wall":
        keep = P[:, 0] < 4.0
        P, N, V = P[keep], N[keep], V[keep]
    sc = Scene("lidar", base.views, P, N, np.ones(len(P), np.float32), V)
    plan = build_plan(sc)
    assert len(plan.rooms) == 1, plan.flags
    room = plan.rooms[0]
    truth = lf.truth_polygon(syn, "R")
    tol = 0.12 if case == "missing_wall" else 0.03
    assert Polygon(room.polygon).symmetric_difference(truth).area < tol * truth.area
    if case == "missing_wall":
        flags = [f for w in room.walls for f in w.flags]
        assert "wall_unobserved" in flags or "wall_face_missing" in flags
    elif case == "no_view_index":
        assert "free_space_from_floor" in plan.flags
    else:
        assert lf.cyclic_match([w.length.value for w in room.walls], [4.2, 3.1, 4.2, 3.1]) < 0.02


def _view() -> CameraView:
    return CameraView("v0", None, 64, 48, np.eye(3), np.eye(4))


@pytest.mark.parametrize("case", ["empty", "nan", "no_normals", "floor_only", "no_views", "few"])
def test_degenerate_input_returns_flagged_plan(case):
    rng = np.random.default_rng(0)
    n = 5000
    P = np.column_stack([rng.uniform(0, 3, n), rng.uniform(0, 3, n), np.zeros(n)])
    N = np.tile([0.0, 0.0, 1.0], (n, 1))
    views = [_view()]
    if case == "empty":
        P, N = np.zeros((0, 3)), np.zeros((0, 3))
    elif case == "nan":
        P[::2] = np.nan
    elif case == "no_normals":
        N = None
    elif case == "no_views":
        views = []
    elif case == "few":
        P, N = P[:50], N[:50]
    k = 0 if P is None else len(P)
    sc = Scene("photo", views, P, N, np.ones(k), np.zeros(k, int))
    plan = build_plan(sc)
    assert isinstance(plan.flags, list) and plan.flags
    assert plan.footprint_area.value >= 0
    assert all(len(r.polygon) >= 3 for r in plan.rooms)
    if case == "floor_only":
        assert len(plan.rooms) == 1
        room = plan.rooms[0]
        assert 7.0 < room.floor_area.value < 10.0 and not room.openings
        assert all("wall_unobserved" in w.flags for w in room.walls)
    if case in ("empty", "few", "no_normals"):
        assert not plan.rooms and "no_rooms" in plan.flags


def test_walls_without_rays_or_floor_fall_back_to_the_seen_extent():
    rng = np.random.default_rng(1)
    t, z = rng.uniform(0, 3, 4000), rng.uniform(0.2, 2.4, 4000)
    zero, three = np.zeros_like(t), np.full_like(t, 3)
    P = np.concatenate([np.column_stack([t, zero, z]), np.column_stack([t, three, z]),
                        np.column_stack([zero, t, z]), np.column_stack([three, t, z])])
    N = np.concatenate([np.tile(v, (4000, 1)) for v in ([0, 1, 0], [0, -1, 0], [1, 0, 0], [-1, 0, 0])])
    sc = Scene("photo", [_view()], P, N.astype(float), np.ones(len(P)), np.full(len(P), -1))
    plan = build_plan(sc, single_room=True)
    assert "room_fallback_extent" in plan.flags and len(plan.rooms) == 1
    assert 6.0 < plan.rooms[0].floor_area.value < 9.5
