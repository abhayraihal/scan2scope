"""Photo-tier stitching on synthetic rooms with exact ground truth and a fake MapAnything runner."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from shapely.geometry import Point, Polygon

from scan2scope.geometry.se3 import make_T, umeyama
from scan2scope.stitch import stitch_rooms
from scan2scope.stitch.doors import (
    door_match_hypotheses,
    door_pair_transform,
    facing_yaw,
    opening_door,
    room_doors,
    room_manhattan,
)
from scan2scope.stitch.doorway import align_run_to_room, as_predictions, find_doorway_photos, view_door_score
from scan2scope.stitch.solver import Door, StitchRoom, apply2, inv2, rigid2, snap_yaw, yaw2
from scan2scope.types import CameraView, Measurement, Opening, Plan, Room, Scene, Wall

K_IMG = np.array([[1300.0, 0.0, 800.0], [0.0, 1300.0, 600.0], [0.0, 0.0, 1.0]])
W_IMG, H_IMG = 1600, 1200
CEILING = 2.5
DOOR_H = 2.0


@dataclass
class Spec:
    hint: str
    rect: tuple[float, float, float, float]  # x0, y0, x1, y1 in the true property frame
    doors: list[tuple[int, float, float, str | None]]  # (wall 0..3, offset from wall start, width, leads to)
    windows: list[tuple[int, float, float]] = field(default_factory=list)


def rect_polygon(rect: tuple[float, float, float, float]) -> np.ndarray:
    x0, y0, x1, y1 = rect
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def look_at(c: np.ndarray, target: np.ndarray) -> np.ndarray:
    f = (target - c) / np.linalg.norm(target - c)
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    T = np.eye(4)
    T[:3, :3] = np.stack([r, np.cross(f, r), f], axis=1)
    T[:3, 3] = c
    return T


def m(v: float, kind: str = "length", unit: str = "m") -> Measurement:
    return Measurement(float(v), unit=unit, kind=kind)


def layout(k_width: float = 0.8, b_width: float = 0.8, w_width: float = 0.7, entry: float = 0.9,
           swap_hall_doors: bool = False) -> list[Spec]:
    """Hallway with a kitchen and a bedroom to the north, a bathroom to the south and an exterior door."""
    hall_doors = [
        (2, 6.0 - 1.0 - k_width, k_width, "02 kitchen"),
        (2, 6.0 - 4.0 - b_width, b_width, "03 bedroom"),
        (0, 4.4, w_width, "04 bathroom"),
        (3, 0.25, entry, None),
    ]
    if swap_hall_doors:
        hall_doors[0], hall_doors[1] = hall_doors[1], hall_doors[0]
    return [
        Spec("01 hallway", (0.0, 0.0, 6.0, 1.4), hall_doors),
        Spec("02 kitchen", (0.0, 1.52, 3.0, 4.82), [(0, 1.0, k_width, "01 hallway")],
             windows=[(2, 1.0, 1.2)]),
        Spec("03 bedroom", (3.12, 1.52, 6.6, 5.12), [(0, 4.0 - 3.12, b_width, "01 hallway")]),
        Spec("04 bathroom", (3.6, -2.32, 6.0, -0.12), [(2, 6.0 - 4.4 - w_width, w_width, "01 hallway")]),
    ]


def symmetric_layout() -> list[Spec]:
    """Two rooms of the same size with identical doors, so swapping them is also a valid layout."""
    return [
        Spec("01 hallway", (0.0, 0.0, 6.24, 1.4),
             [(2, 6.24 - 1.9, 0.8, "02 room"), (2, 6.24 - 5.14, 0.8, "03 room")]),
        Spec("02 room", (0.0, 1.52, 3.06, 4.52), [(0, 1.1, 0.8, "01 hallway")]),
        Spec("03 room", (3.18, 1.52, 6.24, 4.52), [(0, 4.34 - 3.18, 0.8, "01 hallway")]),
    ]


def _box_exit(rect: tuple[float, float, float, float], c: np.ndarray,
              D: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ray parameter where rays c + t D leave the box rect x [0, CEILING], and the face: walls 0-3 in
    rect_polygon edge order, 4 for floor or ceiling."""
    x0, y0, x1, y1 = rect

    def leave(lo: float, hi: float, ck: float, dk: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(dk > 0, (hi - ck) / dk, np.where(dk < 0, (lo - ck) / dk, np.inf))

    tx, ty = leave(x0, x1, c[0], D[:, 0]), leave(y0, y1, c[1], D[:, 1])
    tz = leave(0.0, CEILING, c[2], D[:, 2])
    t = np.minimum(np.minimum(tx, ty), tz)
    face = np.where(t == tz, 4, np.where(tx <= ty, np.where(D[:, 0] > 0, 1, 3), np.where(D[:, 1] > 0, 2, 0)))
    return t, face


def raycast(specs: dict[str, Spec], spec: Spec, T_wc: np.ndarray,
            shape: tuple[int, int] = (12, 16)) -> np.ndarray:
    """Point map (true frame) of a view inside spec's room; rays through a door go on into the next room."""
    h, w = shape
    jj, ii = np.meshgrid((np.arange(w) + 0.5) * W_IMG / w - 0.5, (np.arange(h) + 0.5) * H_IMG / h - 0.5)
    rays = np.stack([jj, ii, np.ones_like(jj)], -1).reshape(-1, 3) @ np.linalg.inv(K_IMG).T
    D = rays / np.linalg.norm(rays, axis=1, keepdims=True) @ T_wc[:3, :3].T
    c = T_wc[:3, 3]
    t, face = _box_exit(spec.rect, c, D)
    P = c + t[:, None] * D
    x0, y0, x1, y1 = spec.rect
    for wi, off, width, leads in spec.doors:
        centre = (x0 + off + width / 2, y0 + off + width / 2, x1 - off - width / 2, y1 - off - width / 2)[wi]
        along = P[:, 0] if wi in (0, 2) else P[:, 1]
        through = (face == wi) & (P[:, 2] < DOOR_H) & (np.abs(along - centre) <= width / 2)
        if not through.any():
            continue
        if leads in specs:
            t2, _ = _box_exit(specs[leads].rect, c, D[through])
        else:
            t2 = t[through] + 3.0
        P[through] = c + t2[:, None] * D[through]
    return P.reshape(h, w, 3)


@dataclass
class Capture:
    scenes: list[Scene]
    plans: list[Plan]
    views: dict[str, dict]  # view id -> true pose, room, door it looks through
    local_from_true: dict[str, tuple[np.ndarray, float, float]]  # hint -> (2-D similarity, z0, scale)


def make_capture(specs: list[Spec], seed: int, *, scale: dict[str, float] | None = None,
                 wall_noise: float = 0.0, door_noise: float = 0.0, pointmaps: bool = True) -> Capture:
    """Scenes and single-room plans in random per-room frames.

    wall_noise and door_noise (metres, 1 sigma) perturb what the layout reports (each wall position, door
    centre along the wall, door width), not reality.
    """
    rng = np.random.default_rng(seed)
    cap = Capture([], [], {}, {})
    by_hint = {sp.hint: sp for sp in specs}
    for spec in specs:
        theta, t, z0 = rng.uniform(-np.pi, np.pi), rng.uniform(-3.0, 3.0, 2), rng.uniform(-1.6, -1.2)
        s = (scale or {}).get(spec.hint, 1.0)
        S = rigid2(theta, t)
        S[:2, :2] *= s
        S3 = make_T(Rotation.from_euler("z", theta).as_matrix(), np.r_[t, z0], s)
        cap.local_from_true[spec.hint] = (S, z0, s)

        def loc(p: np.ndarray, S: np.ndarray = S) -> np.ndarray:
            return apply2(S, p)

        poly = rect_polygon(spec.rect)
        rect_p = tuple(np.asarray(spec.rect) + rng.normal(0.0, wall_noise, 4)) if wall_noise else spec.rect
        poly_p = rect_polygon(rect_p)
        walls = []
        for k in range(4):
            a, b = poly_p[k], poly_p[(k + 1) % 4]
            d = (b - a) / np.linalg.norm(b - a)
            walls.append(Wall(f"R1-W{k + 1}", "R1", loc(a), loc(b), m(s * np.linalg.norm(b - a)),
                              m(s * CEILING, "height"), Rotation.from_euler("z", theta).as_matrix()[:2, :2]
                              @ np.array([-d[1], d[0]]), observed_fraction=0.8))
        openings = []
        door_views = []
        for k, (wi, off, width, leads) in enumerate(spec.doors):
            a, b = poly[wi], poly[(wi + 1) % 4]
            d = (b - a) / np.linalg.norm(b - a)
            n_in = np.array([-d[1], d[0]])
            c = a + d * (off + width / 2)
            a_p = poly_p[wi]
            w_p = width + (rng.normal(0.0, door_noise / 2) if door_noise else 0.0)
            along = (c - a_p) @ d + (rng.normal(0.0, door_noise) if door_noise else 0.0)
            openings.append(Opening(f"R1-O{k + 1}", "R1", f"R1-W{wi + 1}", "door",
                                    m(s * (along - w_p / 2), "offset"), m(s * w_p, "width"),
                                    m(s * DOOR_H, "height"), center=loc(a_p + d * along), confidence=0.9))
            depth = min(1.5, abs((poly - a) @ n_in).max() - 0.3)
            cam = np.r_[c + depth * n_in, 1.4]
            door_views.append((look_at(cam, np.r_[c - 1.0 * n_in, 1.3]), leads))
        for k, (wi, off, width) in enumerate(spec.windows, start=len(openings) + 1):
            a, b = poly[wi], poly[(wi + 1) % 4]
            d = (b - a) / np.linalg.norm(b - a)
            openings.append(Opening(f"R1-O{k}", "R1", f"R1-W{wi + 1}", "window", m(s * off, "offset"),
                                    m(s * width, "width"), m(s * 1.2, "height"), sill=m(s * 0.9, "height"),
                                    center=loc(a + d * (off + width / 2)), confidence=0.8))
        centroid = poly.mean(0)
        views_true = []
        for p in poly:
            inward = (centroid - p) / np.linalg.norm(centroid - p)
            views_true.append((look_at(np.r_[p + 0.45 * inward, 1.4], np.r_[centroid, 1.2]), None, False))
        views_true += [(T, leads, True) for T, leads in door_views]
        views = []
        for k, (T_true, leads, doorway) in enumerate(views_true):
            vid = f"{spec.hint}/{k:02d}"
            T_local = S3 @ T_true
            T_local[:3, :3] /= s
            pm = raycast(by_hint, spec, T_true) if pointmaps else None
            pm_local = pm @ S3[:3, :3].T + S3[:3, 3] if pm is not None else None
            views.append(CameraView(vid, None, W_IMG, H_IMG, K_IMG.copy(), T_local, pointmap=pm_local,
                                    room_hint=spec.hint))
            cap.views[vid] = {"T_true": T_true, "room": spec.hint, "leads_to": leads, "doorway": doorway,
                              "pm_true": pm}
        pts, nrm = [], []
        for w in range(4):
            a, b = poly[w], poly[(w + 1) % 4]
            d = (b - a) / np.linalg.norm(b - a)
            for u in np.arange(0.05, np.linalg.norm(b - a), 0.1):
                for z in np.arange(0.1, CEILING, 0.2):
                    pts.append(np.r_[a + u * d, z])
                    nrm.append(np.r_[-d[1], d[0], 0.0])
        for x in np.arange(spec.rect[0] + 0.1, spec.rect[2], 0.2):
            for y in np.arange(spec.rect[1] + 0.1, spec.rect[3], 0.2):
                for z, nz in ((0.0, 1.0), (CEILING, -1.0)):
                    pts.append(np.r_[x, y, z])
                    nrm.append(np.r_[0.0, 0.0, nz])
        P = np.asarray(pts) @ S3[:3, :3].T + S3[:3, 3]
        N = np.asarray(nrm) @ Rotation.from_euler("z", theta).as_matrix().T
        cap.scenes.append(Scene("photo", views, P.astype(np.float32), N.astype(np.float32), np.ones(len(P)),
                                np.zeros(len(P), int), room_hint=spec.hint,
                                meta={"quality": {"n_views": len(views)}}))
        x0, y0, x1, y1 = rect_p
        area = (x1 - x0) * (y1 - y0)
        room = Room("R1", spec.hint.split(" ", 1)[1], loc(poly_p), walls, openings, z0, z0 + s * CEILING,
                    m(s * CEILING, "height"), m(s * s * area, "area", "m2"),
                    m(s * 2 * ((x1 - x0) + (y1 - y0))), view_ids=[v.id for v in views], source_hint=spec.hint)
        cap.plans.append(Plan([room], [], m(s * s * area, "area", "m2"), m(0.0), m(0.0), flags=["layout_ok"],
                              meta={"layout": {"room": spec.hint}}))
    return cap


def fake_runner(cap: Capture, specs: list[Spec], seed: int = 7, *, noise_pos: float = 0.02,
                noise_deg: float = 0.7):
    """True poses (in a random similarity of the true frame, plus noise) when the doorway photo really looks
    into the candidate room; a plausible but wrong pose for the doorway photo otherwise."""
    rng = np.random.default_rng(seed)
    rects = {s.hint: s.rect for s in specs}
    calls: list[list[str]] = []

    def run(views, key, cache):
        calls.append([v.id for v in views])
        target = {cap.views[v.id]["room"] for v in views[1:]}
        assert len(target) == 1
        target = target.pop()
        G_R = Rotation.random(random_state=rng).as_matrix()
        G_s, G_t = rng.uniform(0.97, 1.03), rng.uniform(-2.0, 2.0, 3)
        out = []
        for i, v in enumerate(views):
            info = cap.views[v.id]
            T = info["T_true"].copy()
            if i == 0 and info["leads_to"] != target:
                x0, y0, x1, y1 = rects[target]
                c = np.r_[rng.uniform(x0 + 0.3, x1 - 0.3), rng.uniform(y0 + 0.3, y1 - 0.3), 1.4]
                yaw = rng.uniform(-np.pi, np.pi)
                T = look_at(c, c + np.r_[np.cos(yaw), np.sin(yaw), -0.1])
            noise = Rotation.from_rotvec(rng.normal(size=3) * np.radians(noise_deg) / np.sqrt(3)).as_matrix()
            Tr = np.eye(4)
            Tr[:3, :3] = G_R @ T[:3, :3] @ noise
            Tr[:3, 3] = G_s * G_R @ (T[:3, 3] + rng.normal(size=3) * noise_pos / np.sqrt(3)) + G_t
            pts = None
            if info["pm_true"] is not None:
                # the view's points move with the pose it was given, wrong or not
                T0 = info["T_true"]
                P = (info["pm_true"] - T0[:3, 3]) @ T0[:3, :3] @ T[:3, :3].T + T[:3, 3]
                pts = G_s * (P + rng.normal(size=P.shape) * 0.01) @ G_R.T + G_t
            out.append(SimpleNamespace(T_wc=Tr, pts3d=pts))
        return out

    run.calls = calls
    return run


def no_runner(views, key, cache):
    raise AssertionError("the runner must not be called")


# --- checks ------------------------------------------------------------------------------------------

def align_to_truth(plan: Plan, specs: list[Spec]) -> np.ndarray:
    by_hint = {r.source_hint: r for r in plan.rooms}
    P = np.vstack([by_hint[s.hint].polygon for s in specs])
    Q = np.vstack([rect_polygon(s.rect) for s in specs])
    T3 = umeyama(np.c_[P, np.zeros(len(P))], np.c_[Q, np.zeros(len(Q))], with_scale=False)
    T = np.eye(3)
    T[:2, :2], T[:2, 2] = T3[:2, :2], T3[:2, 3]
    return T


def centre_errors(plan: Plan, specs: list[Spec]) -> dict[str, float]:
    T = align_to_truth(plan, specs)
    by_hint = {r.source_hint: r for r in plan.rooms}
    return {s.hint: float(np.linalg.norm(apply2(T, by_hint[s.hint].polygon).mean(0)
                                         - rect_polygon(s.rect).mean(0))) for s in specs}


def max_overlap(plan: Plan) -> float:
    polys = [Polygon(r.polygon) for r in plan.rooms]
    return max((a.intersection(b).area for i, a in enumerate(polys) for b in polys[i + 1:]), default=0.0)


def adjacency_hints(plan: Plan) -> set[frozenset]:
    hint = {r.id: r.source_hint for r in plan.rooms}
    return {frozenset((hint[a.room_a], hint[a.room_b])) for a in plan.adjacency}


def true_adjacency(specs: list[Spec]) -> set[frozenset]:
    return {frozenset((s.hint, leads)) for s in specs for *_, leads in s.doors if leads}


def uncertain_rooms(plan: Plan) -> set[str]:
    hint = {r.id: r.source_hint for r in plan.rooms}
    return {hint[f.split(":", 1)[1]] for f in plan.flags if f.startswith("placement_uncertain:")}


# --- end-to-end --------------------------------------------------------------------------------------

def test_doorway_photos_recover_layout(tmp_path):
    specs = layout()  # kitchen and bedroom doors are identical, so door matching alone cannot tell them apart
    cap = make_capture(specs, seed=1)
    runner = fake_runner(cap, specs)
    plan, scenes = stitch_rooms(cap.scenes, cap.plans, tmp_path, cache=None, runner=runner)

    assert [r.id for r in plan.rooms] == ["R1", "R2", "R3", "R4"]
    assert [r.source_hint for r in plan.rooms] == [s.hint for s in specs]
    errs = centre_errors(plan, specs)
    assert max(errs.values()) < 0.10, errs
    assert max_overlap(plan) < 0.05
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert all(a.source == "doorway_photo" for a in plan.adjacency)
    assert all(0.0 <= a.confidence <= 1.0 for a in plan.adjacency)
    assert not uncertain_rooms(plan), plan.flags
    assert runner.calls and all(len(c) <= 8 for c in runner.calls)

    # ids are rewritten consistently and openings point at the room on the other side
    for r in plan.rooms:
        wall_ids = {w.id for w in r.walls}
        assert all(w.id.startswith(f"{r.id}-W") and w.room_id == r.id for w in r.walls)
        assert all(o.id.startswith(f"{r.id}-O") and o.wall_id in wall_ids for o in r.openings)
        assert "layout_ok" in r.flags
    hall = plan.rooms[0]
    connects = {o.id: o.connects_to for o in hall.openings}
    assert connects == {"R1-O1": "R2", "R1-O2": "R3", "R1-O3": "R4", "R1-O4": None}
    assert plan.rooms[1].openings[1].type == "window" and plan.rooms[1].openings[1].connects_to is None

    # footprint, extents and the stitch record
    assert plan.footprint_area.value == pytest.approx(sum(r.floor_area.value for r in plan.rooms))
    assert plan.footprint_area.unit == "m2"
    # the hallway's walls end up axis aligned, but which of them lies along x depends on its local frame
    extents = sorted([plan.extent_x.value, plan.extent_y.value])
    assert extents == pytest.approx([6.6, 7.44], abs=0.15)
    st = plan.meta["stitch"]
    assert {"pairs_tested", "edges", "method_per_room", "scores", "uncertain"} <= set(st)
    assert st["method_per_room"]["R1"] == "root"
    assert {st["method_per_room"][k] for k in ("R2", "R3", "R4")} == {"doorway_photo"}
    assert st["uncertain"] == []
    json.dumps(st)
    assert (tmp_path / "stitch" / "stitch.json").exists()

    # scenes moved with their rooms: wall points sit on the room boundary, floors at z = 0, views inside
    assert len(scenes) == len(cap.scenes)
    for scene, room in zip(scenes, plan.rooms):
        assert scene.room_hint == room.source_hint and scene.meta["stitch"]["room_id"] == room.id
        poly = Polygon(room.polygon)
        walls = np.abs(scene.normals[:, 2]) < 0.5
        d = [poly.exterior.distance(Point(p)) for p in scene.points[walls, :2]]
        assert max(d) < 1e-3
        floor = scene.normals[:, 2] > 0.5
        assert np.abs(scene.points[floor, 2]).max() < 1e-4
        assert np.abs(scene.points[~walls & ~floor, 2] - CEILING).max() < 1e-4
        assert room.floor_z == pytest.approx(0.0, abs=1e-6)
        assert room.ceiling_z == pytest.approx(CEILING, abs=1e-6)
        before = {v.id: v for v in cap.scenes[[s.room_hint for s in cap.scenes].index(scene.room_hint)].views}
        for v in scene.views:
            assert np.allclose(v.T_wc[:3, :3] @ v.T_wc[:3, :3].T, np.eye(3), atol=1e-9)
            assert poly.buffer(0.01).contains(Point(v.center[:2]))
            # point maps moved rigidly with their camera
            v0 = before[v.id]
            cam = (v.pointmap - v.T_wc[:3, 3]) @ v.T_wc[:3, :3]
            cam0 = (v0.pointmap - v0.T_wc[:3, 3]) @ v0.T_wc[:3, :3]
            assert np.allclose(cam, cam0, atol=1e-6)
    # input scenes and plans are not modified
    assert cap.plans[0].rooms[0].id == "R1" and cap.plans[1].rooms[0].id == "R1"
    assert cap.scenes[0].meta.get("stitch") is None


def test_door_matching_alone_with_distinct_widths():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    cap = make_capture(specs, seed=2)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=no_runner, use_doorway_photos=False)
    errs = centre_errors(plan, specs)
    assert max(errs.values()) < 0.02, errs
    assert max_overlap(plan) < 0.05
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert all(a.source == "door_match" for a in plan.adjacency)
    assert not uncertain_rooms(plan), plan.flags
    assert plan.meta["stitch"]["method_per_room"] == {"R1": "root", "R2": "door_match", "R3": "door_match",
                                                     "R4": "door_match"}


def test_identical_doors_are_flagged_not_silently_picked():
    specs = symmetric_layout()
    cap = make_capture(specs, seed=3)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert uncertain_rooms(plan) == {"02 room", "03 room"}
    assert max_overlap(plan) < 0.05
    reasons = {u["room"]: u["reasons"] for u in plan.meta["stitch"]["uncertain"]}
    assert all(any(r.startswith("ambiguous") for r in v) for v in reasons.values())
    assert all(a.confidence <= 0.5 for a in plan.adjacency)
    for r in plan.rooms[1:]:
        assert f"placement_uncertain:{r.id}" in r.flags


def test_two_rooms_claiming_one_door_are_both_flagged():
    specs = [
        Spec("01 hallway", (0.0, 0.0, 4.0, 1.4), [(2, 1.5, 0.8, "02 room")]),
        Spec("02 room", (0.0, 1.52, 3.0, 4.52), [(0, 1.1, 0.8, "01 hallway")]),
        Spec("03 room", (0.0, 1.52, 3.0, 4.52), [(0, 1.1, 0.8, None)]),
    ]
    cap = make_capture(specs, seed=14)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert uncertain_rooms(plan) == {"02 room", "03 room"}
    reasons = {u["room"]: u["reasons"] for u in plan.meta["stitch"]["uncertain"]}
    assert any(r.startswith("ambiguous") for r in reasons["R2"]) and reasons["R3"] == ["unplaced"]
    assert max_overlap(plan) < 0.05


def test_doorway_photos_resolve_identical_doors():
    specs = symmetric_layout()
    cap = make_capture(specs, seed=3)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=fake_runner(cap, specs, seed=11))
    assert not uncertain_rooms(plan), plan.flags
    assert max(centre_errors(plan, specs).values()) < 0.10
    assert adjacency_hints(plan) == true_adjacency(specs)


def test_room_geometry_resolves_identical_doors_and_repairs_greedy_choice():
    # the bedroom's door is listed first in the hallway, so the greedy tree tries the kitchen there first; the
    # swap leaves no room for the bedroom, the local search repairs it, and no ambiguity is flagged (the
    # exterior door is too wide to take either room)
    specs = layout(k_width=0.8, b_width=0.8, entry=1.2, swap_hall_doors=True)
    cap = make_capture(specs, seed=4)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert max(centre_errors(plan, specs).values()) < 0.02
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert not uncertain_rooms(plan), plan.flags


def test_unplaceable_rooms_are_flagged_and_placed_without_overlap():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=1.2)
    # the closet has no compatible door anywhere
    specs.append(Spec("05 closet", (0.0, 0.0, 1.0, 1.0), [(0, 0.3, 0.45, None)]))
    # the study's door only matches doors that are taken, and matches them worse than the rooms behind them
    specs.append(Spec("06 study", (0.0, 0.0, 4.0, 4.0), [(0, 1.0, 0.74, None)]))
    cap = make_capture(specs, seed=5)
    plan, scenes = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert uncertain_rooms(plan) == {"05 closet", "06 study"}, plan.meta["stitch"]["uncertain"]
    assert max_overlap(plan) < 0.05
    placed = [r for r in plan.rooms if r.source_hint not in ("05 closet", "06 study")]
    right = max(r.polygon[:, 0].max() for r in placed)
    for r in plan.rooms[4:]:
        assert r.polygon[:, 0].min() >= right + 0.5
        assert r.evidence["stitch"]["method"] == "unplaced"
    assert plan.meta["stitch"]["method_per_room"]["R5"] == "unplaced"
    assert len(scenes) == 6
    # the four real rooms are still stitched correctly
    errs = centre_errors(Plan(placed, [], plan.footprint_area, plan.extent_x, plan.extent_y), specs[:4])
    assert max(errs.values()) < 0.02


def test_disconnected_pair_keeps_its_adjacency_to_the_right():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=1.2)
    # a wing of two rooms joined by a door that matches nothing in the main group
    specs.append(Spec("05 wing", (0.0, 0.0, 2.5, 2.5), [(1, 1.0, 0.5, "06 store")]))
    specs.append(Spec("06 store", (2.62, 0.0, 4.6, 2.5), [(3, 2.5 - 1.5, 0.5, "05 wing")]))
    cap = make_capture(specs, seed=18)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert uncertain_rooms(plan) == {"05 wing", "06 store"}
    assert sorted(map(sorted, plan.meta["stitch"]["components"])) == [["R1", "R2", "R3", "R4"], ["R5", "R6"]]
    right = max(r.polygon[:, 0].max() for r in plan.rooms[:4])
    assert all(r.polygon[:, 0].min() >= right + 0.5 for r in plan.rooms[4:])
    assert max(centre_errors(Plan(plan.rooms[4:], [], plan.footprint_area, plan.extent_x, plan.extent_y),
                             specs[4:]).values()) < 0.02
    assert max_overlap(plan) <= 0.05


def test_relative_scale_disagreement_is_flagged_not_applied():
    specs = layout()
    cap = make_capture(specs, seed=6, scale={"04 bathroom": 1.25})
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=fake_runner(cap, specs, seed=5))
    bath = plan.rooms[3]
    assert any(f.startswith("scale_disagreement:") and "R4" in f for f in plan.flags), plan.flags
    assert bath.floor_area.value == pytest.approx(1.25 ** 2 * 2.4 * 2.2)  # not rescaled
    assert plan.meta["stitch"]["relative_scale"]["R4"] == pytest.approx(1.25, rel=0.05)
    assert adjacency_hints(plan) == true_adjacency(specs)


def test_layout_noise_keeps_adjacency_and_raw_polygons_apart():
    specs = layout()
    cap = make_capture(specs, seed=128, wall_noise=0.06, door_noise=0.05)
    runner = fake_runner(cap, specs, seed=28, noise_pos=0.05, noise_deg=2.0)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=runner)
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert not uncertain_rooms(plan), plan.flags
    assert max_overlap(plan) <= 0.05
    assert max(centre_errors(plan, specs).values()) < 0.25


def test_overlap_from_an_oversized_room_is_nudged_and_flagged():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    cap = make_capture(specs, seed=15)
    # the layout made the kitchen 25 cm too wide on the bedroom side: the bedroom has to slide to fit
    wide = [Spec(s.hint, (s.rect[0], s.rect[1], s.rect[2] + 0.25, s.rect[3]), s.doors, s.windows)
            if s.hint == "02 kitchen" else s for s in specs]
    cap.plans[1] = make_capture(wide, seed=15).plans[1]
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert "placement_adjusted:R3" in plan.flags or "placement_adjusted:R2" in plan.flags, plan.flags
    assert max_overlap(plan) <= 0.05
    edge = next(e for e in plan.meta["stitch"]["edges"] if "placement_adjusted:" + e["child"] in plan.flags)
    assert 0.0 < edge["nudge_m"] <= 0.36


def test_facing_doors_after_placement_add_loop_adjacency():
    # kitchen and bedroom also share a door, which no tree edge uses
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    specs[1] = Spec("02 kitchen", (0.0, 1.52, 3.0, 4.82),
                    [(0, 1.0, 0.8, "01 hallway"), (1, 2.0, 0.7, "03 bedroom")])
    specs[2] = Spec("03 bedroom", (3.12, 1.52, 6.6, 5.12),
                    [(0, 0.88, 1.02, "01 hallway"), (3, 3.6 - 2.7, 0.7, "02 kitchen")])
    cap = make_capture(specs, seed=16)
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, use_doorway_photos=False)
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert len(plan.adjacency) == 4
    kb = [a for a in plan.adjacency if {a.room_a, a.room_b} == {"R2", "R3"}]
    assert len(kb) == 1 and kb[0].opening_a == "R2-O2" and kb[0].opening_b == "R3-O2"
    assert plan.rooms[1].openings[1].connects_to == "R3" and plan.rooms[2].openings[1].connects_to == "R2"


def test_doorway_photo_places_a_room_whose_layout_missed_the_door():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    cap = make_capture(specs, seed=17)
    cap.plans[3].rooms[0].openings = []  # the bathroom's layout found no door
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=fake_runner(cap, specs, seed=3))
    assert adjacency_hints(plan) == true_adjacency(specs)
    bath = next(a for a in plan.adjacency if "R4" in (a.room_a, a.room_b))
    assert bath.source == "doorway_photo" and bath.opening_b is None and bath.opening_a == "R1-O3"
    assert plan.rooms[0].openings[2].connects_to == "R4"
    assert max(centre_errors(plan, specs).values()) < 0.15
    assert max_overlap(plan) <= 0.05


def test_view_looking_out_of_a_room_without_doors_places_it():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    cap = make_capture(specs, seed=19)
    cap.plans[3].rooms[0].openings = []  # neither layout found the bathroom door
    cap.plans[0].rooms[0].openings = [o for o in cap.plans[0].rooms[0].openings if o.id != "R1-O3"]
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=fake_runner(cap, specs, seed=4))
    assert adjacency_hints(plan) == true_adjacency(specs)
    bath = next(a for a in plan.adjacency if "R4" in (a.room_a, a.room_b))
    assert bath.source == "doorway_photo" and bath.opening_a is None and bath.opening_b is None
    assert max(centre_errors(plan, specs).values()) < 0.15
    assert max_overlap(plan) <= 0.05
    regs = [r for r in plan.meta["stitch"]["doorway_registrations"] if r["room_a"] == "R4"]
    assert regs and all(r["opening_a"] is None for r in regs)


def test_runner_failure_falls_back_to_door_matching():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    cap = make_capture(specs, seed=7)

    calls = []

    def broken(views, key, cache):
        calls.append(1)
        raise RuntimeError("MPS out of memory")

    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=broken)
    assert "doorway_registration_failed:RuntimeError" in plan.flags
    assert "doorway_registration_stopped:repeated_failures" in plan.flags and len(calls) == 3
    assert adjacency_hints(plan) == true_adjacency(specs)
    assert max(centre_errors(plan, specs).values()) < 0.02

    def missing(views, key, cache):
        calls.append(1)
        raise ModuleNotFoundError("No module named 'scan2scope.geometry.mapanything_backend'")

    calls.clear()
    plan, _ = stitch_rooms(cap.scenes, cap.plans, None, runner=missing)
    assert "doorway_registration_unavailable:import_error" in plan.flags and len(calls) == 1
    assert adjacency_hints(plan) == true_adjacency(specs)


def test_degenerate_inputs_do_not_crash():
    plan, scenes = stitch_rooms([], [], None)
    assert plan.rooms == [] and scenes == [] and "no_rooms" in plan.flags
    assert plan.footprint_area.value == 0.0

    specs = layout()[:1]
    cap = make_capture(specs, seed=8)
    plan, scenes = stitch_rooms(cap.scenes, cap.plans, None, runner=no_runner)
    assert [r.id for r in plan.rooms] == ["R1"] and plan.adjacency == []
    assert plan.meta["stitch"]["method_per_room"] == {"R1": "root"}
    # the single room is turned so its walls are axis aligned, landscape, with its corner at the origin
    assert np.allclose(np.sort(plan.rooms[0].polygon[:, 0]), [0, 0, 6, 6], atol=1e-6)
    assert np.allclose(np.sort(plan.rooms[0].polygon[:, 1]), [0, 0, 1.4, 1.4], atol=1e-6)
    assert plan.extent_x.value == pytest.approx(6.0) and plan.extent_y.value == pytest.approx(1.4)

    # mismatched inputs, an empty plan and a room without walls or doors
    specs = layout()
    cap = make_capture(specs, seed=9)
    cap.plans[2] = Plan([], [], m(0.0, "area", "m2"), m(0.0), m(0.0))
    cap.plans[3].rooms[0].walls = []
    cap.plans[3].rooms[0].openings = []
    plan, scenes = stitch_rooms(cap.scenes[:3] + cap.scenes[3:], cap.plans, None, use_doorway_photos=False)
    assert "room_missing:03 bedroom" in plan.flags
    assert len(scenes) == 4 and len(plan.rooms) == 3
    assert max_overlap(plan) < 0.05
    assert scenes[2].meta["stitch"]["room_id"] is None


# --- unit tests --------------------------------------------------------------------------------------

def _door(c, n, w=0.8, kind="door") -> Door:
    return Door("X", kind, np.asarray(c, float), np.asarray(n, float) / np.linalg.norm(n), w, 2.0, 0.9)


def test_door_pair_transform_faces_doors_across_the_wall():
    da = _door([2.0, 1.0], [0.0, -1.0])
    db = _door([-1.0, 3.0], [np.cos(0.7), np.sin(0.7)])
    T = door_pair_transform(da, db, facing_yaw(da, db), 0.12)
    assert np.allclose(apply2(T, db.center), da.center - 0.12 * da.normal)
    assert np.allclose(T[:2, :2] @ db.normal, -da.normal)


def test_snap_yaw_within_five_degrees_only():
    theta, delta, snapped = snap_yaw(np.radians(93.0), 0.0, 0.0)
    assert snapped and theta == pytest.approx(np.radians(90.0)) and np.degrees(delta) == pytest.approx(3.0)
    theta, _, snapped = snap_yaw(np.radians(80.0), 0.0, 0.0)
    assert not snapped and theta == pytest.approx(np.radians(80.0))
    theta, _, snapped = snap_yaw(np.radians(10.0), np.radians(20.0), np.radians(12.0))
    assert snapped and np.degrees(theta) == pytest.approx(8.0)


def test_door_match_respects_width_ratio_and_manhattan_snap():
    specs = layout(k_width=0.8, b_width=1.02, w_width=0.62, entry=0.92)
    cap = make_capture(specs, seed=10)
    rooms = []
    for i, plan in enumerate(cap.plans):
        r = plan.rooms[0]
        rooms.append(StitchRoom(i, f"R{i + 1}", specs[i].hint, r, None, room_manhattan(r), room_doors(r),
                                np.asarray(r.polygon).mean(0)))
    hyps, compared = door_match_hypotheses(rooms)
    assert compared == 4 * 1 + 4 * 1 + 4 * 1 + 3
    for h in hyps:
        wa = rooms[h.a].door(h.opening_a).width
        wb = rooms[h.b].door(h.opening_b).width
        assert min(wa, wb) / max(wa, wb) >= 0.8
        # the relative yaw is a multiple of 90 degrees between the two room frames' Manhattan axes
        rel = (rooms[h.b].manhattan + yaw2(h.T_ab) - rooms[h.a].manhattan) % (np.pi / 2)
        assert min(rel, np.pi / 2 - rel) < 1e-9
    best = {}
    for h in hyps:
        if h.a == 0 and (h.b not in best or h.score > best[h.b].score):
            best[h.b] = h
    assert {(h.opening_a, h.b) for h in best.values()} == {("R1-O1", 1), ("R1-O2", 2), ("R1-O3", 3)}


def test_opening_door_geometry_from_wall_and_offset():
    specs = layout()
    cap = make_capture(specs, seed=12)
    room = cap.plans[0].rooms[0]
    o = room.openings[0]
    d1 = opening_door(room, o)
    o2 = Opening(o.id, o.room_id, o.wall_id, "door", o.offset, o.width, o.height, center=None)
    d2 = opening_door(room, o2)
    assert np.allclose(d1.center, d2.center) and np.allclose(d1.normal, d2.normal)
    assert opening_door(room, Opening(o.id, "R1", "R1-W3", "window", o.offset, o.width, o.height,
                                      center=o.center)) is None
    # unknown wall: the normal comes from the nearest polygon edge, also when the polygon is clockwise
    lost = Opening(o.id, "R1", "nope", "door", o.offset, o.width, o.height, center=o.center)
    assert np.allclose(opening_door(room, lost).normal, d1.normal)
    room.polygon = room.polygon[::-1].copy()
    assert np.allclose(opening_door(room, lost).normal, d1.normal)


def test_doorway_photo_detection_picks_the_view_through_each_door():
    specs = layout()
    cap = make_capture(specs, seed=13)
    for spec, scene, plan in zip(specs, cap.scenes, cap.plans):
        r = plan.rooms[0]
        sr = StitchRoom(0, "R1", spec.hint, r, scene, room_manhattan(r), room_doors(r),
                        np.asarray(r.polygon).mean(0))
        photos = find_doorway_photos(sr)
        assert len(photos) == len(spec.doors)
        for p in photos:
            info = cap.views[p.view.id]
            assert info["doorway"], (spec.hint, p.view.id)
            assert p.door.id == f"R1-O{1 + [d[3] for d in spec.doors].index(info['leads_to'])}"
        # corner views facing away from a door do not count
        assert all(view_door_score(v, sr.doors[0], r.floor_z) < 0.3 for v in scene.views
                   if not cap.views[v.id]["doorway"] and (np.asarray(v.T_wc)[:2, 2] @ sr.doors[0].normal) > 0)


def test_align_run_to_room_recovers_similarity():
    rng = np.random.default_rng(0)
    G = make_T(Rotation.random(random_state=rng).as_matrix(), rng.normal(size=3), 1.3)
    T_room = [look_at(np.r_[rng.uniform(0, 4, 2), 1.4], np.r_[2.0, 2.0, 1.2]) for _ in range(5)]
    T_run = []
    for T in T_room:
        Tr = np.eye(4)
        Tr[:3, :3] = G[:3, :3] / 1.3 @ T[:3, :3]
        Tr[:3, 3] = G[:3, :3] @ T[:3, 3] + G[:3, 3]
        T_run.append(Tr)
    al = align_run_to_room(T_room, T_run)
    assert al.method == "umeyama_centres"
    assert al.s == pytest.approx(1 / 1.3, rel=1e-6) and al.rms < 1e-6 and al.rot_err_deg < 1e-4
    al2 = align_run_to_room(T_room[:2], T_run[:2])
    assert al2.s == pytest.approx(1 / 1.3, rel=1e-6) and al2.rms < 1e-6
    al1 = align_run_to_room(T_room[:1], T_run[:1])
    assert al1.method.endswith("unit_scale") and al1.rot_err_deg < 1e-4


def test_as_predictions_accepts_objects_dicts_and_batches():
    T = np.stack([np.eye(4)] * 3)
    P = np.zeros((3, 4, 5, 3))
    objs = as_predictions([SimpleNamespace(T_wc=T[i], pts3d=P[i]) for i in range(3)], 3)
    dicts = as_predictions([{"camera_poses": T[i][None], "pts3d": P[i][None]} for i in range(3)], 3)
    batch = as_predictions({"T_wc": T, "pts3d": P}, 3)
    for preds in (objs, dicts, batch):
        assert len(preds) == 3 and preds[0].T_wc.shape == (4, 4) and preds[0].pts3d.shape == (4, 5, 3)
    assert as_predictions([SimpleNamespace(T_wc=np.eye(4))] * 2, 3) is None
    assert as_predictions(None, 3) is None


@pytest.mark.ml
def test_default_runner_returns_one_pose_per_view(tmp_path):
    pytest.importorskip("scan2scope.geometry.mapanything_backend")
    from PIL import Image

    from scan2scope.config import setup_env
    from scan2scope.stitch.doorway import mapanything_runner, run_key

    setup_env()
    try:
        from scan2scope.cache import OutputCache

        cache = OutputCache(mode="off", root=tmp_path / "cache")
    except ImportError:
        cache = None
    rng = np.random.default_rng(0)
    views = []
    yy, xx = np.mgrid[0:240, 0:320]
    for i in range(3):
        img = np.stack([(xx + 20 * i) % 256, (yy * 2) % 256, (xx + yy) % 256], -1).astype(np.uint8)
        rows = rng.integers(0, 200, 30)[:, None] + np.arange(8)
        cols = rng.integers(0, 300, 30)[:, None] + np.arange(8)
        img[rows, cols] = 255
        path = tmp_path / f"{i}.png"
        Image.fromarray(img).save(path)
        K = np.array([[260.0, 0.0, 160.0], [0.0, 260.0, 120.0], [0.0, 0.0, 1.0]])
        views.append(CameraView(f"v{i}", path, 320, 240, K, np.eye(4)))
    preds = as_predictions(mapanything_runner(views, run_key(views), cache), 3)
    assert preds is not None and len(preds) == 3
    assert all(p.T_wc.shape == (4, 4) and np.isfinite(p.T_wc).all() for p in preds)


def test_inverse_and_rigid_helpers():
    T = rigid2(0.3, (1.0, -2.0))
    assert np.allclose(inv2(T) @ T, np.eye(3))
    assert yaw2(T) == pytest.approx(0.3)
