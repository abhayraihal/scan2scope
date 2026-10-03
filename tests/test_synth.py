"""Synthetic apartments, renderer and Stray Scanner capture writer."""

from __future__ import annotations

import csv
import re
from pathlib import Path

import av
import cv2
import numpy as np
import pytest
import yaml
from scipy.spatial.transform import Rotation

from scan2scope.synth.apartment import TEMPLATES, free_mask, ground_truth_rooms, random_apartment
from scan2scope.synth.capture import (
    IMU_HEADER,
    ODOMETRY_HEADER,
    CaptureConfig,
    camera_rotations,
    lidar_measurement,
    odometry_poses,
    plan_trajectory,
    sample_drift,
    swift_float,
    write_capture,
)
from scan2scope.synth.generate import generate_benchmark, ground_truth_text
from scan2scope.synth.render import RenderScene, camera_rays, render_depth

TEMPLATE_GT = Path(__file__).resolve().parents[1] / "bench" / "templates" / "ground_truth.yaml"
K_TEST = np.array([[700.0, 0.0, 479.5], [0.0, 700.0, 359.5], [0.0, 0.0, 1.0]])


@pytest.fixture(scope="module")
def apt():
    return random_apartment(3, template="double", n_rooms=4, l_room=True, passage=True)


@pytest.fixture(scope="module")
def scene(apt):
    return RenderScene(apt)


@pytest.fixture(scope="module")
def capture(apt, scene, tmp_path_factory):
    out = tmp_path_factory.mktemp("synth") / "lidar_ideal"
    cfg = CaptureConfig(max_frames=12, noise=False, drift="none")
    return write_capture(apt, out, seed=7, config=cfg, scene=scene)


def _signed_area(P: np.ndarray) -> float:
    return 0.5 * float(np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


# ------------------------------------------------------------------------------------------- apartment


@pytest.mark.parametrize("seed", range(9))
def test_apartment_constraints(seed):
    apt = random_apartment(seed, template=TEMPLATES[seed % 3])
    assert apt.rooms[0].kind == "hallway" and apt.rooms[0].name == "01 hallway"
    assert 3 <= len(apt.rooms) - 1 <= 5
    assert 2400 <= apt.ceiling_mm <= 2800
    names = [r.name for r in apt.rooms]
    assert len(set(names)) == len(names)
    assert all(re.fullmatch(r"\d\d [a-z]+", n) for n in names)
    assert sum(len(r.rects) == 2 for r in apt.rooms) <= 1
    for room in apt.rooms[1:]:
        if len(room.rects) == 1:
            r = room.rects[0]
            assert 2400 <= r.w <= 5500 and 2400 <= r.h <= 5500
        door = apt.openings[room.entry]
        assert door.kind == "door" and {door.low, door.high} == {0, room.index}
        assert 700 <= door.width <= 920 and door.z0 == 0 and 2000 <= door.z1 <= 2100
        assert 100 <= door.thickness <= 150
    for o in apt.openings:
        if o.kind == "window":
            assert -1 in (o.low, o.high) and o.thickness == 200
            assert 800 <= o.width <= 1600 and 1000 <= o.z1 - o.z0 <= 1400 and 800 <= o.z0 <= 1000
            assert o.z1 <= apt.ceiling_mm - 150
        elif o.kind == "opening":
            assert 900 <= o.width <= 1200 and 0 not in (o.low, o.high) and 100 <= o.thickness <= 150
    for room in apt.rooms:
        P = room.polygon
        assert _signed_area(P) < 0  # clockwise seen from above
        assert abs(-_signed_area(P) - room.area) < 1e-6
    for b in apt.boxes:
        if b.kind != "wall":
            shape = apt.rooms[b.room].shape_m().buffer(1e-6)
            assert shape.contains(shape.__class__([(b.lo[0], b.lo[1]), (b.hi[0], b.lo[1]), (b.hi[0], b.hi[1]),
                                                  (b.lo[0], b.hi[1])]))


def test_apartment_is_deterministic():
    a, b = random_apartment(11), random_apartment(11)
    assert [r.name for r in a.rooms] == [r.name for r in b.rooms]
    assert [(x.lo, x.hi) for x in a.boxes] == [(x.lo, x.hi) for x in b.boxes]


def test_ground_truth_matches_apartment(apt):
    text = ground_truth_text(apt, "synth_t", ["lidar_1", "lidar_2", "lidar_drift"], date="2026-10-03")
    gt = yaml.safe_load(text)
    template = yaml.safe_load(TEMPLATE_GT.read_text())
    assert set(template) <= set(gt) and "adjacency" in gt
    assert gt["measured_by"] == "synthetic"
    assert set(template["rooms"][0]) | {"polygon"} == set(gt["rooms"][0])
    t_door = next(o for o in template["rooms"][0]["openings"] if o["type"] == "door")
    t_win = next(o for o in template["rooms"][0]["openings"] if o["type"] == "window")
    assert [c["id"] for c in gt["captures"]] == ["lidar_1", "lidar_2", "lidar_drift"]
    assert all(c["tier"] == "lidar" and c["path"] == f"raw/{c['id']}" for c in gt["captures"])
    assert len(gt["rooms"]) == len(apt.rooms)
    for room, g in zip(apt.rooms, gt["rooms"], strict=True):
        assert g["id"] == room.name
        P = np.asarray(g["polygon"])
        edges = np.linalg.norm(np.roll(P, -1, axis=0) - P, axis=1)
        assert [w["id"] for w in g["walls"]] == [f"W{k + 1}" for k in range(len(P))]
        assert np.allclose(edges, [w["length"] for w in g["walls"]], atol=1e-9)
        assert _signed_area(P) < 0
        assert abs(-_signed_area(P) - room.area) < 1e-6
        assert g["ceiling_height"] == [apt.ceiling] * 3
        if len(room.rects) == 1:
            w, h = room.rects[0].w / 1000, room.rects[0].h / 1000
            assert sorted(wall["length"] for wall in g["walls"]) == sorted([w, w, h, h])
            assert g["diagonal"] is None
        else:
            assert len(g["walls"]) == 6 and g["diagonal"] > 0
        lengths = {w["id"]: w["length"] for w in g["walls"]}
        for o in g["openings"]:
            assert o["offset"] >= 0 and o["offset"] + o["width"] <= lengths[o["wall"]] + 1e-9
            expected = set(t_win) if o["type"] == "window" else set(t_door)
            assert set(o) == expected
        if room.index > 0:
            d1 = g["openings"][0]
            assert (d1["id"], d1["type"], d1["wall"], d1["leads_to"]) == ("D1", "door", "W1", "01 hallway")
    pairs = {tuple(a["rooms"]) for a in gt["adjacency"]}
    assert all(("01 hallway", r.name) in pairs for r in apt.rooms[1:])
    assert any(a["type"] == "opening" for a in gt["adjacency"])


def test_entry_offset_is_from_left_end_seen_from_inside(apt):
    """Offsets run from the wall's left end as seen standing in the room facing the wall."""
    rooms = ground_truth_rooms(apt)
    for room, g in zip(apt.rooms[1:], rooms[1:], strict=True):
        door = apt.openings[room.entry]
        P = room.polygon
        a, b = P[0], P[1]
        d = (b - a) / np.linalg.norm(b - a)
        inward = np.array([d[1], -d[0]])
        facing = -inward
        left = np.array([-facing[1], facing[0]])
        assert np.dot(a - b, left) > 0  # vertex 0 is on the left
        cx, cy = door.centre_on_face(room.index)
        along = np.dot(np.array([cx, cy]) / 1000.0 - a, d)
        assert abs(along - (g["openings"][0]["offset"] + g["openings"][0]["width"] / 2)) < 1e-9


# ------------------------------------------------------------------------------------------- renderer


def _rect_room(apt):
    return next(r for r in apt.rooms[1:] if len(r.rects) == 1)


def test_depth_pixel_matches_analytic_distance(apt, scene):
    room = _rect_room(apt)
    r = room.rects[0]
    x0, y0, x1, y1 = r.x0 / 1000, r.y0 / 1000, r.x1 / 1000, r.y1 / 1000
    pos = np.array([0.5 * (x0 + x1), 0.5 * (y0 + y1), 1.3])
    R = camera_rotations(0.0, 0.0, 0.0, orientation="landscape")[0]
    dr = render_depth(scene, K_TEST, (960, 720), (256, 192), R, pos)
    east = [k for k in range(len(room.polygon)) if np.allclose(room.polygon[k, 0], x1)
            and np.allclose(room.polygon[(k + 1) % len(room.polygon), 0], x1)]
    sid = scene.surfaces.index(next(s for s in scene.surfaces if s.name == f"{room.name}/W{east[0] + 1}"))
    on_wall = dr.sid == sid
    assert on_wall.sum() > 1000
    # z-depth to a plane facing the camera is the same for every pixel
    assert np.allclose(dr.depth[on_wall], x1 - pos[0], atol=1e-7)
    assert np.allclose(dr.cos_incidence[96, 128], 1.0, atol=1e-4) or not on_wall[96, 128]

    R = camera_rotations(0.3, np.radians(-35.0), 0.0, orientation="portrait")[0]
    dr = render_depth(scene, K_TEST, (960, 720), (256, 192), R, pos)
    floor = dr.sid == scene.floor_sid[room.index]
    assert floor.sum() > 1000
    rays = camera_rays(K_TEST, (960, 720), (256, 192))
    d = rays @ R.T
    t = pos[2] / -d[:, 2]  # rays have z = 1 in the camera, so t is already the z-depth
    assert np.allclose(dr.depth.ravel()[floor.ravel()], t[floor.ravel()], atol=1e-7)


def test_camera_rotations_orientation():
    land = camera_rotations(0.4, 0.0, 0.0, orientation="landscape")[0]
    port = camera_rotations(0.4, 0.0, 0.0, orientation="portrait")[0]
    for R in (land, port):
        assert np.allclose(R.T @ R, np.eye(3)) and np.isclose(np.linalg.det(R), 1.0)
        assert np.allclose(R[:, 2], [np.cos(0.4), np.sin(0.4), 0.0])
    assert np.allclose(land[:, 1], [0, 0, -1])  # image y points down
    assert np.allclose(port[:, 0], [0, 0, -1])  # held upright, the sensor's x axis points down


# --------------------------------------------------------------------------------------- sensors and poses


def test_lidar_measurement_rules():
    depth = np.array([[1.0, 2.5, 3.5], [4.7, 5.2, np.inf]])
    cos = np.array([[1.0, 0.4, 1.0], [1.0, 1.0, 0.0]])
    mm, conf = lidar_measurement(depth, cos, None)
    assert mm.dtype == np.uint16 and conf.dtype == np.uint8
    assert mm.tolist() == [[1000, 2500, 3500], [4700, 0, 0]]
    assert conf.tolist() == [[2, 1, 1], [0, 0, 0]]

    rng = np.random.default_rng(0)
    flat = np.full((192, 256), 2.0)
    mm, conf = lidar_measurement(flat, np.ones_like(flat), rng)
    z = mm[mm > 0] / 1000.0
    assert abs(z.mean() - 2.0) < 0.002
    assert 0.014 < z.std() < 0.018  # sigma = 0.004 + 0.006 * 2
    assert set(np.unique(conf)) <= {0, 1, 2}
    assert np.all((mm == 0) == (conf == 0))


@pytest.mark.parametrize("value,double,expected", [
    (0.0, False, "0.0"), (1.0, False, "1.0"), (-0.5, False, "-0.5"), (0.1, False, "0.1"),
    (1e-05, False, "1e-05"), (0.0001, False, "0.0001"), (1597.2357, False, "1597.2357"),
    (-2.3841858e-07, False, "-2.3841858e-07"), (12.0, True, "12.0"),
    (52361.483529125, True, "52361.483529125"), (0.1, True, "0.1")])
def test_swift_float(value, double, expected):
    assert swift_float(value, double) == expected


def test_odometry_quaternion_convention():
    rng = np.random.default_rng(1)
    R = Rotation.random(5, random_state=2).as_matrix()
    pos = rng.normal(size=(5, 3))
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("x", 90, degrees=True).as_matrix()
    T[:3, 3] = [0.3, -0.2, 0.1]
    drift = sample_drift(rng, np.arange(5.0), "none")
    p_w, q = odometry_poses(pos, R, drift, T)
    assert np.allclose(Rotation.from_quat(q).as_matrix(), T[:3, :3] @ R)
    assert np.allclose(p_w, pos @ T[:3, :3].T + T[:3, 3])


def test_drift_magnitudes():
    t = np.arange(0.0, 180.0, 0.1)
    for seed in range(6):
        d = sample_drift(np.random.default_rng(seed), t, "normal")
        assert d.yaw[0] == 0.0 and np.allclose(d.trans[0], 0.0)
        assert 1.0 - 1e-9 <= abs(np.degrees(d.yaw[-1])) <= 3.0 + 1e-9
        assert 0.03 - 1e-9 <= np.linalg.norm(d.trans[-1]) <= 0.06 + 1e-9
    d = sample_drift(np.random.default_rng(0), t, "strong")
    assert np.isclose(abs(np.degrees(d.yaw[-1])), 4.0) and np.isclose(np.linalg.norm(d.trans[-1]), 0.10)
    d = sample_drift(np.random.default_rng(0), t, "none")
    assert not d.yaw.any() and not d.trans.any()


def test_trajectory_visits_rooms_and_returns(apt):
    traj = plan_trajectory(apt, np.random.default_rng(0), orientation="portrait")
    p = traj.pos
    assert apt.room_at(*p[0, :2]) == 0
    visited = {apt.room_at(x, y) for x, y in p[::10, :2]}
    assert set(range(len(apt.rooms))) <= visited
    assert np.linalg.norm(p[0] - p[-1]) < 0.03
    assert abs(np.degrees((traj.yaw[-1] - traj.yaw[0] + np.pi) % (2 * np.pi) - np.pi)) < 2.0
    assert abs(np.degrees(traj.pitch[-1] - traj.pitch[0])) < 1.0
    assert free_mask(apt, p[:, :2]).all() and not traj.flags
    assert p[:, 2].min() > 1.25 and p[:, 2].max() < 1.6
    assert 60.0 < traj.duration <= 210.0
    assert np.degrees(traj.pitch.min()) < -30 and np.degrees(traj.pitch.max()) > 20
    dt = 1.0 / traj.rate
    assert np.degrees(np.abs(np.gradient(traj.yaw, dt))).max() < 150.0  # no spins between samples
    assert np.degrees(np.abs(np.gradient(traj.pitch, dt))).max() < 120.0
    assert np.linalg.norm(np.gradient(p[:, :2], dt, axis=0), axis=1).max() < 1.2


def test_long_routes_are_walked_faster(apt):
    slow = plan_trajectory(apt, np.random.default_rng(4), max_duration=None)
    fast = plan_trajectory(apt, np.random.default_rng(4), max_duration=0.8 * slow.duration)
    assert slow.speed == 1.0 and fast.speed > 1.0
    assert fast.duration <= 0.8 * slow.duration + 1e-6
    # same route; height and sway noise differ because the sample count differs
    assert np.allclose(fast.pos[0], slow.pos[0], atol=0.03)
    assert np.allclose(fast.pos[-1], slow.pos[-1], atol=0.03)


# ------------------------------------------------------------------------------------------- capture files


def test_capture_layout_and_headers(capture):
    d = capture.path
    names = {p.name for p in d.iterdir()}
    assert {"rgb.mp4", "odometry.csv", "imu.csv", "camera_matrix.csv", "depth", "confidence"} <= names
    n = capture.n_frames
    assert sorted(p.name for p in (d / "depth").iterdir()) == [f"{i:06d}.png" for i in range(n)]
    assert sorted(p.name for p in (d / "confidence").iterdir()) == [f"{i:06d}.png" for i in range(n)]

    text = (d / "odometry.csv").read_text()
    assert text.startswith(ODOMETRY_HEADER)
    assert ODOMETRY_HEADER == ("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, "
                               "distortion_center_x, distortion_center_y\n")
    rows = list(csv.reader(text.splitlines()[1:], skipinitialspace=True))
    assert len(rows) == n
    assert all(len(r) == 15 and r[13] == "" and r[14] == "" for r in rows)
    assert [r[1] for r in rows] == [f"{i:06d}" for i in range(n)]
    ts = np.array([float(r[0]) for r in rows])
    assert np.allclose(np.diff(ts), 0.1, atol=1e-6)

    cm = (d / "camera_matrix.csv").read_text()
    assert not cm.endswith("\n") and len(cm.split("\n")) == 3
    K = np.array([[float(v) for v in line.split(", ")] for line in cm.split("\n")])
    fx, fy, cx, cy = (float(v) for v in rows[-1][9:13])
    assert np.allclose(K, [[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    imu = (d / "imu.csv").read_text().splitlines()
    assert imu[0] + "\n" == IMU_HEADER
    vals = np.array([[float(v) for v in line.split(", ")] for line in imu[1:]])
    assert vals.shape[1] == 7 and len(vals) >= 9 * n
    assert abs(np.median(np.linalg.norm(vals[:, 1:4], axis=1)) - 1.0) < 0.05  # g units, gravity included

    depth = cv2.imread(str(d / "depth" / "000000.png"), cv2.IMREAD_UNCHANGED)
    conf = cv2.imread(str(d / "confidence" / "000000.png"), cv2.IMREAD_UNCHANGED)
    assert depth.dtype == np.uint16 and depth.shape == (192, 256)
    assert conf.dtype == np.uint8 and conf.shape == (192, 256)
    assert set(np.unique(conf)) <= {0, 1, 2} and depth.max() > 500

    with av.open(str(d / "rgb.mp4")) as c:
        s = c.streams.video[0]
        assert s.codec_context.name in ("hevc", "h264") and (s.width, s.height) == (960, 720)
        assert s.time_base == 1 / 60 or float(s.time_base) == pytest.approx(1 / 60)
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(frames) == n and frames[0].shape == (720, 960, 3)


def _read_frame(d: Path, i: int):
    rows = list(csv.reader((d / "odometry.csv").read_text().splitlines()[1:], skipinitialspace=True))
    r = [float(v) for v in rows[i][2:13]]
    T = np.eye(4)
    T[:3, :3] = Rotation.from_quat(r[3:7]).as_matrix()
    T[:3, 3] = r[0:3]
    depth = cv2.imread(str(d / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED).astype(float) / 1000.0
    return T, r[7:11], depth


def test_poses_reproduce_rendered_geometry(apt, scene, capture):
    """Back-project depth with the written pose and intrinsics: points land on the apartment's surfaces."""
    T_pa = np.linalg.inv(capture.T_arkit_property)
    lo = np.array([b.lo for b in apt.boxes])
    hi = np.array([b.hi for b in apt.boxes])
    for i in (0, capture.n_frames - 1):
        T_wc, (fx, fy, cx, cy), depth = _read_frame(capture.path, i)
        h, w = depth.shape
        sx, sy = w / 960.0, h / 720.0
        vv, uu = np.nonzero(depth > 0)
        z = depth[vv, uu]
        x = (uu - ((cx + 0.5) * sx - 0.5)) / (fx * sx) * z
        y = (vv - ((cy + 0.5) * sy - 0.5)) / (fy * sy) * z
        pw = np.stack([x, y, z], 1) @ T_wc[:3, :3].T + T_wc[:3, 3]
        pp = pw @ T_pa[:3, :3].T + T_pa[:3, 3]
        dist = np.minimum(np.abs(pp[:, 2]), np.abs(pp[:, 2] - apt.ceiling))
        for b0, b1 in zip(lo, hi, strict=True):
            dist = np.minimum(dist, np.linalg.norm(pp - np.clip(pp, b0, b1), axis=1))
        assert len(pp) > 10000 and dist.max() < 0.002


def test_known_wall_point_reprojects(apt, scene, capture):
    """A wall point found from the true pose projects through the written pose onto its depth pixel."""
    i = capture.n_frames // 2
    T_wc, (fx, fy, cx, cy), depth = _read_frame(capture.path, i)
    rays = camera_rays(capture.K[i], (960, 720), (256, 192))
    j = 96 * 256 + 128
    d = rays[j] / np.linalg.norm(rays[j])
    hit = scene.cast(capture.true_pos[i], (capture.true_R[i] @ d)[None])
    assert scene.surfaces[hit.sid[0]].kind in ("wall", "floor", "ceiling", "furniture", "door_leaf", "trim")
    P = capture.T_arkit_property[:3, :3] @ hit.point[0] + capture.T_arkit_property[:3, 3]
    pc = T_wc[:3, :3].T @ (P - T_wc[:3, 3])
    sx, sy = 256 / 960.0, 192 / 720.0
    u = fx * sx * pc[0] / pc[2] + (cx + 0.5) * sx - 0.5
    v = fy * sy * pc[1] / pc[2] + (cy + 0.5) * sy - 0.5
    assert abs(u - 128) < 1e-3 and abs(v - 96) < 1e-3
    assert abs(depth[96, 128] - pc[2]) <= 0.0006


def test_generate_benchmark_layout(tmp_path):
    props = generate_benchmark(tmp_path, n_properties=1, seed=3, max_frames=6)
    p = props[0]
    assert p.name == "synth_0"
    gt = yaml.safe_load((p / "ground_truth.yaml").read_text())
    assert gt["property"] == "synth_0" and gt["measured_by"] == "synthetic"
    assert [c["id"] for c in gt["captures"]] == ["lidar_1", "lidar_2", "lidar_drift"]
    for c in gt["captures"]:
        d = p / c["path"]
        assert (d / "odometry.csv").exists() and (d / "rgb.mp4").exists()
        assert len(list((d / "depth").iterdir())) == 6
        truth = yaml.safe_load((p / "truth" / f"{c['id']}.yaml").read_text())
        assert truth["frames"] == 6 and np.asarray(truth["T_arkit_from_property"]).shape == (4, 4)
        assert (p / "truth" / f"{c['id']}_poses.csv").exists()
    one, drift = p / "raw" / "lidar_1", p / "raw" / "lidar_drift"
    assert (one / "depth" / "000003.png").read_bytes() == (drift / "depth" / "000003.png").read_bytes()
    assert (one / "odometry.csv").read_text() != (drift / "odometry.csv").read_text()
    assert yaml.safe_load((p / "truth" / "lidar_drift.yaml").read_text())["drift"]["mode"] == "strong"
    assert (p / "plan.png").stat().st_size > 10000
