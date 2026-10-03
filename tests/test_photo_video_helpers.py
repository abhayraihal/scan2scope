"""Photo and video geometry without model weights: a box room rendered analytically stands in for MapAnything."""

import hashlib
import itertools
import json
import math

import numpy as np
import pytest
from PIL import Image

from scan2scope.geometry import mapanything_backend as mb
from scan2scope.geometry import photo, se3, video

LX, LY, LZ = 5.0, 4.0, 2.6
W, H, F = 64, 48, 50.0
K_TRUE = np.array([[F, 0.0, (W - 1) / 2], [0.0, F, (H - 1) / 2], [0.0, 0.0, 1.0]])


def camera_pose(center, yaw, pitch_deg):
    d = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = np.cross(d, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    th = np.radians(pitch_deg)
    z = np.cos(th) * d + np.sin(th) * np.array([0.0, 0.0, 1.0])
    y = np.cross(z, x)
    return se3.make_T(np.stack([x, y, z], 1), np.asarray(center, float))


def render_box(T_wc):
    u, v = np.meshgrid(np.arange(W, dtype=float), np.arange(H, dtype=float))
    rays = np.stack([(u - K_TRUE[0, 2]) / F, (v - K_TRUE[1, 2]) / F, np.ones_like(u)], -1) @ T_wc[:3, :3].T
    c = T_wc[:3, 3]
    t = np.full(rays.shape[:2], np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        for axis, lim in ((0, LX), (1, LY), (2, LZ)):
            for plane in (0.0, lim):
                tt = (plane - c[axis]) / rays[..., axis]
                tt[~np.isfinite(tt) | (tt <= 1e-9)] = np.inf
                t = np.minimum(t, tt)
    return c + t[..., None] * rays


class FakeRunner:
    """Renders each call in the frame of its first camera, with a random per-call Sim(3) and an optional bend
    that grows with the position in the call (yaw, tilt, height and scale), so chained chunks drift."""

    def __init__(self, poses_by_sha, bend=(0.0, 0.0, 0.0, 0.0), seed=0):
        self.poses = poses_by_sha
        self.bend = bend
        self.seed = seed
        self.calls = []

    def infer(self, images, intrinsics=None, key=None, cache=None):
        shas = [mb.array_sha256(mb.as_rgb_uint8(im)) for im in images]
        Ks = list(intrinsics) if intrinsics is not None else [None] * len(images)
        self.calls.append({"n": len(images), "intrinsics": [k is not None for k in Ks], "key": key})
        Ts = [self.poses[s] for s in shas]
        seed = int(hashlib.sha256(("".join(shas) + str(self.seed)).encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        D = se3.make_T(se3.rotvec_to_R(rng.normal(size=3) * np.radians(2.0)), rng.normal(size=3) * 0.1,
                       float(np.exp(rng.normal() * 0.03)))
        to_run = D @ se3.invert(Ts[0])
        c0 = Ts[0][:3, 3]
        yaw, tilt, dz, dlog = self.bend
        out = []
        for i, T in enumerate(Ts):
            Rb = se3.rot_z(np.radians(yaw * i)) @ se3.rotvec_to_R(np.radians([tilt * i, 0.0, 0.0]))
            B = se3.make_T(Rb, np.zeros(3), float(np.exp(dlog * i)))
            B[:3, 3] = c0 - B[:3, :3] @ c0 + np.array([0.0, 0.0, dz * i])
            M = to_run @ B
            pts = se3.apply(M, render_box(T)) + rng.normal(scale=0.002, size=(H, W, 3))
            MT = M @ T
            T_out = se3.make_T(se3.decompose_sim3(MT)[1], MT[:3, 3])
            out.append(mb.ViewPrediction(
                pts3d=pts.astype(np.float32), conf=np.full((H, W), 20.0, np.float32), mask=np.ones((H, W), bool),
                T_wc=T_out, K=K_TRUE.copy(), metric_scale=float(se3.decompose_sim3(D)[0]), image_size=(W, H),
                resized_size=(W, H), crop=(0, 0), intrinsics_given=Ks[i] is not None))
        return out


def _noise_image(rng):
    return rng.integers(0, 255, size=(H, W, 3), dtype=np.uint8)


def _ate(T_est, T_true):
    A = np.array([T[:3, 3] for T in T_est])
    B = np.array([T[:3, 3] for T in T_true])
    S = se3.umeyama(A, B)
    return float(np.sqrt(np.mean(np.sum((se3.apply(S, A) - B) ** 2, axis=1)))), S


# ------------------------------------------------------------------------------------------- backend mapping


def test_target_size_and_resize_geometry_match_mapanything_rules():
    assert mb.target_size([(4032, 3024)]) == (518, 392)
    assert mb.target_size([(1920, 1080)]) == (518, 294)
    assert mb.target_size([(3024, 4032)]) == (392, 518)
    assert mb.target_size([(779, 520)] * 3) == (518, 336)
    assert mb.resize_geometry(4032, 3024, (518, 392)) == ((522, 392), (2, 0))
    assert mb.resize_geometry(1920, 1080, (518, 294)) == ((522, 294), (2, 0))
    assert mb.resize_geometry(779, 520, (518, 336)) == ((518, 345), (0, 4))


def test_preprocess_image_has_target_size():
    img = np.random.default_rng(0).integers(0, 255, size=(300, 400, 3), dtype=np.uint8)
    assert mb.preprocess_image(img, (518, 392)).size == (518, 392)
    assert mb.preprocess_image(img[:50, :60], (518, 392)).size == (518, 392)


def test_pixel_mapping_and_intrinsics_round_trip():
    size, target = (4032, 3024), (518, 392)
    K0 = np.array([[3000.0, 0.0, 2015.5], [0.0, 3000.0, 1511.5], [0.0, 0.0, 1.0]])
    Km = mb.intrinsics_to_model(K0, size, target)
    (w2, h2), crop = mb.resize_geometry(*size, target)
    pred = mb.ViewPrediction(np.zeros((392, 518, 3), np.float32), np.ones((392, 518), np.float32),
                             np.ones((392, 518), bool), np.eye(4), Km, 1.0, size, (w2, h2), crop)
    assert np.allclose(pred.K_image(), K0, atol=1e-9)
    assert np.allclose(pred.scale, (522 / 4032, 392 / 3024))
    u, v = np.array([0.0, 100.0, 517.0]), np.array([0.0, 50.0, 391.0])
    ui, vi = pred.model_to_image(u, v)
    assert np.allclose(pred.image_to_model(ui, vi), (u, v))
    # a point projected with K0 lands on the model pixel predicted by Km
    X = np.array([0.3, -0.2, 2.0])
    p0 = K0 @ X / X[2]
    pm = Km @ X / X[2]
    assert np.allclose(pred.image_to_model(p0[0], p0[1]), pm[:2], atol=1e-9)
    # half-resolution copy of the photo: same content, intrinsics halve
    Kh = pred.K_image(2016, 1512)
    assert np.isclose(Kh[0, 0], 1500.0) and np.isclose(Kh[0, 2], (2015.5 + 0.5) / 2 - 0.5)
    full = pred.uncrop(np.ones((392, 518), bool), False)
    assert full.shape == (392, 522) and full[:, :2].sum() == 0 and full[:, 2:520].all()


def test_confidence_weight_is_bounded_and_monotone():
    w = mb.confidence_weight(np.array([0.5, 1.0, 2.0, 10.0, 30.0, 500.0, np.nan]))
    assert w[0] == 0.0 and w[1] == 0.0 and w[-1] == 0.0
    assert np.all(np.diff(w[1:5]) > 0) and w[4] == 1.0 and w[5] == 1.0


class _DictCache:
    def __init__(self):
        self.store, self.keys = {}, []

    def compute(self, key, fn):
        k = json.dumps(key, sort_keys=True)
        self.keys.append(key)
        if k not in self.store:
            self.store[k] = fn()
        return self.store[k]


def test_runner_infer_caches_and_builds_predictions(monkeypatch):
    runner = mb.MapAnythingRunner()
    calls = []

    def fake_run(imgs, Km, target):
        calls.append((len(imgs), [k is not None for k in Km], target))
        n, (w, h) = len(imgs), target
        return {"pts3d": np.ones((n, h, w, 3), np.float32), "conf": np.full((n, h, w), 5.0, np.float32),
                "mask": np.ones((n, h, w), bool), "T_wc": np.tile(np.eye(4), (n, 1, 1)),
                "K": np.tile(np.eye(3), (n, 1, 1)), "metric_scale": np.full(n, 2.2)}

    monkeypatch.setattr(runner, "_run", fake_run)
    rng = np.random.default_rng(0)
    imgs = [rng.integers(0, 255, size=(120, 160, 3), dtype=np.uint8) for _ in range(3)]
    K = np.array([[150.0, 0, 79.5], [0, 150.0, 59.5], [0, 0, 1]])
    cache = _DictCache()
    preds = runner.infer(imgs, [K, None, K], key={"room": "a"}, cache=cache)
    assert calls == [(3, [True, False, True], (518, 392))]
    assert len(preds) == 3 and preds[0].pts3d.shape == (392, 518, 3) and preds[0].resized_size == (522, 392)
    assert preds[0].intrinsics_given and not preds[1].intrinsics_given
    assert "intrinsics_partial:2/3" in preds[0].meta["flags"]
    key = cache.keys[0]
    assert key["revision"] == mb.MODELS["mapanything"].revision and len(key["images"]) == 3
    assert key["caller"] == {"room": "a"} and key["intrinsics"][1] is None
    again = runner.infer(imgs, [K, None, K], key={"room": "a"}, cache=cache)
    assert len(calls) == 1 and np.array_equal(again[2].pts3d, preds[2].pts3d)
    monkeypatch.setattr(mb, "ALLOW_PARTIAL_INTRINSICS", False)
    strict = runner.infer(imgs, [K, None, K])
    assert calls[-1][1] == [False, False, False] and "intrinsics_dropped_partial" in strict[0].meta["flags"]
    with pytest.raises(ValueError):
        runner.infer([imgs[0]] * (mb.MAX_VIEWS + 1))
    with pytest.raises(ValueError):
        runner.infer([])


# ------------------------------------------------------------------------------------------- photo helpers


def test_list_room_folders_natural_order_and_filters(tmp_path):
    root = tmp_path / "Scan"
    for name in ["10 bath", "2 kitchen", "1 hall", ".hidden", "__MACOSX", "empty"]:
        (root / name).mkdir(parents=True)
    img = Image.fromarray(np.zeros((8, 8, 3), np.uint8))
    for room, files in {"10 bath": ["a.jpg"], "2 kitchen": ["IMG_10.JPG", "IMG_9.jpg", "._IMG_9.jpg", "IMG_9.MOV"],
                        "1 hall": ["x.png"], ".hidden": ["h.jpg"], "__MACOSX": ["m.jpg"]}.items():
        for f in files:
            if f.lower().endswith((".jpg", ".png")) and not f.startswith("._"):
                img.save(root / room / f)
            else:
                (root / room / f).write_bytes(b"junk")
    rooms = photo.list_room_folders(root)
    assert [r[0] for r in rooms] == ["1 hall", "2 kitchen", "10 bath"]
    assert [p.name for p in rooms[1][1]] == ["IMG_9.jpg", "IMG_10.JPG"]
    wrapper = tmp_path / "zip"
    (wrapper / "Scan").mkdir(parents=True)
    (root / "1 hall").rename(wrapper / "Scan" / "1 hall")
    assert [r[0] for r in photo.list_room_folders(wrapper)] == ["1 hall"]
    loose = tmp_path / "loose"
    loose.mkdir()
    img.save(loose / "a.jpg")
    assert photo.list_room_folders(loose) == [("loose", [loose / "a.jpg"])]
    assert photo.list_room_folders(loose / "a.jpg") == [("loose", [loose / "a.jpg"])]


def _save_with_exif(path, arr, *, focal35=None, iso=None, exposure=None, orientation=None):
    exif = Image.Exif()
    exif[0x010F], exif[0x0110] = "Apple", "iPhone 17"
    if orientation:
        exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    if focal35:
        sub[0xA405] = focal35
    if iso:
        sub[0x8827] = iso
    if exposure:
        sub[0x829A] = exposure
    Image.fromarray(arr).save(path, exif=exif, quality=95)


def test_load_photo_exif_orientation_and_intrinsics(tmp_path):
    arr = np.zeros((30, 40, 3), np.uint8)
    arr[:5, :5] = 255
    p = tmp_path / "r.jpg"
    _save_with_exif(p, arr, focal35=26, iso=3200, exposure=1 / 10, orientation=6)
    img, exif, size = photo.load_photo(p)
    assert img.shape == (40, 30, 3) and size == (30, 40)
    assert exif.focal_35mm == 26 and exif.iso == 3200 and math.isclose(exif.exposure_s, 0.1)
    assert exif.low_light and exif.model == "iPhone 17"
    K = photo.intrinsics_from_exif(exif, 30, 40)
    assert math.isclose(K[0, 0], 26 / math.hypot(36, 24) * 50.0) and K[0, 2] == 14.5 and K[1, 2] == 19.5
    assert photo.intrinsics_from_exif(photo.PhotoExif(), 30, 40) is None
    big = tmp_path / "big.png"
    Image.fromarray(np.zeros((100, 5000, 3), np.uint8)).save(big)
    img, _, size = photo.load_photo(big)
    assert size == (5000, 100) and max(img.shape[:2]) == photo.MAX_IMAGE_SIDE


def test_load_photo_odd_formats(tmp_path):
    rng = np.random.default_rng(0)
    cases = {
        "grey.png": Image.fromarray(rng.integers(0, 255, (20, 30), dtype=np.uint8)),
        "rgba.png": Image.fromarray(rng.integers(0, 255, (20, 30, 4), dtype=np.uint8), mode="RGBA"),
        "deep.png": Image.fromarray(rng.integers(0, 65535, (20, 30), dtype=np.uint16)),
        "pal.gif": Image.fromarray(rng.integers(0, 255, (20, 30, 3), dtype=np.uint8)).convert("P"),
        "tiny.jpg": Image.fromarray(np.zeros((2, 3, 3), np.uint8)),
    }
    for name, im in cases.items():
        im.save(tmp_path / name)
        img, exif, size = photo.load_photo(tmp_path / name)
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, name
        assert size == (img.shape[1], img.shape[0]) and exif.focal_35mm is None
    assert mb.as_rgb_uint8(np.ones((4, 5), float)).shape == (4, 5, 3)
    assert mb.as_rgb_uint8(np.full((4, 5, 3), 0.5)).max() == 128


def test_spread_indices():
    idx = photo.spread_indices(12, 8)
    assert len(idx) == 8 and idx[0] == 0 and idx[-1] == 11
    assert photo.spread_indices(5, 8) == [0, 1, 2, 3, 4]


def test_gravity_alignment_levels_a_tilted_room():
    rng = np.random.default_rng(0)
    R_true = se3.rotvec_to_R(np.radians([12.0, -7.0, 40.0]))
    Ts = [R_true_pose for R_true_pose in (se3.make_T(R_true @ camera_pose((0, 0, 0), a, -10)[:3, :3], np.zeros(3))
                                          for a in np.linspace(0, 2 * np.pi, 6, endpoint=False))]
    floor = np.tile([0.0, 0.0, 1.0], (500, 1)) + rng.normal(scale=0.03, size=(500, 3))
    wall = np.c_[rng.normal(size=(800, 2)), np.zeros(800)]
    n = np.vstack([floor, wall])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    R, info = photo.gravity_alignment(Ts, n @ R_true.T, np.ones(len(n)))
    assert info["refined"]
    assert np.degrees(np.arccos(np.clip((R @ R_true @ [0, 0, 1.0])[2], -1, 1))) < 0.5


def test_gravity_alignment_single_steep_photo_uses_wide_cone():
    rng = np.random.default_rng(1)
    T = camera_pose((0, 0, 1.5), 0.3, -40.0)
    floor = np.tile([0.0, 0.0, 1.0], (400, 1)) + rng.normal(scale=0.02, size=(400, 3))
    front = np.tile(-T[:3, 2] * [1, 1, 0], (500, 1)) + rng.normal(scale=0.02, size=(500, 3))
    side = np.tile(T[:3, 0], (300, 1)) + rng.normal(scale=0.02, size=(300, 3))
    n = np.vstack([floor, front, side])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    R, info = photo.gravity_alignment([T], n, np.ones(len(n)))
    assert info["refined"] and info["cone_deg"] == 60.0
    assert abs(info["refine_angle_deg"] - 40.0) < 2.0
    assert np.degrees(np.arccos(np.clip((R @ [0, 0, 1.0])[2], -1, 1))) < 1.0


def test_runner_flags_heavy_aspect_crop(monkeypatch):
    runner = mb.MapAnythingRunner()

    def fake_run(imgs, Km, target):
        n, (w, h) = len(imgs), target
        return {"pts3d": np.ones((n, h, w, 3), np.float32), "conf": np.full((n, h, w), 5.0, np.float32),
                "mask": np.ones((n, h, w), bool), "T_wc": np.tile(np.eye(4), (n, 1, 1)),
                "K": np.tile(np.eye(3), (n, 1, 1)), "metric_scale": np.ones(n)}

    monkeypatch.setattr(runner, "_run", fake_run)
    land = np.zeros((300, 400, 3), np.uint8)
    preds = runner.infer([land, land, land, np.zeros((400, 300, 3), np.uint8)])
    assert "aspect_crop:1/4" in preds[0].meta["flags"]
    assert preds[3].meta["crop_fraction"] > 0.4 and preds[0].meta["crop_fraction"] < 0.05
    # half portrait, half landscape: the mean aspect picks the square size and every view loses a quarter
    mixed = runner.infer([land, np.zeros((400, 300, 3), np.uint8)])
    assert "aspect_crop:2/2" in mixed[0].meta["flags"] and mixed[0].model_size == (518, 518)
    portrait = preds[3]
    assert portrait.uncrop(portrait.mask, False).shape[::-1] == portrait.resized_size


# ------------------------------------------------------------------------------------------- photo scenes


def _room_shots(rng, n):
    """Cameras near the room corners, facing the opposite corner, like the capture protocol."""
    shots = []
    corners = [(0.5, 0.5), (LX - 0.5, 0.5), (LX - 0.5, LY - 0.5), (0.5, LY - 0.5)]
    for k in range(n):
        cx, cy = corners[k % 4]
        c = np.array([cx, cy, 1.4]) + rng.normal(scale=0.1, size=3)
        yaw = math.atan2(LY / 2 - cy, LX / 2 - cx) + rng.normal(scale=0.2)
        shots.append(camera_pose(c, yaw, -20.0 + rng.normal(scale=3.0)))
    return shots


def test_build_room_scenes_with_fake_runner(tmp_path):
    rng = np.random.default_rng(3)
    root = tmp_path / "Scan"
    by_sha, truth = {}, {}
    for room, n in [("02 kitchen", 6), ("10 bath", 1), ("1 hall", 3)]:
        (root / room).mkdir(parents=True)
        for k, T in enumerate(_room_shots(rng, n)):
            p = root / room / f"IMG_{k:04d}.jpg"
            _save_with_exif(p, _noise_image(rng), focal35=27, iso=3200 if (room, k) == ("1 hall", 1) else 100,
                            exposure=1 / 60)
            img, _, _ = photo.load_photo(p)
            by_sha[mb.array_sha256(img)] = T
            truth[p.name, room] = T
    (root / "02 kitchen" / "IMG_0000.MOV").write_bytes(b"live photo")
    (root / "02 kitchen" / "notes.jpg").write_bytes(b"not an image")
    runner = FakeRunner(by_sha)
    scenes = photo.build_room_scenes(root, tmp_path / "work", runner=runner)

    assert [s.room_hint for s in scenes] == ["1 hall", "02 kitchen", "10 bath"]
    hall, kitchen, bath = scenes
    assert all(c["intrinsics"] == [True] * c["n"] for c in runner.calls)
    assert kitchen.meta["quality"]["n_photos"] == 6 and kitchen.meta["quality"]["exif_focal"]
    assert "unreadable_photo:notes.jpg" in kitchen.meta["flags"]
    assert hall.meta["quality"]["low_light"] and not kitchen.meta["quality"]["low_light"]
    assert "thin" in bath.meta["flags"] and bath.meta["quality"]["thin"]
    assert kitchen.scale_log_sigma == photo.SCALE_LOG_SIGMA
    json.dumps(kitchen.meta["quality"])

    for s in scenes:
        assert len(s.points) == len(s.normals) == len(s.weights) == len(s.view_index) > 100
        assert s.view_index.max() < len(s.views) and s.weights.min() >= 0 and s.weights.max() <= 1
        assert np.allclose(np.linalg.norm(s.normals, axis=1), 1.0, atol=1e-4)
        for v in s.views:
            assert v.image_path.exists() and (v.width, v.height) == (W, H)
            assert v.pointmap.shape == (H, W, 3) and v.valid.shape == (H, W) and v.conf.shape == (H, W)
            assert np.allclose(v.K, K_TRUE, atol=1e-6)
            # gravity: the camera's down axis is the true pitch away from world down
            Tt = truth[v.image_path.name, s.room_hint]
            want = np.degrees(np.arccos(np.clip(-Tt[2, 1], -1, 1)))
            got = np.degrees(np.arccos(np.clip(-v.T_wc[2, 1], -1, 1)))
            assert abs(want - got) < 1.0
        # floor points face up and sit on one plane
        down = s.points[:, 2] < np.percentile(s.points[:, 2], 2) + 0.05
        assert np.median(s.normals[down, 2]) > 0.99
        assert np.std(s.points[down, 2]) < 0.02


def test_build_room_scenes_raises_when_nothing_readable(tmp_path):
    (tmp_path / "Scan" / "a").mkdir(parents=True)
    (tmp_path / "Scan" / "a" / "x.jpg").write_bytes(b"broken")
    with pytest.raises(ValueError):
        photo.build_room_scenes(tmp_path / "Scan", tmp_path, runner=FakeRunner({}))
    with pytest.raises(ValueError):
        photo.build_room_scenes(tmp_path / "Scan" / "a" / "x.jpg", tmp_path, runner=FakeRunner({}))
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError):
        photo.build_room_scenes(tmp_path / "empty", tmp_path, runner=FakeRunner({}))


# ------------------------------------------------------------------------------------------- video


def test_plan_chunks_and_owners():
    assert video.plan_chunks(5, 8, 3) == [(0, 5)]
    spans = video.plan_chunks(40, 8, 3)
    assert spans[0] == (0, 8) and spans[-1] == (32, 40)
    assert all(b - a == 8 for a, b in spans)
    assert all(spans[i][1] - spans[i + 1][0] >= 3 for i in range(len(spans) - 1))
    owners = video.assign_owners(40, spans)
    assert owners[0] == 0 and owners[39] == len(spans) - 1
    assert all(spans[c][0] <= f < spans[c][1] for f, c in enumerate(owners))


def _loop_frames(tmp_path, n=40, radius=1.0):
    rng = np.random.default_rng(7)
    frames, by_sha, poses = [], {}, []
    for k in range(n):
        a = 2 * np.pi * k / n
        T = camera_pose((LX / 2 + radius * np.cos(a), LY / 2 + radius * np.sin(a), 1.5), a, -35.0)
        img = _noise_image(rng)
        p = tmp_path / f"{k:04d}.png"
        Image.fromarray(img).save(p)
        by_sha[mb.array_sha256(img)] = T
        frames.append(video.Frame(k, 3 * k, k / 2.0, p))
        poses.append(T)
    return frames, by_sha, poses


def test_video_drift_correction_beats_chaining(tmp_path):
    frames, by_sha, poses = _loop_frames(tmp_path)
    runner = FakeRunner(by_sha, bend=(0.25, 0.08, 0.002, 0.002))
    on = video.build_scene_from_frames(frames, drift_correction=True, runner=runner, chunk_size=8, overlap=3,
                                       loop_frames=4)
    off = video.build_scene_from_frames(frames, drift_correction=False, runner=runner, chunk_size=8, overlap=3,
                                        loop_frames=4)
    d_on, d_off = on.meta["drift"], off.meta["drift"]
    assert d_on["enabled"] and not d_off["enabled"]
    assert d_on["loop_closure"]["attempted"] and d_on["loop_closure"]["accepted"], d_on["loop_closure"]
    assert not d_off["loop_closure"]["attempted"]
    ate_on, S_on = _ate(on.meta["frames"]["T_wc"], poses)
    ate_off, _ = _ate(off.meta["frames"]["T_wc"], poses)
    assert ate_on < 0.5 * ate_off and ate_on < 0.08, (ate_on, ate_off)
    assert d_on["floor_z_spread_after_m"] < d_on["floor_z_spread_before_m"]
    assert d_on["floor_z_spread_after_m"] < 0.02
    assert 0.0 <= d_on["max_yaw_correction_deg"] <= video.YAW_MAX_DEG
    json.dumps(d_on)
    json.dumps(on.meta["quality"])
    q = on.meta["quality"]
    assert q["n_frames"] == 40 and q["n_chunks"] == len(video.plan_chunks(40, 8, 3))
    assert 0.0 < q["mean_conf"] <= 1.0
    # metric scale: the median chunk scale keeps the result close to metric
    assert abs(math.log(se3.decompose_sim3(S_on)[0])) < 0.1
    assert on.scale_log_sigma >= video.SCALE_SIGMA_BASE

    for s in (on, off):
        assert len(s.views) == math.ceil(40 / video.KEYFRAME_EVERY)
        assert s.view_index.max() < len(s.views) and len(s.points) > 1000
        assert np.allclose(np.linalg.norm(s.normals, axis=1), 1.0, atol=1e-4)
        assert all(v.pointmap.shape == (H, W, 3) and v.image_path.exists() for v in s.views)
    # gravity: every camera keeps its true 35 degree pitch
    pitch = [np.degrees(np.arcsin(np.clip(T[2, 2], -1, 1))) for T in on.meta["frames"]["T_wc"]]
    assert np.max(np.abs(np.array(pitch) + 35.0)) < 2.0
    floor = on.points[on.normals[:, 2] > 0.95]
    assert np.std(floor[:, 2]) < 0.03


def test_video_loop_rejected_when_the_walk_does_not_return(tmp_path):
    frames, by_sha, poses = _loop_frames(tmp_path, n=24)
    frames = frames[:18]  # three quarters of the circle: the end never sees the start
    runner = FakeRunner(by_sha)
    s = video.build_scene_from_frames(frames, drift_correction=True, runner=runner, chunk_size=8, overlap=3,
                                      loop_frames=4)
    lc = s.meta["drift"]["loop_closure"]
    assert lc["attempted"] and not lc["accepted"] and lc["reason"].startswith("rejected")
    ate, _ = _ate(s.meta["frames"]["T_wc"], poses[:18])
    assert ate < 0.05


def test_video_loop_rejected_when_the_correction_is_implausible(tmp_path):
    frames, by_sha, _ = _loop_frames(tmp_path)
    runner = FakeRunner(by_sha, bend=(1.5, 0.0, 0.0, 0.0))  # about 50 degrees of yaw drift around the loop
    s = video.build_scene_from_frames(frames, drift_correction=True, runner=runner, chunk_size=8, overlap=3,
                                      loop_frames=4)
    lc = s.meta["drift"]["loop_closure"]
    assert lc["attempted"] and not lc["accepted"] and "rotation" in lc["reason"], lc
    assert lc["error_rot_deg"] > video.LOOP_MAX_ROT_DEG


def test_video_single_chunk_and_tiny_inputs(tmp_path):
    frames, by_sha, _ = _loop_frames(tmp_path, n=5)
    s = video.build_scene_from_frames(frames, runner=FakeRunner(by_sha), chunk_size=8, overlap=3)
    assert s.meta["quality"]["n_chunks"] == 1 and not s.meta["drift"]["loop_closure"]["attempted"]
    assert math.isclose(s.scale_log_sigma, video.SINGLE_RUN_SCALE_SIGMA)
    one = video.build_scene_from_frames(frames[:1], runner=FakeRunner(by_sha), chunk_size=8, overlap=3)
    assert "thin" in one.meta["flags"] and len(one.views) == 1


def _write_video(path, n_frames=45, fps=15, size=(128, 96), rotation=0):
    import av

    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=fps)
        st.width, st.height, st.pix_fmt = size[0], size[1], "yuv420p"
        if rotation:
            st.set_display_rotation(rotation)
        rng = np.random.default_rng(0)
        base = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
        for i in range(n_frames):
            img = np.roll(base, i, axis=1).copy()
            img[:16, :16] = 255  # marker in the top-left corner of the stored frame
            if i % 7 == 3:
                img = (0.5 * img + 0.5 * np.roll(img, 6, axis=1)).astype(np.uint8)
            for pkt in st.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def test_sample_frames_rate_rotation_and_cap(tmp_path):
    clip = tmp_path / "clip.mp4"
    _write_video(clip, rotation=-90)
    frames, info = video.sample_frames(clip, tmp_path / "frames", target_fps=2.0, max_frames=200)
    assert 5 <= len(frames) <= 6 and info["rotation_deg"] == -90
    assert all(b.timestamp > a.timestamp for a, b in itertools.pairwise(frames))
    img = np.asarray(Image.open(frames[0].path))
    assert img.shape == (128, 96, 3)
    # rotated clockwise for display: the stored top-left marker ends up top-right
    assert img[:10, -10:].mean() > 200 and img[:10, :10].mean() < 200
    few, info2 = video.sample_frames(clip, tmp_path / "few", target_fps=2.0, max_frames=3)
    assert len(few) <= 3 and info2["sample_fps"] < 2.0
    folder = tmp_path / "handoff"
    folder.mkdir()
    (folder / ".hidden.mov").write_bytes(b"x")
    clip.rename(folder / "IMG_0001.MOV")
    assert video.resolve_video(folder) == folder / "IMG_0001.MOV"
    with pytest.raises(ValueError):
        video.resolve_video(tmp_path / "frames")
    junk = tmp_path / "junk.mov"
    junk.write_bytes(b"not a video at all" * 10)
    with pytest.raises(ValueError):
        video.sample_frames(junk, tmp_path / "bad")
