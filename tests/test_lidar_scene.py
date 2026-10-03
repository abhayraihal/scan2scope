import cv2
import numpy as np
import pytest
from stray_fixture import DOOR, ROOM, cached_dataset, read_barcode, write_stray_dataset

from scan2scope.geometry import lidar
from scan2scope.geometry.lidar import (
    C_ZUP,
    arkit_to_zup,
    build_scene,
    depth_mask,
    select_keyframes,
    select_views,
)
from scan2scope.geometry.pointmaps import backproject_depth
from scan2scope.geometry.se3 import make_T, rot_z
from scan2scope.ingest.stray import load_stray, scale_intrinsics


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return cached_dataset(tmp_path_factory, "stray_v14")


@pytest.fixture(scope="module")
def scene(ds, tmp_path_factory):
    return build_scene(ds.root, tmp_path_factory.mktemp("scene_work"), drift_correction=False)


def to_room(fx, points, normals=None):
    M = np.linalg.inv(fx.room_to_zup)
    p = points @ M[:3, :3].T + M[:3, 3]
    return p if normals is None else (p, normals @ M[:3, :3].T)


def plane_distance(p, n):
    """Distance of room-frame points to the box surface their normal belongs to, and which surfaces count."""
    ax = np.argmax(np.abs(n), axis=1)
    target = np.where(n[np.arange(len(n)), ax] > 0, 0.0, np.array(ROOM)[ax])  # normals face into the room
    d = np.abs(p[np.arange(len(p)), ax] - target)
    inside = (p[:, 1] > 0.05) & (np.abs(n[np.arange(len(n)), ax]) > 0.95)
    return d, inside


def test_zup_conversion_is_a_proper_rotation():
    R = C_ZUP[:3, :3]
    np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)
    np.testing.assert_allclose(R @ [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])  # ARKit up becomes z up
    np.testing.assert_allclose(R @ [0.0, 0.0, -1.0], [0.0, 1.0, 0.0])  # initial heading becomes +y


def test_scene_contract(scene, ds):
    assert scene.tier == "lidar"
    assert scene.scale_log_sigma == pytest.approx(0.003)
    n = len(scene.points)
    assert n > 10000
    assert scene.points.shape == (n, 3) and scene.normals.shape == (n, 3)
    assert scene.weights.shape == (n,) and scene.view_index.shape == (n,)
    np.testing.assert_allclose(np.linalg.norm(scene.normals, axis=1), 1.0, atol=1e-6)
    assert ((scene.weights >= 0.3 - 1e-6) & (scene.weights <= 1.0 + 1e-6)).all()
    assert scene.view_index.min() >= 0 and scene.view_index.max() < len(scene.views)
    q = scene.meta["quality"]
    assert q["n_frames"] == len(ds.frame_ids) and 0 < q["n_keyframes"] <= 400
    assert q["duration_s"] == pytest.approx(ds.timestamps[-1] - ds.timestamps[0], abs=1e-3)
    assert 0.0 < q["mean_conf"] <= 1.0 and 0.9 < q["valid_depth_fraction"] <= 1.0
    assert scene.meta["drift"] == {"enabled": False}
    assert scene.meta["flags"] == []


def test_points_lie_on_the_true_planes(scene, ds):
    p, n = to_room(ds, scene.points, scene.normals)
    assert np.median(np.abs(p[n[:, 2] > 0.95, 2])) < 0.004  # floor at z = 0, z up
    d, inside = plane_distance(p, n)
    walls = inside & (np.abs(n[:, 2]) < 0.2)
    assert walls.sum() > 5000
    assert np.median(d[walls]) < 0.006 and np.percentile(d[walls], 95) < 0.02
    # the corridor seen through the door survives, outside the room footprint
    assert ((p[:, 1] < -0.3) & (p[:, 0] > DOOR[0]) & (p[:, 0] < DOOR[1])).sum() > 100


def test_normals_face_the_cameras(scene, ds):
    p, n = to_room(ds, scene.points, scene.normals)
    near_x0 = (p[:, 0] < 0.05) & (p[:, 1] > 0.2) & (p[:, 1] < ROOM[1] - 0.2) & (p[:, 2] > 0.3) & (p[:, 2] < 2.0)
    assert np.median(n[near_x0, 0]) > 0.95  # the wall at x = 0 is seen from inside the room


def test_backprojection_matches_analytic_geometry(tmp_path):
    """Noise-free depth, per-frame K scaled to 256x192 with the pixel-centre convention: errors < 1 mm."""
    fx = write_stray_dataset(tmp_path / "exact", n_frames=8, noise=False)
    cap = load_stray(fx.root)
    K_d = scale_intrinsics(cap.K, cap.rgb_size, cap.depth_size)
    T = arkit_to_zup(cap.T_wc)
    errs = []
    for i in range(len(cap)):
        d, c = cap.read_depth(i), cap.read_conf(i)
        valid, _ = depth_mask(d, c)
        pm = backproject_depth(d, K_d[i], T[i])
        p = to_room(fx, pm[valid])
        # nearest box face of the room (the corridor is excluded by y > 0.05)
        q = p[p[:, 1] > 0.05]
        dist = np.min(np.abs(np.stack([q[:, 0], ROOM[0] - q[:, 0], q[:, 1], ROOM[1] - q[:, 1], q[:, 2],
                                       ROOM[2] - q[:, 2]], 1)), axis=1)
        errs.append(dist)
    e = np.concatenate(errs)
    assert np.percentile(e, 99) < 0.0015  # mm rounding of the PNG along oblique rays


def test_views_for_semantics(scene, ds):
    views = scene.views
    duration = ds.timestamps[-1] - ds.timestamps[0]
    assert abs(len(views) - (int(duration) + 1)) <= 2 and len(views) <= 60
    assert len({v.id for v in views}) == len(views)
    t = np.array([v.timestamp for v in views])
    assert (np.diff(t) > 0).all()
    cap = load_stray(ds.root)
    for v in views[::7]:
        i = cap.row_of_frame(v.meta["frame_id"])
        assert v.image_path is not None and v.image_path.is_file()
        img = cv2.imread(str(v.image_path))[..., ::-1]
        assert img.shape == (v.height, v.width, 3) and (v.width, v.height) == ds.rgb_size
        assert read_barcode(img) == v.meta["frame_id"]
        np.testing.assert_allclose(v.K, ds.K_rgb[i], atol=1e-3)
        np.testing.assert_allclose(v.T_wc, C_ZUP @ ds.T_wc[i], atol=1e-6)  # drift correction is off
        assert v.pointmap.shape == (192, 256, 3) and v.valid.shape == (192, 256) and v.valid.dtype == bool
        assert v.conf.shape == (192, 256) and v.conf.max() <= 1.0
        p = to_room(ds, v.pointmap[v.valid].astype(float))
        inside = p[(p[:, 1] > 0.05)]
        dist = np.min(np.abs(np.stack([inside[:, 0], ROOM[0] - inside[:, 0], inside[:, 1], ROOM[1] - inside[:, 1],
                                       inside[:, 2], ROOM[2] - inside[:, 2]], 1)), axis=1)
        assert np.median(dist) < 0.01


def test_keyframes_by_motion_and_cap():
    n = 200
    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, 0, 3] = 0.0101 * np.arange(n)  # just over 1 cm per frame
    kf = select_keyframes(T, np.ones(n, bool))
    assert kf[0] == 0 and np.all(np.diff(kf) == 5)
    for k in range(n):  # just over 1 degree per frame, standing still
        T[k] = make_T(rot_z(np.radians(1.01 * k)), np.zeros(3))
    kf = select_keyframes(T, np.ones(n, bool))
    assert np.all(np.diff(kf) == 5)
    usable = np.ones(n, bool)
    usable[:3] = False
    assert select_keyframes(T, usable)[0] == 3
    assert len(select_keyframes(T, np.ones(n, bool), max_keyframes=10)) <= 10


def test_depth_mask_weights_and_edges():
    d = np.full((20, 20), 2.0, np.float32)
    d[:, 10:] = 3.0  # a 50 % depth step
    d[0, 0] = 0.1  # too close
    d[19, 0] = 6.0  # too far
    c = np.full((20, 20), 2, np.uint8)
    c[5, 3] = 1
    c[6, 3] = 0
    valid, w = depth_mask(d, c)
    assert w[5, 3] == pytest.approx(0.3) and w[2, 2] == pytest.approx(1.0)
    assert not valid[6, 3] and not valid[0, 0] and not valid[19, 0]
    assert not valid[10, 9] and not valid[10, 10] and valid[10, 7]
    v2, w2 = depth_mask(d, None)  # no confidence maps
    assert v2[5, 3] and w2[5, 3] == pytest.approx(lidar.NO_CONF_WEIGHT)


def test_select_views_prefers_still_frames():
    t = np.arange(0, 10, 0.2)
    motion = np.ones_like(t)
    motion[7] = 0.1  # the stillest frame in the second second
    pos = select_views(t, motion, np.ones_like(t))
    assert 7 in pos and len(pos) == 10


def test_missing_depth_frames_and_video_degrade(tmp_path):
    fx = write_stray_dataset(tmp_path / "gappy", n_frames=30)
    for f in (3, 4, 5):
        (fx.root / "depth" / f"{f:06d}.png").unlink()
    (fx.root / "rgb.mp4").unlink()
    sc = build_scene(fx.root, tmp_path / "work", drift_correction=True)
    assert "depth_missing:3/30" in sc.meta["flags"] and "rgb_missing" in sc.meta["flags"]
    assert sc.views and all(v.image_path is None for v in sc.views)
    assert any(f.startswith("views_without_image") for f in sc.meta["flags"])
    assert len(sc.points) > 1000
    assert sc.meta["drift"]["enabled"] is True


def test_no_depth_at_all_raises(tmp_path):
    fx = write_stray_dataset(tmp_path / "nodepth", n_frames=4)
    for p in (fx.root / "depth").iterdir():
        p.unlink()
    with pytest.raises(ValueError, match="depth"):
        build_scene(fx.root, tmp_path / "work")
