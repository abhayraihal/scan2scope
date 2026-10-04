
import av
import cv2
import numpy as np
import pytest
from stray_fixture import cached_dataset, read_barcode, render_depth, write_stray_dataset

from scan2scope.ingest.stray import load_stray, scale_intrinsics


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    return cached_dataset(tmp_path_factory, "stray_v14")


def test_poses_intrinsics_and_sizes(ds):
    cap = load_stray(ds.root)
    assert len(cap) == len(ds.frame_ids)
    np.testing.assert_array_equal(cap.frame_ids, ds.frame_ids)
    np.testing.assert_allclose(cap.timestamps, ds.timestamps, atol=1e-9)
    np.testing.assert_allclose(cap.T_wc, ds.T_wc, atol=1e-6)
    np.testing.assert_allclose(cap.K, ds.K_rgb, atol=1e-3)
    assert cap.rgb_size == ds.rgb_size
    assert cap.depth_size == (256, 192)
    assert cap.format == "v1.4"
    assert cap.flags == []
    assert cap.n_video_frames == len(ds.frame_ids)
    assert cap.fps == pytest.approx(ds.fps, rel=1e-3)
    assert cap.has_depth.all() and cap.has_conf.all()


def test_quaternion_is_opencv_camera_in_arkit_world(ds):
    cap = load_stray(ds.root)
    down, fwd = cap.T_wc[:, :3, 1], cap.T_wc[:, :3, 2]
    assert (down[:, 1] < -0.9).all()  # image down is close to gravity, ARKit -y
    np.testing.assert_allclose(np.degrees(np.arcsin(fwd[:, 1])), -18.0, atol=0.5)  # fixture pitch


def test_depth_metres_and_confidence(ds):
    cap = load_stray(ds.root)
    i = 10
    d, c = cap.read_depth(i), cap.read_conf(i)
    assert d.dtype == np.float32 and d.shape == (192, 256)
    assert c.dtype == np.uint8 and set(np.unique(c)) <= {0, 1, 2}
    truth, _, _ = render_depth(np.linalg.inv(ds.room_to_arkit) @ ds.T_wc_true[i], ds.K_depth[i])
    good = c == 2
    assert np.median(np.abs(d[good] - truth[good])) < 0.006


def test_extract_rgb_keeps_the_frame_hidden_by_the_edit_list(ds, tmp_path):
    cap = load_stray(ds.root)
    ids = [0, 1, 37, int(ds.frame_ids[-1])]
    paths = cap.extract_rgb(ids, tmp_path / "rgb")
    for f, p in zip(ids, paths):
        img = cv2.imread(str(p))[..., ::-1]
        assert img.shape[:2] == (ds.rgb_size[1], ds.rgb_size[0])
        assert read_barcode(img) == f
    # what the reader works around: FFmpeg's default decode drops the frame written at -1/60 s
    with av.open(str(ds.root / "rgb.mp4")) as c:
        assert sum(1 for _ in c.decode(c.streams.video[0])) == len(ds.frame_ids) - 1


def test_imu_in_g_is_converted(ds):
    cap = load_stray(ds.root)
    assert cap.imu_accel_unit == "g"
    assert cap.imu.shape[1] == 7
    assert np.median(np.linalg.norm(cap.imu[:, 1:4], axis=1)) == pytest.approx(9.80665, rel=0.02)


def test_legacy_format_uses_camera_matrix(tmp_path):
    ds = write_stray_dataset(tmp_path / "abc", n_frames=12, legacy=True, imu_unit="m/s2")
    cap = load_stray(ds.root)
    assert cap.format == "legacy"
    assert "intrinsics_from_camera_matrix" in cap.flags
    np.testing.assert_allclose(cap.K, np.broadcast_to(ds.K_rgb[-1], cap.K.shape), atol=1e-3)
    np.testing.assert_allclose(cap.T_wc, ds.T_wc, atol=1e-6)
    assert cap.imu_accel_unit == "m/s2"


def test_messy_files_are_flagged_not_fatal(tmp_path):
    ds = write_stray_dataset(tmp_path / "messy" / "abcdef", n_frames=20)
    rows = (ds.root / "odometry.csv").read_text().splitlines()
    rows[10] = rows[10][:rows[10].rindex(", ,")]  # trailing empty fields cut off
    rows.append(rows[3])  # duplicated frame 2
    rows.insert(5, "garbage, line, here")
    rows.insert(8, "   ")
    (ds.root / "odometry.csv").write_text("\r\n".join(rows) + "\r\n")
    (ds.root / "depth" / "000007.png").unlink()
    (ds.root / "confidence" / "000009.png").unlink()
    (ds.root / "imu.csv").unlink()
    cap = load_stray(tmp_path / "messy")  # the folder holding the dataset is accepted too
    assert len(cap) == 20
    assert "odometry_rows_skipped:1" in cap.flags
    assert "odometry_duplicate_frames:1" in cap.flags
    assert "depth_missing:1/20" in cap.flags
    assert "confidence_missing:1/19" in cap.flags
    i7 = cap.row_of_frame(7)
    assert not cap.has_depth[i7] and cap.read_depth(i7) is None
    assert cap.read_conf(cap.row_of_frame(9)) is None
    assert cap.imu is None
    np.testing.assert_allclose(cap.K, ds.K_rgb, atol=1e-3)


def test_bom_headerless_and_unordered_odometry(tmp_path):
    ds = write_stray_dataset(tmp_path / "bom", n_frames=6)
    path = ds.root / "odometry.csv"
    rows = path.read_text().splitlines()
    path.write_text("\ufeff" + "\n".join(rows) + "\n", encoding="utf-8")
    assert load_stray(ds.root).flags == []
    body = rows[1:]
    body[2], body[3] = body[3], body[2]  # rows out of order; the frame column puts them back
    path.write_text("\n".join(body) + "\n")
    cap = load_stray(ds.root)
    assert "odometry_header_missing" in cap.flags
    np.testing.assert_array_equal(cap.frame_ids, ds.frame_ids)
    np.testing.assert_allclose(cap.T_wc, ds.T_wc, atol=1e-6)
    times = [r.split(", ") for r in body]
    times[4][0] = repr(float(times[1][0]))  # frame 4 stamped like frame 1: one step goes backwards
    path.write_text("\n".join(", ".join(r) for r in times) + "\n")
    assert "timestamps_not_increasing:1" in load_stray(ds.root).flags


def test_missing_video_degrades(tmp_path):
    ds = write_stray_dataset(tmp_path / "novid", n_frames=6)
    (ds.root / "rgb.mp4").unlink()
    cap = load_stray(ds.root)
    assert "rgb_missing" in cap.flags
    assert cap.rgb_size == (640, 480)  # recovered from the principal point
    assert cap.extract_rgb([0, 1], tmp_path / "x") == [None, None]


def test_no_dataset_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_stray(tmp_path)


def test_scale_intrinsics_pixel_centres():
    K = np.array([[1339.3, 0.0, 959.1], [0.0, 1339.3, 725.1], [0.0, 0.0, 1.0]])
    Kd = scale_intrinsics(K, (1920, 1440), (256, 192))
    u, v = 100.0, 50.0  # the ray through this depth pixel centre hits RGB pixel (u + 0.5) * 7.5 - 0.5
    x, y = (u - Kd[0, 2]) / Kd[0, 0], (v - Kd[1, 2]) / Kd[1, 1]
    assert K[0, 0] * x + K[0, 2] == pytest.approx((u + 0.5) * 7.5 - 0.5)
    assert K[1, 1] * y + K[1, 2] == pytest.approx((v + 0.5) * 7.5 - 0.5)
    np.testing.assert_allclose(scale_intrinsics(Kd, (256, 192), (1920, 1440)), K, atol=1e-9)
    assert scale_intrinsics(np.stack([K, K]), (1920, 1440), (256, 192)).shape == (2, 3, 3)


def test_real_style_odometry_line_parses(tmp_path):
    """A header and row copied in the app's format, spaces and empty distortion fields included."""
    root = tmp_path / "4e41d0a7da"
    (root / "depth").mkdir(parents=True)
    head = "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, distortion_center_y"
    row = ("170340.472500583, 000000, 0.013763887, -0.013308318, 0.09741095, 0.94932455, 0.009979882, "
           "-0.006518309, 0.31407145, 1339.3428, 1339.3428, 959.1318, 725.10126, , ")
    (root / "odometry.csv").write_text(head + "\n" + row + "\n")
    cv2.imwrite(str(root / "depth" / "000000.png"), np.full((192, 256), 1500, np.uint16))
    cap = load_stray(root)
    assert cap.frame_ids.tolist() == [0]
    assert cap.K[0, 0, 0] == pytest.approx(1339.3428) and cap.K[0, 1, 2] == pytest.approx(725.10126)
    assert cap.T_wc[0, :3, 3] == pytest.approx([0.013763887, -0.013308318, 0.09741095])
    assert cap.distortion_center is None
    assert cap.rgb_size == (1920, 1440)
    assert cap.read_depth(0)[0, 0] == pytest.approx(1.5)
    assert {"rgb_missing", "confidence_missing:all"} <= set(cap.flags)
