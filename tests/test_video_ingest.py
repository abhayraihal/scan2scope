import itertools
import json
from pathlib import Path

import av
import cv2
import numpy as np
import pytest

from scan2scope.ingest.video import (
    VideoInfo,
    _blur_keep_mask,
    probe,
    rotate_upright,
    sample_frames,
    video_intrinsics,
)

W, H = 96, 64


def _encoder() -> str:
    for name in ("libx264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            return name
        except ValueError:  # UnknownCodecError
            continue
    pytest.skip("no H.264 or MPEG-4 encoder in this PyAV build")


def frame_image(i: int, blur: int = 0) -> np.ndarray:
    """Random block texture that moves one block per frame, red marker block in the top-left corner."""
    rng = np.random.default_rng(1234)
    tex = (rng.random((H // 4, W // 4 + 64)) * 255).astype(np.uint8)
    tex = np.kron(tex, np.ones((4, 4), np.uint8))[:, 4 * (i % 64): 4 * (i % 64) + W]
    img = np.repeat(tex[..., None], 3, axis=2)
    if blur:
        img = cv2.blur(img, (blur, blur))
    img[0:12, 0:12] = (255, 0, 0)
    return img


def make_video(path: Path, n: int, fps: int = 30, blur=lambda i: 0, rotation_ccw: float | None = None,
               pix_fmt: str = "yuv420p", trc: int | None = None, meta: dict | None = None) -> Path:
    codec = _encoder()
    opts = {"movflags": "use_metadata_tags"} if path.suffix == ".mov" else {}
    with av.open(str(path), "w", options=opts) as c:
        s = c.add_stream(codec, rate=fps, options={"crf": "8", "preset": "ultrafast"} if codec == "libx264" else {})
        s.width, s.height, s.pix_fmt = W, H, pix_fmt
        if codec == "mpeg4":
            s.bit_rate = 4_000_000
        if trc is not None:
            s.codec_context.color_trc = trc
        if rotation_ccw is not None:
            s.set_display_rotation(rotation_ccw)
        for k, v in (meta or {}).items():
            c.metadata[k] = v
        for i in range(n):
            frame = av.VideoFrame.from_ndarray(frame_image(i, blur(i)), format="rgb24")
            for p in s.encode(frame):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def test_rotate_upright_is_clockwise():
    a = np.arange(6).reshape(2, 3)  # [[0 1 2], [3 4 5]]
    assert rotate_upright(a, 0) is a
    np.testing.assert_array_equal(rotate_upright(a, 90), [[3, 0], [4, 1], [5, 2]])
    np.testing.assert_array_equal(rotate_upright(a, 180), [[5, 4, 3], [2, 1, 0]])
    np.testing.assert_array_equal(rotate_upright(a, 270), [[2, 5], [1, 4], [0, 3]])
    np.testing.assert_array_equal(rotate_upright(a, -90), rotate_upright(a, 270))
    img = np.zeros((4, 6, 3), np.uint8)
    assert rotate_upright(img, 90).shape == (6, 4, 3) and rotate_upright(img, 90).flags.c_contiguous


def test_probe_reads_stream_and_quicktime_keys(tmp_path):
    meta = {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 17",
            "com.apple.quicktime.software": "26.0",
            "com.apple.quicktime.camera.lens_model": "iPhone 17 back camera 5.96mm f/1.6",
            "com.apple.quicktime.camera.focal_length.35mm_equivalent": "26",
            "com.apple.quicktime.creationdate": "2026-10-04T10:12:33+0900"}
    p = make_video(tmp_path / "IMG_0001.mov", 45, fps=30, meta=meta)
    info = probe(p)
    assert info.duration_s == pytest.approx(1.5, abs=0.05) and info.fps == pytest.approx(30)
    assert info.n_frames == 45 and (info.width, info.height, info.rotation_deg) == (W, H, 0)
    assert info.codec in ("h264", "mpeg4") and info.pix_fmt == "yuv420p" and not info.is_hdr
    assert (info.make, info.model, info.software) == ("Apple", "iPhone 17", "26.0")
    assert info.lens_model.startswith("iPhone 17") and info.focal_35mm == 26.0
    assert info.creation_time == "2026-10-04T10:12:33+0900"
    K = video_intrinsics(info, W, H)
    assert K[0, 0] == pytest.approx(26 * np.hypot(W, H) / 43.2666) and K[0, 2] == pytest.approx((W - 1) / 2)
    assert video_intrinsics(VideoInfo(1, 30, 30, W, H, 0, "h264", "yuv420p", False), W, H) is None


def test_probe_rejects_non_video(tmp_path):
    p = tmp_path / "notes.mov"
    p.write_text("not a video")
    with pytest.raises(ValueError):
        probe(p)


def test_sharpest_frame_per_window_in_order(tmp_path):
    # 4 s at 30 fps, 2 fps windows of 15 frames; frame 15k+7 is the only unblurred one in each window.
    p = make_video(tmp_path / "walk.mp4", 120, blur=lambda i: 0 if i % 15 == 7 else 5)
    recs = sample_frames(p, tmp_path / "frames", target_fps=2.0, max_frames=240)
    assert [r.index for r in recs] == [15 * k + 7 for k in range(8)]
    assert all(b.t > a.t for a, b in itertools.pairwise(recs))
    assert recs[0].t == pytest.approx(7 / 30, abs=1e-3)
    for r in recs:
        assert r.path.name == f"frame_{r.index:05d}.jpg" and r.path.is_file()
        assert (r.width, r.height) == (W, H) and r.sharpness > 0
    manifest = json.loads((tmp_path / "frames" / "frames.json").read_text())
    assert [f["index"] for f in manifest["frames"]] == [r.index for r in recs]
    assert sorted(x.name for x in (tmp_path / "frames").glob("frame_*.jpg")) == [r.path.name for r in recs]


def test_rate_reduced_to_respect_max_frames(tmp_path):
    p = make_video(tmp_path / "walk.mp4", 120)
    recs = sample_frames(p, tmp_path / "f", target_fps=2.0, max_frames=4)
    assert len(recs) == 4
    assert [int(r.t) for r in recs] == [0, 1, 2, 3]  # one keyframe per second instead of two


def test_blurred_windows_dropped(tmp_path):
    heavy = {3, 6}  # every frame of windows 3 and 6 is heavily blurred
    p = make_video(tmp_path / "walk.mp4", 120, blur=lambda i: 21 if i // 15 in heavy else 0)
    out = tmp_path / "f"
    recs = sample_frames(p, out, target_fps=2.0, min_sharpness_rel=0.35)
    assert len(recs) == 6 and not any(r.index // 15 in heavy for r in recs)
    assert len(list(out.glob("frame_*.jpg"))) == 6
    dropped = json.loads((out / "frames.json").read_text())["dropped_blurred"]
    assert sorted(d["index"] // 15 for d in dropped) == [3, 6]
    assert len(sample_frames(p, out, target_fps=2.0, min_sharpness_rel=0.0)) == 8  # filter off; stale files go
    assert len(list(out.glob("frame_*.jpg"))) == 8


def test_blur_mask_uses_local_median():
    s = np.array([100.0] * 20 + [10.0] * 20 + [100.0] * 20)  # a long low-texture stretch is kept
    assert _blur_keep_mask(s, 0.35).all()
    s2 = np.array([100.0] * 30)
    s2[[5, 17]] = 5.0
    keep = _blur_keep_mask(s2, 0.35)
    assert not keep[5] and not keep[17] and keep.sum() == 28
    assert _blur_keep_mask(np.array([0.0, 0.0, 0.0]), 0.35).all()


def test_portrait_rotation_applied(tmp_path):
    # iPhone portrait clips store landscape frames with a display matrix rotating 90 degrees clockwise.
    p = make_video(tmp_path / "portrait.mov", 30, rotation_ccw=-90)
    info = probe(p)
    assert (info.rotation_deg, info.width, info.height) == (90, H, W)
    recs = sample_frames(p, tmp_path / "f", target_fps=1.0)
    assert recs and (recs[0].width, recs[0].height) == (H, W)
    from PIL import Image

    img = np.asarray(Image.open(recs[0].path))
    assert img.shape == (W, H, 3)
    # The stored top-left marker ends up top-right after a clockwise turn.
    assert img[0:10, H - 10:H].mean(axis=(0, 1))[0] > 180 and img[0:10, H - 10:H].mean(axis=(0, 1))[1] < 90


def test_ten_bit_hlg_marked_hdr_and_written_as_8bit(tmp_path):
    try:
        p = make_video(tmp_path / "hdr.mov", 20, pix_fmt="yuv420p10le", trc=18)
    except (ValueError, av.error.FFmpegError) as exc:  # pragma: no cover - no 10-bit encoder
        pytest.skip(f"no 10-bit encoder: {exc}")
    info = probe(p)
    assert info.is_hdr and "10" in info.pix_fmt
    recs = sample_frames(p, tmp_path / "f", target_fps=2.0)
    from PIL import Image

    im = Image.open(recs[0].path)
    assert im.mode == "RGB" and im.size == (W, H)


def test_truncated_video_keeps_decoded_frames(tmp_path):
    p = make_video(tmp_path / "cut.ts", 120)
    data = p.read_bytes()
    p.write_bytes(data[: int(len(data) * 0.5)])
    recs = sample_frames(p, tmp_path / "f", target_fps=2.0)
    assert 1 <= len(recs) <= 8
