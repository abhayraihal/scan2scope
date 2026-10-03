import json
import stat
import zipfile
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

from scan2scope.ingest.detect import detect_capture
from scan2scope.ingest.images import list_room_folders


def photo(path: Path, focal_35: int | None = 26, model: str = "iPhone 17") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = model
    if focal_35:
        exif.get_ifd(0x8769)[0xA405] = focal_35
    exif.get_ifd(0x8769)[0xA434] = f"{model} back camera 5.96mm f/1.6"
    Image.fromarray(np.full((24, 32, 3), 128, np.uint8)).save(path, exif=exif)
    return path


def video(path: Path, n: int = 30, fps: int = 30, meta: dict | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    codec = "mpeg4"
    for name in ("libx264", "mpeg4"):
        try:
            av.codec.Codec(name, "w")
            codec = name
            break
        except ValueError:  # UnknownCodecError
            continue
    opts = {"movflags": "use_metadata_tags"} if path.suffix.lower() == ".mov" else {}
    with av.open(str(path), "w", options=opts) as c:
        s = c.add_stream(codec, rate=fps)
        s.width, s.height, s.pix_fmt = 64, 48, "yuv420p"
        for k, v in (meta or {}).items():
            c.metadata[k] = v
        for i in range(n):
            a = np.full((48, 64, 3), (i * 7) % 255, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(a, format="rgb24")):
                c.mux(p)
        for p in s.encode():
            c.mux(p)
    return path


def stray(root: Path, n: int = 5, rgb: bool = True) -> Path:
    (root / "depth").mkdir(parents=True)
    (root / "confidence").mkdir()
    rows = ["timestamp, frame, x, y, z, qx, qy, qz, qw"]
    for i in range(n):
        rows.append(f"{0.5 + i / 30:.6f}, {i}, 0.0, 0.0, {i * 0.01:.3f}, 0.0, 0.0, 0.0, 1.0")
        Image.fromarray(np.full((192, 256), 1500, np.uint16)).save(root / "depth" / f"{i:06d}.png")
        Image.fromarray(np.full((192, 256), 2, np.uint8)).save(root / "confidence" / f"{i:06d}.png")
    (root / "odometry.csv").write_text("\n".join(rows) + "\n")
    (root / "camera_matrix.csv").write_text("1400.0,0.0,960.0\n0.0,1400.0,720.0\n0.0,0.0,1.0\n")
    (root / "imu.csv").write_text("timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n")
    if rgb:
        video(root / "rgb.mp4", n=n)
    return root


def zip_dir(src: Path, zpath: Path, prefix: str = "", extra: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(zpath, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, prefix + p.relative_to(src).as_posix())
        for name, data in (extra or {}).items():
            zf.writestr(name, data)
    return zpath


@pytest.fixture
def property_scan(tmp_path):
    root = tmp_path / "Scan"
    for i in range(3):
        photo(root / "01 hallway" / f"IMG_{i:04d}.jpg")
    photo(root / "02 kitchen" / "IMG_0100.jpg", focal_35=None)
    for i in range(9):
        photo(root / "10 bath" / f"IMG_{200 + i:04d}.jpg")
    (root / "10 bath" / "notes.txt").write_text("x")
    (root / "10 bath" / "IMG_0200.AAE").write_text("<plist/>")
    video(root / "10 bath" / "IMG_0201.MOV", n=3)  # Live Photo companion
    return root


def test_photo_property(property_scan, tmp_path):
    tier, root, info = detect_capture(property_scan, work_dir=tmp_path / "work")
    assert tier == "photo" and root == property_scan
    assert [r for r, _ in list_room_folders(root)] == ["01 hallway", "02 kitchen", "10 bath"]
    assert info.id == "Scan" and info.tier == "photo" and info.path == str(property_scan)
    assert info.input_stats["n_rooms"] == 3 and info.input_stats["n_photos"] == 13
    assert [r["photos"] for r in info.input_stats["rooms"]] == [3, 1, 9]
    assert info.device["make"] == "Apple" and info.device["model"] == "iPhone 17"
    assert info.device["lens_model"].startswith("iPhone 17 back camera") and info.device["focal_35mm"] == 26
    assert "photo_count_low:02 kitchen" in info.flags and "photo_count_high:10 bath" in info.flags
    assert "exif_focal_missing" in info.flags and "unsupported_files_skipped:1" in info.flags
    assert info.input_stats["photos_without_focal"] == 1
    json.dumps({"device": info.device, "input_stats": info.input_stats, "flags": info.flags})  # goes into result.json


def test_single_room_folder_and_wrapper_folder(property_scan, tmp_path):
    tier, root, info = detect_capture(property_scan / "01 hallway", work_dir=tmp_path / "w")
    assert (tier, root) == ("photo", property_scan / "01 hallway")
    assert info.input_stats["rooms"] == [{"name": "01 hallway", "photos": 3, "photos_without_focal": 0}]
    assert info.flags == []
    wrapper = tmp_path / "export"
    wrapper.mkdir()
    property_scan.rename(wrapper / "Scan")
    tier, root, _ = detect_capture(wrapper, work_dir=tmp_path / "w")
    assert (tier, root) == ("photo", wrapper / "Scan")


def test_live_photo_pairs_stay_photo(tmp_path):
    room = tmp_path / "03 bedroom"
    for i in range(3):
        photo(room / f"IMG_{i:04d}.jpg")
        video(room / f"IMG_{i:04d}.MOV", n=3)
    tier, _, info = detect_capture(room, work_dir=tmp_path / "w")
    assert tier == "photo" and info.input_stats["n_photos"] == 3
    assert not any(f.startswith("unsupported_files_skipped") for f in info.flags)


def test_video_file_and_folder_with_one_video(tmp_path):
    meta = {"com.apple.quicktime.make": "Apple", "com.apple.quicktime.model": "iPhone 17",
            "com.apple.quicktime.camera.focal_length.35mm_equivalent": "26"}
    v = video(tmp_path / "walk" / "IMG_5000.MOV", n=60, meta=meta)
    tier, root, info = detect_capture(v, work_dir=tmp_path / "w")
    assert (tier, root, info.id) == ("video", v, "IMG_5000")
    s = info.input_stats
    assert s["duration_s"] == pytest.approx(2.0, abs=0.05) and s["fps"] == pytest.approx(30)
    assert (s["width"], s["height"], s["n_frames"]) == (64, 48, 60)
    assert info.device["model"] == "iPhone 17" and info.device["focal_35mm"] == 26.0
    assert "video_focal_missing" not in info.flags
    json.dumps({"device": info.device, "input_stats": info.input_stats})
    (tmp_path / "walk" / "readme.txt").write_text("x")
    tier, root, info = detect_capture(tmp_path / "walk", work_dir=tmp_path / "w")
    assert (tier, root) == ("video", v) and "unsupported_files_skipped:1" in info.flags


def test_one_video_wins_over_photos_and_several_videos_pick_longest(tmp_path):
    folder = tmp_path / "mixed"
    photo(folder / "a.jpg")
    photo(folder / "b.jpg")
    v = video(folder / "walk.mp4", n=30)
    tier, root, info = detect_capture(folder, work_dir=tmp_path / "w")
    assert (tier, root) == ("video", v) and "photos_ignored:2" in info.flags
    tier, root, _ = detect_capture(folder, tier="photo", work_dir=tmp_path / "w")
    assert (tier, root) == ("photo", folder)
    clips = tmp_path / "clips"
    video(clips / "short.mp4", n=10)
    long_clip = video(clips / "long.mp4", n=40)
    tier, root, info = detect_capture(clips, work_dir=tmp_path / "w")
    assert (tier, root) == ("video", long_clip) and "multiple_videos:2" in info.flags


def test_unreadable_video_next_to_photos_falls_back_to_photo(tmp_path):
    folder = tmp_path / "room"
    photo(folder / "a.jpg")
    photo(folder / "b.jpg")
    (folder / "walk.mov").write_bytes(b"\0" * 2048)
    tier, root, info = detect_capture(folder, work_dir=tmp_path / "w")
    assert (tier, root) == ("photo", folder) and "video_unreadable:walk.mov" in info.flags
    (folder / "a.jpg").unlink()
    (folder / "b.jpg").unlink()
    with pytest.raises(ValueError, match="cannot read"):
        detect_capture(folder, work_dir=tmp_path / "w")


def test_stray_folder_direct_and_one_level_down(tmp_path):
    s = stray(tmp_path / "2026-10-04-101233")
    tier, root, info = detect_capture(s, work_dir=tmp_path / "w")
    assert (tier, root) == ("lidar", s)
    st = info.input_stats
    assert st["lidar_frames"] == 5 and st["depth_frames"] == 5 and st["confidence_frames"] == 5
    assert st["duration_s"] == pytest.approx(4 / 30, abs=1e-3) and st["rgb_video"] and st["camera_matrix"]
    assert info.device["app"] == "Stray Scanner" and info.flags == []
    tier, root, _ = detect_capture(tmp_path, work_dir=tmp_path / "w")
    assert (tier, root) == ("lidar", s)


def test_stray_zip_with_top_folder_and_macosx(tmp_path):
    s = stray(tmp_path / "src" / "rec")
    z = zip_dir(tmp_path / "src", tmp_path / "rec.zip",
                extra={"__MACOSX/rec/._odometry.csv": b"junk", "rec/._rgb.mp4": b"junk"})
    work = tmp_path / "work"
    tier, root, info = detect_capture(z, work_dir=work)
    assert tier == "lidar" and root == work / "input" / "rec" / "rec"
    assert info.id == "rec" and info.input_stats["source_zip"] == "rec.zip"
    assert not (work / "input" / "rec" / "__MACOSX").exists() and not (root / "._rgb.mp4").exists()
    assert (root / "odometry.csv").read_bytes() == (s / "odometry.csv").read_bytes()
    marker = root / "odometry.csv"
    mtime = marker.stat().st_mtime_ns
    detect_capture(z, work_dir=work)  # reused, not extracted again
    assert marker.stat().st_mtime_ns == mtime


def test_zip_path_traversal_is_blocked(tmp_path):
    src = tmp_path / "src"
    photo(src / "01 hallway" / "a.jpg")
    photo(src / "01 hallway" / "b.jpg")
    z = zip_dir(src, tmp_path / "scan.zip", extra={"../evil.txt": b"x", "/abs.txt": b"x", "a/../../evil2.txt": b"x"})
    with zipfile.ZipFile(z, "a") as zf:
        link = zipfile.ZipInfo("01 hallway/link.jpg")
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        zf.writestr(link, "../../../../etc/hosts")
    work = tmp_path / "deep" / "work"
    tier, _, info = detect_capture(z, work_dir=work)
    assert tier == "photo" and "zip_unsafe_entries_skipped:4" in info.flags
    assert not (tmp_path / "deep" / "evil.txt").exists() and not (work / "evil.txt").exists()
    assert not list(tmp_path.rglob("evil*.txt")) and not list(tmp_path.rglob("abs.txt"))
    assert not list(work.rglob("link.jpg"))


def test_zip_of_room_folders_and_zip_of_one_room(tmp_path):
    src = tmp_path / "src"
    photo(src / "01 hallway" / "a.jpg")
    photo(src / "02 kitchen" / "b.jpg")
    tier, root, info = detect_capture(zip_dir(src, tmp_path / "Scan.zip"), work_dir=tmp_path / "w")
    assert tier == "photo" and [r for r, _ in list_room_folders(root)] == ["01 hallway", "02 kitchen"]
    assert info.id == "Scan"
    one = tmp_path / "one"
    photo(one / "x.jpg")
    photo(one / "y.jpg")
    tier, root, info = detect_capture(zip_dir(one, tmp_path / "04 office.zip"), work_dir=tmp_path / "w")
    assert tier == "photo" and list_room_folders(root)[0][0] == "04 office"


def test_folder_holding_one_zip(tmp_path):
    s = stray(tmp_path / "src" / "rec")
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    zip_dir(tmp_path / "src", inbox / "rec.zip")
    tier, root, _ = detect_capture(inbox, work_dir=tmp_path / "w")
    assert tier == "lidar" and (root / "odometry.csv").read_bytes() == (s / "odometry.csv").read_bytes()


def test_forced_tier_mismatch_explains(property_scan, tmp_path):
    with pytest.raises(ValueError, match="Stray Scanner.*photo"):
        detect_capture(property_scan, tier="lidar", work_dir=tmp_path / "w")
    with pytest.raises(ValueError, match="--tier video"):
        detect_capture(property_scan, tier="video", work_dir=tmp_path / "w")
    s = stray(tmp_path / "rec")
    with pytest.raises(ValueError, match="not photos"):
        detect_capture(s, tier="photo", work_dir=tmp_path / "w")
    with pytest.raises(ValueError, match="tier must be one of"):
        detect_capture(s, tier="drone", work_dir=tmp_path / "w")


def test_forced_video_on_stray_uses_rgb(tmp_path):
    s = stray(tmp_path / "rec")
    tier, root, info = detect_capture(s, tier="video", work_dir=tmp_path / "w")
    assert (tier, root) == ("video", s / "rgb.mp4") and "video_from_stray_rgb" in info.flags
    assert detect_capture(s, tier="lidar", work_dir=tmp_path / "w")[0] == "lidar"


def test_stray_quirks_are_flagged(tmp_path):
    s = stray(tmp_path / "rec", n=6, rgb=False)
    (s / "depth" / "000005.png").unlink()
    for p in (s / "confidence").iterdir():
        p.unlink()
    _, _, info = detect_capture(s, work_dir=tmp_path / "w")
    assert "lidar_rgb_missing" in info.flags and "lidar_confidence_missing" in info.flags
    assert "lidar_frame_count_mismatch" not in " ".join(info.flags)  # 6 vs 5 is within tolerance
    assert info.input_stats["lidar_frames"] == 6 and info.input_stats["depth_frames"] == 5


def test_unrecognised_input_raises_with_expectations(tmp_path):
    junk = tmp_path / "junk"
    junk.mkdir()
    (junk / "a.txt").write_text("x")
    (junk / "b.pdf").write_bytes(b"%PDF")
    with pytest.raises(ValueError, match="expected a folder of room folders.*2 other files"):
        detect_capture(junk, work_dir=tmp_path / "w")
    with pytest.raises(ValueError, match="does not exist"):
        detect_capture(tmp_path / "missing", work_dir=tmp_path / "w")
    with pytest.raises(ValueError, match="not a video"):
        detect_capture(junk / "a.txt", work_dir=tmp_path / "w")
    bad_zip = tmp_path / "bad.zip"
    bad_zip.write_bytes(b"PK not really")
    with pytest.raises(ValueError):
        detect_capture(bad_zip, work_dir=tmp_path / "w")


def test_single_image_becomes_one_room(tmp_path):
    img = photo(tmp_path / "IMG_0042.jpg")
    tier, root, info = detect_capture(img, work_dir=tmp_path / "w")
    assert tier == "photo" and root == tmp_path / "w" / "input" / "IMG_0042"
    assert list_room_folders(root) == [("IMG_0042", [root / "IMG_0042.jpg"])]
    assert "single_image_input" in info.flags and "photo_count_low:IMG_0042" in info.flags
