import math
import struct
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from PIL.TiffImagePlugin import IFDRational

from scan2scope.ingest.images import (
    ExifInfo,
    apple_acceleration,
    dedupe_images,
    intrinsics_from_exif,
    list_room_folders,
    load_image,
    natural_key,
    parse_apple_makernote,
    read_exif,
    scan_photo_folders,
)


def makernote(endian: str = ">", accel=(-0.0125, -0.98, 0.25), marker: bytes | None = None) -> bytes:
    """Apple iOS MakerNote: header, IFD at 14, value offsets from the MakerNote start."""
    e = endian
    n = 3
    data_start = 16 + 12 * n + 4
    content_id = b"8C1D2E3F-ABCD\0"
    accel_off = data_start
    cid_off = accel_off + 24
    entries = struct.pack(e + "HHIi", 0x0001, 9, 1, 14)  # MakerNoteVersion, SLONG inline
    entries += struct.pack(e + "HHII", 0x0008, 10, 3, accel_off)  # AccelerationVector, 3 SRATIONAL
    entries += struct.pack(e + "HHII", 0x0011, 2, len(content_id), cid_off)  # ContentIdentifier
    values = b"".join(struct.pack(e + "ii", round(v * 1_000_000), 1_000_000) for v in accel)
    mark = marker if marker is not None else (b"MM" if e == ">" else b"II")
    body = struct.pack(e + "H", n) + entries + b"\0\0\0\0" + values + content_id
    return b"Apple iOS\0" + b"\x00\x01" + mark + body


def write_jpeg(path: Path, size=(40, 30), orientation=None, focal_35=None, extra=None, marker_block=True):
    w, h = size
    a = np.full((h, w, 3), 255, np.uint8)
    if marker_block:
        a[0:8, 0:8] = (255, 0, 0)
    exif = Image.Exif()
    exif[0x010F] = "Apple"
    exif[0x0110] = "iPhone 17"
    exif[0x0131] = "26.0"
    if orientation is not None:
        exif[0x0112] = orientation
    sub = exif.get_ifd(0x8769)
    if focal_35 is not None:
        sub[0xA405] = focal_35
    sub[0x920A] = IFDRational(596, 100)
    sub[0xA434] = "iPhone 17 back camera 5.96mm f/1.6"
    sub[0x9003] = "2026:10:04 10:12:33"
    sub[0x9011] = "+09:00"
    sub[0x829A] = IFDRational(1, 60)
    sub[0x8827] = 125
    for k, v in (extra or {}).items():
        sub[k] = v
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(a).save(path, exif=exif, quality=95)
    return path


def test_orientation_6_is_rotated_upright_with_intrinsics(tmp_path):
    p = write_jpeg(tmp_path / "IMG_0001.JPG", size=(40, 30), orientation=6, focal_35=26)
    rgb, info = load_image(p)
    assert rgb.dtype == np.uint8 and rgb.shape == (40, 30, 3)
    # Orientation 6: rotate 90 degrees clockwise, so the stored top-left corner ends up top-right.
    assert rgb[0:8, 22:30].mean(axis=(0, 1))[1] < 60
    assert rgb[0:8, 0:8].mean(axis=(0, 1))[1] > 200
    assert (info.width, info.height, info.orientation) == (30, 40, 6)
    assert info.make == "Apple" and info.model == "iPhone 17" and info.software == "26.0"
    assert info.lens_model.startswith("iPhone 17 back camera")
    assert info.focal_35mm == 26 and info.focal_mm == pytest.approx(5.96)
    assert info.iso == 125 and info.exposure_s == pytest.approx(1 / 60)
    assert info.datetime_original == "2026-10-04T10:12:33+09:00"
    assert info.apple_acceleration is None

    K = intrinsics_from_exif(info, 30, 40)
    f = 26 * math.hypot(30, 40) / 43.2666
    np.testing.assert_allclose(K, [[f, 0, 14.5], [0, f, 19.5], [0, 0, 1]])
    K2 = intrinsics_from_exif(info, 60, 80)  # same camera at twice the resolution
    assert K2[0, 0] == pytest.approx(2 * f)


def test_read_exif_matches_load_without_decoding(tmp_path):
    p = write_jpeg(tmp_path / "a.jpg", size=(64, 48), orientation=8, focal_35=24)
    info = read_exif(p)
    assert (info.width, info.height, info.orientation, info.focal_35mm) == (48, 64, 8, 24)
    rgb, _ = load_image(p)
    assert rgb.shape[:2] == (64, 48)


def test_missing_focal_gives_no_intrinsics(tmp_path):
    p = tmp_path / "plain.png"
    Image.fromarray(np.zeros((10, 20, 3), np.uint8)).save(p)
    rgb, info = load_image(p)
    assert rgb.shape == (10, 20, 3)
    assert info.focal_35mm is None and info.make is None and info.orientation == 1
    assert intrinsics_from_exif(info, 20, 10) is None
    assert intrinsics_from_exif(None, 20, 10) is None
    assert intrinsics_from_exif(ExifInfo(focal_35mm=0.0), 20, 10) is None


def test_grey_16bit_and_rgba_become_rgb_uint8(tmp_path):
    p16 = tmp_path / "d.png"
    Image.fromarray(np.full((4, 5), 65535, np.uint16)).save(p16)
    rgb, _ = load_image(p16)
    assert rgb.shape == (4, 5, 3) and rgb.dtype == np.uint8 and rgb.max() == 255
    pa = tmp_path / "a.png"
    Image.fromarray(np.full((4, 5, 4), 200, np.uint8), "RGBA").save(pa)
    assert load_image(pa)[0].shape == (4, 5, 3)


def test_heic_uses_decoded_pixels_without_second_rotation(tmp_path):
    pillow_heif = pytest.importorskip("pillow_heif")
    pillow_heif.register_heif_opener()
    a = np.full((30, 40, 3), 255, np.uint8)
    a[0:8, 0:8] = (255, 0, 0)
    exif = Image.Exif()
    exif[0x0112] = 6
    exif.get_ifd(0x8769)[0xA405] = 26
    p = tmp_path / "IMG_0002.HEIC"
    try:
        Image.fromarray(a).save(p, exif=exif, quality=95)
    except (OSError, RuntimeError, ValueError) as exc:  # pragma: no cover - libheif without an encoder
        pytest.skip(f"cannot encode HEIC here: {exc}")
    rgb, info = load_image(p)
    # libheif applies the container transform and pillow-heif resets the EXIF tag, so nothing is rotated twice.
    assert rgb.shape[:2] == (info.height, info.width)
    assert info.orientation == 6 and info.focal_35mm == 26
    assert read_exif(p).width == info.width


def test_parse_apple_makernote_big_and_little_endian():
    for e in (">", "<"):
        tags = parse_apple_makernote(makernote(e))
        assert tags[0x0001] == 14
        assert tags[0x0011] == "8C1D2E3F-ABCD"
        np.testing.assert_allclose(tags[0x0008], (-0.0125, -0.98, 0.25))
        assert apple_acceleration(makernote(e)) == pytest.approx((-0.0125, -0.98, 0.25))


def test_parse_apple_makernote_guesses_unknown_byte_order():
    assert apple_acceleration(makernote(">", marker=b"\0\0")) == pytest.approx((-0.0125, -0.98, 0.25))


def test_parse_apple_makernote_rejects_bad_input():
    good = makernote()
    assert parse_apple_makernote(b"") == {}
    assert parse_apple_makernote(b"Nikon\0\x02\x10\0\0MM\0\0\0\x08" + b"\0" * 40) == {}
    assert parse_apple_makernote(good[:20]) == {}
    # Values cut off: the inline version survives, the out-of-range vector and string are dropped.
    truncated = parse_apple_makernote(good[:60])
    assert truncated.get(0x0001) == 14 and 0x0008 not in truncated
    assert apple_acceleration(good[:60]) is None
    rng = np.random.default_rng(1)
    for _ in range(200):
        junk = b"Apple iOS\0\0\x01MM" + rng.integers(0, 256, int(rng.integers(0, 120)), dtype=np.uint8).tobytes()
        parse_apple_makernote(junk)  # must not raise
    zero_den = bytearray(good)
    off = 16 + 12 * 3 + 4
    zero_den[off + 4:off + 8] = b"\0\0\0\0"
    assert apple_acceleration(bytes(zero_den)) is None


def test_makernote_read_from_jpeg(tmp_path):
    p = write_jpeg(tmp_path / "m.jpg", focal_35=26, extra={0x927C: makernote(">")})
    _, info = load_image(p)
    assert info.apple_acceleration == pytest.approx((-0.0125, -0.98, 0.25))


def test_natural_key_orders_numbers():
    names = ["10 bath", "2 kitchen", "01 hallway", "Bedroom", "bedroom 2"]
    assert sorted(names, key=natural_key) == ["01 hallway", "2 kitchen", "10 bath", "Bedroom", "bedroom 2"]


def _touch(path: Path, data: bytes = b"x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def test_list_room_folders_skips_sidecars_and_companions(tmp_path):
    root = tmp_path / "Scan"
    for name in ("IMG_0010.JPG", "IMG_0009.jpg"):
        write_jpeg(root / "02 kitchen" / name)
    _touch(root / "02 kitchen" / "IMG_0009.MOV")  # Live Photo companion
    _touch(root / "02 kitchen" / "IMG_0010.AAE")
    _touch(root / "02 kitchen" / "._IMG_0010.JPG")
    _touch(root / "02 kitchen" / ".DS_Store")
    _touch(root / "02 kitchen" / "notes.txt")
    write_jpeg(root / "10 bath" / "a.png")
    write_jpeg(root / "1 hallway" / "IMG_0001.jpeg")
    (root / "empty").mkdir()
    _touch(root / "__MACOSX" / "02 kitchen" / "._IMG_0009.jpg")
    write_jpeg(root / ".hidden" / "x.jpg")

    rooms = list_room_folders(root)
    assert [r for r, _ in rooms] == ["1 hallway", "02 kitchen", "10 bath"]
    assert [p.name for p in rooms[1][1]] == ["IMG_0009.jpg", "IMG_0010.JPG"]
    scan = scan_photo_folders(root)
    assert [p.name for p in scan.skipped] == ["notes.txt"]
    assert scan.empty_dirs == ["empty"]


def test_folder_of_images_is_one_room(tmp_path):
    room = tmp_path / "03 bedroom"
    for i in range(3):
        write_jpeg(room / f"IMG_{i:04d}.jpg")
    rooms = list_room_folders(room)
    assert rooms == [("03 bedroom", sorted(room.glob("*.jpg")))]
    assert list_room_folders(tmp_path / "missing") == []


def test_loose_images_next_to_room_folders_are_left_out(tmp_path):
    write_jpeg(tmp_path / "01 hallway" / "a.jpg")
    write_jpeg(tmp_path / "stray.jpg")
    scan = scan_photo_folders(tmp_path)
    assert [r for r, _ in scan.rooms] == ["01 hallway"]
    assert [p.name for p in scan.loose_images] == ["stray.jpg"]


def test_dedupe_prefers_original_format_and_unedited_copy(tmp_path):
    files = [_touch(tmp_path / n) for n in ("IMG_0001.JPG", "IMG_0001.HEIC", "IMG_E0002.HEIC", "IMG_0002.HEIC",
                                             "IMG_E0003.JPG")]
    kept, dups = dedupe_images(files)
    assert [p.name for p in kept] == ["IMG_0001.HEIC", "IMG_0002.HEIC", "IMG_E0003.JPG"]
    assert sorted(p.name for p in dups) == ["IMG_0001.JPG", "IMG_E0002.HEIC"]
