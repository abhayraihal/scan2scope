"""Photo ingest: upright RGB, EXIF fields, the Apple MakerNote acceleration vector, intrinsics and room folders."""

from __future__ import annotations

import logging
import math
import re
import struct
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

log = logging.getLogger("scan2scope.ingest")

IMAGE_EXTS = (".heic", ".heif", ".jpg", ".jpeg", ".png")  # also the preference order among duplicates
VIDEO_EXTS = (".mov", ".mp4", ".m4v", ".mkv", ".avi", ".webm", ".3gp")
SIDECAR_EXTS = (".aae", ".xmp")
JUNK_NAMES = frozenset({"thumbs.db", "desktop.ini", "icon\r"})
STRAY_MARKERS = ("odometry.csv", "camera_matrix.csv", "imu.csv")
FULL_FRAME_DIAGONAL_MM = 43.2666  # diagonal of a 36 x 24 mm frame

_EXIF_IFD = 0x8769
_MAKE, _MODEL, _ORIENTATION, _SOFTWARE = 0x010F, 0x0110, 0x0112, 0x0131
_EXPOSURE, _ISO, _DATETIME_ORIGINAL, _OFFSET_ORIGINAL = 0x829A, 0x8827, 0x9003, 0x9011
_FOCAL, _MAKERNOTE, _FOCAL_35MM, _LENS_MODEL = 0x920A, 0x927C, 0xA405, 0xA434
_APPLE_ACCELERATION = 0x0008

_heif_registered = False


def _register_heif() -> None:
    global _heif_registered
    if _heif_registered:
        return
    try:
        import pillow_heif

        pillow_heif.register_heif_opener()
    except ImportError:
        log.warning("pillow-heif is not installed; HEIC photos cannot be read")
    _heif_registered = True


@dataclass
class ExifInfo:
    make: str | None = None
    model: str | None = None
    lens_model: str | None = None
    focal_mm: float | None = None
    focal_35mm: float | None = None
    width: int = 0  # upright, after the orientation is applied
    height: int = 0
    orientation: int = 1  # EXIF orientation of the stored pixels (HEIF: the container transform)
    datetime_original: str | None = None
    iso: int | None = None
    exposure_s: float | None = None
    apple_acceleration: tuple[float, float, float] | None = None  # MakerNote 0x0008, in g, phone axes
    software: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------------------------------------
# Apple MakerNote (EXIF 0x927C): b"Apple iOS\0", 2 version bytes, b"MM", then a TIFF IFD at offset 14 whose
# value offsets count from the start of the MakerNote (ExifTool MakerNotes.pm: Start +14, Base start-14).

_TIFF_TYPES: dict[int, tuple[str, int]] = {
    1: ("B", 1), 2: ("s", 1), 3: ("H", 2), 4: ("I", 4), 5: ("I", 8), 6: ("b", 1), 7: ("s", 1),
    8: ("h", 2), 9: ("i", 4), 10: ("i", 8), 11: ("f", 4), 12: ("d", 8), 13: ("I", 4),
}


def _decode_tiff_value(typ: int, count: int, buf: bytes, endian: str) -> Any:
    if typ == 2:
        return buf.split(b"\0", 1)[0].decode("utf-8", "replace")
    if typ == 7:
        return bytes(buf)
    code = _TIFF_TYPES[typ][0]
    if typ in (5, 10):
        ints = struct.unpack(f"{endian}{2 * count}{code}", buf)
        vals: tuple[Any, ...] = tuple(n / d if d else math.nan for n, d in zip(ints[::2], ints[1::2]))
    else:
        vals = struct.unpack(f"{endian}{count}{code}", buf)
    return vals[0] if count == 1 else vals


def _ifd_entry_count(data: bytes, endian: str, clamp: bool) -> int | None:
    (n,) = struct.unpack_from(endian + "H", data, 14)
    available = (len(data) - 16) // 12
    if clamp:
        return min(n, available) or None
    return n if 0 < n <= available else None


def parse_apple_makernote(data: bytes) -> dict[int, Any]:
    """Tag id -> value for an Apple iOS MakerNote. Returns {} for other or malformed data; never raises."""
    if not isinstance(data, (bytes, bytearray)) or bytes(data[:10]) != b"Apple iOS\0" or len(data) < 28:
        return {}
    data = bytes(data)
    marker = data[12:14]
    if marker in (b"MM", b"II"):
        endian = ">" if marker == b"MM" else "<"
        n = _ifd_entry_count(data, endian, clamp=True)  # a truncated note keeps its complete entries
    else:  # ExifTool treats the byte order as unknown; take the one with a plausible entry count
        endian, n = ">", _ifd_entry_count(data, ">", clamp=False)
        if n is None:
            endian, n = "<", _ifd_entry_count(data, "<", clamp=False)
    if n is None:
        return {}
    tags: dict[int, Any] = {}
    for i in range(n):
        tag, typ, count, raw = struct.unpack_from(endian + "HHI4s", data, 16 + 12 * i)
        spec = _TIFF_TYPES.get(typ)
        if spec is None or count == 0 or count > 65536:
            continue
        size = spec[1] * count
        if size <= 4:
            buf = raw[:size]
        else:
            (offset,) = struct.unpack(endian + "I", raw)
            if offset + size > len(data):
                continue
            buf = data[offset:offset + size]
        try:
            tags[tag] = _decode_tiff_value(typ, count, buf, endian)
        except struct.error:
            continue
    return tags


def apple_acceleration(makernote: bytes) -> tuple[float, float, float] | None:
    """AccelerationVector in g: +x towards the phone's left side, +y towards its bottom, +z into its face."""
    v = parse_apple_makernote(makernote).get(_APPLE_ACCELERATION)
    if isinstance(v, tuple) and len(v) == 3 and all(math.isfinite(x) for x in v):
        return (float(v[0]), float(v[1]), float(v[2]))
    return None


# ---------------------------------------------------------------------------------------------------------
# EXIF


def _text(v: Any) -> str | None:
    if v is None:
        return None
    if isinstance(v, bytes):
        v = v.split(b"\0", 1)[0].decode("utf-8", "replace")
    s = str(v).strip("\x00 \t\r\n")
    return s or None


def _num(v: Any) -> float | None:
    if isinstance(v, (tuple, list)):
        v = v[0] if v else None
    try:
        f = float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None
    return f if math.isfinite(f) else None


def _iso_datetime(value: Any, offset: Any) -> str | None:
    raw = _text(value)
    if raw is None:
        return None
    try:
        s = datetime.strptime(raw[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return raw
    off = _text(offset)
    if off and re.fullmatch(r"[+-]\d{2}:\d{2}", off):
        s += off
    return s


def _exif_from_image(im: Image.Image) -> ExifInfo:
    try:
        exif = im.getexif()
        sub = exif.get_ifd(_EXIF_IFD)
    except Exception as exc:  # a malformed EXIF block should not make the photo unusable
        log.debug("unreadable EXIF in %s: %s", getattr(im, "filename", "?"), exc)
        exif, sub = Image.Exif(), {}
    tag_orientation = int(_num(exif.get(_ORIENTATION)) or 1)
    # pillow-heif applies the HEIF transform when decoding and resets the EXIF tag to 1, keeping the original.
    original = _num(im.info.get("original_orientation"))
    orientation = int(original) if original else tag_orientation
    if orientation not in range(1, 9):
        orientation = 1
    width, height = im.size
    if tag_orientation in (5, 6, 7, 8):
        width, height = height, width
    focal_35mm = _num(sub.get(_FOCAL_35MM))
    iso = _num(sub.get(_ISO))
    makernote = sub.get(_MAKERNOTE)
    return ExifInfo(
        make=_text(exif.get(_MAKE)),
        model=_text(exif.get(_MODEL)),
        lens_model=_text(sub.get(_LENS_MODEL)),
        focal_mm=_num(sub.get(_FOCAL)),
        focal_35mm=focal_35mm if focal_35mm and focal_35mm > 0 else None,
        width=int(width),
        height=int(height),
        orientation=orientation,
        datetime_original=_iso_datetime(sub.get(_DATETIME_ORIGINAL), sub.get(_OFFSET_ORIGINAL)),
        iso=int(iso) if iso is not None else None,
        exposure_s=_num(sub.get(_EXPOSURE)),
        apple_acceleration=apple_acceleration(makernote) if isinstance(makernote, bytes) else None,
        software=_text(exif.get(_SOFTWARE)),
    )


def read_exif(path: str | Path) -> ExifInfo:
    """EXIF of a photo without decoding its pixels."""
    _register_heif()
    with Image.open(path) as im:
        return _exif_from_image(im)


def _to_rgb_uint8(im: Image.Image) -> np.ndarray:
    if im.mode in ("I;16", "I;16L", "I;16B", "I", "F"):
        a = np.asarray(im, dtype=np.float64)
        scale = 255.0 / 65535.0 if im.mode.startswith("I;16") else 255.0 / max(float(a.max(initial=0.0)), 1.0)
        g = np.clip(a * scale, 0, 255).astype(np.uint8)
        return np.repeat(g[..., None], 3, axis=2)
    return np.array(im.convert("RGB"))


def load_image(path: str | Path) -> tuple[np.ndarray, ExifInfo]:
    """Upright RGB uint8 (H, W, 3) and its EXIF. HEIC/HEIF, JPEG and PNG."""
    _register_heif()
    with Image.open(path) as im:
        info = _exif_from_image(im)
        try:
            upright = ImageOps.exif_transpose(im)
        except Exception as exc:
            log.warning("could not apply EXIF orientation to %s: %s", path, exc)
            upright = im
        rgb = _to_rgb_uint8(upright)
    info.width, info.height = int(rgb.shape[1]), int(rgb.shape[0])
    return rgb, info


# ---------------------------------------------------------------------------------------------------------
# Intrinsics


def focal_px_from_35mm(focal_35mm: float, width: int, height: int) -> float:
    """35 mm equivalent focal length -> pixels, matched on the image diagonal."""
    return float(focal_35mm) * math.hypot(width, height) / FULL_FRAME_DIAGONAL_MM


def intrinsics_from_35mm(focal_35mm: float | None, width: int, height: int) -> np.ndarray | None:
    if focal_35mm is None or not math.isfinite(focal_35mm) or focal_35mm <= 0 or width <= 0 or height <= 0:
        return None
    f = focal_px_from_35mm(focal_35mm, width, height)
    # Pixel centres at integer coordinates (OpenCV, MapAnything), so the image centre is ((w-1)/2, (h-1)/2).
    return np.array([[f, 0.0, (width - 1) / 2.0], [0.0, f, (height - 1) / 2.0], [0.0, 0.0, 1.0]])


def intrinsics_from_exif(exif: ExifInfo | None, width: int, height: int) -> np.ndarray | None:
    """Pinhole K for an upright image of width x height, or None when the 35 mm focal length is missing."""
    return None if exif is None else intrinsics_from_35mm(exif.focal_35mm, width, height)


# ---------------------------------------------------------------------------------------------------------
# Folders


def natural_key(name: str) -> tuple[tuple[Any, ...], str]:
    """Sort key with digit runs compared as numbers: '2 kitchen' < '10 bath'."""
    parts = re.split(r"(\d+)", name.casefold())
    return tuple(int(p) if i % 2 else p for i, p in enumerate(parts)), name


@dataclass
class FolderListing:
    images: list[Path] = field(default_factory=list)
    videos: list[Path] = field(default_factory=list)  # not Live Photo companions
    companions: list[Path] = field(default_factory=list)  # Live Photo .mov sharing a stem with a photo
    sidecars: list[Path] = field(default_factory=list)  # .aae/.xmp edit sidecars and OS junk files
    other: list[Path] = field(default_factory=list)
    dirs: list[Path] = field(default_factory=list)


def list_dir(folder: str | Path) -> FolderListing:
    """Classify the entries of one folder, skipping hidden files, AppleDouble ._ files and __MACOSX."""
    out = FolderListing()
    try:
        entries = list(Path(folder).iterdir())
    except OSError as exc:
        log.warning("cannot list %s: %s", folder, exc)
        return out
    for p in entries:
        name = p.name
        if name.startswith(".") or name == "__MACOSX":
            continue
        if p.is_dir():
            out.dirs.append(p)
        elif p.is_file():
            ext = p.suffix.lower()
            if ext in IMAGE_EXTS:
                out.images.append(p)
            elif ext in VIDEO_EXTS:
                out.videos.append(p)
            elif ext in SIDECAR_EXTS or name.casefold() in JUNK_NAMES:
                out.sidecars.append(p)
            else:
                out.other.append(p)
    stems = {p.stem.casefold() for p in out.images}
    out.companions = [v for v in out.videos if v.stem.casefold() in stems]
    out.videos = [v for v in out.videos if v.stem.casefold() not in stems]
    for lst in (out.images, out.videos, out.companions, out.sidecars, out.other, out.dirs):
        lst.sort(key=lambda p: natural_key(p.name))
    return out


_EXT_RANK = {e: i for i, e in enumerate(IMAGE_EXTS)}
_EDITED_COPY = re.compile(r"^(img_)e(\d+)$", re.IGNORECASE)  # iOS exports an edited IMG_0001 as IMG_E0001


def dedupe_images(images: list[Path]) -> tuple[list[Path], list[Path]]:
    """Drop second copies of a photo: the same stem in another format, or an iOS IMG_E edit of an original."""
    first: dict[str, Path] = {}
    dups: list[Path] = []
    for p in sorted(images, key=lambda p: (_EXT_RANK.get(p.suffix.lower(), 99), natural_key(p.name))):
        stem = p.stem.casefold()
        if stem in first:
            dups.append(p)
        else:
            first[stem] = p
    kept = []
    for stem, p in first.items():
        m = _EDITED_COPY.match(stem)
        if m and f"{m.group(1)}{m.group(2)}" in first:
            dups.append(p)
        else:
            kept.append(p)
    kept.sort(key=lambda p: natural_key(p.name))
    dups.sort(key=lambda p: natural_key(p.name))
    return kept, dups


@dataclass
class PhotoScan:
    root: Path
    rooms: list[tuple[str, list[Path]]]
    loose_images: list[Path] = field(default_factory=list)  # photos next to room folders, not used
    duplicates: list[Path] = field(default_factory=list)
    skipped: list[Path] = field(default_factory=list)  # unsupported files inside the photo folders
    videos: list[Path] = field(default_factory=list)  # videos at the top level, not used by the photo tier
    empty_dirs: list[str] = field(default_factory=list)  # subfolders without photos


def scan_photo_folders(root: str | Path) -> PhotoScan:
    """Rooms of a photo capture with everything that was left out. See list_room_folders."""
    root = Path(root)
    scan = PhotoScan(root=root, rooms=[])
    if not root.is_dir():
        return scan
    top = list_dir(root)
    stray_like = any((root / m).is_file() for m in STRAY_MARKERS)
    for d in top.dirs:
        if stray_like and d.name.casefold() in ("depth", "confidence"):
            continue
        sub = list_dir(d)
        images, dups = dedupe_images(sub.images)
        if not images:
            scan.empty_dirs.append(d.name)
            continue
        scan.rooms.append((d.name, images))
        scan.duplicates += dups
        scan.skipped += sub.other + sub.videos
    if scan.rooms:
        scan.loose_images = top.images
        scan.skipped += top.other
        scan.videos = top.videos
    elif top.images:
        images, dups = dedupe_images(top.images)
        scan.rooms = [(root.name, images)]
        scan.duplicates += dups
        scan.skipped += top.other
        scan.videos = top.videos
    return scan


def list_room_folders(root: str | Path) -> list[tuple[str, list[Path]]]:
    """(room name, photos) per room, both naturally sorted.

    Each subfolder holding photos is a room named after the folder. A folder holding photos directly (and no
    photo subfolders) is a single room named after the folder. Hidden files, AppleDouble ._ files, .aae/.xmp
    sidecars and Live Photo .mov companions are skipped, and a second copy of a photo is dropped.
    """
    return scan_photo_folders(root).rooms
