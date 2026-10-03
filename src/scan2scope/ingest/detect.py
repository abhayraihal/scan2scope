"""Capture detection: unpack zips safely, work out the tier from the files and summarise the input."""

from __future__ import annotations

import json
import logging
import re
import shutil
import stat
import tempfile
import zipfile
import zlib
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from scan2scope.ingest.images import (
    IMAGE_EXTS,
    VIDEO_EXTS,
    PhotoScan,
    list_dir,
    read_exif,
    scan_photo_folders,
)
from scan2scope.types import TIERS, CaptureInfo

log = logging.getLogger("scan2scope.ingest")

PHOTO_MIN, PHOTO_MAX = 2, 8  # photos per room the protocol asks for; outside this range is flagged, not refused
_ZIP_STAMP = ".scan2scope_zip.json"
_MAX_DESCEND = 3
_DISK_MARGIN = 512 << 20

EXPECTED = (
    "a folder of room folders with photos (photo tier), one folder of photos (photo tier, one room), "
    "a video file or a folder holding one video (video tier), or a Stray Scanner export with odometry.csv "
    "and depth/ (LiDAR tier); any of these may also be a .zip"
)
_FORCED = {
    "lidar": "--tier lidar expects a Stray Scanner export: a folder (or its zip) with odometry.csv and a depth/ "
             "folder, at most one level down",
    "video": "--tier video expects a video file (.mov, .mp4, ...) or a folder holding one",
    "photo": "--tier photo expects a folder of room folders holding photos (.heic, .jpg, .png), or one folder "
             "of photos",
}


# ---------------------------------------------------------------------------------------------------------
# Zip extraction


def _member_parts(info: zipfile.ZipInfo) -> tuple[str, ...] | None:
    """Path parts of a zip member, or None when it could land outside the extraction folder."""
    name = info.filename.replace("\\", "/")
    if name.startswith("/") or re.match(r"^[A-Za-z]:", name) or "\0" in name:
        return None
    parts = tuple(p for p in PurePosixPath(name).parts if p not in ("", "."))
    if not parts or ".." in parts:
        return None
    if stat.S_ISLNK(info.external_attr >> 16):  # a symlink could point anywhere
        return None
    return parts


def extract_zip(zip_path: Path, dest_root: Path, flags: list[str]) -> Path:
    """Unpack zip_path into dest_root/<zip stem>/ and return that folder, or its single top-level folder.

    Skips absolute paths, '..' components, symlinks, __MACOSX and AppleDouble files. A previous extraction of
    the same zip (same path, size and mtime) is reused.
    """
    stem = zip_path.stem.strip() or "capture"
    dest = dest_root / stem
    n = 1
    while zip_path.resolve().is_relative_to(dest.resolve()):  # a zip nested in an earlier extraction
        n += 1
        dest = dest_root / f"{stem}-{n}"
    st = zip_path.stat()
    stamp = {"source": str(zip_path.resolve()), "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    stamp_file = dest / _ZIP_STAMP
    try:
        reuse = json.loads(stamp_file.read_text()) == stamp
    except (OSError, ValueError):
        reuse = False
    if reuse:
        log.info("reusing %s, extracted from %s earlier", dest, zip_path.name)
    else:
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        dest_resolved = dest.resolve()
        try:
            with zipfile.ZipFile(zip_path) as zf:
                members, unsafe = [], 0
                for info in zf.infolist():
                    if info.is_dir():
                        continue
                    parts = _member_parts(info)
                    if parts is None:
                        unsafe += 1
                        continue
                    if parts[0] == "__MACOSX" or any(p.startswith("._") for p in parts) or parts[-1] == ".DS_Store":
                        continue
                    target = dest.joinpath(*parts)
                    if not target.resolve().is_relative_to(dest_resolved):
                        unsafe += 1
                        continue
                    members.append((info, target))
                if unsafe:
                    log.warning("skipped %d zip entries with unsafe paths in %s", unsafe, zip_path.name)
                    flags.append(f"zip_unsafe_entries_skipped:{unsafe}")
                total = sum(info.file_size for info, _ in members)
                free = shutil.disk_usage(dest).free
                if total > free - _DISK_MARGIN:
                    raise ValueError(f"{zip_path.name} unpacks to {total / 1e9:.1f} GB but only "
                                     f"{free / 1e9:.1f} GB is free under {dest_root}")
                for info, target in members:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
        except (zipfile.BadZipFile, zlib.error, EOFError) as exc:
            raise ValueError(f"{zip_path.name} is not a readable zip: {exc}") from exc
        except (RuntimeError, NotImplementedError) as exc:  # encrypted member, unsupported compression
            raise ValueError(f"cannot unpack {zip_path.name}: {exc}") from exc
        stamp_file.write_text(json.dumps(stamp))
        log.info("extracted %d files from %s into %s", len(members), zip_path.name, dest)
    listing = list_dir(dest)
    if len(listing.dirs) == 1 and not (listing.images or listing.videos or listing.other):
        return listing.dirs[0]
    return dest


def _is_zip(path: Path) -> bool:
    if path.suffix.lower() == ".zip":
        return True
    return path.suffix.lower() not in VIDEO_EXTS + IMAGE_EXTS and zipfile.is_zipfile(path)


# ---------------------------------------------------------------------------------------------------------
# Tier detection


def _is_stray(folder: Path) -> bool:
    return (folder / "odometry.csv").is_file() and (folder / "depth").is_dir()


def _stray_candidates(base: Path) -> list[Path]:
    if _is_stray(base):
        return [base]
    return [d for d in list_dir(base).dirs if _is_stray(d)]


def _count_files(folder: Path, exts: tuple[str, ...]) -> int:
    try:
        return sum(1 for p in folder.iterdir() if p.suffix.lower() in exts and not p.name.startswith("."))
    except OSError:
        return 0


def _video_duration(path: Path) -> float:
    from scan2scope.ingest.video import probe

    try:
        return probe(path).duration_s
    except ValueError:
        return -1.0


def _probe_video(path: Path) -> bool:
    """True for a file of unrecognised extension that decodes as a video longer than one frame."""
    from scan2scope.ingest.video import probe

    try:
        info = probe(path)
    except ValueError:
        return False
    return info.n_frames > 1 and info.codec not in ("png", "mjpeg", "gif", "bmp", "tiff", "webp")


def _pick_video(videos: list[Path], flags: list[str]) -> Path | None:
    if len(videos) == 1:
        return videos[0]
    durations = {v: _video_duration(v) for v in videos}
    readable = [v for v in videos if durations[v] >= 0]
    if not readable:
        return None
    best = max(readable, key=lambda v: (durations[v], v.stat().st_size))
    log.warning("%d videos found; using the longest, %s", len(videos), best.name)
    flags.append(f"multiple_videos:{len(videos)}")
    return best


def _video_choice(listing: Any, flags: list[str], pick_one: bool) -> tuple[str, Path, Any] | None:
    """Video tier result for a folder: (tier, video, files skipped next to it)."""
    if not listing.videos or (len(listing.videos) > 1 and not pick_one):
        return None
    video = _pick_video(listing.videos, flags)
    if video is None:
        return None
    return "video", video, [p for p in listing.other if p.suffix.lower() != ".zip"]


def _classify(base: Path, tier: str | None, flags: list[str]) -> tuple[str, Path, Any] | None:
    """(tier, root, detail) for this folder, or None when nothing at this level matches."""
    stray = _stray_candidates(base)
    if stray and tier in (None, "lidar"):
        if len(stray) > 1:
            flags.append(f"multiple_lidar_captures:{len(stray)}")
            stray.sort(key=lambda d: -_count_files(d / "depth", (".png", ".npy")))
            log.warning("%d Stray Scanner captures found; using %s", len(stray), stray[0].name)
        return "lidar", stray[0], None
    if tier == "lidar":
        return None
    listing = list_dir(base)
    if tier == "video":
        # The Stray RGB video, unless the folder holds its own videos next to a Stray export one level down.
        if stray and (stray[0] == base or not listing.videos) and (stray[0] / "rgb.mp4").is_file():
            flags.append("video_from_stray_rgb")
            return "video", stray[0] / "rgb.mp4", []
        return _video_choice(listing, flags, pick_one=True)
    scan = scan_photo_folders(base)
    if tier == "photo":
        if scan.rooms:
            return "photo", base, scan
        if stray:
            raise ValueError(f"{_FORCED['photo']}; {base} is a Stray Scanner export, whose depth/ and confidence/ "
                             "PNGs are not photos. Use --tier lidar, or --tier video for its rgb.mp4")
        return None
    # Automatic, in contract order: lidar (above), exactly one video, photos, then the longest of several videos.
    if len(listing.videos) == 1:
        if not scan.rooms or _video_duration(listing.videos[0]) >= 0:
            if scan.rooms:
                flags.append(f"photos_ignored:{sum(len(p) for _, p in scan.rooms)}")
            if (base / "depth").is_dir() or any((base / m).is_file() for m in ("camera_matrix.csv", "imu.csv")):
                flags.append("stray_export_incomplete")  # Stray-like folder without odometry.csv: RGB only
            return _video_choice(listing, flags, pick_one=False)
        flags.append(f"video_unreadable:{listing.videos[0].name}")
    if scan.rooms:
        return "photo", base, scan
    return _video_choice(listing, flags, pick_one=True)


def _n(count: int, one: str, many: str) -> str:
    return f"{count} {one if count == 1 else many}"


def _describe(folder: Path) -> str:
    """Short inventory of a folder for error messages."""
    listing = list_dir(folder)
    scan = scan_photo_folders(folder)
    room_dirs = {name for name, _ in scan.rooms} if listing.dirs else set()
    exts = Counter(p.suffix.lower() or "(no extension)" for p in listing.other)
    bits = []
    if listing.images:
        bits.append(_n(len(listing.images), "photo", "photos"))
    if listing.videos or listing.companions:
        bits.append(_n(len(listing.videos) + len(listing.companions), "video", "videos"))
    if listing.other:
        kinds = ", ".join(f"{n} {e}" for e, n in exts.most_common(4))
        bits.append(f"{_n(len(listing.other), 'other file', 'other files')} ({kinds})")
    rooms = [d.name for d in listing.dirs if d.name in room_dirs]
    empty = [d.name for d in listing.dirs if d.name not in room_dirs]
    for names, one, many in ((rooms, "room folder with photos", "room folders with photos"),
                             (empty, "folder without photos", "folders without photos")):
        if names:
            shown = ", ".join(names[:4]) + (", ..." if len(names) > 4 else "")
            bits.append(f"{_n(len(names), one, many)} ({shown})")
    return "; ".join(bits) or "nothing usable (empty or hidden files only)"


def _single_child(folder: Path) -> tuple[str, Path] | None:
    """The one subfolder or zip a wrapper folder holds, when it holds nothing else."""
    listing = list_dir(folder)
    zips = [p for p in listing.other if p.suffix.lower() == ".zip"]
    if listing.images or listing.videos or len(listing.other) - len(zips) > 0:
        return None
    if len(listing.dirs) == 1 and not zips:
        return "dir", listing.dirs[0]
    if len(zips) == 1 and not listing.dirs:
        return "zip", zips[0]
    return None


# ---------------------------------------------------------------------------------------------------------
# Input summaries


def _photo_summary(scan: PhotoScan, flags: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    rooms, devices, lenses, softwares, focals, sizes = [], Counter(), Counter(), Counter(), Counter(), Counter()
    no_focal = unreadable = 0
    for name, photos in scan.rooms:
        room_no_focal = 0
        for p in photos:
            try:
                ex = read_exif(p)
            except Exception as exc:  # any decoder error: count the photo as unreadable, keep going
                log.warning("cannot read %s: %s", p, exc)
                unreadable += 1
                continue
            if ex.make or ex.model:
                devices[(ex.make, ex.model)] += 1
            if ex.lens_model:
                lenses[ex.lens_model] += 1
            if ex.software:
                softwares[ex.software] += 1
            if ex.focal_35mm:
                focals[ex.focal_35mm] += 1
            else:
                room_no_focal += 1
            sizes[f"{ex.width}x{ex.height}"] += 1
        no_focal += room_no_focal
        rooms.append({"name": name, "photos": len(photos), "photos_without_focal": room_no_focal})
        if len(photos) < PHOTO_MIN:
            flags.append(f"photo_count_low:{name}")
        elif len(photos) > PHOTO_MAX:
            flags.append(f"photo_count_high:{name}")
    if no_focal:
        flags.append("exif_focal_missing")
    if unreadable:
        flags.append(f"unreadable_images:{unreadable}")
    if scan.duplicates:
        flags.append(f"duplicate_images_skipped:{len(scan.duplicates)}")
    if scan.loose_images:
        flags.append(f"photo_loose_images_ignored:{len(scan.loose_images)}")
    if scan.videos:
        flags.append(f"videos_ignored:{len(scan.videos)}")
    if scan.skipped:
        flags.append(f"unsupported_files_skipped:{len(scan.skipped)}")
    for name in scan.empty_dirs:
        flags.append(f"photo_folder_empty:{name}")
    if len(devices) > 1:
        flags.append("multiple_devices")
    make, model = devices.most_common(1)[0][0] if devices else (None, None)
    device = {
        "make": make,
        "model": model,
        "lens_model": lenses.most_common(1)[0][0] if lenses else None,
        "software": softwares.most_common(1)[0][0] if softwares else None,
        "focal_35mm": focals.most_common(1)[0][0] if focals else None,
        "source": "exif",
    }
    stats = {
        "rooms": rooms,
        "n_rooms": len(rooms),
        "n_photos": sum(r["photos"] for r in rooms),
        "photos_without_focal": no_focal,
        "image_sizes": dict(sizes.most_common()),
        "lens_models": dict(lenses.most_common()),
    }
    return stats, device


def _video_summary(video: Path, skipped: list[Path] | None, flags: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    from scan2scope.ingest.video import probe

    info = probe(video)
    stats = {
        "video_file": video.name,
        "duration_s": info.duration_s,
        "fps": round(info.fps, 3),
        "n_frames": info.n_frames,
        "width": info.width,
        "height": info.height,
        "rotation_deg": info.rotation_deg,
        "codec": info.codec,
        "pix_fmt": info.pix_fmt,
        "is_hdr": info.is_hdr,
        "file_size_mb": round(video.stat().st_size / 1e6, 1),
    }
    device = {"make": info.make, "model": info.model, "lens_model": info.lens_model, "software": info.software,
              "focal_35mm": info.focal_35mm, "creation_time": info.creation_time, "source": "quicktime"}
    if info.is_hdr:
        flags.append("video_hdr")
    if info.focal_35mm is None:
        flags.append("video_focal_missing")
    if info.n_frames == 0 or info.duration_s <= 0:
        flags.append("video_duration_unknown")
    if skipped:
        flags.append(f"unsupported_files_skipped:{len(skipped)}")
    return stats, device


def _csv_rows(path: Path) -> tuple[int, float | None]:
    """Data rows of a Stray Scanner CSV and the time span of its first column."""
    try:
        lines = [ln for ln in path.read_text(errors="replace").splitlines() if ln.strip()]
    except OSError:
        return 0, None
    rows = lines[1:] if lines and not lines[0].lstrip()[:1].isdigit() else lines
    span = None
    try:
        span = float(rows[-1].split(",")[0]) - float(rows[0].split(",")[0])
    except (IndexError, ValueError):
        pass
    return len(rows), span


def _lidar_summary(root: Path, flags: list[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    n_odom, span = _csv_rows(root / "odometry.csv")
    n_depth = _count_files(root / "depth", (".png", ".npy"))
    n_conf = _count_files(root / "confidence", (".png", ".npy"))
    rgb = root / "rgb.mp4"
    stats: dict[str, Any] = {
        "lidar_frames": n_odom or n_depth,
        "odometry_rows": n_odom,
        "depth_frames": n_depth,
        "confidence_frames": n_conf,
        "duration_s": round(span, 3) if span is not None else None,
        "rgb_video": rgb.is_file(),
        "camera_matrix": (root / "camera_matrix.csv").is_file(),
        "imu": (root / "imu.csv").is_file(),
    }
    device: dict[str, Any] = {"app": "Stray Scanner", "make": None, "model": None, "source": "stray"}
    if rgb.is_file():
        from scan2scope.ingest.video import probe

        try:
            info = probe(rgb)
            stats.update(rgb_width=info.width, rgb_height=info.height, rgb_fps=round(info.fps, 3),
                         rgb_frames=info.n_frames)
            device.update(make=info.make, model=info.model)
        except ValueError as exc:
            log.warning("cannot read %s: %s", rgb, exc)
            flags.append("lidar_rgb_unreadable")
    else:
        flags.append("lidar_rgb_missing")
    if n_odom and n_depth and abs(n_odom - n_depth) > max(2, 0.01 * n_odom):
        flags.append(f"lidar_frame_count_mismatch:{n_odom}/{n_depth}")
    if n_depth == 0:
        flags.append("lidar_depth_empty")
    if n_conf == 0:
        flags.append("lidar_confidence_missing")
    if not stats["camera_matrix"]:
        flags.append("lidar_camera_matrix_missing")
    return stats, device


# ---------------------------------------------------------------------------------------------------------


def _stage_single_image(image: Path, work_dir: Path) -> Path:
    room = work_dir / "input" / image.stem
    room.mkdir(parents=True, exist_ok=True)
    target = room / image.name
    if not target.exists():
        shutil.copy2(image, target)
    return room


def detect_capture(path: str | Path, tier: str | None = None,
                   work_dir: str | Path | None = None) -> tuple[str, Path, CaptureInfo]:
    """Work out the tier of a capture and where its data is.

    Returns (tier, root, info): root is the Stray Scanner folder (lidar), the video file (video) or the folder
    whose subfolders are rooms or which is itself one room (photo). A zip is extracted into work_dir/input
    first. info.input_stats holds rooms and photo counts (photo), duration, fps and resolution (video) or
    frame counts (lidar). Raises ValueError when the files match no tier, or not the forced one.
    """
    source = Path(path).expanduser()
    if tier is not None and tier not in TIERS:
        raise ValueError(f"tier must be one of {', '.join(TIERS)}, got {tier!r}")
    if not source.exists():
        raise ValueError(f"{source} does not exist; expected {EXPECTED}")
    flags: list[str] = []
    work = Path(work_dir) if work_dir is not None else None

    def workspace() -> Path:
        nonlocal work
        if work is None:
            work = Path(tempfile.mkdtemp(prefix="scan2scope-"))
            log.info("unpacking into %s", work)
        return work

    current = source
    stats_extra: dict[str, Any] = {}
    if current.is_file() and _is_zip(current):
        stats_extra["source_zip"] = current.name
        current = extract_zip(current, workspace() / "input", flags)

    result: tuple[str, Path, Any] | None = None
    if current.is_file():
        ext = current.suffix.lower()
        kind = "image" if ext in IMAGE_EXTS else "video" if ext in VIDEO_EXTS or _probe_video(current) else None
        if kind == "video" and tier in (None, "video"):
            if ext not in VIDEO_EXTS:
                flags.append(f"video_extension_unrecognised:{ext or 'none'}")
            result = "video", current, []
        elif kind == "image" and tier in (None, "photo"):
            flags.append("single_image_input")
            current = _stage_single_image(current, workspace())
            result = "photo", current, scan_photo_folders(current)
        elif tier is not None:
            what = f"a {kind}" if kind else f"a single {ext or 'extensionless'} file"
            raise ValueError(f"{_FORCED[tier]}; {source} is {what}")
        else:
            raise ValueError(f"{source} is a {ext or 'extensionless'} file that is not a video; expected {EXPECTED}")
    else:
        for _ in range(_MAX_DESCEND + 1):
            result = _classify(current, tier, flags)
            if result is not None:
                break
            child = _single_child(current)
            if child is None:
                break
            kind, nxt = child
            if kind == "zip":
                stats_extra["source_zip"] = nxt.name
                nxt = extract_zip(nxt, workspace() / "input", flags)
            log.info("descending into %s", nxt)
            current = nxt
        if result is None:
            found = _describe(current)
            if tier is not None:
                auto = None
                try:
                    auto = _classify(current, None, [])
                except ValueError:
                    pass
                hint = f"; it looks like a {auto[0]} capture, so drop --tier or use --tier {auto[0]}" if auto else ""
                raise ValueError(f"{_FORCED[tier]}. {current} has: {found}{hint}")
            raise ValueError(f"cannot tell what kind of capture {current} is; expected {EXPECTED}. Found: {found}")

    tier_found, root, detail = result
    if tier_found == "photo":
        stats, device = _photo_summary(detail, flags)
    elif tier_found == "video":
        stats, device = _video_summary(root, detail, flags)
    else:
        stats, device = _lidar_summary(root, flags)
    stats.update(stats_extra)
    capture_id = (source.stem if source.is_file() else source.name) or "capture"
    info = CaptureInfo(id=capture_id, tier=tier_found, path=str(source), device=device, input_stats=stats,
                       flags=flags)
    log.info("capture %s: %s tier at %s%s", capture_id, tier_found, root,
             f" (flags: {', '.join(flags)})" if flags else "")
    return tier_found, root, info
