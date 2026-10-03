"""Reader for Stray Scanner datasets, the LiDAR tier input.

Conventions follow the app's own encoders (github.com/strayrobots/scanner, MIT, main branch read 2026-10-03:
docs/format.md, StrayScanner/Helpers/{OdometryEncoder,DatasetEncoder,VideoEncoder,DepthEncoder,
ConfidenceEncoder,IMUEncoder}.swift, StrayScanner/CCode/PngEncoder.mm, Controllers/RecordSessionViewController
.swift) and the official viewer (github.com/kekeblom/StrayVisualizer stray_visualize.py, MIT):

- odometry.csv: header "timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x,
  distortion_center_y". Every field after a comma has a leading space. Versions before April 2026 stop after qw;
  the distortion centre is empty when the device gives no calibration data. x, y, z is the position of
  ARCamera.transform; the quaternion is q_WA * q_AC with q_AC a 180 degree rotation about x, so it is the
  camera-to-world rotation with OpenCV camera axes (x right, y down, z forward), as the viewer uses it. The world
  is ARWorldTrackingConfiguration's default (worldAlignment .gravity): y up, gravity aligned, x and z set by the
  heading when the AR session started. fx, fy, cx, cy are ARCamera.intrinsics, valid for the rgb.mp4 resolution.
- frame is the saved-frame counter: frames skipped by the fps divider or dropped by the encoder's back-pressure
  semaphore get no row, no depth file and no video frame, so `frame` names depth/NNNNNN.png,
  confidence/NNNNNN.png and the frame's position in rgb.mp4.
- rgb.mp4: HEVC written by AVAssetWriter at the captured-image resolution (1920x1440 on current iPhones). The
  first frame is appended at -1/60 s, before the writer session starts at 0, and the file gets an edit list
  that hides it. FFmpeg honours the edit list and drops that frame, which shifts every image one frame against
  odometry.csv (the viewer's zip of poses and frames has this offset), so the video is opened with the mov
  demuxer option ignore_editlist. Checked on a real 3481-frame recording: 3480 frames decode by default, 3481
  with the option.
- depth/NNNNNN.png: 16-bit grey PNG of round(depth_m * 1000), 256x192, z-depth along the optical axis.
  Versions from 2021 wrote depth/NNNNNN.npy instead.
- confidence/NNNNNN.png: 8-bit grey ARConfidenceLevel, 0 low, 1 medium, 2 high.
- camera_matrix.csv: K of the last frame, rows "fx, 0.0, cx", "0.0, fy, cy", "0.0, 0.0, 1.0". It is the only
  intrinsics source for files without the fx..cy columns.
- imu.csv: "timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z", device clock shared with odometry.csv. Since
  February 2025 the app logs raw CMAccelerometerData, which is in g although format.md says m/s^2; earlier
  versions logged (userAcceleration + gravity) * 9.81. The unit is told apart by the median magnitude.
- distortion/NNNNNN.bin (optional): AVCameraCalibrationData lookup tables; not applied here, ARKit intrinsics are
  used as a pinhole model.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

log = logging.getLogger("scan2scope.ingest.stray")

REQUIRED_COLUMNS = ("timestamp", "frame", "x", "y", "z", "qx", "qy", "qz", "qw")
INTRINSIC_COLUMNS = ("fx", "fy", "cx", "cy")
DEFAULT_DEPTH_SIZE = (256, 192)
DEFAULT_RGB_SIZE = (1920, 1440)
# fx / width of the 1x camera in real recordings (1339 px at 1920); used only when no intrinsics exist at all
FALLBACK_FX_PER_WIDTH = 0.697
G = 9.80665
# open rgb.mp4 with the edit list ignored so the frame AVAssetWriter wrote at -1/60 s is decoded too
VIDEO_OPTIONS = {"ignore_editlist": "1"}


def scale_intrinsics(K: np.ndarray, src_size: tuple[int, int], dst_size: tuple[int, int]) -> np.ndarray:
    """Intrinsics for an image resampled from src_size (w, h) to dst_size, pixel centres at integer coordinates.

    Pixel j of the destination covers source pixel (j + 0.5) * src_w / dst_w - 0.5, as in types.CameraView.
    Works on (3, 3) or (..., 3, 3).
    """
    K = np.asarray(K, float)
    sx, sy = dst_size[0] / src_size[0], dst_size[1] / src_size[1]
    out = K.copy()
    out[..., 0, 0] = K[..., 0, 0] * sx
    out[..., 0, 1] = K[..., 0, 1] * sx
    out[..., 1, 1] = K[..., 1, 1] * sy
    out[..., 0, 2] = (K[..., 0, 2] + 0.5) * sx - 0.5
    out[..., 1, 2] = (K[..., 1, 2] + 0.5) * sy - 0.5
    return out


def _float(s: str) -> float:
    try:
        v = float(s)
    except ValueError:
        return math.nan
    return v


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]], bool]:
    """Lower-case header names (empty when the first line is numeric), stripped fields per row, has_header."""
    text = path.read_text(encoding="utf-8-sig", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln for ln in text.split("\n") if ln.strip()]
    if not lines:
        return [], [], True
    first = [f.strip().lower() for f in lines[0].split(",")]
    has_header = not first or math.isnan(_float(first[0]))
    rows = [[f.strip() for f in ln.split(",")] for ln in (lines[1:] if has_header else lines)]
    return (first if has_header else []), rows, has_header


def _files_by_number(folder: Path, suffixes: tuple[str, ...]) -> dict[int, Path]:
    out: dict[int, Path] = {}
    if not folder.is_dir():
        return out
    for p in folder.iterdir():
        if p.name.startswith(".") or p.suffix.lower() not in suffixes:
            continue
        try:
            n = int(p.stem)
        except ValueError:
            continue
        if n not in out or p.suffix.lower() == ".png":
            out[n] = p
    return out


def find_dataset_root(root: Path) -> Path:
    """The folder holding odometry.csv: root itself or a single subfolder (zips add one level)."""
    root = Path(root)
    if (root / "odometry.csv").is_file():
        return root
    subs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith((".", "__MACOSX"))
            and (p / "odometry.csv").is_file()] if root.is_dir() else []
    if len(subs) == 1:
        return subs[0]
    raise FileNotFoundError(f"no Stray Scanner dataset (odometry.csv) in {root}"
                            + (f"; {len(subs)} candidate folders" if subs else ""))


@dataclass
class StrayCapture:
    """A parsed Stray Scanner dataset. Arrays are per odometry row, sorted by frame id.

    Row i uses depth/{frame_ids[i]:06d}.png and video frame frame_ids[i]. read_depth and read_conf take the row
    index i; extract_rgb takes frame ids.
    """

    root: Path
    frame_ids: np.ndarray  # (N,) int64
    timestamps: np.ndarray  # (N,) float64 seconds, device clock
    T_wc: np.ndarray  # (N, 4, 4) camera-to-world, OpenCV camera axes, ARKit world (y up)
    K: np.ndarray  # (N, 3, 3) intrinsics at rgb_size
    rgb_size: tuple[int, int]  # (width, height) of rgb.mp4; from the principal point if unreadable (flagged)
    depth_size: tuple[int, int] | None  # (width, height) of the depth maps
    has_depth: np.ndarray  # (N,) bool
    has_conf: np.ndarray  # (N,) bool
    n_video_frames: int | None = None
    imu: np.ndarray | None = None  # (M, 7): t, a_x, a_y, a_z in m/s^2, w_x, w_y, w_z in rad/s
    imu_accel_unit: str | None = None  # unit found in imu.csv: "g" or "m/s2"
    distortion_center: np.ndarray | None = None  # (N, 2), NaN where missing
    format: str = "v1.4"  # "v1.4" with per-frame intrinsics, "legacy" without
    flags: list[str] = field(default_factory=list)
    _depth_files: dict[int, Path] = field(default_factory=dict, repr=False)
    _conf_files: dict[int, Path] = field(default_factory=dict, repr=False)

    def __len__(self) -> int:
        return len(self.frame_ids)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps.max() - self.timestamps.min()) if len(self) > 1 else 0.0

    @property
    def fps(self) -> float:
        if len(self) < 2:
            return 0.0
        dt = np.diff(self.timestamps)
        dt = dt[dt > 0]
        return float(1.0 / np.median(dt)) if len(dt) else 0.0

    @property
    def video_path(self) -> Path:
        return self.root / "rgb.mp4"

    def depth_path(self, i: int) -> Path | None:
        return self._depth_files.get(int(self.frame_ids[i]))

    def read_depth(self, i: int) -> np.ndarray | None:
        """Depth of row i in metres, float32 (h, w); 0 where there is no measurement; None if the file is missing."""
        p = self.depth_path(i)
        if p is None:
            return None
        try:
            if p.suffix.lower() == ".npy":
                a = np.load(p)
                a = np.asarray(a, np.float32)
                # 2021 builds stored millimetres; treat small float values as metres
                d = a if (a.dtype.kind == "f" and np.nanmax(a, initial=0.0) < 100.0) else a / 1000.0
            else:
                raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
                if raw is None:
                    return None
                if raw.ndim == 3:
                    raw = raw[..., 0]
                d = raw.astype(np.float32) / (1000.0 if raw.dtype == np.uint16 else 1.0)
        except Exception as exc:  # unreadable file: the caller skips the frame
            log.debug("depth %s unreadable: %s", p, exc)
            return None
        d = np.nan_to_num(d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        d[d < 0] = 0.0
        return d

    def read_conf(self, i: int) -> np.ndarray | None:
        """Confidence of row i, uint8 (h, w) with 0 low, 1 medium, 2 high; None if the file is missing."""
        p = self._conf_files.get(int(self.frame_ids[i]))
        if p is None:
            return None
        try:
            c = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        except Exception as exc:
            log.debug("confidence %s unreadable: %s", p, exc)
            return None
        if c is None:
            return None
        if c.ndim == 3:
            c = c[..., 0]
        return np.clip(c, 0, 2).astype(np.uint8)

    def row_of_frame(self, frame_id: int) -> int | None:
        i = int(np.searchsorted(self.frame_ids, frame_id))
        return i if i < len(self) and self.frame_ids[i] == frame_id else None

    def extract_rgb(self, frame_ids: Sequence[int], out_dir: Path, *, turns: Sequence[int] | None = None,
                    quality: int = 92) -> list[Path | None]:
        """Decode rgb.mp4 once and write the requested frames as JPEGs; None for frames that cannot be decoded.

        Frames are stored in the landscape sensor orientation, which is the orientation K is for; turns[i] rotates
        frame i by that many quarter turns counter-clockwise (np.rot90) before saving.
        """
        import av
        from PIL import Image

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        ids = [int(f) for f in frame_ids]
        paths: dict[int, Path] = {}
        if not ids:
            return []
        if not self.video_path.is_file():
            self._flag("rgb_missing")
            return [None] * len(ids)
        wanted = set(ids)
        turn_of = dict(zip(ids, turns)) if turns is not None else {}
        last = max(wanted)
        n_err = 0
        try:
            with av.open(str(self.video_path), options=VIDEO_OPTIONS) as container:
                stream = container.streams.video[0]
                stream.thread_type = "AUTO"
                index = 0
                for packet in container.demux(stream):
                    try:
                        frames = packet.decode()
                    except av.error.FFmpegError:
                        n_err += 1
                        continue
                    for frame in frames:
                        if index in wanted:
                            p = out_dir / f"frame_{index:06d}.jpg"
                            img = np.rot90(frame.to_ndarray(format="rgb24"), turn_of.get(index, 0) % 4)
                            Image.fromarray(np.ascontiguousarray(img)).save(p, quality=quality)
                            paths[index] = p
                        index += 1
                    if index > last:
                        break
        except Exception as exc:
            log.warning("rgb.mp4 could not be decoded: %s", exc)
            self._flag(f"rgb_unreadable:{type(exc).__name__}")
        if n_err:
            self._flag(f"rgb_decode_errors:{n_err}")
        missing = len(wanted - set(paths))
        if missing:
            self._flag(f"rgb_frames_not_decoded:{missing}")
        return [paths.get(i) for i in ids]

    def _flag(self, flag: str) -> None:
        if flag not in self.flags:
            self.flags.append(flag)


def _parse_odometry(path: Path, flags: list[str]) -> dict[str, np.ndarray]:
    names, rows, has_header = _read_csv_rows(path)
    if not has_header:
        flags.append("odometry_header_missing")
        width = max((len(r) for r in rows), default=0)
        names = list(REQUIRED_COLUMNS) + (list(INTRINSIC_COLUMNS) + ["distortion_center_x", "distortion_center_y"]
                                          if width >= 13 else [])
    col = {n: i for i, n in enumerate(names) if n}
    missing = [c for c in REQUIRED_COLUMNS if c not in col]
    if missing:
        raise ValueError(f"{path} lacks columns {missing}; header was {names}")
    has_k = all(c in col for c in INTRINSIC_COLUMNS)
    has_dc = "distortion_center_x" in col and "distortion_center_y" in col
    vals: list[list[float]] = []
    bad = 0
    for r in rows:
        if len(r) < len(names):
            r = r + [""] * (len(names) - len(r))
        req = [_float(r[col[c]]) for c in REQUIRED_COLUMNS]
        if not all(math.isfinite(v) for v in req):
            bad += 1
            continue
        k = [_float(r[col[c]]) for c in INTRINSIC_COLUMNS] if has_k else [math.nan] * 4
        dc = [_float(r[col["distortion_center_x"]]), _float(r[col["distortion_center_y"]])] if has_dc \
            else [math.nan] * 2
        vals.append(req + k + dc)
    if bad:
        flags.append(f"odometry_rows_skipped:{bad}")
    a = np.array(vals, float).reshape(-1, 15)
    q = a[:, 5:9]
    qn = np.linalg.norm(q, axis=1)
    ok = (qn > 0.5) & (qn < 1.5) & (a[:, 1] >= 0)
    if (~ok).any():
        flags.append(f"odometry_bad_rotation:{int((~ok).sum())}")
    a = a[ok]
    order = np.argsort(a[:, 1], kind="stable")
    a = a[order]
    fid = np.round(a[:, 1]).astype(np.int64)
    keep = np.r_[True, fid[1:] != fid[:-1]] if len(fid) else np.zeros(0, bool)
    if (~keep).any():
        flags.append(f"odometry_duplicate_frames:{int((~keep).sum())}")
    a, fid = a[keep], fid[keep]
    if len(fid) > 1 and (np.diff(fid) != 1).any():
        flags.append(f"odometry_frame_gaps:{int((np.diff(fid) != 1).sum())}")
    if len(fid) > 1 and (np.diff(a[:, 0]) <= 0).any():
        flags.append(f"timestamps_not_increasing:{int((np.diff(a[:, 0]) <= 0).sum())}")
    return {"frame": fid, "t": a[:, 0], "xyz": a[:, 2:5], "q": a[:, 5:9], "k": a[:, 9:13], "dc": a[:, 13:15],
            "has_k": np.array(has_k)}


def _read_camera_matrix(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    try:
        nums = [_float(x) for x in path.read_text().replace("\n", ",").split(",") if x.strip()]
    except OSError:
        return None
    if len(nums) < 9 or not all(math.isfinite(v) for v in nums[:9]):
        return None
    K = np.array(nums[:9], float).reshape(3, 3)
    return K if K[0, 0] > 0 and K[1, 1] > 0 else None


def _parse_imu(path: Path, flags: list[str]) -> tuple[np.ndarray | None, str | None]:
    try:
        names, rows, has_header = _read_csv_rows(path)
    except OSError:
        return None, None
    cols = ("timestamp", "a_x", "a_y", "a_z", "alpha_x", "alpha_y", "alpha_z")
    col = {n: i for i, n in enumerate(names)} if has_header else {c: i for i, c in enumerate(cols)}
    if not all(c in col for c in cols):
        flags.append("imu_columns_unknown")
        return None, None
    out = []
    for r in rows:
        if len(r) <= max(col[c] for c in cols):
            continue
        v = [_float(r[col[c]]) for c in cols]
        if all(math.isfinite(x) for x in v):
            out.append(v)
    if not out:
        return None, None
    imu = np.array(out, float)
    mag = float(np.median(np.linalg.norm(imu[:, 1:4], axis=1)))
    if 0.5 < mag < 2.0:
        imu[:, 1:4] *= G
        return imu, "g"
    if 5.0 < mag < 20.0:
        return imu, "m/s2"
    flags.append(f"imu_unit_unknown:{mag:.3g}")
    return imu, None


def _probe_video(path: Path, flags: list[str]) -> tuple[tuple[int, int] | None, int | None]:
    if not path.is_file():
        flags.append("rgb_missing")
        return None, None
    try:
        import av

        with av.open(str(path), options=VIDEO_OPTIONS) as container:
            s = container.streams.video[0]
            size = (int(s.codec_context.width or s.width), int(s.codec_context.height or s.height))
            return (size if size[0] > 0 and size[1] > 0 else None), (int(s.frames) or None)
    except Exception as exc:
        log.warning("cannot probe %s: %s", path, exc)
        flags.append(f"rgb_unreadable:{type(exc).__name__}")
        return None, None


def load_stray(root: str | Path) -> StrayCapture:
    """Parse a Stray Scanner dataset folder (or a folder holding exactly one) without decoding images."""
    root = find_dataset_root(Path(root))
    flags: list[str] = []
    od = _parse_odometry(root / "odometry.csv", flags)
    n = len(od["frame"])
    if n == 0:
        raise ValueError(f"{root / 'odometry.csv'} has no usable rows")

    T = np.tile(np.eye(4), (n, 1, 1))
    T[:, :3, :3] = Rotation.from_quat(od["q"]).as_matrix()  # scipy order x, y, z, w as in the file
    T[:, :3, 3] = od["xyz"]

    rgb_size, n_video = _probe_video(root / "rgb.mp4", flags)
    expected = int(od["frame"].max()) + 1
    if n_video is not None and n_video != expected:
        flags.append(f"rgb_frame_count_mismatch:{n_video}/{expected}")

    K = np.tile(np.eye(3), (n, 1, 1))
    k = od["k"]
    k_ok = np.isfinite(k).all(1) & (k[:, 0] > 0) & (k[:, 1] > 0)
    fmt = "v1.4" if bool(od["has_k"]) else "legacy"
    cam = _read_camera_matrix(root / "camera_matrix.csv")
    if k_ok.any():
        if not k_ok.all():
            flags.append(f"intrinsics_filled:{int((~k_ok).sum())}")
            idx = np.flatnonzero(k_ok)
            near = idx[np.clip(np.searchsorted(idx, np.arange(n)), 0, len(idx) - 1)]
            k = k[near]
        K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2] = k[:, 0], k[:, 1], k[:, 2], k[:, 3]
    elif cam is not None:
        flags.append("intrinsics_from_camera_matrix")
        K[:] = cam
    else:
        w, h = rgb_size or DEFAULT_RGB_SIZE
        flags.append("intrinsics_assumed")
        K[:, 0, 0] = K[:, 1, 1] = FALLBACK_FX_PER_WIDTH * w
        K[:, 0, 2], K[:, 1, 2] = (w - 1) / 2, (h - 1) / 2

    if rgb_size is None:
        # principal point near the image centre gives the size the intrinsics were made for
        cx, cy = float(np.median(K[:, 0, 2])), float(np.median(K[:, 1, 2]))
        rgb_size = DEFAULT_RGB_SIZE if abs(cx - 959.5) < 40 and abs(cy - 719.5) < 40 else \
            (round(2 * cx + 1), round(2 * cy + 1))
        flags.append(f"rgb_size_assumed:{rgb_size[0]}x{rgb_size[1]}")
    else:
        cx, cy = float(np.median(K[:, 0, 2])), float(np.median(K[:, 1, 2]))
        if abs(cx / rgb_size[0] - 0.5) > 0.1 or abs(cy / rgb_size[1] - 0.5) > 0.1:
            flags.append(f"intrinsics_rgb_size_mismatch:{cx:.0f},{cy:.0f}@{rgb_size[0]}x{rgb_size[1]}")

    depth_files = _files_by_number(root / "depth", (".png", ".npy"))
    conf_files = _files_by_number(root / "confidence", (".png",))
    has_depth = np.array([int(f) in depth_files for f in od["frame"]], bool)
    has_conf = np.array([int(f) in conf_files for f in od["frame"]], bool)
    if not has_depth.any():
        flags.append("depth_missing:all")
    elif not has_depth.all():
        flags.append(f"depth_missing:{int((~has_depth).sum())}/{n}")
    if not conf_files:
        flags.append("confidence_missing:all")
    elif not has_conf[has_depth].all():
        flags.append(f"confidence_missing:{int((~has_conf[has_depth]).sum())}/{int(has_depth.sum())}")
    extra = len(set(depth_files) - {int(f) for f in od["frame"]})
    if extra:
        flags.append(f"depth_without_pose:{extra}")

    imu, unit = (None, None)
    if (root / "imu.csv").is_file():
        imu, unit = _parse_imu(root / "imu.csv", flags)

    dc = od["dc"]
    cap = StrayCapture(root=root, frame_ids=od["frame"], timestamps=od["t"], T_wc=T, K=K, rgb_size=rgb_size,
                       depth_size=None, has_depth=has_depth, has_conf=has_conf, n_video_frames=n_video, imu=imu,
                       imu_accel_unit=unit, distortion_center=dc if np.isfinite(dc).any() else None, format=fmt,
                       flags=flags, _depth_files=depth_files, _conf_files=conf_files)
    first = np.flatnonzero(has_depth)
    for i in first[:5]:
        d = cap.read_depth(int(i))
        if d is not None:
            cap.depth_size = (int(d.shape[1]), int(d.shape[0]))
            break
    if cap.depth_size is None and has_depth.any():
        cap.depth_size = DEFAULT_DEPTH_SIZE
        flags.append("depth_unreadable")
    log.info("stray capture %s: %d frames over %.1fs (%.1f fps), rgb %s, depth %s, %s format%s", root.name, n,
             cap.duration_s, cap.fps, rgb_size, cap.depth_size, fmt, f", flags {flags}" if flags else "")
    return cap
