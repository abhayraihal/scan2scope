"""Tiny synthetic Stray Scanner datasets for tests (not the benchmark generator).

A 4.0 x 3.0 x 2.5 m box room with a door hole into a short corridor, seen from a loop of poses that ends where
it started. Depth is ray-cast analytically at 256x192, RGB is a small video with the frame number drawn as a
barcode, and the files mimic what the app writes (see scan2scope.ingest.stray for the sources): ", "-separated
CSVs with empty trailing distortion fields, 16-bit millimetre depth PNGs, 0/1/2 confidence PNGs, IMU in g, and
the first video frame at a presentation time of -1/60 s so the mp4 gets the same edit list as the app's files.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.spatial.transform import Rotation

ROOM = (4.0, 3.0, 2.5)  # x, y, z extent of the room, room frame z up, floor at z = 0
DOOR = (1.5, 2.4, 2.0)  # x0, x1 and height of the door hole in the wall y = 0
CORRIDOR = (0.5, 3.5, -1.2)  # x0, x1 and far wall y of the corridor behind the door
DEPTH_SIZE = (256, 192)
ROOM_YAW_DEG = 25.0
BARCODE_BITS = 12

# z-up scan2scope world from ARKit's y-up world (x, y, z) -> (x, -z, y)
C_ZUP = np.array([[1.0, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


@dataclass
class StrayFixture:
    root: Path
    frame_ids: np.ndarray
    timestamps: np.ndarray
    T_wc: np.ndarray  # (N, 4, 4) poses written to odometry.csv, ARKit world, OpenCV camera axes
    T_wc_true: np.ndarray  # (N, 4, 4) poses the depth was rendered from
    K_rgb: np.ndarray  # (N, 3, 3)
    K_depth: np.ndarray  # (N, 3, 3)
    rgb_size: tuple[int, int]
    depth_size: tuple[int, int]
    room_to_arkit: np.ndarray  # (4, 4) room frame (z up, room corner at the origin) -> ARKit world
    fps: float

    @property
    def room_to_zup(self) -> np.ndarray:
        """Room frame -> the z-up world produced by geometry.lidar."""
        return C_ZUP @ self.room_to_arkit


def _rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def look_rotation(yaw: float, pitch: float, roll: float = 0.0) -> np.ndarray:
    """Camera-to-world rotation in a z-up world with OpenCV camera axes (x right, y down, z forward)."""
    f = np.array([np.cos(pitch) * np.cos(yaw), np.cos(pitch) * np.sin(yaw), np.sin(pitch)])
    r = np.cross(f, [0.0, 0.0, 1.0])
    r /= np.linalg.norm(r)
    d = np.cross(f, r)
    R = np.stack([r, d, f], axis=1)
    return R @ Rotation.from_rotvec([0.0, 0.0, roll]).as_matrix()


def loop_poses(n: int, roll_deg: float = 0.0) -> np.ndarray:
    """Room-frame camera-to-world poses on an ellipse around the room centre, last pose equal to the first.

    roll_deg turns the phone about the optical axis (90 is portrait, the sensor image then lies sideways).
    """
    T = np.tile(np.eye(4), (n, 1, 1))
    cx, cy = ROOM[0] / 2, ROOM[1] / 2
    for k in range(n):
        s = k / (n - 1)
        phi = 2 * np.pi * s - np.pi / 2
        p = np.array([cx + 1.1 * np.cos(phi), cy + 0.65 * np.sin(phi), 1.4 + 0.02 * np.sin(3 * phi)])
        yaw = np.arctan2(cy - p[1], cx - p[0]) + np.radians(25.0) * np.sin(2 * phi)
        T[k, :3, :3] = look_rotation(yaw, np.radians(-18.0), np.radians(roll_deg + 2.0 * np.sin(phi)))
        T[k, :3, 3] = p
    return T


def depth_K(K_rgb: np.ndarray, rgb_size: tuple[int, int], depth_size: tuple[int, int]) -> np.ndarray:
    """Intrinsics at depth resolution, pixel centres at integer coordinates (types.CameraView convention)."""
    sx, sy = depth_size[0] / rgb_size[0], depth_size[1] / rgb_size[1]
    K = np.array(K_rgb, float).copy()
    K[..., 0, 0] *= sx
    K[..., 1, 1] *= sy
    K[..., 0, 2] = (K_rgb[..., 0, 2] + 0.5) * sx - 0.5
    K[..., 1, 2] = (K_rgb[..., 1, 2] + 0.5) * sy - 0.5
    return K


def _box_exit(o: np.ndarray, d: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distance along d to the exit face of an axis-aligned box containing o; face id = 2 * axis + (d > 0)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        t_hi = (hi - o) / d
        t_lo = (lo - o) / d
    t = np.where(d > 0, t_hi, np.where(d < 0, t_lo, np.inf))
    axis = np.argmin(t, axis=-1)
    tmin = np.take_along_axis(t, axis[..., None], -1)[..., 0]
    side = np.take_along_axis(d, axis[..., None], -1)[..., 0] > 0
    return tmin, 2 * axis + side


def render_depth(T_wc_room: np.ndarray, K_d: np.ndarray, size: tuple[int, int] = DEPTH_SIZE
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Analytic z-depth (h, w) of the room plus corridor, a surface id per pixel and |cos| of the incidence."""
    w, h = size
    u, v = np.meshgrid(np.arange(w, dtype=float), np.arange(h, dtype=float))
    rc = np.stack([(u - K_d[0, 2]) / K_d[0, 0], (v - K_d[1, 2]) / K_d[1, 1], np.ones_like(u)], -1)
    d = rc @ T_wc_room[:3, :3].T
    o = np.broadcast_to(T_wc_room[:3, 3], d.shape)
    t, face = _box_exit(o, d, np.zeros(3), np.array(ROOM))
    hit = o + t[..., None] * d
    through = (face == 2) & (hit[..., 0] > DOOR[0]) & (hit[..., 0] < DOOR[1]) & (hit[..., 2] < DOOR[2])
    if through.any():
        lo = np.array([CORRIDOR[0], CORRIDOR[2], 0.0])
        hi = np.array([CORRIDOR[1], 0.0, ROOM[2]])
        t2, face2 = _box_exit(hit[through], d[through], lo, hi)
        t[through] = t[through] + t2
        face[through] = 6 + face2
    normal_axis = np.where(face >= 6, face - 6, face) // 2
    dn = np.take_along_axis(d, normal_axis[..., None], -1)[..., 0]
    inc = np.abs(dn) / np.linalg.norm(d, axis=-1)
    return t, face, inc  # rc has z = 1, so the ray parameter is the camera z-depth


_FACE_COLOURS = np.array([
    [200, 60, 60], [60, 200, 60], [60, 60, 200], [200, 200, 60], [200, 60, 200], [60, 200, 200],
    [90, 90, 90], [150, 150, 150], [120, 80, 40], [40, 120, 80], [80, 40, 120], [180, 120, 60],
], np.uint8)


def draw_barcode(img: np.ndarray, value: int) -> None:
    h, w = img.shape[:2]
    bw, bh = w // (BARCODE_BITS + 4), h // 10
    for b in range(BARCODE_BITS):
        x0 = (b + 2) * bw
        img[bh // 2:bh // 2 + bh, x0:x0 + bw] = 255 if (value >> b) & 1 else 0


def read_barcode(img: np.ndarray) -> int:
    h, w = img.shape[:2]
    bw, bh = w // (BARCODE_BITS + 4), h // 10
    grey = img.mean(-1) if img.ndim == 3 else img
    value = 0
    for b in range(BARCODE_BITS):
        x0 = (b + 2) * bw
        if grey[bh // 2 + bh // 4:bh // 2 + 3 * bh // 4, x0 + bw // 4:x0 + 3 * bw // 4].mean() > 127:
            value |= 1 << b
    return value


def _fmt(v: float) -> str:
    """Swift's description of a Float: the shortest repr of the float32 value."""
    return repr(float(np.float32(v)))


def _write_video(path: Path, frames: list[np.ndarray], divider: int) -> None:
    out = av.open(str(path), "w")
    stream = None
    for codec in ("libx265", "libx264", "mpeg4"):
        try:
            stream = out.add_stream(codec, rate=Fraction(60, divider))
            break
        except Exception:  # noqa: BLE001, S112 - try the next encoder
            continue
    assert stream is not None, "no usable video encoder in PyAV"
    stream.width, stream.height = frames[0].shape[1], frames[0].shape[0]
    stream.pix_fmt = "yuv420p"
    stream.time_base = Fraction(1, 60)
    stream.codec_context.time_base = Fraction(1, 60)
    if stream.codec_context.name in ("libx265", "hevc"):
        stream.options = {"preset": "ultrafast", "x265-params": "log-level=none"}
    elif stream.codec_context.name in ("libx264", "h264"):
        stream.options = {"preset": "ultrafast"}
    for k, img in enumerate(frames):
        f = av.VideoFrame.from_ndarray(img, format="rgb24")
        f.pts = k * divider - 1  # VideoEncoder.swift: the first saved frame is appended at -1/60 s
        f.time_base = Fraction(1, 60)
        for pkt in stream.encode(f):
            out.mux(pkt)
    for pkt in stream.encode():
        out.mux(pkt)
    out.close()


def write_stray_dataset(root: Path, *, n_frames: int = 160, fps: float = 5.0, rgb_size: tuple[int, int] = (640, 480),
                        legacy: bool = False, seed: int = 0, noise: bool = True, imu_unit: str = "g",
                        t0: float = 1000.0, roll_deg: float = 0.0) -> StrayFixture:
    """Write a dataset folder at root and return the ground truth."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    (root / "depth").mkdir(parents=True, exist_ok=True)
    (root / "confidence").mkdir(exist_ok=True)

    T_room = loop_poses(n_frames, roll_deg)
    p0 = T_room[0, :3, 3]
    A = np.eye(4)
    A[:3, :3] = _rot_z(np.radians(ROOM_YAW_DEG))
    A[:3, 3] = -A[:3, :3] @ p0 + np.array([0.03, -0.02, 0.01])  # session origin near the first frame
    room_to_arkit = C_ZUP.T @ A
    T_wc = room_to_arkit @ T_room

    W, H = rgb_size
    K_rgb = np.tile(np.eye(3), (n_frames, 1, 1))
    k = np.arange(n_frames)
    K_rgb[:, 0, 0] = K_rgb[:, 1, 1] = 0.6975 * W + 0.15 * np.sin(k / 7.0)
    K_rgb[:, 0, 2] = W / 2 - 0.45 + 0.05 * np.sin(k / 5.0)
    K_rgb[:, 1, 2] = H / 2 - 0.3 + 0.05 * np.cos(k / 5.0)
    K_d = depth_K(K_rgb, rgb_size, DEPTH_SIZE)

    divider = max(1, round(60 / fps))
    timestamps = t0 + k * divider / 60.0 + rng.normal(0, 1e-5, n_frames)
    rgb_frames = []
    for i in range(n_frames):
        depth, face, inc = render_depth(T_room[i], K_d[i])
        conf = np.full(depth.shape, 2, np.uint8)
        conf[(inc < 0.35) | (depth > 3.5)] = 1
        conf[(inc < 0.12) | (depth > 5.0)] = 0
        jump = np.zeros(depth.shape, bool)
        jump[:, 1:] |= np.abs(np.diff(depth, axis=1)) > 0.1 * depth[:, 1:]
        jump[:, :-1] |= np.abs(np.diff(depth, axis=1)) > 0.1 * depth[:, :-1]
        jump[1:, :] |= np.abs(np.diff(depth, axis=0)) > 0.1 * depth[1:, :]
        jump[:-1, :] |= np.abs(np.diff(depth, axis=0)) > 0.1 * depth[:-1, :]
        conf[jump] = 0
        if noise:
            sigma = np.where(conf == 2, 0.002 + 0.002 * depth, 0.01 + 0.004 * depth)
            depth = depth + rng.normal(0, 1, depth.shape) * sigma
        mm = np.clip(np.round(depth * 1000.0), 0, 65535).astype(np.uint16)
        cv2.imwrite(str(root / "depth" / f"{i:06d}.png"), mm)
        cv2.imwrite(str(root / "confidence" / f"{i:06d}.png"), conf)
        small = _FACE_COLOURS[face % len(_FACE_COLOURS)]
        img = cv2.resize(small, (W, H), interpolation=cv2.INTER_NEAREST)
        draw_barcode(img, i)
        rgb_frames.append(img)
    _write_video(root / "rgb.mp4", rgb_frames, divider)

    _write_odometry(root / "odometry.csv", k, timestamps, T_wc, None if legacy else K_rgb)
    Kl = K_rgb[-1]
    (root / "camera_matrix.csv").write_text("\n".join(", ".join(_fmt(x) for x in row) for row in Kl))
    _write_imu(root / "imu.csv", timestamps, T_wc, rng, unit=imu_unit)
    return StrayFixture(root, k.copy(), timestamps, T_wc, T_wc.copy(), K_rgb, K_d, rgb_size, DEPTH_SIZE,
                        room_to_arkit, fps)


def _write_odometry(path: Path, frame_ids: np.ndarray, timestamps: np.ndarray, T_wc: np.ndarray,
                    K: np.ndarray | None) -> None:
    head = "timestamp, frame, x, y, z, qx, qy, qz, qw"
    lines = [head + (", fx, fy, cx, cy, distortion_center_x, distortion_center_y" if K is not None else "")]
    q = Rotation.from_matrix(T_wc[:, :3, :3]).as_quat()  # x, y, z, w
    for i, fid in enumerate(frame_ids):
        p = T_wc[i, :3, 3]
        vals = [repr(float(timestamps[i])), f"{int(fid):06d}"] + [_fmt(x) for x in p] + [_fmt(x) for x in q[i]]
        if K is not None:
            vals += [_fmt(K[i, 0, 0]), _fmt(K[i, 1, 1]), _fmt(K[i, 0, 2]), _fmt(K[i, 1, 2]), "", ""]
        lines.append(", ".join(vals))
    path.write_text("\n".join(lines) + "\n")


def _write_imu(path: Path, timestamps: np.ndarray, T_wc: np.ndarray, rng: np.random.Generator, unit: str) -> None:
    t = np.arange(timestamps[0] - 0.2, timestamps[-1] + 0.2, 0.01)
    idx = np.clip(np.searchsorted(timestamps, t), 0, len(timestamps) - 1)
    # gravity (ARKit -y) in the camera frame stands in for the device frame; magnitude is what tests check
    g = np.einsum("nji,j->ni", T_wc[idx, :3, :3], np.array([0.0, -1.0, 0.0]))
    a = g + rng.normal(0, 0.01, g.shape)
    if unit != "g":
        a = a * 9.81
    gyro = rng.normal(0, 0.02, g.shape)
    rows = ["timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z"]
    rows += [", ".join(repr(float(x)) for x in (t[i], *a[i], *gyro[i])) for i in range(len(t))]
    path.write_text("\n".join(rows) + "\n")


def drift_poses(T_true_zup: np.ndarray, timestamps: np.ndarray, yaw_deg: float,
                trans: tuple[float, float, float]) -> np.ndarray:
    """VIO-like drift in the z-up world: yaw error grows linearly in time and every true step is applied with
    the current yaw error, plus a constant translation drift rate reaching `trans` at the last frame."""
    s = (timestamps - timestamps[0]) / max(timestamps[-1] - timestamps[0], 1e-9)
    psi = np.radians(yaw_deg) * s
    out = T_true_zup.copy()
    for k in range(1, len(s)):
        step = T_true_zup[k, :3, 3] - T_true_zup[k - 1, :3, 3]
        out[k, :3, 3] = out[k - 1, :3, 3] + _rot_z(psi[k - 1]) @ step + np.asarray(trans) * (s[k] - s[k - 1])
        out[k, :3, :3] = _rot_z(psi[k]) @ T_true_zup[k, :3, :3]
    return out


def write_drifted_copy(src: StrayFixture, dst: Path, *, yaw_deg: float = 3.0,
                       trans: tuple[float, float, float] = (0.05, -0.04, 0.015)) -> StrayFixture:
    """Copy src with odometry that drifts in yaw and translation; depth and RGB stay rendered from the truth."""
    dst = Path(dst)
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src.root, dst)
    T_drift = C_ZUP.T @ drift_poses(C_ZUP @ src.T_wc_true, src.timestamps, yaw_deg, trans)
    has_k = "fx" in (src.root / "odometry.csv").read_text().splitlines()[0]
    _write_odometry(dst / "odometry.csv", src.frame_ids, src.timestamps, T_drift, src.K_rgb if has_k else None)
    return replace(src, root=dst, T_wc=T_drift)


_CACHE: dict[tuple, StrayFixture] = {}


def cached_dataset(tmp_path_factory, name: str, **kwargs) -> StrayFixture:
    """Write a dataset once per test session."""
    key = (name, tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = write_stray_dataset(tmp_path_factory.mktemp(name) / "a1b2c3d4e5", **kwargs)
    return _CACHE[key]


def cached_drifted(tmp_path_factory, base: StrayFixture, **kwargs) -> StrayFixture:
    key = ("drift", str(base.root), tuple(sorted(kwargs.items())))
    if key not in _CACHE:
        _CACHE[key] = write_drifted_copy(base, tmp_path_factory.mktemp("drifted") / "f6e5d4c3b2", **kwargs)
    return _CACHE[key]
