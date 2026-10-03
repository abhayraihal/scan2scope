"""Synthetic Stray Scanner 1.4 captures: walking route, ARKit-like poses with drift, LiDAR depth and confidence.

The files mirror the Stray Scanner 1.4 encoders (github.com/strayrobots/scanner, tag v1.4, StrayScanner/Helpers,
MIT licence):
- odometry.csv: one row per saved frame, ", " separated, Swift float printing, empty distortion columns (ARKit
  gives no lens calibration for the rear camera). The quaternion is ARKit's camera rotation times a 180 degree
  rotation about x, i.e. OpenCV camera axes, camera-to-world in ARKit's gravity-aligned y-up world.
- depth/NNNNNN.png: 16-bit grey, millimetres, round(metres * 1000), 0 where there is no return.
- confidence/NNNNNN.png: 8-bit grey with values 0, 1, 2.
- camera_matrix.csv: intrinsics of the last frame, three rows, no trailing newline.
- imu.csv: raw CoreMotion samples, acceleration in g including gravity, rotation rate in rad/s, device axes.
- rgb.mp4: HEVC on a 1/60 s timescale.
"""

from __future__ import annotations

import logging
import os
import shutil
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import av
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d, maximum_filter, minimum_filter
from scipy.spatial.transform import Rotation

from scan2scope.synth.apartment import Apartment, SynthRoom, arm_path, free_mask, loop_shape
from scan2scope.synth.render import RenderScene, render_depth, render_rgb

log = logging.getLogger("scan2scope.synth")

ODOMETRY_HEADER = ("timestamp, frame, x, y, z, qx, qy, qz, qw, fx, fy, cx, cy, distortion_center_x, "
                   "distortion_center_y\n")
IMU_HEADER = "timestamp, a_x, a_y, a_z, alpha_x, alpha_y, alpha_z\n"
RGB_SIZE = (960, 720)
DEPTH_SIZE = (256, 192)
IMU_RATE = 100.0
G0 = 9.80665
# OpenCV camera axes expressed in CoreMotion device axes (x right, y top, z out of the screen, portrait).
R_DEV_CV = np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
DRIFT_MODES = ("none", "normal", "strong")

HOLD, CORRIDOR, DOOR, LOOP, TURN, PAN = range(6)
_PAN_AMP = np.radians([0.0, 22.0, 3.0, 14.0, 0.0, 0.0])
_PITCH_BASE = np.radians([-14.0, -8.0, -6.0, -4.0, -8.0, -14.0])
_PITCH_AMP = np.radians([0.0, 14.0, 4.0, 31.0, 6.0, 4.0])


# --------------------------------------------------------------------------------------------- formatting


def swift_float(x: float, double: bool = False) -> str:
    """Format like Swift's Float/Double description: shortest round-trip digits, exponent below 1e-4."""
    v = np.float64(x) if double else np.float32(x)
    if not np.isfinite(v):
        return "nan" if np.isnan(v) else ("inf" if v > 0 else "-inf")
    if v == 0:
        return "-0.0" if np.signbit(v) else "0.0"
    s = np.format_float_scientific(v, unique=True, trim="-", exp_digits=2)
    mant, exp = s.split("e")
    e = int(exp)
    neg = mant.startswith("-")
    digits = mant.lstrip("-").replace(".", "")
    if e < -4 or abs(float(v)) > (2.0 ** 53 if double else 2.0 ** 24):
        m = digits[0] + ("." + digits[1:] if len(digits) > 1 else "")
        out = f"{m}e{'-' if e < 0 else '+'}{abs(e):02d}"
    elif e >= 0:
        out = digits[:e + 1].ljust(e + 1, "0") + "." + (digits[e + 1:] or "0")
    else:
        out = "0." + "0" * (-e - 1) + digits
    return ("-" if neg else "") + out


# --------------------------------------------------------------------------------------------- trajectory


@dataclass
class Trajectory:
    """Camera centre and viewing angles sampled at `rate` Hz; t = 0 is the first recorded frame."""

    t: np.ndarray
    pos: np.ndarray  # (M, 3) property frame
    yaw: np.ndarray  # heading of the optical axis, radians from +x towards +y
    pitch: np.ndarray  # optical axis elevation, radians, up is positive
    roll: np.ndarray
    mode: np.ndarray
    orientation: str  # portrait | landscape (how the phone is held)
    rate: float = IMU_RATE
    flags: list[str] = field(default_factory=list)

    @property
    def duration(self) -> float:
        return float(self.t[-1])

    def at(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Camera centres (N, 3) and OpenCV camera-to-property rotations (N, 3, 3) at the given times."""
        times = np.asarray(times, float)
        pos = np.stack([np.interp(times, self.t, self.pos[:, i]) for i in range(3)], 1)
        ang = [np.interp(times, self.t, a) for a in (self.yaw, self.pitch, self.roll)]
        return pos, camera_rotations(*ang, orientation=self.orientation)

    def rotations(self) -> np.ndarray:
        return camera_rotations(self.yaw, self.pitch, self.roll, orientation=self.orientation)


def camera_rotations(yaw: np.ndarray, pitch: np.ndarray, roll: np.ndarray, *, orientation: str) -> np.ndarray:
    """OpenCV camera-to-property rotations. Portrait: image x points down in the world, as on a held iPhone."""
    yaw, pitch, roll = (np.atleast_1d(np.asarray(a, float)) for a in (yaw, pitch, roll))
    f = np.stack([np.cos(yaw) * np.cos(pitch), np.sin(yaw) * np.cos(pitch), np.sin(pitch)], -1)
    r = np.stack([np.sin(yaw), -np.cos(yaw), np.zeros_like(yaw)], -1)
    dn = np.cross(f, r)
    c, s = np.cos(roll)[:, None], np.sin(roll)[:, None]
    r, dn = r * c + dn * s, dn * c - r * s
    cols = (dn, -r, f) if orientation == "portrait" else (r, dn, f)
    return np.stack(cols, -1)


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


class _Path:
    """Dense walking path: positions, yaw, mode and the time step to reach each node."""

    def __init__(self, x: float, y: float, yaw: float) -> None:
        self.x, self.y, self.yaw, self.mode, self.dt = [x], [y], [yaw], [HOLD], [0.0]

    @property
    def pos(self) -> np.ndarray:
        return np.array([self.x[-1], self.y[-1]])

    def _push(self, x: float, y: float, yaw: float, mode: int, dt: float) -> None:
        self.x.append(float(x))
        self.y.append(float(y))
        self.yaw.append(float(yaw))
        self.mode.append(mode)
        self.dt.append(float(dt))

    def hold(self, seconds: float, mode: int = HOLD) -> None:
        n = max(1, round(seconds / 0.05))
        for _ in range(n):
            self._push(self.x[-1], self.y[-1], self.yaw[-1], mode, seconds / n)

    def turn_to(self, target: float, rate: float, mode: int = TURN) -> None:
        y0 = self.yaw[-1]
        delta = _wrap(target - y0)
        n = int(np.ceil(abs(delta) / np.radians(2.0)))
        for i in range(1, n + 1):
            self._push(self.x[-1], self.y[-1], y0 + delta * i / n, mode, abs(delta) / n / rate)

    def line_to(self, x: float, y: float, speed: float, rate: float, mode: int, yaw: float | None = None) -> None:
        """Walk straight to (x, y), facing the walking direction unless `yaw` is given."""
        p0, p1 = self.pos, np.array([x, y], float)
        length = float(np.linalg.norm(p1 - p0))
        if length < 1e-4:
            return
        if yaw is None:
            yaw = float(np.arctan2(*(p1 - p0)[::-1]))
            if abs(_wrap(yaw - self.yaw[-1])) > np.radians(3):
                self.turn_to(yaw, rate, TURN)
        y0 = self.yaw[-1]
        dy = _wrap(yaw - y0)
        n = int(np.ceil(length / 0.02))
        for i in range(1, n + 1):
            p = p0 + (p1 - p0) * i / n
            self._push(p[0], p[1], y0 + dy * i / n, mode, max(length / n / speed, abs(dy) / n / rate))

    def follow(self, pts: np.ndarray, yaws: np.ndarray, speed: float, rate: float, mode: int) -> None:
        """Walk through points (K, 2) with continuous yaw targets (K,), turning on the spot to yaws[0] first."""
        self.line_to(pts[0, 0], pts[0, 1], speed, rate, mode, yaw=self.yaw[-1])
        self.turn_to(yaws[0], rate, TURN)
        # turn_to may land on yaws[0] +- 2 pi (a half turn can go either way); continue from where it landed
        yaws = yaws + 2 * np.pi * np.round((self.yaw[-1] - yaws[0]) / (2 * np.pi))
        for i in range(1, len(pts)):
            ds = float(np.linalg.norm(pts[i] - pts[i - 1]))
            dyaw = float(yaws[i] - yaws[i - 1])
            self._push(pts[i, 0], pts[i, 1], yaws[i], mode, max(ds / speed, abs(dyaw) / rate, 1e-3))


def _loop_ring(room: SynthRoom, inset: float, step: float = 0.02) -> tuple[np.ndarray, np.ndarray] | None:
    """Counter-clockwise loop points at `inset` from the walls and their outward (wall-facing) yaw."""
    poly = loop_shape(room.shape_m(), inset)
    if poly is None:
        return None
    ring = np.asarray(poly.exterior.coords)[:-1]
    area2 = np.sum(ring[:, 0] * np.roll(ring[:, 1], -1) - np.roll(ring[:, 0], -1) * ring[:, 1])
    if area2 < 0:
        ring = ring[::-1]
    closed = np.vstack([ring, ring[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    si = np.arange(0.0, s[-1], step)
    pts = np.stack([np.interp(si, s, closed[:, 0]), np.interp(si, s, closed[:, 1])], 1)
    tan = np.roll(pts, -1, axis=0) - np.roll(pts, 1, axis=0)
    tan /= np.maximum(np.linalg.norm(tan, axis=1, keepdims=True), 1e-9)
    normal = np.stack([tan[:, 1], -tan[:, 0]], 1)
    return pts, np.arctan2(normal[:, 1], normal[:, 0])


def _smooth_noise(rng: np.random.Generator, shape: tuple[int, ...], sigma: float) -> np.ndarray:
    """Gaussian-smoothed white noise along axis 0, scaled to unit variance for long sequences."""
    n = rng.standard_normal(shape)
    if sigma <= 0:
        return n
    return gaussian_filter1d(n, sigma, axis=0, mode="wrap") * np.sqrt(2.0 * np.sqrt(np.pi) * sigma)


def plan_trajectory(apt: Apartment, rng: np.random.Generator, *, orientation: str = "portrait",
                    rate: float = IMU_RATE) -> Trajectory:
    """Walk from the hallway entrance through every room's door, loop each room facing the walls, come back.

    Inside each room the camera walks a rounded loop about 1.1 to 1.3 m from the walls (less in small rooms),
    facing the walls with yaw panning and pitch sweeps so the floor and ceiling junctions are seen; L-shaped
    rooms get a walk into the arm. The capture ends on the starting view.
    """
    if orientation not in ("portrait", "landscape"):
        raise ValueError(f"orientation must be portrait or landscape, got {orientation!r}")
    v_cor, v_door, v_loop = rng.uniform(0.50, 0.62), rng.uniform(0.36, 0.45), rng.uniform(0.30, 0.38)
    w_turn, w_walk = np.radians(rng.uniform(45, 60)), np.radians(rng.uniform(35, 45))
    inset = rng.uniform(1.1, 1.3)
    hall = apt.hallway.rects[0]
    hx0, hy0, hx1, hy1 = hall.x0 / 1000, hall.y0 / 1000, hall.x1 / 1000, hall.y1 / 1000
    y_cc = 0.5 * (hy0 + hy1)
    x_s = hx0 + min(rng.uniform(1.3, 1.8), 0.45 * (hx1 - hx0))
    y_s = y_cc + rng.uniform(-0.05, 0.05)
    yaw_s = np.pi + rng.uniform(-0.1, 0.1)
    pan = rng.uniform(0.45, 0.7)
    path = _Path(x_s, y_s, yaw_s)
    path.hold(rng.uniform(1.5, 2.0))
    for target in (yaw_s + pan, yaw_s - pan, yaw_s):
        path.turn_to(target, 0.6 * w_turn, PAN)
    flags: list[str] = []
    for room in apt.rooms[1:]:
        _visit_room(path, apt, room, rng, inset, y_cc, (v_cor, v_door, v_loop), (w_turn, w_walk), flags)
    path.line_to(x_s, y_s, v_cor, w_walk, CORRIDOR)
    path.turn_to(yaw_s, w_turn, TURN)
    for target in (yaw_s - pan, yaw_s + pan, yaw_s):
        path.turn_to(target, 0.6 * w_turn, PAN)
    path.hold(rng.uniform(2.0, 2.5))

    t_nodes = np.cumsum(path.dt)
    t = np.arange(0.0, t_nodes[-1] + 1e-9, 1.0 / rate)
    x = np.interp(t, t_nodes, path.x)
    y = np.interp(t, t_nodes, path.y)
    yaw = np.interp(t, t_nodes, path.yaw)
    mode = np.asarray(path.mode)[np.clip(np.searchsorted(t_nodes, t, side="right") - 1, 0, len(t_nodes) - 1)]
    sig = 0.25 * rate
    x, y, yaw = (gaussian_filter1d(a, sig, mode="nearest") for a in (x, y, yaw))
    pan_amp, p_base, p_amp = (gaussian_filter1d(tab[mode], 0.8 * rate, mode="nearest")
                              for tab in (_PAN_AMP, _PITCH_BASE, _PITCH_AMP))
    speed = np.hypot(np.gradient(x), np.gradient(y)) * rate
    walking = np.clip(speed / 0.3, 0.0, 1.0)
    activity = gaussian_filter1d((mode != HOLD).astype(float), 0.8 * rate, mode="nearest")
    n = len(t)
    yaw = yaw + pan_amp * np.sin(2 * np.pi * t / rng.uniform(5.0, 8.0) + rng.uniform(0, 2 * np.pi))
    pitch = p_base + p_amp * np.sin(2 * np.pi * t / rng.uniform(4.0, 5.5) + rng.uniform(0, 2 * np.pi))
    roll = np.radians(1.5) * _smooth_noise(rng, (n,), 1.0 * rate) * (0.3 + 0.7 * activity)
    f_step = rng.uniform(1.6, 2.0)
    z = (rng.uniform(1.35, 1.45) + 0.012 * walking * np.sin(2 * np.pi * f_step * t)
         + 0.015 * _smooth_noise(rng, (n,), 2.0 * rate) * (0.3 + 0.7 * activity))
    sway = 0.008 * _smooth_noise(rng, (n, 2), 1.0 * rate) * activity[:, None]
    pos = np.stack([x + sway[:, 0], y + sway[:, 1], z], 1)
    traj = Trajectory(t=t, pos=pos, yaw=yaw, pitch=pitch, roll=roll, mode=mode, orientation=orientation, rate=rate,
                      flags=flags)
    bad = ~free_mask(apt, pos[::5, :2], margin=0.05)
    if bad.any():
        log.warning("trajectory leaves free space at %d of %d samples", int(bad.sum()), len(bad))
        traj.flags.append(f"trajectory_collision:{int(bad.sum())}")
    return traj


def _visit_room(path: _Path, apt: Apartment, room: SynthRoom, rng: np.random.Generator, inset: float,
                y_cc: float, speeds: tuple[float, float, float], rates: tuple[float, float], flags: list[str]) -> None:
    v_cor, v_door, v_loop = speeds
    w_turn, w_walk = rates
    o = apt.openings[room.entry]
    ri = room.index
    sgn = o.into(ri)
    s_c = 0.5 * (o.s0 + o.s1) / 1000.0
    mid = 0.5 * (o.n0 + o.n1) / 1000.0
    face = o.face(ri) / 1000.0
    hall_face = o.face(o.other(ri)) / 1000.0
    r = min(inset, room.max_inset)
    if o.axis == 1:
        corridor, door, inside = (s_c, y_cc), (s_c, mid), (s_c, face + sgn * r)
        yaw_in = np.pi / 2 if sgn > 0 else -np.pi / 2
    else:
        corridor, door, inside = (hall_face - sgn * 0.75, s_c), (mid, s_c), (face + sgn * r, s_c)
        yaw_in = 0.0 if sgn > 0 else np.pi
    path.line_to(*corridor, v_cor, w_walk, CORRIDOR)
    path.turn_to(yaw_in, w_turn, TURN)
    path.hold(rng.uniform(1.0, 1.5), DOOR)
    path.line_to(*door, v_door, w_walk, DOOR, yaw=yaw_in)
    path.line_to(*inside, v_door, w_walk, DOOR, yaw=yaw_in)
    ring = _loop_ring(room, r)
    if ring is not None:
        pts, yaw_n = ring
        j = int(np.argmin(np.linalg.norm(pts - np.asarray(inside), axis=1)))
        direction = 1 if rng.random() < 0.5 else -1
        idx = (j + direction * np.arange(len(pts) + 1)) % len(pts)
        pts, yaws = pts[idx], np.unwrap(yaw_n[idx])
        arm = arm_path(room)
        if arm is None:
            path.follow(pts, yaws, v_loop, w_walk, LOOP)
        else:
            ja = int(np.argmin(np.linalg.norm(pts - arm[0], axis=1)))
            path.follow(pts[:ja + 1], yaws[:ja + 1], v_loop, w_walk, LOOP)
            back = path.pos.copy()
            path.line_to(*arm[0], v_door, w_walk, CORRIDOR)
            path.line_to(*arm[1], v_door, w_walk, CORRIDOR)
            path.line_to(*arm[0], v_door, w_walk, CORRIDOR)
            path.line_to(*back, v_door, w_walk, CORRIDOR)
            path.follow(pts[ja:], yaws[ja:], v_loop, w_walk, LOOP)
    else:
        path.turn_to(yaw_in + np.pi, w_turn, TURN)
        flags.append(f"no_loop:{room.name}")
    path.line_to(*inside, v_door, w_walk, DOOR)
    path.line_to(*door, v_door, w_walk, DOOR)
    path.line_to(*corridor, v_door, w_walk, DOOR)


# --------------------------------------------------------------------------------------------- poses and drift


def arkit_world(rng: np.random.Generator, p0: np.ndarray, R0: np.ndarray) -> np.ndarray:
    """4x4 transform from the property frame to an ARKit world (y up, -z = initial heading, origin near p0).

    The AR session starts a moment before recording, so the origin and heading are offset from the first frame.
    """
    fwd = R0[:, 2]
    ang = np.arctan2(fwd[1], fwd[0]) + rng.uniform(-0.35, 0.35)
    h = np.array([np.cos(ang), np.sin(ang), 0.0])
    origin = p0 + np.array([rng.uniform(-0.25, 0.25), rng.uniform(-0.25, 0.25), rng.uniform(-0.05, 0.05)])
    y_w = np.array([0.0, 0.0, 1.0])
    z_w = -h
    x_w = np.cross(y_w, z_w)
    T = np.eye(4)
    T[:3, :3] = np.stack([x_w, y_w, z_w])
    T[:3, 3] = -T[:3, :3] @ origin
    return T


@dataclass
class Drift:
    mode: str
    yaw: np.ndarray  # (N,) radians, rotation about gravity applied to each camera
    trans: np.ndarray  # (N, 3) metres, property frame

    def summary(self) -> dict:
        return {"mode": self.mode, "final_yaw_deg": round(float(np.degrees(self.yaw[-1])), 4),
                "final_translation_m": round(float(np.linalg.norm(self.trans[-1])), 5),
                "max_yaw_deg": round(float(np.degrees(np.abs(self.yaw).max())), 4),
                "max_translation_m": round(float(np.linalg.norm(self.trans, axis=1).max()), 5)}


def sample_drift(rng: np.random.Generator, t: np.ndarray, mode: str) -> Drift:
    """VIO-like drift: a yaw random walk and a translation random walk, each pinned to a final magnitude.

    normal: 1 to 3 degrees for a 3-minute capture (scaled by sqrt(duration / 3 min), as a random walk grows) and
    1 to 2 cm per minute; strong: 4 degrees and 10 cm by the end. Each frame's rotation turns about gravity
    around its own camera centre; the translation is added to the camera centre.
    """
    if mode not in DRIFT_MODES:
        raise ValueError(f"drift must be one of {DRIFT_MODES}, got {mode!r}")
    n = len(t)
    if mode == "none" or n < 2:
        return Drift(mode, np.zeros(n), np.zeros((n, 3)))
    T = max(float(t[-1] - t[0]), 1e-6)
    u = (t - t[0]) / T

    def walk() -> np.ndarray:
        w = np.concatenate([[0.0], np.cumsum(rng.standard_normal(n - 1))]) / np.sqrt(n)
        return w - u * w[-1]

    if mode == "normal":
        yaw_end = np.radians(rng.uniform(1.0, 3.0)) * np.sqrt(T / 180.0)
        trans_end = rng.uniform(0.01, 0.02) * T / 60.0
    else:
        yaw_end, trans_end = np.radians(4.0), 0.10
    yaw = yaw_end * (rng.choice([-1.0, 1.0]) * u + 0.3 * walk())
    ang = rng.uniform(0, 2 * np.pi)
    d = np.array([np.cos(ang), np.sin(ang), 0.0])
    side = np.array([-np.sin(ang), np.cos(ang), 0.0])
    trans = trans_end * ((u + 0.3 * walk())[:, None] * d + 0.25 * walk()[:, None] * side
                         + 0.15 * walk()[:, None] * np.array([0.0, 0.0, 1.0]))
    trans *= trans_end / max(float(np.linalg.norm(trans[-1])), 1e-9)
    return Drift(mode, yaw, trans)


def _rot_z(a: np.ndarray) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    R = np.zeros((len(a), 3, 3))
    R[:, 0, 0], R[:, 0, 1], R[:, 1, 0], R[:, 1, 1], R[:, 2, 2] = c, -s, s, c, 1.0
    return R


def odometry_poses(pos: np.ndarray, R: np.ndarray, drift: Drift, T_wp: np.ndarray,
                   jitter: tuple[np.ndarray, np.ndarray] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Written (estimated) poses in the ARKit world: positions (N, 3) and quaternions (N, 4) x, y, z, w."""
    R_est = _rot_z(drift.yaw) @ R
    p_est = pos + drift.trans
    if jitter is not None:
        R_est = Rotation.from_rotvec(jitter[1]).as_matrix() @ R_est
        p_est = p_est + jitter[0]
    R_w = T_wp[:3, :3] @ R_est
    p_w = p_est @ T_wp[:3, :3].T + T_wp[:3, 3]
    # Stray writes q_WA * q_AC with q_AC = 180 degrees about x, which equals the OpenCV camera rotation.
    q_wa = Rotation.from_matrix(R_w @ np.diag([1.0, -1.0, -1.0])).as_quat()
    x1, y1, z1, w1 = q_wa.T
    return p_w, np.stack([w1, z1, -y1, -x1], 1)


# --------------------------------------------------------------------------------------------- sensors


def lidar_measurement(depth: np.ndarray, cos_inc: np.ndarray, rng: np.random.Generator | None, *,
                      max_range: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """ARKit-like depth (uint16 mm) and confidence (uint8 0/1/2) from a noise-free z-depth render.

    Noise sigma is 0.004 + 0.006 * depth (a quarter of the variance white, the rest spatially smooth, as ARKit
    depth is a densified, filtered map); depth edges get mixed pixels,
    grazing angles drop out, nothing returns beyond max_range. Confidence is 2 under 3 m at incidence under
    60 degrees, 1 under 4.5 m, else 0. rng=None gives the noise-free measurement with the same confidence rule.
    """
    z = np.asarray(depth, np.float64)
    valid = np.isfinite(z) & (z > 0) & (z <= max_range)
    zt = np.where(valid, z, 0.0)
    theta = np.degrees(np.arccos(np.clip(cos_inc, 0.0, 1.0)))
    conf = np.where((zt < 3.0) & (theta < 60.0), 2, np.where(zt < 4.5, 1, 0)).astype(np.uint8)
    zm = zt.copy()
    if rng is not None:
        sigma = 0.004 + 0.006 * zt
        smooth = gaussian_filter(rng.standard_normal(z.shape), 4.0)
        smooth /= max(float(smooth.std()), 1e-9)
        zm = zt + sigma * (0.5 * rng.standard_normal(z.shape) + np.sqrt(0.75) * smooth)
        big = 1e6
        zmin = minimum_filter(np.where(valid, zt, big), 3)
        zmax = maximum_filter(np.where(valid, zt, -big), 3)
        edge = valid & (zmin < big) & (zmax > -big) & (zmax - zmin > 0.05 + 0.03 * zt)
        mix = edge & (rng.random(z.shape) < 0.5)
        zm[mix] = zmin[mix] + rng.random(int(mix.sum())) * (zmax[mix] - zmin[mix])
        lower = mix & (rng.random(z.shape) < 0.8)
        conf[lower] = np.minimum(conf[lower], 1)
        p_drop = 0.9 * np.clip((theta - 70.0) / 15.0, 0.0, 1.0)
        valid &= rng.random(z.shape) >= p_drop
        valid &= zm <= max_range
    conf[~valid] = 0
    mm = np.where(valid, np.round(zm * 1000.0), 0.0)
    return np.clip(mm, 0, 65535).astype(np.uint16), conf


def imu_samples(traj: Trajectory, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw accelerometer (g, gravity included) and gyroscope (rad/s) in device axes at the trajectory rate."""
    dt = 1.0 / traj.rate
    R_pd = traj.rotations() @ R_DEV_CV.T
    acc_p = np.stack([np.gradient(np.gradient(traj.pos[:, i], dt), dt) for i in range(3)], 1)
    acc_p = gaussian_filter1d(acc_p, 2.0, axis=0, mode="nearest")
    g = np.array([0.0, 0.0, -G0])
    acc = np.einsum("nji,nj->ni", R_pd, g - acc_p) / G0
    rel = Rotation.from_matrix(R_pd[:-1]).inv() * Rotation.from_matrix(R_pd[1:])
    gyro = rel.as_rotvec() / dt
    gyro = np.vstack([gyro, gyro[-1:]])
    if rng is not None:
        acc = acc + rng.uniform(-0.01, 0.01, 3) + rng.normal(0.0, 0.003, acc.shape)
        gyro = gyro + rng.uniform(-0.005, 0.005, 3) + rng.normal(0.0, 0.003, gyro.shape)
    return traj.t.copy(), acc, gyro


# --------------------------------------------------------------------------------------------- writers


class VideoWriter:
    """rgb.mp4 through PyAV: HEVC (hvc1) when available, else H.264, on a 1/60 s timescale like Stray Scanner."""

    def __init__(self, path: Path, size: tuple[int, int], fps: float, codec: str | None = None,
                 crf: int = 26) -> None:
        ticks = 60 if abs(60.0 / fps - round(60.0 / fps)) < 1e-9 else 600
        self.time_base = Fraction(1, ticks)
        self.ticks_per_frame = ticks / fps
        self.container = av.open(str(path), mode="w", options={"video_track_timescale": str(ticks)})
        self.flags: list[str] = []
        names = [codec] if codec else ["libx265", "libx264", "mpeg4"]
        self.stream = None
        for name in names:
            try:
                self.stream = self.container.add_stream(name, rate=Fraction(fps).limit_denominator(1000))
                self.codec = name
                break
            except (ValueError, av.FFmpegError) as exc:  # encoder missing from this FFmpeg build
                log.debug("encoder %s unavailable: %s", name, exc)
        if self.stream is None:
            raise RuntimeError(f"no video encoder available from {names}")
        if self.codec != names[0]:
            self.flags.append(f"video_codec_fallback:{self.codec}")
        self.stream.width, self.stream.height = size
        self.stream.pix_fmt = "yuv420p"
        self.stream.time_base = self.time_base
        if self.codec == "libx265":
            self.stream.codec_tag = "hvc1"
            self.stream.options = {"crf": str(crf), "preset": "fast", "x265-params": "log-level=error"}
        elif self.codec == "libx264":
            self.stream.options = {"crf": str(crf - 4), "preset": "fast"}

    def write(self, rgb: np.ndarray, slot: int) -> None:
        frame = av.VideoFrame.from_ndarray(np.ascontiguousarray(rgb), format="rgb24")
        frame.pts = round(slot * self.ticks_per_frame)
        frame.time_base = self.time_base
        for packet in self.stream.encode(frame):
            self.container.mux(packet)

    def close(self) -> None:
        for packet in self.stream.encode():
            self.container.mux(packet)
        self.container.close()


def _odometry_text(timestamps: np.ndarray, p_w: np.ndarray, q: np.ndarray, K: np.ndarray) -> str:
    lines = [ODOMETRY_HEADER]
    f = swift_float
    for i in range(len(timestamps)):
        x, y, z = p_w[i]
        qx, qy, qz, qw = q[i]
        k = K[i]
        lines.append(f"{f(timestamps[i], True)}, {i:06d}, {f(x)}, {f(y)}, {f(z)}, {f(qx)}, {f(qy)}, {f(qz)}, "
                     f"{f(qw)}, {f(k[0, 0])}, {f(k[1, 1])}, {f(k[0, 2])}, {f(k[1, 2])}, , \n")
    return "".join(lines)


def _camera_matrix_text(K: np.ndarray) -> str:
    return "\n".join(", ".join(swift_float(v) for v in row) for row in K)


def _imu_text(timestamps: np.ndarray, acc: np.ndarray, gyro: np.ndarray) -> str:
    f = swift_float
    rows = [IMU_HEADER]
    for t, a, w in zip(timestamps, acc, gyro):
        rows.append(f"{f(t, True)}, {f(a[0], True)}, {f(a[1], True)}, {f(a[2], True)}, "
                    f"{f(w[0], True)}, {f(w[1], True)}, {f(w[2], True)}\n")
    return "".join(rows)


# --------------------------------------------------------------------------------------------- capture


@dataclass
class CaptureConfig:
    fps: float = 10.0
    rgb_size: tuple[int, int] = RGB_SIZE
    depth_size: tuple[int, int] = DEPTH_SIZE
    orientation: str = "portrait"
    drift: str = "normal"  # none | normal | strong
    noise: bool = True  # depth noise and artefacts, pose jitter, IMU noise, dropped frames
    rgb_scale: float = 0.5  # RGB is rendered at this fraction of rgb_size and resized up
    max_frames: int | None = None
    max_range: float = 5.0
    png_compression: int = 6
    video_codec: str | None = None
    drop_rate: float = 0.003  # frames the encoder drops when it falls behind (timestamp gaps)


@dataclass
class CaptureResult:
    path: Path
    config: CaptureConfig
    frame_times: np.ndarray  # (N,) seconds since the first frame
    timestamps: np.ndarray  # (N,) as written (device uptime)
    slots: np.ndarray  # (N,) frame slot on the fps grid (gaps where frames were dropped)
    true_pos: np.ndarray  # (N, 3) property frame
    true_R: np.ndarray  # (N, 3, 3) OpenCV camera-to-property
    K: np.ndarray  # (N, 3, 3) at rgb_size
    T_arkit_property: np.ndarray  # (4, 4)
    drift: Drift
    trajectory: Trajectory
    jitter: tuple[np.ndarray, np.ndarray] | None
    flags: list[str] = field(default_factory=list)
    seconds: float = 0.0

    @property
    def n_frames(self) -> int:
        return len(self.frame_times)


def _intrinsics(rng: np.random.Generator, n: int, size: tuple[int, int], noise: bool) -> np.ndarray:
    W, H = size
    s = W / 960.0
    fx = rng.uniform(670.0, 810.0) * s
    fy = fx * (1.0 + rng.uniform(-0.002, 0.002))
    cx = (W - 1) / 2.0 + rng.uniform(-4.0, 4.0) * s
    cy = (H - 1) / 2.0 + rng.uniform(-4.0, 4.0) * s
    breathe = 1.0 + (0.0004 * _smooth_noise(rng, (n,), 30.0) if noise and n > 1 else np.zeros(n))
    K = np.zeros((n, 3, 3))
    K[:, 0, 0], K[:, 1, 1] = fx * breathe, fy * breathe
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = cx, cy, 1.0
    return K


def write_capture(apt: Apartment, out_dir: str | Path, *, seed: int, config: CaptureConfig | None = None,
                  scene: RenderScene | None = None) -> CaptureResult:
    """Render one capture of `apt` into `out_dir` in the Stray Scanner layout."""
    cfg = config or CaptureConfig()
    out = Path(out_dir)
    t_start = time.perf_counter()
    rng_traj, rng_noise, rng_drift, rng_misc, rng_imu = (np.random.default_rng(s)
                                                         for s in np.random.SeedSequence(seed).spawn(5))
    traj = plan_trajectory(apt, rng_traj, orientation=cfg.orientation)
    period = 1.0 / cfg.fps
    slots, s = [], 0
    while s * period <= traj.duration + 1e-9:
        slots.append(s)
        s += 2 if (cfg.noise and rng_misc.random() < cfg.drop_rate) else 1
    slots = np.asarray(slots[:cfg.max_frames] if cfg.max_frames else slots)
    frame_t = slots * period
    n = len(frame_t)
    pos, R = traj.at(frame_t)
    K = _intrinsics(rng_misc, n, cfg.rgb_size, cfg.noise)
    T_wp = arkit_world(rng_misc, pos[0], R[0])
    drift = sample_drift(rng_drift, frame_t, cfg.drift)
    jitter = None
    if cfg.noise:
        jitter = (0.0015 * _smooth_noise(rng_misc, (n, 3), 3.0), np.radians(0.05) * _smooth_noise(rng_misc, (n, 3), 3.0))
    t0 = rng_misc.uniform(5000.0, 400000.0)
    stamps = t0 + frame_t + (rng_misc.normal(0.0, 2e-5, n) if cfg.noise else 0.0)
    result = CaptureResult(path=out, config=cfg, frame_times=frame_t, timestamps=stamps, slots=slots, true_pos=pos,
                           true_R=R, K=K, T_arkit_property=T_wp, drift=drift, trajectory=traj, jitter=jitter,
                           flags=list(traj.flags))

    out.mkdir(parents=True, exist_ok=True)
    (out / "depth").mkdir(exist_ok=True)
    (out / "confidence").mkdir(exist_ok=True)
    scene = scene or RenderScene(apt)
    video = VideoWriter(out / "rgb.mp4", cfg.rgb_size, cfg.fps, codec=cfg.video_codec)
    result.flags += video.flags
    png = [cv2.IMWRITE_PNG_COMPRESSION, int(cfg.png_compression)]
    noise_rng = rng_noise if cfg.noise else None
    try:
        for i in range(n):
            dr = render_depth(scene, K[i], cfg.rgb_size, cfg.depth_size, R[i], pos[i])
            depth_mm, conf = lidar_measurement(dr.depth, dr.cos_incidence, noise_rng, max_range=cfg.max_range)
            cv2.imwrite(str(out / "depth" / f"{i:06d}.png"), depth_mm, png)
            cv2.imwrite(str(out / "confidence" / f"{i:06d}.png"), conf, png)
            rgb = render_rgb(scene, K[i], cfg.rgb_size, R[i], pos[i], boxes=dr.boxes_seen, scale=cfg.rgb_scale,
                             rng=noise_rng)
            video.write(rgb, int(slots[i]))
            if (i + 1) % 300 == 0:
                log.info("%s: frame %d/%d (%.0f s)", out.name, i + 1, n, time.perf_counter() - t_start)
    finally:
        video.close()
    write_odometry(result, out / "odometry.csv")
    (out / "camera_matrix.csv").write_text(_camera_matrix_text(K[-1]))
    t_imu, acc, gyro = imu_samples(traj, rng_imu if cfg.noise else None)
    keep = t_imu <= frame_t[-1] + 0.05
    jit = rng_imu.uniform(0.0, 0.003, int(keep.sum())) if cfg.noise else 0.0
    (out / "imu.csv").write_text(_imu_text(t0 + t_imu[keep] + jit, acc[keep], gyro[keep]))
    result.seconds = time.perf_counter() - t_start
    log.info("%s: %d frames, %.1f s of capture, rendered in %.0f s", out.name, n, frame_t[-1], result.seconds)
    return result


def write_odometry(result: CaptureResult, path: Path) -> None:
    p_w, q = odometry_poses(result.true_pos, result.true_R, result.drift, result.T_arkit_property, result.jitter)
    path.write_text(_odometry_text(result.timestamps, p_w, q, result.K))


def redrift_capture(src: CaptureResult, out_dir: str | Path, *, seed: int, drift: str = "strong") -> CaptureResult:
    """Same frames as `src` with a different pose drift: sensor files are hard-linked (copied if that fails)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("depth", "confidence"):
        (out / sub).mkdir(exist_ok=True)
        for f in sorted((src.path / sub).iterdir()):
            _link(f, out / sub / f.name)
    for name in ("rgb.mp4", "camera_matrix.csv", "imu.csv"):
        _link(src.path / name, out / name)
    rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(3)[2])
    new = CaptureResult(path=out, config=src.config, frame_times=src.frame_times, timestamps=src.timestamps,
                        slots=src.slots, true_pos=src.true_pos, true_R=src.true_R, K=src.K,
                        T_arkit_property=src.T_arkit_property, drift=sample_drift(rng, src.frame_times, drift),
                        trajectory=src.trajectory, jitter=src.jitter, flags=list(src.flags))
    write_odometry(new, out / "odometry.csv")
    return new


def _link(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
