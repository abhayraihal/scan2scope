"""Synthetic rooms, cameras and point maps for the semantics, rules and scope tests."""

from __future__ import annotations

import cv2
import numpy as np

from scan2scope.types import CameraView, DamageRegion, Measurement, Opening, Plan, Room, Wall


def meas(value: float, lo: float | None = None, hi: float | None = None, unit: str = "m", kind: str = "length"):
    return Measurement(value=value, lo=lo, hi=hi, unit=unit, kind=kind)


def rect_room(room_id: str = "R1", x0: float = 0.0, y0: float = 0.0, w: float = 4.0, d: float = 3.0,
              floor_z: float = 0.0, height: float = 2.5, rel: float = 0.0) -> Room:
    """Counter-clockwise rectangle; W1 south (y = y0), W2 east, W3 north, W4 west. rel sets +-intervals."""
    poly = np.array([[x0, y0], [x0 + w, y0], [x0 + w, y0 + d], [x0, y0 + d]], float)
    walls = []
    for k in range(4):
        s, e = poly[k], poly[(k + 1) % 4]
        t = (e - s) / np.linalg.norm(e - s)
        L = float(np.linalg.norm(e - s))
        walls.append(Wall(id=f"{room_id}-W{k + 1}", room_id=room_id, start=s.copy(), end=e.copy(),
                          length=meas(L, L * (1 - rel), L * (1 + rel)),
                          height=meas(height, height * (1 - rel), height * (1 + rel), kind="height"),
                          normal_in=np.array([-t[1], t[0]]), observed_fraction=1.0))
    area = w * d
    return Room(id=room_id, label="room", polygon=poly, walls=walls, openings=[], floor_z=floor_z,
                ceiling_z=floor_z + height,
                ceiling_height=meas(height, height * (1 - rel), height * (1 + rel), kind="height"),
                floor_area=meas(area, area * (1 - 2 * rel), area * (1 + 2 * rel), unit="m2", kind="area"),
                perimeter=meas(2 * (w + d)))


def add_opening(room: Room, wall_k: int, otype: str, offset: float, width: float, height: float,
                sill: float | None = None, rel: float = 0.0) -> Opening:
    wall = room.walls[wall_k - 1]
    t = (wall.end - wall.start) / np.linalg.norm(wall.end - wall.start)
    o = Opening(id=f"{room.id}-O{len(room.openings) + 1}", room_id=room.id, wall_id=wall.id, type=otype,
                offset=meas(offset), width=meas(width, width * (1 - rel), width * (1 + rel), kind="width"),
                height=meas(height, height * (1 - rel), height * (1 + rel), kind="height"),
                sill=meas(sill) if sill is not None else None, center=wall.start + t * (offset + width / 2),
                confidence=0.9)
    room.openings.append(o)
    return o


def make_plan(*rooms: Room) -> Plan:
    area = sum(r.floor_area.value for r in rooms)
    return Plan(rooms=list(rooms), adjacency=[], footprint_area=meas(area, unit="m2", kind="area"),
                extent_x=meas(1.0), extent_y=meas(1.0))


def look_at(cam: np.ndarray, target: np.ndarray, up: tuple[float, float, float] = (0.0, 0.0, 1.0)) -> np.ndarray:
    """Camera-to-world pose in OpenCV axes (x right, y down, z forward)."""
    cam = np.asarray(cam, float)
    z = np.asarray(target, float) - cam
    z /= np.linalg.norm(z)
    down = -np.asarray(up, float)
    y = down - (down @ z) * z
    y /= np.linalg.norm(y)
    x = np.cross(y, z)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, cam
    return T


def intrinsics(width: int, height: int, f: float) -> np.ndarray:
    return np.array([[f, 0, width / 2 - 0.5], [0, f, height / 2 - 0.5], [0, 0, 1.0]])


def raycast_box(T_wc: np.ndarray, K: np.ndarray, width: int, height: int, pm_w: int, pm_h: int,
                lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    """World point map of the inside of an axis-aligned box, sampled per the CameraView pixel mapping."""
    j, i = np.meshgrid(np.arange(pm_w), np.arange(pm_h))
    x = (j + 0.5) * width / pm_w - 0.5
    y = (i + 0.5) * height / pm_h - 0.5
    d_c = np.stack([(x - K[0, 2]) / K[0, 0], (y - K[1, 2]) / K[1, 1], np.ones_like(x, float)], -1)
    d = d_c @ T_wc[:3, :3].T
    o = T_wc[:3, 3]
    t = np.full(d.shape[:2], np.inf)
    for k in range(3):
        with np.errstate(divide="ignore", invalid="ignore"):
            tk = np.where(d[..., k] > 0, (hi[k] - o[k]) / d[..., k],
                          np.where(d[..., k] < 0, (lo[k] - o[k]) / d[..., k], np.inf))
        t = np.minimum(t, tk)
    return o + d * t[..., None]


def project(P: np.ndarray, T_wc: np.ndarray, K: np.ndarray) -> np.ndarray:
    R, c = T_wc[:3, :3], T_wc[:3, 3]
    pc = (np.asarray(P, float) - c) @ R
    uv = pc[:, :2] / pc[:, 2:3]
    return uv * [K[0, 0], K[1, 1]] + [K[0, 2], K[1, 2]]


def polygon_mask(width: int, height: int, pts: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Filled polygon of image points (in pixels of a width x height image) rasterised at `scale`."""
    w, h = round(width * scale), round(height * scale)
    m = np.zeros((h, w), np.uint8)
    p = (np.asarray(pts, float) + 0.5) * scale - 0.5
    cv2.fillPoly(m, [np.round(p * 16).astype(np.int32)], 1, lineType=cv2.LINE_8, shift=4)
    return m.astype(bool)


def box_view(view_id: str, cam, target, room_lo, room_hi, *, width: int = 640, height: int = 480, f: float = 500.0,
             pm_w: int = 160, pm_h: int = 120, image_path=None) -> CameraView:
    T = look_at(np.asarray(cam, float), np.asarray(target, float))
    K = intrinsics(width, height, f)
    pm = raycast_box(T, K, width, height, pm_w, pm_h, np.asarray(room_lo, float), np.asarray(room_hi, float))
    return CameraView(id=view_id, image_path=image_path, width=width, height=height, K=K, T_wc=T,
                      pointmap=pm.astype(np.float32), valid=np.ones((pm_h, pm_w), bool))


def rect_on_wall_y(y: float, x0: float, x1: float, z0: float, z1: float) -> np.ndarray:
    return np.array([[x0, y, z0], [x1, y, z0], [x1, y, z1], [x0, y, z1]], float)


def region(rid: str, cls: str, surface_id: str, u: tuple[float, float], v: tuple[float, float], *,
           room_id: str = "R1", area: float | None = None, length: float | None = None, rel: float = 0.0,
           score: float = 0.8, evidence: dict | None = None) -> DamageRegion:
    w, h = u[1] - u[0], v[1] - v[0]
    a = w * h if area is None else area
    return DamageRegion(
        id=rid, room_id=room_id, surface_id=surface_id, cls=cls, score=score,
        area=meas(a, a * (1 - 2 * rel), a * (1 + 2 * rel), unit="m2", kind="area"),
        width=meas(w, w * (1 - rel), w * (1 + rel), kind="width"),
        height=meas(h, h * (1 - rel), h * (1 + rel), kind="height"), u_range=u, v_range=v,
        length=meas(length, length * (1 - rel), length * (1 + rel)) if length is not None else None,
        view_ids=["v1", "v2"], evidence=dict(evidence or {}))
