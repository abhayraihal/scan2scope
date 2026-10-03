"""Door matching: two openings of similar width in two rooms are hypothesised to be the same door.

Room B is placed so b's centre faces a's centre across a wall of thickness t with the two inward normals
opposite, and the relative yaw is snapped to A's Manhattan frame when it is within 5 degrees of it.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from scan2scope.geometry.gravity import manhattan_angle
from scan2scope.stitch.solver import DOOR_TYPES, Door, Hypothesis, StitchRoom, rigid2, snap_yaw
from scan2scope.types import Opening, Room

log = logging.getLogger("scan2scope.stitch")

WALL_THICKNESS_PRIOR = 0.12
WALL_THICKNESS_RANGE = (0.08, 0.30)
MIN_WIDTH_RATIO = 0.8
SNAP_TOL = np.radians(5.0)
BASE_SCORE = 0.6  # door matching alone is weaker evidence than a doorway-photo registration


def _vec2(v: Any) -> np.ndarray | None:
    if v is None:
        return None
    a = np.asarray(v, float).reshape(-1)
    if a.size < 2 or not np.isfinite(a[:2]).all():
        return None
    return a[:2].copy()


def _value(m: Any) -> float | None:
    try:
        v = float(m.value)
    except (AttributeError, TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def room_polygon(room: Room) -> np.ndarray | None:
    P = np.asarray(room.polygon, float)
    if P.ndim != 2 or P.shape[0] < 3 or P.shape[1] < 2 or not np.isfinite(P[:, :2]).all():
        return None
    return P[:, :2]


def nearest_edge(P: np.ndarray, p: np.ndarray) -> tuple[float, np.ndarray] | None:
    """Distance from p to the polygon boundary and the inward normal of the nearest edge (CCW polygon)."""
    A, B = P, np.roll(P, -1, axis=0)
    D = B - A
    L2 = (D ** 2).sum(1)
    ok = L2 > 1e-12
    if not ok.any():
        return None
    t = np.clip(((p - A) * D).sum(1) / np.maximum(L2, 1e-12), 0.0, 1.0)
    dist = np.linalg.norm(A + t[:, None] * D - p, axis=1)
    dist[~ok] = np.inf
    k = int(np.argmin(dist))
    d = D[k] / np.sqrt(L2[k])
    return float(dist[k]), np.array([-d[1], d[0]])


def room_manhattan(room: Room) -> float:
    """Dominant wall direction of a room modulo 90 degrees, from its walls or else its polygon edges."""
    dirs, weights = [], []
    for w in room.walls:
        s, e = _vec2(w.start), _vec2(w.end)
        if s is None or e is None or np.linalg.norm(e - s) < 1e-6:
            continue
        dirs.append(e - s)
        weights.append(np.linalg.norm(e - s))
    if not dirs:
        P = room_polygon(room)
        if P is None:
            return 0.0
        D = np.roll(P, -1, axis=0) - P
        keep = np.linalg.norm(D, axis=1) > 1e-6
        dirs, weights = list(D[keep]), list(np.linalg.norm(D[keep], axis=1))
    if not dirs:
        return 0.0
    D = np.asarray(dirs)
    return manhattan_angle(np.c_[-D[:, 1], D[:, 0]], np.asarray(weights))


def opening_door(room: Room, o: Opening) -> Door | None:
    """Door geometry (centre, inward normal, width) of a door or open passage, or None when unusable."""
    if o.type not in DOOR_TYPES:
        return None
    width = _value(o.width)
    if width is None or not 0.3 <= width <= 5.0:
        return None
    center = _vec2(o.center)
    normal = None
    wall = next((w for w in room.walls if w.id == o.wall_id), None)
    if wall is not None:
        s, e = _vec2(wall.start), _vec2(wall.end)
        if s is not None and e is not None and np.linalg.norm(e - s) > 1e-6:
            u = (e - s) / np.linalg.norm(e - s)
            n = _vec2(wall.normal_in)
            ok = n is not None and np.linalg.norm(n) > 1e-6
            normal = n / np.linalg.norm(n) if ok else np.array([-u[1], u[0]])
            offset = _value(o.offset)
            if center is None and offset is not None:
                center = s + u * (offset + width / 2)
    if normal is None and center is not None:
        P = room_polygon(room)
        hit = nearest_edge(P, center) if P is not None else None
        normal = hit[1] if hit is not None else None
    if center is None or normal is None:
        return None
    height = _value(o.height)
    conf = float(o.confidence) if o.confidence is not None and np.isfinite(o.confidence) else 0.0
    return Door(o.id, o.type, center, normal, width, height if height and height > 0.3 else None,
                float(np.clip(conf, 0.0, 1.0)))


def room_doors(room: Room) -> list[Door]:
    return [d for d in (opening_door(room, o) for o in room.openings) if d is not None]


def facing_yaw(door_a: Door, door_b: Door) -> float:
    """Rotation of b's frame into a's that makes the two inward normals opposite."""
    na, nb = door_a.normal, door_b.normal
    return float(np.arctan2(-na[1], -na[0]) - np.arctan2(nb[1], nb[0]))


def door_pair_transform(door_a: Door, door_b: Door, theta: float, thickness: float) -> np.ndarray:
    """T_ab that puts b's centre across the wall from a's centre, `thickness` out of room a."""
    T = rigid2(theta)
    T[:2, 2] = door_a.center - thickness * door_a.normal - T[:2, :2] @ door_b.center
    return T


def door_match_score(door_a: Door, door_b: Door, ratio: float, snapped: bool) -> float:
    f_width = np.exp(-0.5 * ((1.0 - ratio) / 0.08) ** 2)
    f_type = 1.0 if door_a.type == door_b.type else 0.8
    f_manhattan = 1.0 if snapped else 0.5
    f_conf = 0.5 + 0.5 * np.sqrt(door_a.confidence * door_b.confidence)
    f_height = 1.0
    if door_a.height and door_b.height:
        h_ratio = min(door_a.height, door_b.height) / max(door_a.height, door_b.height)
        f_height = np.exp(-0.5 * ((1.0 - h_ratio) / 0.1) ** 2)
    return float(BASE_SCORE * f_width * f_type * f_manhattan * f_conf * f_height)


def door_match_hypotheses(rooms: list[StitchRoom], *, min_ratio: float = MIN_WIDTH_RATIO,
                          thickness: float = WALL_THICKNESS_PRIOR) -> tuple[list[Hypothesis], int]:
    """Every door pair (a in A, b in B) with width ratio >= min_ratio.

    Returns the hypotheses and the number of door pairs compared.
    """
    out: list[Hypothesis] = []
    compared = 0
    for ia, A in enumerate(rooms):
        for B in rooms[ia + 1:]:
            for da in A.doors:
                for db in B.doors:
                    compared += 1
                    ratio = min(da.width, db.width) / max(da.width, db.width)
                    if ratio < min_ratio:
                        continue
                    theta, delta, snapped = snap_yaw(facing_yaw(da, db), A.manhattan, B.manhattan, SNAP_TOL)
                    out.append(Hypothesis(
                        A.index, B.index, door_pair_transform(da, db, theta, thickness),
                        door_match_score(da, db, ratio, snapped), "door_match", da.id, db.id,
                        evidence={"width_a": round(da.width, 3), "width_b": round(db.width, 3),
                                  "width_ratio": round(ratio, 3),
                                  "yaw_snap_deg": round(float(np.degrees(delta)), 2),
                                  "snapped": snapped, "thickness": thickness},
                    ))
    log.debug("stitch: %d door-match hypotheses from %d door pairs", len(out), compared)
    return out, compared
