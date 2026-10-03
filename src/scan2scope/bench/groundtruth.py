"""Loads bench/data/<property>/ground_truth.yaml (format: bench/templates/ground_truth.yaml).

Conventions from docs/ground_truth_protocol.md: walls are listed starting at the wall with the entry door, then
the next wall on the right seen from inside, so the order is clockwise seen from above. Opening and damage
offsets run from the wall's left end seen from inside, which is the start of the clockwise traversal. The
synthetic generator writes the same format plus a per-room `polygon` (vertex k is the left end of wall W(k+1))
and a property-level `adjacency` list.

Zero or negative lengths are template placeholders and load as missing values with a flag.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scan2scope.types import TIERS

log = logging.getLogger("scan2scope.bench")

OPENING_TYPES = ("door", "window", "opening")
MAX_RECTILINEAR_WALLS = 16
CLOSURE_TIE_M = 0.002


def num(x: Any) -> float | None:
    """Finite float from a number or a numeric string (decimal comma allowed), else None."""
    if x is None or isinstance(x, bool):
        return None
    if isinstance(x, str):
        s = x.strip().replace(",", ".")
        if not s:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
    else:
        try:
            v = float(x)
        except (TypeError, ValueError):
            return None
    return v if math.isfinite(v) else None


def positive(x: Any) -> float | None:
    v = num(x)
    return v if v is not None and v > 0 else None


def room_label(room_id: str) -> str:
    """'02 kitchen' -> 'kitchen' (the folder number is dropped)."""
    s = re.sub(r"^\s*\d+[\s._-]*", "", str(room_id)).strip()
    return s or str(room_id).strip()


def norm_name(s: Any) -> str:
    return re.sub(r"[\s_\-.]+", " ", str(s)).strip().lower()


@dataclass
class GTWall:
    id: str
    length: float | None


@dataclass
class GTOpening:
    id: str
    type: str
    wall: str
    offset: float | None  # wall's left end seen from inside to the near edge
    width: float | None
    height: float | None
    sill: float | None = None
    leads_to: str | None = None
    wall_thickness: float | None = None

    @property
    def center(self) -> float | None:
        """Centre along the wall from its left end seen from inside."""
        if self.offset is None or self.width is None:
            return None
        return self.offset + self.width / 2.0


@dataclass
class GTDamage:
    id: str
    cls: str
    surface: str  # wall id, "ceiling" or "floor"
    offset: float | None
    bottom: float | None
    width: float | None
    height: float | None
    length: float | None = None
    area: float | None = None  # mask area when the source knows it (synthetic); otherwise the bbox is used

    @property
    def bbox_area(self) -> float | None:
        if self.width is None or self.height is None:
            return None
        return self.width * self.height


@dataclass
class GTRoom:
    id: str
    label: str
    walls: list[GTWall]
    ceiling_readings: list[float]
    ceiling_height: float | None
    openings: list[GTOpening]
    damage: list[GTDamage]
    polygon: np.ndarray | None = None  # (k, 2) clockwise, vertex k = left end of wall k (0-based)
    diagonal: float | None = None
    floor_area: float | None = None
    area_method: str = "unavailable"  # polygon | opposite_walls | quad_diagonal | rectilinear | unavailable
    closure_error: float | None = None
    shape: np.ndarray | None = None  # polygon used for the aspect ratio (given or reconstructed)
    flags: list[str] = field(default_factory=list)

    def wall(self, wall_id: str) -> GTWall | None:
        return next((w for w in self.walls if w.id == wall_id), None)

    def wall_index(self, wall_id: str) -> int | None:
        return next((k for k, w in enumerate(self.walls) if w.id == wall_id), None)

    @property
    def perimeter(self) -> float | None:
        if not self.walls or any(w.length is None for w in self.walls):
            return None
        return float(sum(w.length for w in self.walls))

    @property
    def aspect(self) -> float | None:
        """Long side over short side of the minimum rotated rectangle of the room shape."""
        if self.shape is not None:
            return polygon_aspect(self.shape)
        if len(self.walls) == 4 and all(w.length for w in self.walls):
            a = (self.walls[0].length + self.walls[2].length) / 2.0
            b = (self.walls[1].length + self.walls[3].length) / 2.0
            return max(a, b) / min(a, b)
        return None


@dataclass
class GTCapture:
    id: str
    tier: str
    path: Path
    rooms: list[str] | None = None  # GT room ids covered; None = every room
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class GroundTruth:
    property: str
    path: Path
    rooms: list[GTRoom]
    captures: list[GTCapture]
    adjacency: list[dict[str, Any]]  # {"rooms": (a, b), "openings": (oa, ob), "type": str, "source": str}
    measured_by: str = ""
    instrument: str = ""
    date: str = ""
    synthetic: bool = False
    flags: list[str] = field(default_factory=list)

    @property
    def root(self) -> Path:
        return self.path.parent

    def room(self, room_id: str) -> GTRoom | None:
        return next((r for r in self.rooms if r.id == room_id), None)

    def capture(self, capture_id: str) -> GTCapture | None:
        return next((c for c in self.captures if c.id == capture_id), None)

    def capture_rooms(self, capture: GTCapture | None) -> list[GTRoom]:
        if capture is None or capture.rooms is None:
            return list(self.rooms)
        keep = set(capture.rooms)
        return [r for r in self.rooms if r.id in keep]

    def adjacency_pairs(self, room_ids: list[str] | None = None) -> set[frozenset[str]]:
        keep = None if room_ids is None else set(room_ids)
        out = set()
        for a in self.adjacency:
            pair = frozenset(a["rooms"])
            if keep is None or pair <= keep:
                out.add(pair)
        return out

    def footprint(self, capture: GTCapture | None = None) -> float | None:
        rooms = self.capture_rooms(capture)
        if not rooms or any(r.floor_area is None for r in rooms):
            return None
        return float(sum(r.floor_area for r in rooms))


def shoelace(P: np.ndarray) -> float:
    """Signed area, positive for counter-clockwise."""
    P = np.asarray(P, float)
    if len(P) < 3:
        return 0.0
    x, y = P[:, 0], P[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def polygon_aspect(P: np.ndarray) -> float | None:
    """Long side over short side of the minimum-area bounding box (one side lies along a polygon edge)."""
    P = np.asarray(P, float).reshape(-1, 2)
    if len(P) < 3 or not np.isfinite(P).all():
        return None
    edges = np.roll(P, -1, axis=0) - P
    norms = np.linalg.norm(edges, axis=1)
    keep = norms > 1e-9
    if not keep.any():
        return None
    u = edges[keep] / norms[keep, None]
    v = np.stack([-u[:, 1], u[:, 0]], axis=1)
    a, b = P @ u.T, P @ v.T
    w, h = a.max(axis=0) - a.min(axis=0), b.max(axis=0) - b.min(axis=0)
    k = int(np.argmin(w * h))
    if min(w[k], h[k]) <= 1e-9:
        return None
    return float(max(w[k], h[k]) / min(w[k], h[k]))


def _heron(a: float, b: float, c: float) -> float | None:
    s = (a + b + c) / 2.0
    q = s * (s - a) * (s - b) * (s - c)
    return math.sqrt(q) if q > 0 else None


def rectilinear_polygon(lengths: list[float | None], diagonal: float | None = None
                        ) -> tuple[np.ndarray, float] | None:
    """Corners of a clockwise rectilinear room from its wall lengths in protocol order, or None.

    Every corner is 90 degrees, so walls alternate between the two axes; each corner turns right (convex) or
    left (reflex) with four more right turns than left. The turn pattern that closes the outline best and gives
    a simple polygon wins; the measured diagonal (start of W1 to the farthest corner) breaks ties.
    Returns (corners (n, 2) starting at the left end of W1, closure error in metres).
    """
    n = len(lengths)
    if n < 4 or n % 2 or n > MAX_RECTILINEAR_WALLS or any(v is None or v <= 0 for v in lengths):
        return None
    from shapely.geometry import Polygon

    L = np.asarray(lengths, float)
    dirs = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])
    turns = np.array(list(itertools.product((-1, 1), repeat=n - 1)), dtype=int)
    last = -4 - turns.sum(axis=1)
    turns = turns[np.abs(last) == 1]
    if len(turns) == 0:
        return None
    heading = np.concatenate([np.zeros((len(turns), 1), int), np.cumsum(turns, axis=1)], axis=1) % 4
    steps = dirs[heading] * L[None, :, None]
    ends = np.cumsum(steps, axis=1)
    closure = np.linalg.norm(ends[:, -1], axis=1)
    order = np.argsort(np.round(closure / CLOSURE_TIE_M), kind="stable")
    best: tuple[float, float, np.ndarray] | None = None
    for k in order:
        if best is not None and closure[k] > best[0] + CLOSURE_TIE_M:
            break
        corners = np.vstack([[0.0, 0.0], ends[k, :-1]])
        poly = Polygon(corners)
        if not poly.is_valid or abs(poly.area) < 1e-6:
            continue
        diag_err = 0.0
        if diagonal is not None:
            diag_err = abs(float(np.max(np.linalg.norm(corners - corners[0], axis=1))) - diagonal)
        if best is None or (closure[k] < best[0] - CLOSURE_TIE_M) or diag_err < best[1] - 1e-9:
            best = (float(closure[k]), diag_err, corners)
    if best is None:
        return None
    return best[2], best[0]


def floor_area(walls: list[GTWall], polygon: np.ndarray | None, diagonal: float | None
               ) -> tuple[float | None, str, float | None, np.ndarray | None, list[str]]:
    """(area, method, closure error, shape polygon, flags) for one ground-truth room."""
    flags: list[str] = []
    lengths = [w.length for w in walls]
    if polygon is not None and len(polygon) >= 3:
        area = abs(shoelace(polygon))
        if area > 0:
            closure = None
            if len(polygon) == len(walls) and all(v is not None for v in lengths):
                edges = np.linalg.norm(np.roll(polygon, -1, axis=0) - polygon, axis=1)
                closure = float(np.max(np.abs(edges - np.asarray(lengths, float))))
                if closure > 0.01:
                    flags.append(f"polygon_walls_disagree:{closure:.3f}")
            return area, "polygon", closure, polygon, flags
        flags.append("degenerate_polygon")
    if any(v is None for v in lengths):
        return None, "unavailable", None, None, flags + ["missing_wall_length"]
    if len(walls) == 4:
        l1, l2, l3, l4 = lengths
        closure = float(math.hypot(l1 - l3, l2 - l4))
        if diagonal is not None:
            t1, t2 = _heron(l1, l2, diagonal), _heron(l3, l4, diagonal)
            if t1 is not None and t2 is not None:
                return t1 + t2, "quad_diagonal", closure, None, flags
            flags.append("diagonal_inconsistent")
        a, b = (l1 + l3) / 2.0, (l2 + l4) / 2.0
        shape = np.array([[0.0, 0.0], [a, 0.0], [a, -b], [0.0, -b]])
        return a * b, "opposite_walls", closure, shape, flags
    rec = rectilinear_polygon(lengths, diagonal)
    if rec is None:
        return None, "unavailable", None, None, flags + ["not_rectilinear"]
    corners, closure = rec
    if closure > 0.05:
        flags.append(f"rectilinear_closure:{closure:.3f}")
    return abs(shoelace(corners)), "rectilinear", closure, corners, flags


def _parse_polygon(raw: Any) -> np.ndarray | None:
    if raw is None:
        return None
    try:
        P = np.asarray([[num(p[0]), num(p[1])] for p in raw], dtype=float)
    except (TypeError, IndexError, KeyError, ValueError):
        return None
    if P.ndim != 2 or len(P) < 3 or not np.isfinite(P).all():
        return None
    return P


def _as_list(x: Any) -> list:
    if x is None:
        return []
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


def _parse_room(raw: dict[str, Any], index: int, flags: list[str]) -> GTRoom | None:
    if not isinstance(raw, dict) or raw.get("id") in (None, ""):
        flags.append(f"room_without_id:{index}")
        return None
    rid = str(raw["id"])
    rflags: list[str] = []
    walls = []
    for k, w in enumerate(_as_list(raw.get("walls"))):
        if isinstance(w, dict):
            wid, length = str(w.get("id") or f"W{k + 1}"), positive(w.get("length"))
        else:
            wid, length = f"W{k + 1}", positive(w)
        if length is None:
            rflags.append(f"missing:{wid}.length")
        walls.append(GTWall(wid, length))
    readings = []
    for v in _as_list(raw.get("ceiling_height")):
        h = positive(v)
        if h is None:
            rflags.append("ceiling_reading_placeholder")
        else:
            readings.append(h)
    ceiling = float(np.median(readings)) if readings else None
    wall_ids = {w.id for w in walls}
    if len(wall_ids) < len(walls):
        rflags.append("duplicate_wall_ids")
    openings = []
    for k, o in enumerate(_as_list(raw.get("openings"))):
        if not isinstance(o, dict):
            rflags.append(f"bad_opening:{k}")
            continue
        oid = str(o.get("id") or f"O{k + 1}")
        if any(p.id == oid for p in openings):
            rflags.append(f"duplicate_opening_id:{oid}->{oid}~{k + 1}")
            oid = f"{oid}~{k + 1}"
        kind = str(o.get("type") or "door").strip().lower()
        if kind not in OPENING_TYPES:
            rflags.append(f"opening_type:{oid}:{kind}->opening")
            kind = "opening"
        wall = str(o.get("wall") or "")
        if wall not in wall_ids:
            rflags.append(f"opening_unknown_wall:{oid}:{wall}")
        op = GTOpening(oid, kind, wall, num(o.get("offset")), positive(o.get("width")), positive(o.get("height")),
                       num(o.get("sill")), None if o.get("leads_to") in (None, "") else str(o.get("leads_to")),
                       positive(o.get("wall_thickness")))
        if op.width is None:
            rflags.append(f"missing:{oid}.width")
        openings.append(op)
    damage = []
    for k, d in enumerate(_as_list(raw.get("damage"))):
        if not isinstance(d, dict):
            rflags.append(f"bad_damage:{k}")
            continue
        did = str(d.get("id") or f"X{k + 1}")
        if any(x.id == did for x in damage):
            rflags.append(f"duplicate_damage_id:{did}->{did}~{k + 1}")
            did = f"{did}~{k + 1}"
        cls = str(d.get("class") or d.get("cls") or "").strip().lower().replace(" ", "_").replace("-", "_")
        surface = str(d.get("surface") or "").strip()
        if surface.lower() in ("ceiling", "ceil", "floor"):
            surface = "ceiling" if surface.lower().startswith("ceil") else "floor"
        damage.append(GTDamage(did, cls, surface, num(d.get("offset")), num(d.get("bottom")),
                               positive(d.get("width")), positive(d.get("height")), positive(d.get("length")),
                               positive(d.get("area"))))
    polygon = _parse_polygon(raw.get("polygon"))
    if raw.get("polygon") is not None and polygon is None:
        rflags.append("bad_polygon")
    diagonal = positive(raw.get("diagonal"))
    area, method, closure, shape, aflags = floor_area(walls, polygon, diagonal)
    rflags += aflags
    if area is None:
        rflags.append("floor_area_unavailable")
    room = GTRoom(rid, room_label(rid), walls, readings, ceiling, openings, damage, polygon, diagonal, area,
                  method, closure, shape, rflags)
    flags.extend(f"{rid}:{f}" for f in rflags)
    return room


def _parse_adjacency(raw: Any, rooms: list[GTRoom], flags: list[str]) -> list[dict[str, Any]]:
    """Explicit `adjacency` entries plus pairs implied by openings whose leads_to names another room."""
    by_id = {r.id: r for r in rooms}
    out: dict[frozenset[str], dict[str, Any]] = {}
    for k, a in enumerate(_as_list(raw)):
        names = a.get("rooms") if isinstance(a, dict) else a
        if not isinstance(names, (list, tuple)) or len(names) != 2:
            flags.append(f"bad_adjacency:{k}")
            continue
        ra, rb = str(names[0]), str(names[1])
        ops = list(_as_list(a.get("openings")) if isinstance(a, dict) else []) + [None, None]
        oa, ob = (None if v in (None, "") else str(v) for v in ops[:2])
        if ra == rb or ra not in by_id or rb not in by_id:
            flags.append(f"adjacency_unknown_room:{ra}|{rb}")
            continue
        bad = False
        for room, oid, other in ((by_id[ra], oa, rb), (by_id[rb], ob, ra)):
            op = next((o for o in room.openings if o.id == oid), None) if oid else None
            if oid and op is None:
                flags.append(f"adjacency_unknown_opening:{room.id}/{oid}")
            if op is not None and op.leads_to not in (None, other):
                bad = True
        if bad:
            flags.append(f"adjacency_contradicts_leads_to:{ra}|{rb}")
            continue
        kind = str(a.get("type") or "door") if isinstance(a, dict) else "door"
        out[frozenset((ra, rb))] = {"rooms": tuple(sorted((ra, rb))), "openings": (oa, ob) if ra <= rb else (ob, oa),
                                    "type": kind, "source": "adjacency"}
    for room in rooms:
        for op in room.openings:
            if op.type == "window" or op.leads_to is None or op.leads_to == room.id:
                continue
            if op.leads_to not in by_id:
                continue
            pair = frozenset((room.id, op.leads_to))
            a, b = sorted(pair)
            side = 0 if room.id == a else 1
            if pair not in out:
                ops = [None, None]
                ops[side] = op.id
                out[pair] = {"rooms": (a, b), "openings": tuple(ops), "type": op.type, "source": "leads_to"}
            elif out[pair]["openings"][side] is None:
                ops = list(out[pair]["openings"])
                ops[side] = op.id
                out[pair]["openings"] = tuple(ops)
    return [out[p] for p in sorted(out, key=lambda p: tuple(sorted(p)))]


def load_ground_truth(path: str | Path) -> GroundTruth:
    """Parse one ground_truth.yaml. Problems are recorded in flags; only an unreadable file raises."""
    path = Path(path)
    doc = yaml.safe_load(path.read_text()) or {}
    if not isinstance(doc, dict):
        raise TypeError(f"{path}: expected a mapping at the top level")
    flags: list[str] = []
    rooms: list[GTRoom] = []
    seen: set[str] = set()
    for k, raw in enumerate(_as_list(doc.get("rooms"))):
        room = _parse_room(raw, k, flags)
        if room is None:
            continue
        if room.id in seen:
            flags.append(f"duplicate_room:{room.id}")
            continue
        seen.add(room.id)
        rooms.append(room)
    captures = []
    for k, c in enumerate(_as_list(doc.get("captures"))):
        if not isinstance(c, dict) or not c.get("id") or not c.get("path"):
            flags.append(f"bad_capture:{k}")
            continue
        tier = str(c.get("tier") or "").strip().lower()
        if tier not in TIERS:
            flags.append(f"capture_tier:{c.get('id')}:{tier}")
            continue
        p = Path(str(c["path"])).expanduser()
        cap_rooms = None
        if c.get("rooms") is not None:
            cap_rooms = [str(r) for r in _as_list(c.get("rooms")) if str(r) in seen]
            dropped = [str(r) for r in _as_list(c.get("rooms")) if str(r) not in seen]
            if dropped:
                flags.append(f"capture_unknown_rooms:{c['id']}:{','.join(dropped)}")
        meta = {key: v for key, v in c.items() if key not in ("id", "tier", "path", "rooms")}
        captures.append(GTCapture(str(c["id"]), tier, p if p.is_absolute() else path.parent / p, cap_rooms, meta))
    adjacency = _parse_adjacency(doc.get("adjacency"), rooms, flags)
    measured_by = str(doc.get("measured_by") or "")
    synthetic = bool(doc.get("synthetic")) or measured_by.strip().lower() == "synthetic"
    gt = GroundTruth(str(doc.get("property") or path.parent.name), path, rooms, captures, adjacency, measured_by,
                     str(doc.get("instrument") or ""), str(doc.get("date") or ""), synthetic, flags)
    if flags:
        log.info("%s: %d ground-truth flags (%s%s)", gt.property, len(flags), ", ".join(flags[:4]),
                 ", ..." if len(flags) > 4 else "")
    return gt
