"""Random Manhattan apartments made of axis-aligned boxes, with exact ground truth.

Property frame: metres, x east, y north, z up, floor at z = 0. Layouts are generated in integer millimetres, so
every ground-truth number is exact at the millimetre precision the ground-truth file records. Walls are the solid
left over when the room interiors are cut out of the footprint; openings cut z-ranges out of that solid.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import pairwise

import numpy as np
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry import box as shapely_box
from shapely.ops import unary_union

log = logging.getLogger("scan2scope.synth")

EXTERIOR_MM = 200
ROOM_SIDE_MM = (2400, 5500)
TEMPLATES = ("double", "single_end", "single")
LOOP_INSET_M = (1.1, 1.3)  # nominal camera loop distance from the walls; captures draw from this range
LOOP_CORNER_M = 0.25  # corner radius of the camera loop
FURNITURE_CLEARANCE_M = 0.35  # free space kept between furniture and the camera loop

# label, width, depth, height ranges in mm
FURNITURE: dict[str, list[tuple[str, tuple[int, int], tuple[int, int], tuple[int, int]]]] = {
    "bedroom": [("wardrobe", (1000, 2000), (550, 620), (1900, 2150)), ("dresser", (800, 1400), (400, 500), (750, 950)),
                ("desk", (1000, 1400), (550, 650), (730, 760)), ("shelf", (600, 1000), (280, 350), (1700, 2000))],
    "living": [("sofa", (1600, 2200), (700, 800), (800, 900)), ("tv_unit", (1200, 1800), (400, 450), (450, 550)),
               ("shelf", (600, 1000), (280, 350), (1700, 2000)), ("cabinet", (800, 1200), (400, 450), (800, 1000))],
    "kitchen": [("counter", (1800, 3000), (600, 620), (880, 920)), ("tall_unit", (600, 700), (600, 650), (1900, 2100)),
                ("counter", (1200, 2000), (600, 620), (880, 920))],
    "bathroom": [("vanity", (600, 1000), (450, 550), (820, 880)), ("bathtub", (1500, 1700), (700, 750), (550, 600)),
                 ("cabinet", (400, 600), (300, 350), (1600, 1900))],
    "office": [("desk", (1200, 1600), (600, 700), (730, 760)), ("shelf", (800, 1200), (300, 350), (1800, 2000)),
               ("cabinet", (800, 1000), (400, 450), (700, 1000))],
}
TABLE = ("table", (1000, 1400), (700, 900), (730, 760))


class _Retry(Exception):
    pass


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle in integer millimetres."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def w(self) -> int:
        return self.x1 - self.x0

    @property
    def h(self) -> int:
        return self.y1 - self.y0

    def contains(self, x: float, y: float) -> bool:
        return self.x0 < x < self.x1 and self.y0 < y < self.y1

    def shapely_m(self) -> Polygon:
        return shapely_box(self.x0 / 1000, self.y0 / 1000, self.x1 / 1000, self.y1 / 1000)


@dataclass
class SynthOpening:
    """A door, open passage or window: a cut through one wall slab."""

    index: int
    kind: str  # door | opening | window
    axis: int  # axis of the wall normal: 0 -> wall plane x = const, 1 -> wall plane y = const
    n0: int  # wall faces along the normal axis (mm), n0 < n1
    n1: int
    s0: int  # extent along the wall (mm), s0 < s1
    s1: int
    z0: int
    z1: int
    low: int  # room index on the low-coordinate side of the wall, -1 = outside
    high: int

    @property
    def width(self) -> int:
        return self.s1 - self.s0

    @property
    def thickness(self) -> int:
        return self.n1 - self.n0

    def plan_rect(self) -> Rect:
        if self.axis == 0:
            return Rect(self.n0, self.s0, self.n1, self.s1)
        return Rect(self.s0, self.n0, self.s1, self.n1)

    def other(self, room: int) -> int:
        return self.high if room == self.low else self.low

    def face(self, room: int) -> int:
        """Coordinate of the wall face seen from `room`."""
        return self.n0 if room == self.low else self.n1

    def into(self, room: int) -> int:
        """Sign of the normal-axis direction pointing from the wall into `room`."""
        return -1 if room == self.low else 1

    def centre_on_face(self, room: int) -> tuple[float, float]:
        s = 0.5 * (self.s0 + self.s1)
        f = self.face(room)
        return (f, s) if self.axis == 0 else (s, f)


@dataclass
class SynthRoom:
    index: int
    kind: str
    rects: list[Rect]
    name: str = ""
    entry: int | None = None  # opening used to enter the room; None for the hallway (W1 is the entrance wall)
    max_inset: float = LOOP_INSET_M[1]  # largest camera loop inset (m) that still leaves a walkable loop
    polygon_mm: np.ndarray | None = None  # (K, 2) int, clockwise from above, vertex 0 = left end of W1

    def shape_m(self) -> Polygon:
        return unary_union([r.shapely_m() for r in self.rects])

    @property
    def polygon(self) -> np.ndarray:
        """Interior polygon in metres, ground-truth order (clockwise, starting at the left end of W1)."""
        return self.polygon_mm / 1000.0

    @property
    def area(self) -> float:
        return float(sum(r.w * r.h for r in self.rects)) / 1e6

    def inset_range(self) -> tuple[float, float]:
        lo, hi = LOOP_INSET_M
        return min(lo, self.max_inset), min(hi, self.max_inset)


@dataclass
class SynthBox:
    lo: tuple[float, float, float]  # metres
    hi: tuple[float, float, float]
    kind: str  # wall | furniture | door_leaf
    room: int = -1
    label: str = "wall"


@dataclass
class Apartment:
    seed: int
    template: str
    ceiling_mm: int
    envelope: Rect  # interior envelope; the exterior walls surround it
    rooms: list[SynthRoom]  # rooms[0] is the hallway, rooms[1:] are in walking order
    openings: list[SynthOpening]
    entrance: tuple[tuple[int, int], tuple[int, int]]  # hallway entrance wall (left end, right end) seen from inside
    boxes: list[SynthBox] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    @property
    def ceiling(self) -> float:
        return self.ceiling_mm / 1000.0

    @property
    def footprint(self) -> Rect:
        e = self.envelope
        return Rect(e.x0 - EXTERIOR_MM, e.y0 - EXTERIOR_MM, e.x1 + EXTERIOR_MM, e.y1 + EXTERIOR_MM)

    @property
    def hallway(self) -> SynthRoom:
        return self.rooms[0]

    def room_openings(self, room: int) -> list[SynthOpening]:
        return [o for o in self.openings if room in (o.low, o.high)]

    def room_at(self, x: float, y: float) -> int:
        """Room index containing the plan point (metres), -1 if none."""
        xm, ym = x * 1000.0, y * 1000.0
        for room in self.rooms:
            if any(r.contains(xm, ym) for r in room.rects):
                return room.index
        return -1

    def is_free(self, x: float, y: float, margin: float = 0.0) -> bool:
        """True when the plan point (metres) is walkable: inside a room or a door gap, clear of furniture."""
        xm, ym = x * 1000.0, y * 1000.0
        inside = self.room_at(x, y) >= 0 or any(
            o.kind != "window" and o.plan_rect().contains(xm, ym) for o in self.openings)
        if not inside:
            return False
        for b in self.boxes:
            if b.kind == "wall":
                continue
            if b.lo[0] - margin < x < b.hi[0] + margin and b.lo[1] - margin < y < b.hi[1] + margin:
                return False
        return True


# --------------------------------------------------------------------------------------------- generation


def random_apartment(seed: int, *, template: str | None = None, n_rooms: int | None = None,
                     l_room: bool | None = None, passage: bool | None = None, furniture: bool = True,
                     door_leaves: bool = True) -> Apartment:
    """A random hallway plus 3 to 5 rooms. Same seed and options give the same apartment."""
    if template is not None and template not in TEMPLATES:
        raise ValueError(f"template must be one of {TEMPLATES}, got {template!r}")
    rng = np.random.default_rng(seed)
    last: Exception | None = None
    for _ in range(300):
        try:
            apt = _build(rng, seed, template, n_rooms, l_room, passage, furniture, door_leaves)
            log.debug("apartment seed=%d template=%s rooms=%d", seed, apt.template, len(apt.rooms) - 1)
            return apt
        except _Retry as exc:
            last = exc
    raise RuntimeError(f"could not build an apartment for seed {seed}: {last}")


def _split(rng: np.random.Generator, total: int, k: int, lo: int = ROOM_SIDE_MM[0],
           hi: int = ROOM_SIDE_MM[1]) -> list[int]:
    if not k * lo <= total <= k * hi:
        raise _Retry("row length out of range")
    for _ in range(200):
        w = lo + rng.dirichlet(np.full(k, 3.0)) * (total - k * lo)
        w = np.round(w).astype(int)
        w[-1] = total - int(w[:-1].sum())
        if np.all((w >= lo) & (w <= hi)):
            return [int(v) for v in w]
    raise _Retry("could not split row")


def _row(rng: np.random.Generator, widths: list[int], walls: list[int], y0: int, y1: int, x0: int = 0) -> list[Rect]:
    rects, x = [], x0
    for i, w in enumerate(widths):
        rects.append(Rect(x, y0, x + w, y1))
        x += w + (walls[i] if i < len(walls) else 0)
    return rects


def _shared_walls(a: Rect, b: Rect, max_gap: int = 160) -> list[tuple[int, int, int, int, int, bool]]:
    """Wall segments separating rects a and b: (axis, n0, n1, s0, s1, a_is_low)."""
    out = []
    for lo_r, hi_r, a_low in ((a, b, True), (b, a, False)):
        gap = hi_r.x0 - lo_r.x1
        if 0 < gap <= max_gap:
            s0, s1 = max(lo_r.y0, hi_r.y0), min(lo_r.y1, hi_r.y1)
            if s1 > s0:
                out.append((0, lo_r.x1, hi_r.x0, s0, s1, a_low))
        gap = hi_r.y0 - lo_r.y1
        if 0 < gap <= max_gap:
            s0, s1 = max(lo_r.x0, hi_r.x0), min(lo_r.x1, hi_r.x1)
            if s1 > s0:
                out.append((1, lo_r.y1, hi_r.y0, s0, s1, a_low))
    return out


def _exterior_segments(room: SynthRoom, env: Rect) -> list[tuple[int, int, int, int, int, int, int]]:
    """Exterior wall segments of a room: (axis, n0, n1, s0, s1, low, high), collinear pieces merged."""
    e = EXTERIOR_MM
    segs = []
    for r in room.rects:
        if r.x0 == env.x0:
            segs.append((0, env.x0 - e, env.x0, r.y0, r.y1, -1, room.index))
        if r.x1 == env.x1:
            segs.append((0, env.x1, env.x1 + e, r.y0, r.y1, room.index, -1))
        if r.y0 == env.y0:
            segs.append((1, env.y0 - e, env.y0, r.x0, r.x1, -1, room.index))
        if r.y1 == env.y1:
            segs.append((1, env.y1, env.y1 + e, r.x0, r.x1, room.index, -1))
    segs.sort()
    merged: list[list[int]] = []
    for s in segs:
        if merged and merged[-1][:3] == list(s[:3]) and merged[-1][4] == s[3]:
            merged[-1][4] = s[4]
        else:
            merged.append(list(s))
    return [tuple(m) for m in merged]


def _build(rng: np.random.Generator, seed: int, template: str | None, n_rooms: int | None, l_room: bool | None,
           passage: bool | None, furniture: bool, door_leaves: bool) -> Apartment:
    tpl = template if template is not None else str(rng.choice(TEMPLATES))
    n_max = 5 if tpl == "double" else 4
    n = int(n_rooms) if n_rooms is not None else int(rng.integers(3, n_max + 1))
    if not 3 <= n <= n_max:
        raise ValueError(f"template {tpl} supports 3 to {n_max} rooms, got {n}")
    H = int(rng.integers(2400, 2801))
    c = int(rng.integers(1000, 1601))

    def tw() -> int:
        return int(rng.integers(100, 151))

    rows: list[dict] = []  # {"rects": [...], "corridor_low": bool (corridor on the low-y side), "y0", "y1"}
    if tpl == "double":
        k_n = int(rng.choice([n // 2, n - n // 2]))
        k_s = n - k_n
        d_s, d_n = (int(v) for v in rng.integers(ROOM_SIDE_MM[0], ROOM_SIDE_MM[1] + 1, 2))
        t_cs, t_cn = tw(), tw()
        walls_n, walls_s = [tw() for _ in range(k_n - 1)], [tw() for _ in range(k_s - 1)]
        lo = max(k_n * ROOM_SIDE_MM[0] + sum(walls_n), k_s * ROOM_SIDE_MM[0] + sum(walls_s))
        hi = min(k_n * ROOM_SIDE_MM[1] + sum(walls_n), k_s * ROOM_SIDE_MM[1] + sum(walls_s))
        if lo > hi:
            raise _Retry("rows cannot match")
        L = int(lo + (hi - lo) * rng.beta(2.0, 2.0))
        y_c0 = d_s + t_cs
        y_c1 = y_c0 + c
        y_n0 = y_c1 + t_cn
        hall = Rect(0, y_c0, L, y_c1)
        rows.append({"rects": _row(rng, _split(rng, L - sum(walls_s), k_s), walls_s, 0, d_s), "corridor_low": False,
                     "y0": 0, "y1": d_s})
        rows.append({"rects": _row(rng, _split(rng, L - sum(walls_n), k_n), walls_n, y_n0, y_n0 + d_n),
                     "corridor_low": True, "y0": y_n0, "y1": y_n0 + d_n})
        env = Rect(0, 0, L, y_n0 + d_n)
        end_rects: list[Rect] = []
    else:
        k = n if tpl == "single" else n - 1
        t_cn = tw()
        d_max = ROOM_SIDE_MM[1] if tpl == "single" else ROOM_SIDE_MM[1] - c - t_cn
        d_n = int(rng.integers(ROOM_SIDE_MM[0], d_max + 1))
        widths = [int(v) for v in rng.integers(ROOM_SIDE_MM[0], ROOM_SIDE_MM[1] + 1, k)]
        walls = [tw() for _ in range(k - 1)]
        L = sum(widths) + sum(walls)
        y_n0 = c + t_cn
        hall = Rect(0, 0, L, c)
        rows.append({"rects": _row(rng, widths, walls, y_n0, y_n0 + d_n), "corridor_low": True, "y0": y_n0,
                     "y1": y_n0 + d_n})
        end_rects = []
        x_max = L
        if tpl == "single_end":
            t_e = tw()
            w_e = int(rng.integers(ROOM_SIDE_MM[0], ROOM_SIDE_MM[1] + 1))
            end_rects = [Rect(L + t_e, 0, L + t_e + w_e, y_n0 + d_n)]
            x_max = L + t_e + w_e
        env = Rect(0, 0, x_max, y_n0 + d_n)

    room_rects: list[list[Rect]] = [[r] for row in rows for r in row["rects"]] + [[r] for r in end_rects]
    row_of: list[int] = [i for i, row in enumerate(rows) for _ in row["rects"]] + [-1] * len(end_rects)

    want_l = l_room or (l_room is None and rng.random() < 0.5)
    if want_l and not _make_l_room(rng, rows, room_rects, row_of, tw()) and l_room:
        raise _Retry("no row deep enough for an L-shaped room")

    rooms = [SynthRoom(0, "hallway", [hall])]
    for rects in room_rects:
        rooms.append(SynthRoom(len(rooms), "room", rects))

    openings: list[SynthOpening] = []

    def add(kind: str, axis: int, n0: int, n1: int, s0: int, s1: int, z0: int, z1: int, low: int, high: int) -> int:
        openings.append(SynthOpening(len(openings), kind, axis, n0, n1, s0, s1, z0, z1, low, high))
        return len(openings) - 1

    for room in rooms[1:]:
        segs = [s for r in room.rects for s in _shared_walls(hall, r)]
        if not segs:
            raise _Retry("room not adjacent to the hallway")
        axis, n0, n1, s0, s1, hall_low = max(segs, key=lambda s: s[4] - s[3])
        length = s1 - s0
        w_d = int(rng.integers(700, 921))
        w_d = min(w_d, length - 300)
        if w_d < 700:
            raise _Retry("no room for a door")
        off = int(rng.integers(150, length - 150 - w_d + 1))
        h_d = int(rng.integers(2000, 2101))
        low, high = (0, room.index) if hall_low else (room.index, 0)
        room.entry = add("door", axis, n0, n1, s0 + off, s0 + off + w_d, 0, h_d, low, high)

    want_p = passage if passage is not None else bool(rng.random() < 0.5)
    if want_p:
        cands = []
        for a in rooms[1:]:
            for b in rooms[a.index + 1:]:
                for ra in a.rects:
                    for rb in b.rects:
                        for seg in _shared_walls(ra, rb):
                            if seg[4] - seg[3] >= 900 + 500:
                                cands.append((a.index, b.index, seg))
        if cands:
            ia, ib, (axis, n0, n1, s0, s1, a_low) = cands[int(rng.integers(len(cands)))]
            w_p = int(rng.integers(900, min(1200, s1 - s0 - 500) + 1))
            off = int(rng.integers(250, s1 - s0 - 250 - w_p + 1))
            h_p = int(rng.integers(2000, min(2250, H - 150) + 1))
            low, high = (ia, ib) if a_low else (ib, ia)
            add("opening", axis, n0, n1, s0 + off, s0 + off + w_p, 0, h_p, low, high)

    entrance_y0, entrance_y1 = hall.y0, hall.y1
    for room in rooms:
        for axis, n0, n1, s0, s1, low, high in _exterior_segments(room, env):
            if room.index == 0 and axis == 0 and n1 == env.x0:
                continue  # the entrance wall where every capture starts and ends
            length = s1 - s0
            p_first = 0.35 if room.index == 0 else 0.8
            taken: list[tuple[int, int]] = []
            for p in (p_first, 0.3):
                if length < 1400 or rng.random() >= p:
                    break
                w = int(rng.integers(800, min(1600, length - 600) + 1))
                lo_s, hi_s = s0 + 300, s1 - 300 - w
                ok = [s for s in range(lo_s, hi_s + 1, 10)
                      if all(s + w + 300 <= a or s >= b + 300 for a, b in taken)]
                if not ok:
                    break
                s = int(ok[int(rng.integers(len(ok)))])
                sill = int(rng.integers(800, 1001))
                h = int(rng.integers(1000, min(1400, H - 200 - sill) + 1))
                add("window", axis, n0, n1, s, s + w, sill, sill + h, low, high)
                taken.append((s, s + w))
                if length < 3400:
                    break

    apt = Apartment(seed=seed, template=tpl, ceiling_mm=H, envelope=env, rooms=rooms, openings=openings,
                    entrance=((hall.x0, entrance_y0), (hall.x0, entrance_y1)))
    _order_and_name(apt, rng)
    for room in apt.rooms:
        room.polygon_mm = _gt_polygon(apt, room)
        room.max_inset = _max_inset(room)
    apt.boxes = [SynthBox((b[0] / 1000, b[1] / 1000, b[4] / 1000), (b[2] / 1000, b[3] / 1000, b[5] / 1000), "wall")
                 for b in _wall_boxes(apt)]
    if door_leaves:
        _place_door_leaves(apt, rng)
    if furniture:
        _place_furniture(apt, rng)
    return apt


def _make_l_room(rng: np.random.Generator, rows: list[dict], room_rects: list[list[Rect]], row_of: list[int],
                 t_split: int) -> bool:
    """Merge the exterior-side part of a room into its neighbour, making that neighbour L-shaped."""
    options = []
    for ri, row in enumerate(rows):
        depth = row["y1"] - row["y0"]
        if len(row["rects"]) < 2 or depth < ROOM_SIDE_MM[0] + t_split + 1000:
            continue
        idx = [i for i, r in enumerate(row_of) if r == ri]
        for a, b in pairwise(idx):
            options += [(ri, a, b), (ri, b, a)]
    if not options:
        return False
    ri, ia, ib = options[int(rng.integers(len(options)))]
    row = rows[ri]
    A, B = room_rects[ia][0], room_rects[ib][0]
    depth = row["y1"] - row["y0"]
    d_bc = int(rng.integers(ROOM_SIDE_MM[0], depth - t_split - 1000 + 1))
    xs = (A.x1, B.x1) if A.x1 < B.x0 else (B.x0, A.x0)
    if row["corridor_low"]:
        b_c = Rect(B.x0, row["y0"], B.x1, row["y0"] + d_bc)
        arm = Rect(xs[0], row["y0"] + d_bc + t_split, xs[1], row["y1"])
    else:
        b_c = Rect(B.x0, row["y1"] - d_bc, B.x1, row["y1"])
        arm = Rect(xs[0], row["y0"], xs[1], row["y1"] - d_bc - t_split)
    room_rects[ia] = [A, arm]
    room_rects[ib] = [b_c]
    return True


def _order_and_name(apt: Apartment, rng: np.random.Generator) -> None:
    """Renumber rooms in walking order (by entry door position along the corridor) and name them."""

    def key(room: SynthRoom) -> tuple[float, float]:
        o = apt.openings[room.entry]
        x = 0.5 * (o.s0 + o.s1) if o.axis == 1 else o.n0
        y = o.n0 if o.axis == 1 else 0.5 * (o.s0 + o.s1)
        return (x, y)

    others = sorted(apt.rooms[1:], key=key)
    order = [apt.rooms[0]] + others
    remap = {room.index: i for i, room in enumerate(order)}
    for o in apt.openings:
        o.low = remap.get(o.low, -1)
        o.high = remap.get(o.high, -1)
    for i, room in enumerate(order):
        room.index = i
    apt.rooms = order

    n = len(others)
    areas = np.array([r.area for r in others])
    rank = np.argsort(areas)
    kinds = ["bedroom"] * n
    kinds[rank[0]] = "bathroom"
    kinds[rank[-1]] = "living"
    if n >= 4:
        kinds[rank[1]] = "kitchen"
    if n >= 5:
        kinds[rank[2]] = str(rng.choice(["bedroom", "office"]))
    for room, kind in zip(others, kinds):
        room.kind = kind
    for i, room in enumerate(order):
        room.name = f"{i + 1:02d} {room.kind}"


def _gt_polygon(apt: Apartment, room: SynthRoom) -> np.ndarray:
    """Interior polygon in mm, clockwise from above, rotated so edge 0 is the wall holding the entry."""
    shape = unary_union([shapely_box(r.x0, r.y0, r.x1, r.y1) for r in room.rects])
    if shape.geom_type != "Polygon":
        raise _Retry("room is not one polygon")
    pts = np.round(np.asarray(shape.exterior.coords)[:-1]).astype(np.int64)
    area2 = np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
    if area2 > 0:
        pts = pts[::-1]
    keep = []
    for i in range(len(pts)):
        a, b, c = pts[i - 1], pts[i], pts[(i + 1) % len(pts)]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            keep.append(i)
    pts = pts[keep]
    if room.entry is None:
        (x0, y0), (x1, y1) = apt.entrance
        mid = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    else:
        mid = apt.openings[room.entry].centre_on_face(room.index)
    k = _edge_containing(pts, mid)
    if k is None:
        raise _Retry("entry wall not on the room polygon")
    return np.roll(pts, -k, axis=0)


def _edge_containing(pts: np.ndarray, p: tuple[float, float], tol: float = 0.5) -> int | None:
    for k in range(len(pts)):
        a, b = pts[k], pts[(k + 1) % len(pts)]
        if a[0] == b[0] and abs(p[0] - a[0]) <= tol and min(a[1], b[1]) - tol <= p[1] <= max(a[1], b[1]) + tol:
            return k
        if a[1] == b[1] and abs(p[1] - a[1]) <= tol and min(a[0], b[0]) - tol <= p[0] <= max(a[0], b[0]) + tol:
            return k
    return None


def loop_shape(shape: Polygon, inset: float) -> Polygon | None:
    """Rounded walking loop region at `inset` metres from the walls (its boundary is the camera path)."""
    core = shape.buffer(-(inset + LOOP_CORNER_M), join_style=2)
    if core.is_empty:
        return None
    if core.geom_type != "Polygon":
        core = max(core.geoms, key=lambda g: g.area)
    loop = core.buffer(LOOP_CORNER_M, join_style=1)
    return loop if loop.geom_type == "Polygon" and loop.length > 0.5 else None


def _max_inset(room: SynthRoom) -> float:
    shape = room.shape_m()
    r = LOOP_INSET_M[1]
    while r > 0.45:
        core = shape.buffer(-(r + LOOP_CORNER_M), join_style=2)
        if not core.is_empty and core.area > 0.01:
            return round(r, 3)
        r -= 0.025
    return 0.45


def _subtract(intervals: list[tuple[int, int]], a: int, b: int) -> list[tuple[int, int]]:
    out = []
    for z0, z1 in intervals:
        if b <= z0 or a >= z1:
            out.append((z0, z1))
            continue
        if a > z0:
            out.append((z0, a))
        if b < z1:
            out.append((b, z1))
    return out


def _wall_boxes(apt: Apartment) -> list[tuple[int, int, int, int, int, int]]:
    """Solid wall material as boxes (x0, y0, x1, y1, z0, z1) in mm, merged greedily on the plan grid."""
    fp = apt.footprint
    rects = [r for room in apt.rooms for r in room.rects]
    cuts = [(o.plan_rect(), o.z0, o.z1) for o in apt.openings]
    xs, ys = {fp.x0, fp.x1}, {fp.y0, fp.y1}
    for r in rects + [c[0] for c in cuts]:
        xs |= {r.x0, r.x1}
        ys |= {r.y0, r.y1}
    xs_l, ys_l = sorted(xs), sorted(ys)
    nx, ny = len(xs_l) - 1, len(ys_l) - 1
    H = apt.ceiling_mm
    cells: dict[tuple[int, int], tuple[tuple[int, int], ...]] = {}
    for j in range(ny):
        cy = 0.5 * (ys_l[j] + ys_l[j + 1])
        for i in range(nx):
            cx = 0.5 * (xs_l[i] + xs_l[i + 1])
            if any(r.contains(cx, cy) for r in rects):
                continue
            iv = [(0, H)]
            for pr, z0, z1 in cuts:
                if pr.contains(cx, cy):
                    iv = _subtract(iv, z0, z1)
            if iv:
                cells[j, i] = tuple(iv)
    runs = []
    for j in range(ny):
        i = 0
        while i < nx:
            s = cells.get((j, i))
            if s is None:
                i += 1
                continue
            i0 = i
            while i + 1 < nx and cells.get((j, i + 1)) == s:
                i += 1
            runs.append((j, i0, i, s))
            i += 1
    open_runs: dict[tuple, list[int]] = {}
    merged = []
    for j, i0, i1, s in runs:
        key = (i0, i1, s)
        span = open_runs.get(key)
        if span is not None and span[1] == j - 1:
            span[1] = j
        else:
            if span is not None:
                merged.append((key, span[0], span[1]))
            open_runs[key] = [j, j]
    merged += [(key, span[0], span[1]) for key, span in open_runs.items()]
    boxes = []
    for (i0, i1, s), j0, j1 in merged:
        for z0, z1 in s:
            boxes.append((xs_l[i0], ys_l[j0], xs_l[i1 + 1], ys_l[j1 + 1], z0, z1))
    return boxes


def _room_edges(room: SynthRoom) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Edges (start, end, inward unit normal) in metres for a clockwise polygon."""
    P = room.polygon
    out = []
    for k in range(len(P)):
        a, b = P[k], P[(k + 1) % len(P)]
        d = (b - a) / np.linalg.norm(b - a)
        out.append((a, b, np.array([d[1], -d[0]])))
    return out


def _opening_zone(o: SynthOpening, room: int, depth: float, pad: float) -> Polygon:
    """Plan region in front of an opening on the side of `room`."""
    f = o.face(room) / 1000.0
    sgn = o.into(room)
    s0, s1 = o.s0 / 1000.0 - pad, o.s1 / 1000.0 + pad
    n0, n1 = sorted((f, f + sgn * depth))
    return shapely_box(n0, s0, n1, s1) if o.axis == 0 else shapely_box(s0, n0, s1, n1)


def _box_m(axis: int, n0: float, n1: float, s0: float, s1: float) -> Polygon:
    n0, n1 = sorted((n0, n1))
    s0, s1 = sorted((s0, s1))
    return shapely_box(n0, s0, n1, s1) if axis == 0 else shapely_box(s0, n0, s1, n1)


def _place_door_leaves(apt: Apartment, rng: np.random.Generator) -> None:
    """Open door leaves: swung 90 degrees into the room when there is space, else folded flat against the wall."""
    for o in apt.openings:
        if o.kind != "door" or rng.random() >= 0.6:
            continue
        room = o.high if o.low == 0 else o.low
        if room <= 0:
            continue
        shape = apt.rooms[room].shape_m()
        r_min = apt.rooms[room].inset_range()[0]
        w = o.width / 1000.0
        f = o.face(room) / 1000.0
        sgn = o.into(room)
        hinge_low = bool(rng.random() < 0.5)
        hinge = o.s0 / 1000.0 if hinge_low else o.s1 / 1000.0
        away = -1.0 if hinge_low else 1.0
        if r_min - (w - 0.02) >= 0.3:
            poly = _box_m(o.axis, f, f + sgn * (w - 0.02), hinge + away * 0.005, hinge + away * 0.045)
        else:
            poly = _box_m(o.axis, f, f + sgn * 0.04, hinge + away * 0.005, hinge + away * (w - 0.02))
        if not shape.buffer(1e-6).contains(poly) or any(_conflicts(apt, room, poly, skip=o.index)):
            continue
        x0, y0, x1, y1 = poly.bounds
        apt.boxes.append(SynthBox((x0, y0, 0.01), (x1, y1, o.z1 / 1000.0 - 0.01), "door_leaf", room, "door_leaf"))


def arm_path(room: SynthRoom) -> tuple[np.ndarray, np.ndarray] | None:
    """Walking line into the arm of an L-shaped room: (start inside the main part, end near the arm's far wall)."""
    if len(room.rects) != 2:
        return None
    main, arm = room.rects
    y = 0.5 * (arm.y0 + arm.y1) / 1000.0
    if arm.x0 == main.x1:
        return np.array([main.x1 / 1000.0 - 0.4, y]), np.array([arm.x1 / 1000.0 - 0.7, y])
    return np.array([main.x0 / 1000.0 + 0.4, y]), np.array([arm.x0 / 1000.0 + 0.7, y])


def free_mask(apt: Apartment, xy: np.ndarray, margin: float = 0.05) -> np.ndarray:
    """Plan points (N, 2) in metres that are inside a room or a door gap and clear of furniture by `margin`."""
    x, y = xy[:, 0:1] * 1000.0, xy[:, 1:2] * 1000.0
    rects = [r for room in apt.rooms for r in room.rects]
    rects += [o.plan_rect() for o in apt.openings if o.kind != "window"]
    R = np.array([(r.x0, r.y0, r.x1, r.y1) for r in rects], float)
    inside = ((x > R[:, 0]) & (x < R[:, 2]) & (y > R[:, 1]) & (y < R[:, 3])).any(1)
    obj = [b for b in apt.boxes if b.kind != "wall"]
    if obj:
        B = np.array([(b.lo[0], b.lo[1], b.hi[0], b.hi[1]) for b in obj]) * 1000.0
        m = margin * 1000.0
        hit = ((x > B[:, 0] - m) & (x < B[:, 2] + m) & (y > B[:, 1] - m) & (y < B[:, 3] + m)).any(1)
        inside &= ~hit
    return inside


def _conflicts(apt: Apartment, room: int, poly: Polygon, skip: int | None = None, height: float = 3.0):
    """Yield reasons why a furniture footprint in `room` is not allowed."""
    r_hi = apt.rooms[room].inset_range()[1]
    arm = arm_path(apt.rooms[room])
    if arm is not None:
        line = LineString(arm).buffer(0.45)
        if poly.intersects(line):
            yield "arm_path"
    for o in apt.room_openings(room):
        if o.index == skip:
            continue
        if o.kind == "window":
            if height > o.z0 / 1000.0 - 0.05 and poly.intersects(_opening_zone(o, room, 1.0, 0.1)):
                yield "window"
        elif poly.intersects(_opening_zone(o, room, r_hi + 0.3, 0.2)):
            yield "door"
    for b in apt.boxes:
        if (b.kind != "wall" and b.room == room
                and poly.intersects(shapely_box(b.lo[0], b.lo[1], b.hi[0], b.hi[1]).buffer(0.05))):
            yield "furniture"


def _place_furniture(apt: Apartment, rng: np.random.Generator) -> None:
    for room in apt.rooms[1:]:
        shape = room.shape_m()
        r_lo, r_hi = room.inset_range()
        max_depth = r_lo - FURNITURE_CLEARANCE_M
        catalogue = FURNITURE.get(room.kind, FURNITURE["bedroom"])
        n_items = int(rng.integers(1, 4))
        edges = _room_edges(room)
        for _ in range(n_items):
            label, wr, dr, hr = catalogue[int(rng.integers(len(catalogue)))]
            w = rng.integers(wr[0], wr[1] + 1) / 1000.0
            d = min(rng.integers(dr[0], dr[1] + 1) / 1000.0, max_depth)
            h = rng.integers(hr[0], hr[1] + 1) / 1000.0
            if d < 0.25:
                continue
            for _attempt in range(40):
                a, b, nin = edges[int(rng.integers(len(edges)))]
                length = float(np.linalg.norm(b - a))
                if length < w + 0.1:
                    continue
                s = rng.uniform(0.05, length - w - 0.05)
                u = (b - a) / length
                p0 = a + u * s
                p1 = p0 + u * w + nin * d
                poly = shapely_box(min(p0[0], p1[0]), min(p0[1], p1[1]), max(p0[0], p1[0]), max(p0[1], p1[1]))
                if not shape.buffer(1e-6).contains(poly) or any(_conflicts(apt, room.index, poly, height=h)):
                    continue
                x0, y0, x1, y1 = poly.bounds
                apt.boxes.append(SynthBox((x0, y0, 0.0), (x1, y1, h), "furniture", room.index, label))
                break
        if room.kind in ("living", "kitchen") and rng.random() < 0.4:
            core = shape.buffer(-(r_hi + FURNITURE_CLEARANCE_M), join_style=2)
            label, wr, dr, hr = TABLE
            w, d, h = (rng.integers(v[0], v[1] + 1) / 1000.0 for v in (wr, dr, hr))
            if not core.is_empty:
                c = core.representative_point() if not core.contains(core.centroid) else core.centroid
                poly = shapely_box(c.x - w / 2, c.y - d / 2, c.x + w / 2, c.y + d / 2)
                if core.contains(poly) and not any(_conflicts(apt, room.index, poly, height=h)):
                    x0, y0, x1, y1 = poly.bounds
                    apt.boxes.append(SynthBox((x0, y0, 0.0), (x1, y1, h), "furniture", room.index, label))


# --------------------------------------------------------------------------------------------- ground truth


def ground_truth_rooms(apt: Apartment) -> list[dict]:
    """Per-room ground truth in the bench/templates/ground_truth.yaml layout (plus `polygon`), metres."""
    out = []
    for room in apt.rooms:
        P = room.polygon_mm
        walls = []
        for k in range(len(P)):
            a, b = P[k], P[(k + 1) % len(P)]
            walls.append({"id": f"W{k + 1}", "length": float(np.abs(b - a).sum()) / 1000.0})
        openings = []
        for o in apt.room_openings(room.index):
            mid = o.centre_on_face(room.index)
            k = _edge_containing(P, mid)
            if k is None:
                log.warning("opening %d not on a wall of %s", o.index, room.name)
                apt.flags.append(f"opening_off_wall:{room.name}:{o.index}")
                continue
            a = P[k]
            along = 1 if o.axis == 0 else 0
            d0, d1 = abs(o.s0 - int(a[along])), abs(o.s1 - int(a[along]))
            entry = {"kind": o.kind, "wall": k, "offset": min(d0, d1) / 1000.0, "width": o.width / 1000.0,
                     "index": o.index}
            if o.kind == "window":
                entry.update(height=(o.z1 - o.z0) / 1000.0, sill=o.z0 / 1000.0)
            else:
                other = o.other(room.index)
                entry.update(height=o.z1 / 1000.0, leads_to=apt.rooms[other].name if other >= 0 else "outside",
                             wall_thickness=o.thickness / 1000.0)
            openings.append(entry)
        openings.sort(key=lambda e: (e["index"] != room.entry, e["wall"], e["offset"]))
        counters = {"door": 0, "window": 0, "opening": 0}
        prefix = {"door": "D", "window": "N", "opening": "O"}
        items = []
        for e in openings:
            counters[e["kind"]] += 1
            item = {"id": f"{prefix[e['kind']]}{counters[e['kind']]}", "type": e["kind"], "wall": f"W{e['wall'] + 1}",
                    "offset": e["offset"], "width": e["width"], "height": e["height"]}
            if e["kind"] == "window":
                item["sill"] = e["sill"]
            else:
                item["leads_to"] = e["leads_to"]
                item["wall_thickness"] = e["wall_thickness"]
            item["_opening"] = e["index"]
            items.append(item)
        diagonal = None
        if len(P) != 4:
            diagonal = float(np.max(np.linalg.norm(P - P[0], axis=1))) / 1000.0
        H = apt.ceiling
        out.append({"id": room.name, "walls": walls, "diagonal": diagonal, "ceiling_height": [H, H, H],
                    "openings": items, "damage": [], "polygon": (P / 1000.0).tolist()})
    return out


def adjacency(apt: Apartment, rooms_gt: list[dict]) -> list[dict]:
    """Room pairs joined by a door or open passage, with the opening id on each side."""
    ids: dict[tuple[int, int], str] = {}
    for room, gt in zip(apt.rooms, rooms_gt):
        for item in gt["openings"]:
            ids[room.index, item["_opening"]] = item["id"]
    out = []
    for o in apt.openings:
        if o.kind == "window":
            continue
        a, b = sorted((o.low, o.high))
        out.append({"rooms": [apt.rooms[a].name, apt.rooms[b].name], "type": o.kind,
                    "openings": [ids.get((a, o.index)), ids.get((b, o.index))]})
    return out


def point_in_room(apt: Apartment, room: int, x: float, y: float) -> bool:
    return apt.rooms[room].shape_m().contains(Point(x, y))
