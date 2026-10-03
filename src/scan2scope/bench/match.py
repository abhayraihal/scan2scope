"""Deterministic matching of a result.json dict to ground truth: rooms, walls, openings and damage.

Rooms: photo tier by Room.source_hint (the folder name) equal to the GT room id, with a label and shape
fallback; video and LiDAR by Hungarian assignment on |log area ratio| + 0.5 |log aspect ratio| plus an
adjacency-consistency term, refined for a few rounds against the current assignment.

Walls: ours run counter-clockwise, the protocol lists GT walls clockwise. Every cyclic shift of our walls is
aligned to the GT order in both orientations ("reversed" is the protocol case, "same" covers a GT listed the
other way round), with gaps when the counts differ, and the smallest summed length difference wins. Near-ties
(within 2 cm + 2% of the GT perimeter) go to the alignment whose openings line up best.

Openings: same matched wall, compatible type (door and opening are interchangeable), centre along the wall
within half the GT width. GT offsets run from the wall's left end seen from inside, which is the start of the
GT traversal; in the reversed orientation that end is the end of our wall, so a GT centre c becomes L_gt - c.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from scan2scope.bench.groundtruth import (
    GroundTruth,
    GTCapture,
    GTOpening,
    GTRoom,
    norm_name,
    num,
    polygon_aspect,
    room_label,
    shoelace,
)

log = logging.getLogger("scan2scope.bench")

TIE_ABS = 0.02
TIE_REL = 0.02
NOPOS_COST = 1.0  # cost of an opening pair whose position is unknown on one side (matched by type only)
BIG = 1e9
ASPECT_WEIGHT = 0.5
ADJ_WEIGHT = 0.5
ADJ_ROUNDS = 10
MAX_AREA_RATIO = 3.0  # shape matches further apart than this are rejected
DAMAGE_POS_TOL = 0.10


def meas(m: Any) -> tuple[float, float, float] | None:
    """(value, lo, hi) of a measurement dict; lo/hi fall back to the value."""
    if not isinstance(m, dict):
        return None
    v = num(m.get("value"))
    if v is None:
        return None
    lo, hi = num(m.get("lo")), num(m.get("hi"))
    return v, v if lo is None else lo, v if hi is None else hi


def mval(m: Any) -> float | None:
    t = meas(m)
    return None if t is None else t[0]


@dataclass
class PredRoom:
    id: str
    label: str
    source_hint: str | None
    polygon: np.ndarray
    area: float | None
    walls: list[dict[str, Any]]
    openings: list[dict[str, Any]]
    raw: dict[str, Any]

    @property
    def aspect(self) -> float | None:
        return polygon_aspect(self.polygon) if len(self.polygon) >= 3 else None

    @property
    def perimeter(self) -> float | None:
        lengths = self.wall_lengths
        return None if not lengths or any(v is None for v in lengths) else float(sum(lengths))

    @property
    def wall_lengths(self) -> list[float | None]:
        out = []
        for w in self.walls:
            v = mval(w.get("length"))
            if v is None:
                try:
                    v = float(np.linalg.norm(np.subtract(w["end"], w["start"])))
                except (KeyError, TypeError, ValueError):
                    v = None
            out.append(v)
        return out


def pred_rooms(result: dict[str, Any]) -> list[PredRoom]:
    out = []
    for k, r in enumerate(result.get("rooms") or []):
        if not isinstance(r, dict):
            continue
        try:
            P = np.asarray(r.get("polygon") or [], float).reshape(-1, 2)
        except (TypeError, ValueError):
            P = np.zeros((0, 2))
        area = mval(r.get("floor_area"))
        if area is None and len(P) >= 3:
            area = abs(shoelace(P))
        walls = [w for w in r.get("walls") or [] if isinstance(w, dict)]
        openings = [o for o in r.get("openings") or [] if isinstance(o, dict)]
        rid = str(r.get("id") or f"R{k + 1}")
        out.append(PredRoom(rid, str(r.get("label") or rid), None if r.get("source_hint") is None
                            else str(r["source_hint"]), P, area, walls, openings, r))
    return out


@dataclass
class WallAlignment:
    orientation: str  # reversed | same
    shift: int
    pairs: list[tuple[int, int]]  # (GT wall index, predicted wall index)
    missed: list[int]  # GT wall indices without a predicted wall
    extra: list[int]  # predicted wall indices without a GT wall
    cost: float
    ties: int = 1  # candidates within the length tolerance of the best
    ambiguous: bool = False  # another tied candidate with a different assignment lines the openings up as well
    mirrored: bool = False  # GT offsets read from the right end (a measuring slip); set only when it matches more


@dataclass
class OpeningMatch:
    status: str  # matched | missed | phantom
    gt: str | None = None
    pred: str | None = None
    gt_wall: str | None = None
    pred_wall: str | None = None
    gt_type: str | None = None
    pred_type: str | None = None
    center_err: float | None = None  # predicted minus GT centre along our wall, metres


@dataclass
class DamageMatch:
    status: str  # matched | missed | phantom
    gt: str | None = None
    pred: str | None = None
    gt_room: str | None = None
    pred_room: str | None = None
    gt_surface: str | None = None
    pred_surface: str | None = None
    gt_class: str | None = None
    pred_class: str | None = None

    @property
    def class_ok(self) -> bool:
        return self.status == "matched" and self.gt_class == self.pred_class


@dataclass
class RoomMatch:
    gt_room: str
    pred_room: str | None
    method: str  # source_hint | label | hungarian | unmatched
    cost: float | None = None
    walls: WallAlignment | None = None
    openings: list[OpeningMatch] = field(default_factory=list)
    damage: list[DamageMatch] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def pred_wall_index(self, gt_index: int) -> int | None:
        if self.walls is None:
            return None
        return next((p for g, p in self.walls.pairs if g == gt_index), None)


@dataclass
class CaptureMatch:
    rooms: list[RoomMatch]
    extra_rooms: list[str]
    extra_openings: list[OpeningMatch]
    room_map: dict[str, str]  # predicted room id -> GT room id
    extra_damage: list[DamageMatch] = field(default_factory=list)
    flags: list[str] = field(default_factory=list)

    def room(self, gt_id: str) -> RoomMatch | None:
        return next((r for r in self.rooms if r.gt_room == gt_id), None)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def types_compatible(gt_type: str | None, pred_type: str | None) -> bool:
    a, b = str(gt_type or "").lower(), str(pred_type or "").lower()
    if a == b:
        return True
    return {a, b} <= {"door", "opening"}


def _from_end(orientation: str, mirrored: bool) -> bool:
    """True when the GT wall's left end seen from inside is the end of our wall."""
    return (orientation == "reversed") != mirrored


def gt_center_ours(op: GTOpening, gt_len: float | None, orientation: str, mirrored: bool = False) -> float | None:
    """GT opening centre as a distance from the start of our wall."""
    c = op.center
    if c is None:
        return None
    if _from_end(orientation, mirrored):
        return None if gt_len is None else gt_len - c
    return c


def gt_span_ours(start: float | None, width: float | None, gt_len: float | None, orientation: str,
                 mirrored: bool = False) -> tuple[float, float] | None:
    """GT span [start, start + width] measured from the wall's left end, as a range from the start of our wall."""
    if start is None or width is None:
        return None
    if _from_end(orientation, mirrored):
        if gt_len is None:
            return None
        return gt_len - start - width, gt_len - start
    return start, start + width


def pred_center(op: dict[str, Any]) -> float | None:
    off, w = mval(op.get("offset")), mval(op.get("width"))
    if off is None or w is None:
        return None
    return off + w / 2.0


def _dp_align(g: list[float | None], q: list[float | None], skip_g: list[float], skip_q: list[float]
              ) -> tuple[float, list[tuple[int, int]]]:
    """Order-preserving alignment of two wall sequences with gaps; cost = summed |length difference| + gaps."""
    m, n = len(g), len(q)
    D = np.full((m + 1, n + 1), np.inf)
    back = np.zeros((m + 1, n + 1), dtype=int)
    D[0, 0] = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                continue
            best, arg = np.inf, -1
            if i > 0 and j > 0:
                a, b = g[i - 1], q[j - 1]
                c = D[i - 1, j - 1] + (abs(a - b) if a is not None and b is not None else 0.0)
                best, arg = c, 0
            if i > 0 and D[i - 1, j] + skip_g[i - 1] < best - 1e-12:
                best, arg = D[i - 1, j] + skip_g[i - 1], 1
            if j > 0 and D[i, j - 1] + skip_q[j - 1] < best - 1e-12:
                best, arg = D[i, j - 1] + skip_q[j - 1], 2
            D[i, j], back[i, j] = best, arg
    pairs = []
    i, j = m, n
    while i > 0 or j > 0:
        a = back[i, j]
        if a == 0:
            pairs.append((i - 1, j - 1))
            i, j = i - 1, j - 1
        elif a == 1:
            i -= 1
        else:
            j -= 1
    return float(D[m, n]), pairs[::-1]


def _wall_candidates(gt_room: GTRoom, pred: PredRoom) -> list[WallAlignment]:
    G = [w.length for w in gt_room.walls]
    P = pred.wall_lengths
    m, n = len(G), len(P)
    known = [v for v in G if v is not None]
    fill = float(np.mean(known)) if known else 1.0
    skip_g = [fill if v is None else v for v in G]
    skip_q = [0.0 if v is None else v for v in P]
    if n == 0:
        return [WallAlignment("reversed", 0, [], list(range(m)), [], float(sum(skip_g)))]
    seen: dict[tuple, WallAlignment] = {}
    for orientation in ("reversed", "same"):
        for s in range(n):
            seq = [(s - j) % n if orientation == "reversed" else (s + j) % n for j in range(n)]
            cost, pairs = _dp_align(G, [P[k] for k in seq], skip_g, [skip_q[k] for k in seq])
            pairs = [(i, seq[j]) for i, j in pairs]
            key = (orientation, tuple(pairs))
            if key in seen:
                continue
            gm = {i for i, _ in pairs}
            pm = {j for _, j in pairs}
            seen[key] = WallAlignment(orientation, s, pairs, [i for i in range(m) if i not in gm],
                                      [j for j in range(n) if j not in pm], cost)
    return list(seen.values())


def match_openings(gt_room: GTRoom, pred: PredRoom, walls: WallAlignment) -> list[OpeningMatch]:
    """Matched, missed and phantom openings of one room pair under a wall alignment."""
    wall_map = dict(walls.pairs)
    pred_index = {str(w.get("id")): k for k, w in enumerate(pred.walls)}
    pred_by_wall: dict[int | None, list[dict[str, Any]]] = defaultdict(list)
    for o in pred.openings:
        pred_by_wall[pred_index.get(str(o.get("wall_id")))].append(o)
    gt_by_wall: dict[int | None, list[GTOpening]] = defaultdict(list)
    for op in gt_room.openings:
        gt_by_wall[gt_room.wall_index(op.wall)].append(op)
    out: list[OpeningMatch] = []
    used: set[int] = set()
    for gi in sorted(gt_by_wall, key=lambda x: (x is None, x if x is not None else 0)):
        gops = gt_by_wall[gi]
        pj = wall_map.get(gi) if gi is not None else None
        cands = pred_by_wall.get(pj, []) if pj is not None else []
        gt_len = gt_room.walls[gi].length if gi is not None else None
        C = np.full((len(gops), max(len(cands), 1)), BIG)
        for a, g in enumerate(gops):
            cg = gt_center_ours(g, gt_len, walls.orientation, walls.mirrored)
            for b, p in enumerate(cands):
                if not types_compatible(g.type, p.get("type")):
                    continue
                cp = pred_center(p)
                if cg is None or cp is None:
                    C[a, b] = NOPOS_COST
                elif g.width is not None and abs(cp - cg) <= g.width / 2.0 + 1e-9:
                    C[a, b] = abs(cp - cg)
        done: set[int] = set()
        if cands:
            for a, b in zip(*linear_sum_assignment(C)):
                if C[a, b] >= BIG:
                    continue
                g, p = gops[a], cands[b]
                cg = gt_center_ours(g, gt_len, walls.orientation, walls.mirrored)
                cp = pred_center(p)
                out.append(OpeningMatch("matched", g.id, str(p.get("id")), g.wall, str(p.get("wall_id")), g.type,
                                        str(p.get("type")), None if cg is None or cp is None else cp - cg))
                used.add(id(p))
                done.add(a)
        for a, g in enumerate(gops):
            if a not in done:
                out.append(OpeningMatch("missed", g.id, None, g.wall, None, g.type, None))
    for o in pred.openings:
        if id(o) not in used:
            out.append(OpeningMatch("phantom", None, str(o.get("id")), None, str(o.get("wall_id")), None,
                                    str(o.get("type"))))
    return out


def _opening_score(ops: list[OpeningMatch]) -> tuple[int, float]:
    bad = sum(o.status != "matched" for o in ops)
    err = sum(abs(o.center_err) for o in ops if o.status == "matched" and o.center_err is not None)
    return bad, round(err, 4)


def align_walls(gt_room: GTRoom, pred: PredRoom) -> tuple[WallAlignment, list[OpeningMatch]]:
    """Best wall alignment and the opening matches under it."""
    cands = _wall_candidates(gt_room, pred)
    best_cost = min(c.cost for c in cands)
    perim = sum(w.length for w in gt_room.walls if w.length is not None)
    tol = TIE_ABS + TIE_REL * perim
    tied = [c for c in cands if c.cost <= best_cost + tol]
    scored = []
    for c in tied:
        ops = match_openings(gt_room, pred, c)
        scored.append(((*_opening_score(ops), round(c.cost, 6), c.orientation != "reversed", c.shift), c, ops))
    scored.sort(key=lambda t: t[0])
    _, best, ops = scored[0]
    best.ties = len(tied)
    best.ambiguous = any(k[:2] == scored[0][0][:2] and set(c.pairs) != set(best.pairs) for k, c, _ in scored[1:])
    if any(o.status == "missed" for o in ops):
        alt = dataclasses.replace(best, mirrored=True)
        alt_ops = match_openings(gt_room, pred, alt)
        if _opening_score(alt_ops)[0] < _opening_score(ops)[0]:
            return alt, alt_ops
    return best, ops


def _unary(p: PredRoom, g: GTRoom) -> float:
    if p.area and g.floor_area:
        cost = abs(math.log(p.area / g.floor_area))
    elif p.perimeter and g.perimeter:
        cost = 2.0 * abs(math.log(p.perimeter / g.perimeter))
    else:
        cost = 1.0
    pa, ga = p.aspect, g.aspect
    if pa and ga:
        cost += ASPECT_WEIGHT * abs(math.log(pa / ga))
    return cost


def _inconsistency(p: int, g: int, assign: dict[int, int], pred_nb: list[set[int]], gt_nb: list[set[int]]) -> float:
    """Share of p's matched neighbours that do not land on g's GT neighbours, and the converse."""
    inv = {gg: pp for pp, gg in assign.items() if pp != p and gg != g}
    others = {pp: gg for pp, gg in assign.items() if pp != p and gg != g}
    bad = total = 0
    for q in pred_nb[p]:
        if q in others:
            total += 1
            bad += others[q] not in gt_nb[g]
    for h in gt_nb[g]:
        if h in inv:
            total += 1
            bad += inv[h] not in pred_nb[p]
    return bad / total if total else 0.0


def _hungarian(cost: np.ndarray) -> dict[int, int]:
    if cost.size == 0:
        return {}
    rows, cols = linear_sum_assignment(cost)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def shape_assignment(preds: list[PredRoom], gts: list[GTRoom], pred_pairs: set[frozenset[str]],
                     gt_pairs: set[frozenset[str]]) -> dict[int, tuple[int, float]]:
    """Hungarian assignment pred index -> (GT index, unary cost), refined with adjacency consistency."""
    if not preds or not gts:
        return {}
    U = np.array([[_unary(p, g) for g in gts] for p in preds])
    pid = {p.id: k for k, p in enumerate(preds)}
    gid = {g.id: k for k, g in enumerate(gts)}
    pred_nb: list[set[int]] = [set() for _ in preds]
    gt_nb: list[set[int]] = [set() for _ in gts]
    for pair in pred_pairs:
        a, b = sorted(pair)
        if a in pid and b in pid:
            pred_nb[pid[a]].add(pid[b])
            pred_nb[pid[b]].add(pid[a])
    for pair in gt_pairs:
        a, b = sorted(pair)
        if a in gid and b in gid:
            gt_nb[gid[a]].add(gid[b])
            gt_nb[gid[b]].add(gid[a])

    def objective(assign: dict[int, int]) -> float:
        return sum(U[p, g] + ADJ_WEIGHT * _inconsistency(p, g, assign, pred_nb, gt_nb) for p, g in assign.items())

    assign = _hungarian(U)
    best, best_j = assign, objective(assign)
    seen = {tuple(sorted(assign.items()))}
    for _ in range(ADJ_ROUNDS):
        C = U + ADJ_WEIGHT * np.array([[_inconsistency(p, g, assign, pred_nb, gt_nb) for g in range(len(gts))]
                                       for p in range(len(preds))])
        assign = _hungarian(C)
        key = tuple(sorted(assign.items()))
        if key in seen:
            break
        seen.add(key)
        j = objective(assign)
        if j < best_j - 1e-12:
            best, best_j = assign, j
    return {p: (g, float(U[p, g])) for p, g in best.items()}


def _area_ratio_ok(p: PredRoom, g: GTRoom) -> bool:
    if not p.area or not g.floor_area:
        return True
    return abs(math.log(p.area / g.floor_area)) <= math.log(MAX_AREA_RATIO)


def match_rooms(gts: list[GTRoom], preds: list[PredRoom], tier: str, gt_pairs: set[frozenset[str]],
                pred_pairs: set[frozenset[str]]) -> tuple[dict[str, tuple[str, str, float | None]], list[str]]:
    """GT room id -> (pred room id, method, cost), plus flags."""
    flags: list[str] = []
    out: dict[str, tuple[str, str, float | None]] = {}
    taken: set[str] = set()
    if tier == "photo":
        by_norm = defaultdict(list)
        by_label = defaultdict(list)
        for g in gts:
            by_norm[norm_name(g.id)].append(g.id)
            by_label[norm_name(g.label)].append(g.id)
        for p in preds:
            if p.source_hint is None:
                continue
            hit, method = None, "source_hint"
            if any(g.id == p.source_hint for g in gts):
                hit = p.source_hint
            elif len(by_norm[norm_name(p.source_hint)]) == 1:
                hit = by_norm[norm_name(p.source_hint)][0]
            elif len(by_label[norm_name(room_label(p.source_hint))]) == 1:
                hit, method = by_label[norm_name(room_label(p.source_hint))][0], "label"
            if hit is None:
                flags.append(f"source_hint_unknown:{p.id}:{p.source_hint}")
                continue
            if hit in out:
                flags.append(f"duplicate_source_hint:{p.id}:{p.source_hint}")
                continue
            out[hit] = (p.id, method, None)
            taken.add(p.id)
    rest_g = [g for g in gts if g.id not in out]
    rest_p = [p for p in preds if p.id not in taken]
    for pi, (gi, cost) in sorted(shape_assignment(rest_p, rest_g, pred_pairs, gt_pairs).items()):
        p, g = rest_p[pi], rest_g[gi]
        if not _area_ratio_ok(p, g):
            flags.append(f"room_match_rejected:{p.id}->{g.id}")
            continue
        out[g.id] = (p.id, "hungarian", round(cost, 4))
        if tier == "photo":
            flags.append(f"matched_by_shape:{p.id}->{g.id}")
    return out, flags


def gt_surface_ours(gt_room: GTRoom, rm: RoomMatch, pred: PredRoom, surface: str) -> str | None:
    """Our surface id for a GT damage surface (wall id, 'ceiling' or 'floor')."""
    if surface == "ceiling":
        return f"{pred.id}-CEIL"
    if surface == "floor":
        return f"{pred.id}-FLOOR"
    gi = gt_room.wall_index(surface)
    pj = None if gi is None else rm.pred_wall_index(gi)
    return None if pj is None else str(pred.walls[pj].get("id"))


def _range(r: Any) -> tuple[float, float] | None:
    try:
        a, b = num(r[0]), num(r[1])
    except (TypeError, IndexError, KeyError):
        return None
    return None if a is None or b is None else (min(a, b), max(a, b))


def _box_gap(a: tuple[float, float], b: tuple[float, float]) -> float:
    return max(0.0, max(a[0], b[0]) - min(a[1], b[1]))


def match_damage(gt_room: GTRoom, rm: RoomMatch, pred: PredRoom, damage: list[dict[str, Any]]) -> list[DamageMatch]:
    """GT damage of one matched room against the predicted damage on that room's surfaces."""
    mine = [d for d in damage if str(d.get("room_id")) == pred.id]
    gt_len = {w.id: w.length for w in gt_room.walls}
    orientation = rm.walls.orientation if rm.walls is not None else "reversed"
    mirrored = rm.walls.mirrored if rm.walls is not None else False
    C = np.full((len(gt_room.damage), max(len(mine), 1)), BIG)
    for a, g in enumerate(gt_room.damage):
        sid = gt_surface_ours(gt_room, rm, pred, g.surface)
        on_wall = g.surface not in ("ceiling", "floor")
        u_gt = gt_span_ours(g.offset, g.width, gt_len.get(g.surface), orientation, mirrored) if on_wall else None
        v_gt = None if g.bottom is None or g.height is None else (g.bottom, g.bottom + g.height)
        for b, d in enumerate(mine):
            if sid is None or str(d.get("surface_id")) != sid:
                continue
            u, v = _range(d.get("u_range")), _range(d.get("v_range"))
            cls_pen = 0.0 if str(d.get("class")) == g.cls else 0.5
            if u_gt is None or v_gt is None or u is None or v is None:
                C[a, b] = NOPOS_COST + cls_pen
                continue
            gu, gv = _box_gap(u, u_gt), _box_gap(v, v_gt)
            if gu <= DAMAGE_POS_TOL and gv <= DAMAGE_POS_TOL:
                du = abs((u[0] + u[1]) / 2 - (u_gt[0] + u_gt[1]) / 2)
                dv = abs((v[0] + v[1]) / 2 - (v_gt[0] + v_gt[1]) / 2)
                C[a, b] = math.hypot(du, dv) + cls_pen
    out: list[DamageMatch] = []
    done_g: set[int] = set()
    done_p: set[int] = set()
    if mine and gt_room.damage:
        for a, b in zip(*linear_sum_assignment(C)):
            if C[a, b] >= BIG:
                continue
            g, d = gt_room.damage[a], mine[b]
            out.append(DamageMatch("matched", g.id, str(d.get("id")), gt_room.id, pred.id, g.surface,
                                   str(d.get("surface_id")), g.cls, str(d.get("class"))))
            done_g.add(a)
            done_p.add(b)
    for a, g in enumerate(gt_room.damage):
        if a not in done_g:
            out.append(DamageMatch("missed", g.id, None, gt_room.id, pred.id, g.surface, None, g.cls, None))
    for b, d in enumerate(mine):
        if b not in done_p:
            out.append(DamageMatch("phantom", None, str(d.get("id")), gt_room.id, pred.id, None,
                                   str(d.get("surface_id")), None, str(d.get("class"))))
    return out


def pred_adjacency(result: dict[str, Any]) -> set[frozenset[str]]:
    prop = result.get("property") if isinstance(result.get("property"), dict) else {}
    out = set()
    for a in prop.get("adjacency") or []:
        if isinstance(a, dict) and a.get("room_a") is not None and a.get("room_b") is not None:
            ra, rb = str(a["room_a"]), str(a["room_b"])
            if ra != rb:
                out.add(frozenset((ra, rb)))
    return out


def match_capture(gt: GroundTruth, capture: GTCapture | None, result: dict[str, Any],
                  tier: str | None = None) -> CaptureMatch:
    tier = tier or (capture.tier if capture is not None else str((result.get("capture") or {}).get("tier")))
    gts = gt.capture_rooms(capture)
    preds = pred_rooms(result)
    ids = [g.id for g in gts]
    pairs, flags = match_rooms(gts, preds, tier, gt.adjacency_pairs(ids), pred_adjacency(result))
    by_id = {p.id: p for p in preds}
    damage = [d for d in result.get("damage") or [] if isinstance(d, dict)]
    rooms = []
    for g in gts:
        if g.id not in pairs:
            rooms.append(RoomMatch(g.id, None, "unmatched",
                                   damage=[DamageMatch("missed", d.id, None, g.id, None, d.surface, None, d.cls)
                                           for d in g.damage]))
            continue
        pid, method, cost = pairs[g.id]
        walls, ops = align_walls(g, by_id[pid])
        rm = RoomMatch(g.id, pid, method, cost, walls, ops)
        rm.damage = match_damage(g, rm, by_id[pid], damage)
        if walls.orientation != "reversed":
            rm.flags.append("gt_walls_counter_clockwise")
        if walls.ambiguous:
            rm.flags.append("wall_alignment_ambiguous")
        if walls.mirrored:
            rm.flags.append("gt_offsets_mirrored")
        rooms.append(rm)
    room_map = {r.pred_room: r.gt_room for r in rooms if r.pred_room is not None}
    extra = [p.id for p in preds if p.id not in room_map]
    extra_ops = [OpeningMatch("phantom", None, str(o.get("id")), None, str(o.get("wall_id")), None, str(o.get("type")))
                 for p in preds if p.id in extra for o in p.openings]
    extra_damage = [DamageMatch("phantom", None, str(d.get("id")), None, str(d.get("room_id")), None,
                                str(d.get("surface_id")), None, str(d.get("class")))
                    for d in damage if str(d.get("room_id")) not in room_map]
    return CaptureMatch(rooms, extra, extra_ops, room_map, extra_damage, flags)
