"""Room graph and placement for photo-tier stitching.

Rooms are nodes, placement hypotheses are edges. Hypotheses that place the same pair of rooms the same way are
merged into one edge. Rooms are placed by a maximum spanning tree on edge score, grown from the room with the
most connections, with no-overlap as a hard constraint. A single-edge local search repairs greedy mistakes and
detects placements whose alternatives score within a margin, which are flagged instead of silently picked.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid

from scan2scope.types import Room, Scene

log = logging.getLogger("scan2scope.stitch")

DOOR_TYPES = ("door", "opening")


# --- 2-D rigid transforms (3x3 homogeneous, plan coordinates) ---------------------------------------------------

def rigid2(theta: float, t: Any = (0.0, 0.0)) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, float(t[0])], [s, c, float(t[1])], [0.0, 0.0, 1.0]])


def apply2(T: np.ndarray, P: Any) -> np.ndarray:
    P = np.asarray(P, float)
    return P @ T[:2, :2].T + T[:2, 2]


def rotate2(T: np.ndarray, v: Any) -> np.ndarray:
    return np.asarray(v, float) @ T[:2, :2].T


def inv2(T: np.ndarray) -> np.ndarray:
    Ti = np.eye(3)
    Ti[:2, :2] = T[:2, :2].T
    Ti[:2, 2] = -T[:2, :2].T @ T[:2, 2]
    return Ti


def yaw2(T: np.ndarray) -> float:
    return float(np.arctan2(T[1, 0], T[0, 0]))


def wrap_angle(a: float, period: float = 2 * np.pi) -> float:
    """Wrap into [-period/2, period/2)."""
    return float((a + period / 2) % period - period / 2)


def snap_yaw(theta: float, m_a: float, m_b: float, tol: float = np.radians(5.0)) -> tuple[float, float, bool]:
    """Snap the rotation of b into a so b's Manhattan frame lands on a's modulo 90 degrees, when within tol."""
    delta = wrap_angle(m_b + theta - m_a, np.pi / 2)
    if abs(delta) <= tol:
        return theta - delta, delta, True
    return theta, delta, False


# --- rooms and hypotheses --------------------------------------------------------------------------------------

@dataclass
class Door:
    """A door or open passage of one room, in that room's plan frame."""

    id: str
    type: str
    center: np.ndarray
    normal: np.ndarray  # unit, pointing into the room
    width: float
    height: float | None
    confidence: float

    @property
    def tangent(self) -> np.ndarray:
        return np.array([self.normal[1], -self.normal[0]])


@dataclass
class StitchRoom:
    index: int
    id: str  # new id, R1..Rn in folder order
    hint: str
    room: Room  # renamed copy, still in the room's own frame
    scene: Scene | None
    manhattan: float  # dominant wall direction modulo 90 degrees, radians
    doors: list[Door]
    centroid: np.ndarray
    aux: dict[str, Any] = field(default_factory=dict)  # per-room caches (point KD-tree)

    def door(self, opening_id: str | None) -> Door | None:
        for d in self.doors:
            if d.id == opening_id:
                return d
        return None


@dataclass
class Hypothesis:
    a: int  # reference room index
    b: int  # room placed relative to a
    T_ab: np.ndarray  # (3, 3) maps b's plan coordinates into a's
    score: float
    source: str  # doorway_photo | door_match
    opening_a: str | None
    opening_b: str | None
    scale_ba: float | None = None  # lengths in b's frame over lengths in a's frame (doorway photo only)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Edge:
    """Merged hypotheses that place room j the same way relative to room i."""

    i: int
    j: int
    T_ij: np.ndarray  # maps j's frame into i's
    score: float
    source: str
    opening_i: str | None
    opening_j: str | None
    members: list[int] = field(default_factory=list)
    scales: list[float] = field(default_factory=list)  # lengths in j over lengths in i, from doorway photos

    def other(self, k: int) -> int:
        return self.j if k == self.i else self.i

    def T_into(self, placed: int) -> np.ndarray:
        """Transform from the other room's frame into `placed`'s frame."""
        return self.T_ij if placed == self.i else inv2(self.T_ij)

    def opening_of(self, k: int) -> str | None:
        return self.opening_i if k == self.i else self.opening_j


@dataclass
class Params:
    shrink: float = 0.02
    max_overlap_m2: float = 0.05
    margin: float = 0.25  # alternatives within this share of the used score count as ambiguous
    min_score: float = 0.08
    weak_score: float = 0.2
    merge_dist: float = 0.3
    merge_angle: float = np.radians(10.0)
    differ_dist: float = 0.25
    differ_angle: float = np.radians(10.0)
    nudge_sigma: float = 0.15
    gap: float = 1.0
    max_forced_runs: int = 200


_NUDGES = sorted(
    ((du, dn) for du in (0.0, -0.1, 0.1, -0.2, 0.2, -0.3, 0.3) for dn in (0.0, 0.06, 0.12, 0.18)
     if (du, dn) != (0.0, 0.0)),
    key=lambda s: (np.hypot(*s), abs(s[0])),
)


def cluster_hypotheses(rooms: list[StitchRoom], hyps: list[Hypothesis], params: Params) -> list[Edge]:
    edges: list[Edge] = []
    by_pair: dict[tuple[int, int], list[Edge]] = {}
    order = sorted(range(len(hyps)), key=lambda k: -hyps[k].score)
    for k in order:
        h = hyps[k]
        if h.a == h.b or not np.isfinite(h.score) or h.score < params.min_score:
            continue
        i, j = sorted((h.a, h.b))
        forward = h.a == i
        T = h.T_ab if forward else inv2(h.T_ab)
        oi, oj = (h.opening_a, h.opening_b) if forward else (h.opening_b, h.opening_a)
        scale = None
        if h.scale_ba is not None and h.scale_ba > 0 and np.isfinite(h.scale_ba):
            scale = h.scale_ba if forward else 1.0 / h.scale_ba
        cj = rooms[j].centroid
        for e in by_pair.setdefault((i, j), []):
            close = np.linalg.norm(apply2(T, cj) - apply2(e.T_ij, cj)) < params.merge_dist
            if close and abs(wrap_angle(yaw2(T) - yaw2(e.T_ij))) < params.merge_angle:
                e.score = min(0.99, 1.0 - (1.0 - e.score) * (1.0 - h.score))
                e.members.append(k)
                if h.source == "doorway_photo":
                    e.source = "doorway_photo"
                if e.opening_i is None and oi is not None:
                    e.opening_i = oi
                if e.opening_j is None and oj is not None:
                    e.opening_j = oj
                if scale is not None:
                    e.scales.append(scale)
                break
        else:
            e = Edge(i, j, T, float(h.score), h.source, oi, oj, [k], [scale] if scale is not None else [])
            by_pair[(i, j)].append(e)
            edges.append(e)
    edges.sort(key=lambda e: (-e.score, e.i, e.j))
    return edges


# --- geometry checks -------------------------------------------------------------------------------------------

def shrunk_polygon(room: StitchRoom, T: np.ndarray, shrink: float):
    P = np.asarray(room.room.polygon, float)
    if P.ndim != 2 or len(P) < 3 or not np.isfinite(P).all():
        return None
    poly = Polygon(apply2(T, P))
    if not poly.is_valid:
        poly = make_valid(poly)
    g = poly.buffer(-shrink, join_style="mitre")
    if g.is_empty or g.area <= 0:
        return None
    return g


def _blockers(poly, polys: dict[int, Any], limit: float) -> list[int]:
    if poly is None:
        return []
    return [r for r, q in polys.items() if q is not None and poly.intersection(q).area > limit]


def _differs(room: StitchRoom, Ta: np.ndarray, Tb: np.ndarray, params: Params) -> bool:
    d = np.linalg.norm(apply2(Ta, room.centroid) - apply2(Tb, room.centroid))
    return bool(d > params.differ_dist or abs(wrap_angle(yaw2(Ta) - yaw2(Tb))) > params.differ_angle)


def _away_direction(rooms: list[StitchRoom], e: Edge, placed: int, child: int, Tc: np.ndarray,
                    T_placed: np.ndarray) -> np.ndarray:
    """Unit vector that moves the child away from its parent across their shared door."""
    d_child = rooms[child].door(e.opening_of(child))
    if d_child is not None:
        return rotate2(Tc, d_child.normal)
    d_parent = rooms[placed].door(e.opening_of(placed))
    if d_parent is not None:
        return -rotate2(T_placed, d_parent.normal)
    v = apply2(Tc, rooms[child].centroid) - apply2(T_placed, rooms[placed].centroid)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


@dataclass
class Tree:
    root: int
    T: dict[int, np.ndarray]
    parent: dict[int, tuple[int, int]]  # child -> (parent room, edge index)
    eff_score: dict[int, float]  # child -> edge score after the nudge penalty
    shift: dict[int, np.ndarray]
    rejected: dict[int, list[int]]  # edge index -> rooms it would overlap
    order: list[int]

    @property
    def total(self) -> float:
        return float(sum(self.eff_score.values()))

    def key(self) -> tuple[int, float]:
        return len(self.T), round(self.total, 9)


def grow(rooms: list[StitchRoom], edges: list[Edge], root: int, members: set[int], params: Params,
         force: dict[int, int] | None = None) -> Tree:
    """Prim's algorithm on edge score with the no-overlap constraint.

    force pins a room to one edge, which is then used as soon as it touches the tree.
    """
    T = {root: np.eye(3)}
    polys = {root: shrunk_polygon(rooms[root], T[root], params.shrink)}
    tree = Tree(root, T, {}, {}, {}, {}, [root])
    force = force or {}
    while True:
        best = None
        for k, e in enumerate(edges):
            if k in tree.rejected or (e.i in T) == (e.j in T):
                continue
            placed, child = (e.i, e.j) if e.i in T else (e.j, e.i)
            if child not in members or (child in force and force[child] != k):
                continue
            if child in force:
                best = (k, placed, child)
                break
            if best is None or e.score > edges[best[0]].score:
                best = (k, placed, child)
        if best is None:
            return tree
        k, placed, child = best
        e = edges[k]
        Tc = T[placed] @ e.T_into(placed)
        poly = shrunk_polygon(rooms[child], Tc, params.shrink)
        blockers = _blockers(poly, polys, params.max_overlap_m2)
        shift = np.zeros(2)
        if blockers:
            n_dir = _away_direction(rooms, e, placed, child, Tc, T[placed])
            u_dir = np.array([-n_dir[1], n_dir[0]])
            for du, dn in _NUDGES:
                s = du * u_dir + dn * n_dir
                T2 = Tc.copy()
                T2[:2, 2] += s
                p2 = shrunk_polygon(rooms[child], T2, params.shrink)
                if not _blockers(p2, polys, params.max_overlap_m2):
                    Tc, poly, shift = T2, p2, s
                    break
            else:
                tree.rejected[k] = blockers
                continue
        T[child] = Tc
        polys[child] = poly
        tree.parent[child] = (placed, k)
        tree.eff_score[child] = float(e.score * np.exp(-0.5 * (np.linalg.norm(shift) / params.nudge_sigma) ** 2))
        if np.linalg.norm(shift) > 0:
            tree.shift[child] = shift
        tree.order.append(child)


def _forcings(rooms: list[StitchRoom], edges: list[Edge], tree: Tree, members: set[int], params: Params,
              min_ratio: float) -> list[tuple[int, int]]:
    """(room, edge) pairs worth forcing: an unused edge that attaches a room to the tree in another place."""
    used = {k for _, k in tree.parent.values()}
    out = []
    for k, e in enumerate(edges):
        if k in used:
            continue
        for x, y in ((e.i, e.j), (e.j, e.i)):
            if x == tree.root or x not in members or y not in tree.T:
                continue
            if x in tree.T and (e.score < min_ratio * tree.eff_score.get(x, 0.0)
                                or not _differs(rooms[x], tree.T[y] @ e.T_into(y), tree.T[x], params)):
                continue
            out.append((x, k))
    return out


def _changed(rooms: list[StitchRoom], a: Tree, b: Tree, params: Params) -> list[int]:
    keys = (set(a.T) | set(b.T)) - {a.root}
    return sorted(r for r in keys if (r in a.T) != (r in b.T) or _differs(rooms[r], a.T[r], b.T[r], params))


def _improve_and_flag(rooms: list[StitchRoom], edges: list[Edge], tree: Tree, members: set[int],
                      params: Params, budget: list[int]) -> tuple[Tree, dict[int, str]]:
    """Single-edge local search, then flag rooms whose placement has a near-equal feasible alternative."""
    for _ in range(10):
        better = None
        for x, k in _forcings(rooms, edges, tree, members, params, 0.5):
            if budget[0] <= 0:
                break
            budget[0] -= 1
            forced = grow(rooms, edges, tree.root, members, params, force={x: k})
            if forced.key() > tree.key():
                better = forced
                break
        if better is None:
            break
        log.info("stitch: local search improved placement (%d rooms, score %.3f)", len(better.T), better.total)
        tree = better
    ambiguous: dict[int, str] = {}
    for x, k in _forcings(rooms, edges, tree, members, params, 1.0 - params.margin):
        if budget[0] <= 0:
            break
        budget[0] -= 1
        forced = grow(rooms, edges, tree.root, members, params, force={x: k})
        if x not in forced.T or len(forced.T) < len(tree.T):
            continue
        if forced.total < (1.0 - params.margin) * tree.total:
            continue
        # the rooms that move (or swap in and out) must be about as well supported as they are now
        changed = _changed(rooms, tree, forced, params)
        base_s = sum(tree.eff_score.get(r, 0.0) for r in changed)
        if not changed or sum(forced.eff_score.get(r, 0.0) for r in changed) < (1.0 - params.margin) * base_s:
            continue
        e = edges[k]
        alt = f"{rooms[e.i].id}-{rooms[e.j].id}:{e.source}:{e.opening_i}/{e.opening_j}:{e.score:.2f}"
        for r in changed:
            if r in tree.T:
                ambiguous.setdefault(r, alt)
    return tree, ambiguous


def _pick_root(cands: set[int], edges: list[Edge]) -> int:
    def key(r: int) -> tuple[int, float, int]:
        inc = [e for e in edges if r in (e.i, e.j) and e.other(r) in cands]
        return len({e.other(r) for e in inc}), sum(e.score for e in inc), -r

    return max(cands, key=key)


@dataclass
class Solution:
    T: dict[int, np.ndarray]  # property frame from room frame, every room
    parent: dict[int, tuple[int, int]]
    eff_score: dict[int, float]
    method: dict[int, str]
    uncertain: dict[int, list[str]]
    shift: dict[int, np.ndarray]
    edges: list[Edge]
    edge_status: dict[int, str]
    components: list[list[int]]
    root: int | None


def _bbox(rooms: list[StitchRoom], T: dict[int, np.ndarray], ids: list[int]) -> np.ndarray | None:
    pts = [apply2(T[r], rooms[r].room.polygon) for r in ids if len(np.asarray(rooms[r].room.polygon)) > 0]
    pts = [p for p in pts if np.isfinite(p).all()]
    if not pts:
        return None
    P = np.vstack(pts)
    return np.array([P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()])


def solve(rooms: list[StitchRoom], hyps: list[Hypothesis], params: Params | None = None) -> Solution:
    params = params or Params()
    edges = cluster_hypotheses(rooms, hyps, params)
    sol = Solution({}, {}, {}, {}, {}, {}, edges, {}, [], None)
    if not rooms:
        return sol
    remaining = set(range(len(rooms)))
    budget = [params.max_forced_runs]
    trees: list[tuple[Tree, dict[int, str]]] = []
    while remaining:
        root = _pick_root(remaining, edges)
        tree = grow(rooms, edges, root, remaining, params)
        tree, ambiguous = _improve_and_flag(rooms, edges, tree, remaining, params, budget)
        trees.append((tree, ambiguous))
        remaining -= set(tree.T)
    if budget[0] <= 0:
        log.warning("stitch: local search budget exhausted, ambiguity checks may be incomplete")

    used_edges: set[int] = set()
    right: float | None = None
    base_y = 0.0
    for ci, (tree, ambiguous) in enumerate(trees):
        # each component is turned so its root's walls are axis aligned and it is wider than tall;
        # later components go to the right
        R0 = rigid2(-rooms[tree.root].manhattan)
        bb = _bbox(rooms, {r: R0 @ tree.T[r] for r in tree.order}, tree.order)
        if bb is not None and bb[3] - bb[1] > bb[2] - bb[0] + 1e-6:
            R0 = rigid2(np.pi / 2) @ R0
        Tc = {r: R0 @ tree.T[r] for r in tree.order}
        bb = _bbox(rooms, Tc, tree.order)
        if bb is not None and right is not None:
            shift = rigid2(0.0, (right + params.gap - bb[0], base_y - bb[1]))
            Tc = {r: shift @ T for r, T in Tc.items()}
            bb = _bbox(rooms, Tc, tree.order)
        if bb is not None:
            if right is None:
                base_y = bb[1]
            right = bb[2] if right is None else max(right, bb[2])
        if ci == 0:
            sol.root = tree.root
        sol.T.update(Tc)
        ids = tree.order
        sol.components.append(list(ids))
        for r in ids:
            reasons = sol.uncertain.setdefault(r, [])
            if r == tree.root:
                sol.method[r] = "root" if ci == 0 else "unplaced"
                if ci > 0:
                    reasons.append("unplaced")
                continue
            p, k = tree.parent[r]
            used_edges.add(k)
            sol.parent[r] = (p, k)
            sol.eff_score[r] = tree.eff_score[r]
            sol.method[r] = edges[k].source
            if ci > 0:
                reasons.append("disconnected_component")
            if r in ambiguous:
                reasons.append(f"ambiguous:{ambiguous[r]}")
            if tree.eff_score[r] < params.weak_score:
                reasons.append("weak_evidence")
            if r in tree.shift:
                sol.shift[r] = rotate2(R0, tree.shift[r])
    bb = _bbox(rooms, sol.T, list(sol.T))
    if bb is not None:
        to_origin = rigid2(0.0, (-bb[0], -bb[1]))
        sol.T = {r: to_origin @ T for r, T in sol.T.items()}
    for r in list(sol.uncertain):
        if not sol.uncertain[r]:
            del sol.uncertain[r]
    rejected = {k for tree, _ in trees for k in tree.rejected}
    for k in range(len(edges)):
        sol.edge_status[k] = "used" if k in used_edges else ("rejected_overlap" if k in rejected else "unused")
    return sol
