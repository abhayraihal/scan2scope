"""Floor-plan drawings from a result dict: plan.svg, plan.png (150 dpi) and rooms/<room id>.svg.

Drawn from the result alone, so a saved result.json can be re-rendered. Plans are at a fixed scale in plan
metres (PT_PER_M, capped for large properties). Uses the matplotlib object API on the Agg canvas, no pyplot.
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
from matplotlib import rc_context
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Arc, Patch, Rectangle
from matplotlib.patches import Polygon as PolyPatch
from shapely.errors import GEOSException
from shapely.geometry import Point
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.ops import polylabel

from scan2scope.output.console import _f, drift_state, fmt_iv, fmt_pm, mvals, short_id

log = logging.getLogger("scan2scope.output")

PT_PER_M = 100.0
MAX_SIDE_IN = 30.0
MIN_WIDTH_IN = 9.5
MARGIN_M = 1.1
WALL_T = 0.09  # drawn wall thickness, outside the measured interior face
PNG_DPI = 150
FILLS = ("#f6e7c8", "#d8ead3", "#d5e3f2", "#f3d9dc", "#e3dcf0", "#fdf3c4", "#d4eeee", "#ecdcc8")
INK = "#262626"
SOFT = "#5a5a5a"
DAMAGE = "#d0021b"
UNCERTAIN = "#c26a00"
WINDOW_FILL = "#e3f0fb"
RC = {"svg.fonttype": "none", "font.family": "DejaVu Sans", "hatch.linewidth": 0.6}
LABEL_BOX = {"boxstyle": "square,pad=0.12", "fc": "white", "ec": "none", "alpha": 0.8}


@dataclass
class _Wall:
    id: str
    short: str
    p0: np.ndarray
    p1: np.ndarray
    L: float
    u: np.ndarray
    n_in: np.ndarray
    length: dict | None
    gaps: list[tuple[float, float, dict]] = field(default_factory=list)

    def at(self, s: float, off: float = 0.0) -> np.ndarray:
        """Point s metres along the wall from its start and off metres along the inward normal."""
        return self.p0 + self.u * s + self.n_in * off


@dataclass
class _Room:
    id: str
    label: str
    poly: np.ndarray
    shape: Any
    walls: list[_Wall]
    fill: str
    uncertain: bool
    data: dict


def _inside(shape: Any, p: np.ndarray) -> bool:
    try:
        return bool(shape.contains(Point(float(p[0]), float(p[1]))))
    except (GEOSException, ValueError, TypeError):
        return False


def _signed_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y))


def _uncertain_ids(result: dict[str, Any]) -> set[str]:
    rooms = [r for r in result.get("rooms") or [] if isinstance(r, dict)]
    names = {str(x): str(r.get("id")) for r in rooms for x in (r.get("id"), r.get("label"), r.get("source_hint")) if x}
    out = {str(r.get("id")) for r in rooms if any(str(f).startswith("placement_uncertain") for f in r.get("flags") or [])}
    for f in (result.get("property") or {}).get("flags") or []:
        name, _, target = str(f).partition(":")
        if name == "placement_uncertain" and target in names:
            out.add(names[target])
    return out


def _parse_room(room: dict[str, Any], fill: str, uncertain: bool) -> _Room:
    rid = str(room.get("id"))
    try:
        poly = np.asarray(room.get("polygon") or [], float).reshape(-1, 2)
    except (TypeError, ValueError):
        poly = np.zeros((0, 2))
    poly = poly[np.isfinite(poly).all(1)]
    try:
        shape = ShapelyPolygon(poly).buffer(0) if len(poly) >= 3 else ShapelyPolygon()
    except (GEOSException, ValueError, TypeError):
        shape = ShapelyPolygon()
    walls = []
    for w in room.get("walls") or []:
        try:
            p0, p1 = np.asarray(w["start"], float), np.asarray(w["end"], float)
        except (KeyError, TypeError, ValueError):
            continue
        d = p1 - p0
        length = float(np.hypot(d[0], d[1]))
        if not np.isfinite(length) or length < 1e-6:
            continue
        u = d / length
        left = np.array([-u[1], u[0]])
        mid = (p0 + p1) / 2
        if _inside(shape, mid + 0.05 * left):
            n_in = left
        elif _inside(shape, mid - 0.05 * left):
            n_in = -left
        else:
            n_in = left if _signed_area(poly) >= 0 else -left
        walls.append(_Wall(str(w.get("id")), short_id(w.get("id"), rid), p0, p1, length, u, n_in, w.get("length")))
    by_id = {w.id: w for w in walls}
    for o in room.get("openings") or []:
        wall = by_id.get(str(o.get("wall_id")))
        off, width = mvals(o.get("offset")), mvals(o.get("width"))
        if wall is None or off is None or width is None:
            continue
        a, b = max(0.0, off[0]), min(wall.L, off[0] + width[0])
        if b - a > 0.02:
            wall.gaps.append((a, b, o))
    return _Room(rid, str(room.get("label") or rid), poly, shape, walls, fill, uncertain, room)


class _Labels:
    """Greedy label placement: the first candidate position whose box is free wins."""

    def __init__(self, ax: Any, pt_per_m: float) -> None:
        self.ax, self.k, self.boxes = ax, 1.0 / pt_per_m, []

    def size(self, text: str, fs: float) -> tuple[float, float]:
        lines = text.split("\n")
        return max(len(s) for s in lines) * 0.6 * fs * self.k, len(lines) * 1.3 * fs * self.k

    @staticmethod
    def box(xy: np.ndarray, w: float, h: float, ang: float) -> tuple[float, float, float, float]:
        c, s = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        dx, dy = abs(c) * w / 2 + abs(s) * h / 2, abs(s) * w / 2 + abs(c) * h / 2
        return xy[0] - dx, xy[1] - dy, xy[0] + dx, xy[1] + dy

    def free(self, b: tuple[float, float, float, float], pad: float = 0.02) -> bool:
        return all(b[2] + pad <= o[0] or o[2] + pad <= b[0] or b[3] + pad <= o[1] or o[3] + pad <= b[1]
                   for o in self.boxes)

    def block(self, b: tuple[float, float, float, float]) -> None:
        self.boxes.append(b)

    def place(self, cands: list[np.ndarray], text: str, fs: float, ang: float = 0.0, **kw: Any) -> None:
        w, h = self.size(text, fs)
        boxes = [(xy, self.box(xy, w, h, ang)) for xy in cands]
        xy, b = next(((xy, b) for xy, b in boxes if self.free(b)), boxes[0])
        self.boxes.append(b)
        self.ax.text(float(xy[0]), float(xy[1]), text, fontsize=fs, rotation=ang, rotation_mode="anchor",
                     ha="center", va="center", zorder=8, **kw)


def _upright(u: np.ndarray) -> float:
    ang = math.degrees(math.atan2(u[1], u[0]))
    if ang > 90:
        ang -= 180
    elif ang <= -90:
        ang += 180
    return ang


def _quad(ax: Any, pts: list[np.ndarray], **kw: Any) -> None:
    ax.add_patch(PolyPatch(np.array(pts), closed=True, **kw))


def _line(ax: Any, a: np.ndarray, b: np.ndarray, **kw: Any) -> None:
    ax.add_line(Line2D([a[0], b[0]], [a[1], b[1]], **kw))


def _extends(room: _Room, vertex: np.ndarray, direction: np.ndarray, n_out: np.ndarray) -> bool:
    """Whether the band should run past this corner: only at convex corners, where the square is outside."""
    return not _inside(room.shape, vertex + direction * WALL_T / 2 + n_out * WALL_T / 2)


def _draw_walls(ax: Any, room: _Room, labels: _Labels) -> None:
    colour = SOFT if room.uncertain else INK
    for w in room.walls:
        n_out = -w.n_in
        pieces, cur = [], 0.0
        for a, b, _ in sorted(w.gaps, key=lambda g: g[0]):
            if a > cur:
                pieces.append((cur, a))
            cur = max(cur, b)
        if cur < w.L:
            pieces.append((cur, w.L))
        for a, b in pieces:
            a2 = a - WALL_T if a <= 1e-6 and _extends(room, w.p0, -w.u, n_out) else a
            b2 = b + WALL_T if b >= w.L - 1e-6 and _extends(room, w.p1, w.u, n_out) else b
            _quad(ax, [w.at(a2), w.at(b2), w.at(b2, -WALL_T), w.at(a2, -WALL_T)], facecolor=colour,
                  edgecolor=colour, lw=0.3, zorder=4)
            # Label obstacles: one box for an axis-aligned band, short boxes along a slanted one.
            n_seg = 1 if (abs(w.u[0]) < 0.09 or abs(w.u[1]) < 0.09) else max(1, math.ceil((b2 - a2) / 0.25))
            for s0, s1 in itertools.pairwise(np.linspace(a2, b2, n_seg + 1)):
                arr = np.array([w.at(s0), w.at(s1), w.at(s1, -WALL_T), w.at(s0, -WALL_T)])
                labels.block((*arr.min(0), *arr.max(0)))


def _draw_opening(ax: Any, w: _Wall, a: float, b: float, op: dict, swing: bool, labels: _Labels) -> None:
    kind = op.get("type")
    for s in (a, b):
        _line(ax, w.at(s), w.at(s, -WALL_T), color=INK, lw=0.6, zorder=5)
    if kind == "window":
        _quad(ax, [w.at(a), w.at(b), w.at(b, -WALL_T), w.at(a, -WALL_T)], facecolor=WINDOW_FILL, edgecolor="none",
              zorder=5)
        for off in (0.0, -WALL_T):
            _line(ax, w.at(a, off), w.at(b, off), color=INK, lw=0.7, zorder=5)
    elif kind == "door" and swing:
        r = b - a
        hinge = w.at(a)
        _line(ax, hinge, w.at(a, r), color=INK, lw=1.0, zorder=5)
        t1 = math.degrees(math.atan2(w.u[1], w.u[0]))
        t2 = math.degrees(math.atan2(w.n_in[1], w.n_in[0]))
        if (t2 - t1) % 360 > 180:
            t1, t2 = t2, t1
        ax.add_patch(Arc((float(hinge[0]), float(hinge[1])), 2 * r, 2 * r, theta1=t1, theta2=t2, color=INK, lw=0.6,
                         zorder=5))
        corners = np.array([w.at(a), w.at(b), w.at(b, r), w.at(a, r)])
        labels.block((*corners.min(0), *corners.max(0)))


def _swing_skip(result: dict[str, Any]) -> set[str]:
    """Second opening of each adjacency pair: the shared door is drawn swinging into one room only."""
    skip = set()
    for a in (result.get("property") or {}).get("adjacency") or []:
        if isinstance(a, dict) and a.get("opening_a") and a.get("opening_b"):
            skip.add(str(a["opening_b"]))
    return skip


def _surface_damage(ax: Any, d: dict, labels: _Labels, labels_todo: list) -> None:
    u0, u1 = sorted((_f(d["u_range"][0], 0.0), _f(d["u_range"][1], 0.0)))
    v0, v1 = sorted((_f(d["v_range"][0], 0.0), _f(d["v_range"][1], 0.0)))
    labels.block((u0, v0, u1, v1))
    ceiling = str(d.get("surface_id", "")).endswith("-CEIL")
    if ceiling:
        ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, facecolor="none", edgecolor=DAMAGE, hatch="////",
                               linestyle="--", lw=0.9, zorder=3))
    else:
        ax.add_patch(Rectangle((u0, v0), u1 - u0, v1 - v0, facecolor=DAMAGE, alpha=0.35, edgecolor=DAMAGE, lw=0.9,
                               zorder=3))
    text = f"{d.get('id')} {str(d.get('class', '')).replace('_', ' ')}" + (" (ceiling)" if ceiling else "")
    c = np.array([(u0 + u1) / 2, (v0 + v1) / 2])
    h = (v1 - v0) / 2 + 0.12
    labels_todo.append(([c + [0, h], c - [0, h], c], text))


def _wall_damage(ax: Any, w: _Wall, d: dict, labels: _Labels, labels_todo: list) -> None:
    u0, u1 = sorted((_f(d["u_range"][0], 0.0), _f(d["u_range"][1], 0.0)))
    u0, u1 = max(0.0, u0), min(w.L, u1)
    if u1 - u0 < 0.04:
        c = (u0 + u1) / 2
        u0, u1 = c - 0.02, c + 0.02
    strip = np.array([w.at(u0), w.at(u1), w.at(u1, 0.07), w.at(u0, 0.07)])
    _quad(ax, list(strip), facecolor=DAMAGE, edgecolor=DAMAGE, lw=0.3, zorder=6)
    labels.block((*strip.min(0), *strip.max(0)))
    text = f"{d.get('id')} {str(d.get('class', '')).replace('_', ' ')}"
    c = (u0 + u1) / 2
    labels_todo.append(([w.at(c, 0.2), w.at(c, 0.35), w.at(c + 0.4, 0.2), w.at(c - 0.4, 0.2)], text))


def _room_label(ax: Any, room: _Room, labels: _Labels) -> None:
    try:
        p = polylabel(room.shape, tolerance=0.05) if room.shape.area > 0 else None
        xy = np.array([p.x, p.y]) if p is not None else room.poly.mean(0)
    except (GEOSException, ValueError, TypeError):
        xy = room.poly.mean(0) if len(room.poly) else np.zeros(2)
    lines = [(f"{room.id} {room.label}", 8.0, {"fontweight": "bold", "color": INK})]
    fa = mvals(room.data.get("floor_area"))
    if fa is not None:
        lines.append((f"{fa[0]:.2f} m² [{fa[1]:.2f}, {fa[2]:.2f}]", 6.8, {"color": INK}))
    if mvals(room.data.get("ceiling_height")) is not None:
        lines.append((f"ceiling {fmt_pm(room.data.get('ceiling_height'))}", 6.8, {"color": INK}))
    if room.uncertain:
        lines.append(("placement uncertain", 6.5, {"color": UNCERTAIN, "fontstyle": "italic"}))
    sizes = [labels.size(text, fs) for text, fs, _ in lines]
    bw, bh = max(w for w, _ in sizes), sum(h for _, h in sizes)
    # The block moves off damage and door swings when it can, but stays inside the room.
    centre = xy
    for dx, dy in ((0, 0), (0, 0.8), (0, -0.8), (0.45, 0), (-0.45, 0), (0, 1.6), (0, -1.6), (0.45, 0.8),
                   (-0.45, -0.8)):
        c = xy + np.array([dx * bw, dy * bh])
        inside = (dx, dy) == (0, 0) or _inside(room.shape, c)
        if inside and labels.free((c[0] - bw / 2, c[1] - bh / 2, c[0] + bw / 2, c[1] + bh / 2)):
            centre = c
            break
    y = centre[1] + bh / 2
    for (text, fs, kw), (_, h) in zip(lines, sizes):
        y -= h / 2
        labels.place([np.array([centre[0], y])], text, fs, **kw)
        y -= h / 2


def _wall_dims(ax: Any, room: _Room, labels: _Labels, others: list[Any]) -> None:
    fs = 6.5
    for w in room.walls:
        if mvals(w.length) is None:
            continue
        text = f"{w.short} {fmt_pm(w.length)}"
        _, h = labels.size(text, fs)
        mid = (w.p0 + w.p1) / 2
        probe = mid - w.n_in * (WALL_T + 0.3)
        sides = [(w.n_in, 0.06 + h / 2)]
        if not any(_inside(s, probe) for s in others):
            sides.insert(0, (-w.n_in, WALL_T + 0.05 + h / 2))  # outside first, own room as the fallback
        cands = [mid + n * (base + k * h * 1.1) + w.u * s for n, base in sides
                 for s in (0.0, 0.22 * w.L, -0.22 * w.L) for k in range(4)]
        labels.place(cands, text, fs, _upright(w.u), color=INK, bbox=LABEL_BOX)


def _opening_labels(room: _Room, labels: _Labels) -> None:
    for w in room.walls:
        for a, b, o in w.gaps:
            width = mvals(o.get("width"))
            text = f"{short_id(o.get('id'), room.id)} {width[0]:.2f}" if width else short_id(o.get("id"), room.id)
            c = (a + b) / 2
            r = b - a
            cands = [w.at(c, 0.16), w.at(c - r / 2 - 0.3, 0.16), w.at(c + r / 2 + 0.3, 0.16), w.at(c, r + 0.15),
                     w.at(c, 0.4)]
            labels.place(cands, text, 6.0, _upright(w.u), color=SOFT, bbox=LABEL_BOX)


def _scale_bar(ax: Any, x0: float, y0: float, labels: _Labels) -> None:
    x, y, h = x0 + 0.35, y0 + 0.35, 0.06
    ax.add_patch(Rectangle((x, y), 0.5, h, facecolor=INK, edgecolor=INK, lw=0.6, zorder=7))
    ax.add_patch(Rectangle((x + 0.5, y), 0.5, h, facecolor="white", edgecolor=INK, lw=0.6, zorder=7))
    ax.text(x, y - 0.04, "0", ha="center", va="top", fontsize=6.5, color=INK)
    ax.text(x + 1.0, y - 0.04, "1 m", ha="center", va="top", fontsize=6.5, color=INK)
    labels.block((x - 0.1, y - 0.2, x + 1.2, y + h + 0.05))


def _legend_handles() -> list[Any]:
    return [
        Patch(facecolor=INK, edgecolor=INK, label="wall"),
        Line2D([0], [0], color=INK, lw=1.0, label="door and swing"),
        Patch(facecolor=WINDOW_FILL, edgecolor=INK, lw=0.7, label="window"),
        Patch(facecolor="white", edgecolor=SOFT, lw=0.7, label="opening, no door"),
        Patch(facecolor=DAMAGE, edgecolor=DAMAGE, label="damage on wall"),
        Patch(facecolor=DAMAGE, alpha=0.35, edgecolor=DAMAGE, label="damage on floor"),
        Patch(facecolor="none", edgecolor=DAMAGE, hatch="////", linestyle="--", label="damage on ceiling"),
        Patch(facecolor="none", edgecolor=UNCERTAIN, linestyle="--", lw=1.2, label="placement uncertain"),
    ]


def _info_lines(result: dict[str, Any], room: dict[str, Any] | None) -> list[str]:
    cap = result.get("capture") or {}
    interval = (result.get("conventions") or {}).get("interval") or {}
    prop = result.get("property") or {}
    q = _f(interval.get("q"))
    qtxt = f", q {q:.2f} ({interval.get('calibration', 'prior')})" if q is not None else ""
    ex, ey = mvals(prop.get("extent_x")), mvals(prop.get("extent_y"))
    extents = f"   extents {ex[0]:.2f} x {ey[0]:.2f} m" if ex and ey else ""
    lines = [f"capture {cap.get('id', '?')}   tier {cap.get('tier', '?')}",
             f"intervals {(_f(interval.get('level'), 0.9) or 0.9):.0%} nominal{qtxt}; labels give value ± half-width",
             f"footprint {fmt_iv(prop.get('footprint_area'), 2)} m²{extents}",
             f"drift correction: {drift_state(result)}"]
    if room is None:
        rooms = [r for r in result.get("rooms") or [] if isinstance(r, dict)]
        n_open = sum(len(r.get("openings") or []) for r in rooms)
        lines.append(f"{len(rooms)} rooms, {n_open} openings, {len(result.get('damage') or [])} damage regions, "
                     f"{len(result.get('concealed_damage_flags') or [])} concealed-damage flags")
        flags = [str(f) for f in prop.get("flags") or []]
        if flags:
            text = ", ".join(flags)
            lines.append("flags: " + (text if len(text) <= 80 else text[:77] + "..."))
    return lines


def _sheet_lines(result: dict[str, Any], room: dict[str, Any]) -> list[str]:
    rid = room.get("id")
    out = [(f"floor area {fmt_iv(room.get('floor_area'), 2)} m2   ceiling {fmt_iv(room.get('ceiling_height'))} m   "
            f"perimeter {fmt_iv(room.get('perimeter'))} m")]
    for w in room.get("walls") or []:
        out.append(f"{w.get('id')!s:<10} length {fmt_iv(w.get('length'))}   height {fmt_iv(w.get('height'))}")
    for o in room.get("openings") or []:
        sill = f"   sill {fmt_iv(o.get('sill'))}" if o.get("sill") else ""
        out.append(f"{o.get('id')!s:<10} {o.get('type')!s:<7} in {short_id(o.get('wall_id'), rid):<4} "
                   f"width {fmt_iv(o.get('width'))}   height {fmt_iv(o.get('height'))}{sill}")
    for d in result.get("damage") or []:
        if isinstance(d, dict) and d.get("room_id") == rid:
            out.append(f"{d.get('id')!s:<10} {d.get('class')!s:<13} on {d.get('surface_id')}   "
                       f"area {fmt_iv(d.get('area'), 3)} m2")
    return out


def _figure(result: dict[str, Any], shown: list[_Room], others: list[Any], *, sheet: dict[str, Any] | None,
            skip_swing: set[str]) -> Figure:
    pts = [r.poly for r in shown if len(r.poly)] + [np.array([w.p0, w.p1]) for r in shown for w in r.walls]
    if pts:
        allp = np.vstack(pts)
        (bx0, by0), (bx1, by1) = allp.min(0), allp.max(0)
    else:
        bx0, by0, bx1, by1 = 0.0, 0.0, 4.0, 3.0
    x0, y0, x1, y1 = bx0 - MARGIN_M, by0 - MARGIN_M, bx1 + MARGIN_M, by1 + MARGIN_M
    w_m, h_m = x1 - x0, y1 - y0
    S = min(PT_PER_M, MAX_SIDE_IN * 72.0 / max(w_m, h_m))
    plan_w, plan_h = w_m * S / 72.0, h_m * S / 72.0
    fig_w = max(plan_w, MIN_WIDTH_IN)
    if fig_w > plan_w:
        pad = (fig_w * 72.0 / S - w_m) / 2
        x0, x1 = x0 - pad, x1 + pad

    info = _info_lines(result, sheet)
    mono = _sheet_lines(result, sheet) if sheet is not None else []
    band = max(1.05, 0.42 + len(info) * 0.141 + (len(mono) * 0.122 + 0.08 if mono else 0.0) + 0.12)
    fig_h = plan_h + band
    fig = Figure(figsize=(fig_w, fig_h), facecolor="white")
    ax = fig.add_axes((0.0, band / fig_h, 1.0, plan_h / fig_h))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal")
    ax.set_axis_off()
    labels = _Labels(ax, S)

    for r in shown:
        if len(r.poly) >= 3:
            ax.add_patch(PolyPatch(r.poly, closed=True, facecolor=r.fill, edgecolor="none", zorder=1))
    by_wall = {w.id: (r, w) for r in shown for w in r.walls}
    shown_ids = {r.id for r in shown}
    damage_labels: list = []
    for d in result.get("damage") or []:
        if not isinstance(d, dict) or not isinstance(d.get("u_range"), list) or not isinstance(d.get("v_range"), list):
            continue
        sid = str(d.get("surface_id"))
        if sid in by_wall:
            _wall_damage(ax, by_wall[sid][1], d, labels, damage_labels)
        elif str(d.get("room_id")) in shown_ids and sid.endswith(("-FLOOR", "-CEIL")):
            _surface_damage(ax, d, labels, damage_labels)
    for r in shown:
        _draw_walls(ax, r, labels)
        for w in r.walls:
            for a, b, o in w.gaps:
                _draw_opening(ax, w, a, b, o, str(o.get("id")) not in skip_swing, labels)
        if r.uncertain and r.shape.area > 0:
            grown = r.shape.buffer(WALL_T + 0.05, join_style="mitre")
            for g in getattr(grown, "geoms", [grown]):
                ax.add_patch(PolyPatch(np.asarray(g.exterior.coords), closed=True, facecolor="none",
                                       edgecolor=UNCERTAIN, linestyle="--", lw=1.2, zorder=6))
    for r in shown:
        _room_label(ax, r, labels)
    for r in shown:
        _wall_dims(ax, r, labels, [s for s in others if s is not r.shape])
    for r in shown:
        _opening_labels(r, labels)
    for cands, text in damage_labels:
        labels.place(cands, text, 6.0, color=DAMAGE, bbox=LABEL_BOX)
    if not shown:
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, "no rooms in this result", ha="center", va="center", fontsize=10,
                color=SOFT)
    _scale_bar(ax, x0, y0, labels)

    t = fig.dpi_scale_trans
    title = f"{sheet.get('id')} {sheet.get('label')}" if sheet is not None else "scan2scope floor plan"
    fig.add_artist(Line2D([0.12, fig_w - 0.12], [band, band], transform=t, color="#bbbbbb", lw=0.6))
    fig.text(0.15, band - 0.1, title, transform=t, fontsize=9.5, fontweight="bold", va="top", color=INK)
    fig.text(0.15, band - 0.32, "\n".join(info), transform=t, fontsize=7, va="top", linespacing=1.45, color=INK)
    if mono:
        fig.text(0.15, band - 0.36 - len(info) * 0.141, "\n".join(mono), transform=t, fontsize=6.5,
                 family="DejaVu Sans Mono", va="top", linespacing=1.35, color=INK)
    fig.legend(handles=_legend_handles(), loc="upper right", bbox_to_anchor=(fig_w - 0.1, band - 0.06),
               bbox_transform=t, fontsize=6.5, frameon=False, ncol=2, handlelength=2.0, columnspacing=1.0)
    return fig


def _error_figure(message: str) -> Figure:
    fig = Figure(figsize=(7, 1.5), facecolor="white")
    fig.text(0.03, 0.5, f"plan could not be drawn: {message}"[:180], fontsize=9, va="center", color=INK)
    return fig


def _safe_name(ident: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(ident)) or "room"


def render_all(result: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    """Write plan.svg, plan.png and rooms/<id>.svg; a drawing that fails is replaced by a note and logged."""
    out = Path(out_dir)
    (out / "rooms").mkdir(parents=True, exist_ok=True)
    rooms_raw = [r for r in result.get("rooms") or [] if isinstance(r, dict)]
    uncertain = _uncertain_ids(result)
    errors: list[str] = []
    files: list[Path] = []
    with rc_context(RC):
        parsed: list[_Room] = []
        for i, r in enumerate(rooms_raw):
            try:
                parsed.append(_parse_room(r, FILLS[i % len(FILLS)], str(r.get("id")) in uncertain))
            except Exception as exc:  # noqa: BLE001 - skip the room, keep drawing the rest
                errors.append(f"room {r.get('id')}: {exc}")
                log.warning("could not draw room %s: %s", r.get("id"), exc)
        shapes = [r.shape for r in parsed]
        jobs = [("plan", parsed, None, _swing_skip(result))] + [(r.id, [r], r.data, set()) for r in parsed]
        for name, shown, sheet, skip in jobs:
            if sheet is None:
                targets = [(out / "plan.svg", "svg"), (out / "plan.png", "png")]
            else:
                targets = [(out / "rooms" / f"{_safe_name(name)}.svg", "svg")]
            try:  # matplotlib draws inside savefig, so saving is part of the guarded work
                fig = _figure(result, shown, shapes if sheet is None else [], sheet=sheet, skip_swing=skip)
                for path, fmt in targets:
                    fig.savefig(path, format=fmt, dpi=PNG_DPI, facecolor="white")
            except Exception as exc:  # a drawing that fails must not stop the run
                errors.append(f"{name}: {type(exc).__name__}: {exc}")
                log.warning("drawing %s failed: %s", name, exc, exc_info=log.isEnabledFor(logging.DEBUG))
                fig = _error_figure(f"{type(exc).__name__}: {exc}")
                for path, fmt in targets:
                    fig.savefig(path, format=fmt, dpi=PNG_DPI, facecolor="white")
            files += [path for path, _ in targets]
    return {"files": files, "errors": errors}
