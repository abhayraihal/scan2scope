"""Line items from damage regions and concealed-damage flags, keyed to surfaces (scope/catalog.yaml).

Scope runs after the uncertainty stage. Every quantity formula is monotone in its inputs, so it is evaluated
three times: on the measured values, on the lower interval ends and on the upper ends (opening sizes, which are
subtracted, use the opposite end). Missing intervals fall back to the value.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

from scan2scope.types import ConcealedFlag, DamageRegion, LineItem, Measurement, Plan, Room, Wall

log = logging.getLogger("scan2scope.scope")

CATALOG_PATH = Path(__file__).with_name("catalog.yaml")
M2_TO_SF = 10.7639
M_TO_LF = 3.28084
BOUNDS = ("value", "lo", "hi")
FORMULAS = {"damage_area_margin", "surface_area", "crack_length", "flood_cut_length", "flood_cut_area", "each"}


@functools.lru_cache(maxsize=4)
def load_catalog(path: str | None = None) -> dict[str, Any]:
    doc = yaml.safe_load(Path(path or CATALOG_PATH).read_text())
    for entry in doc.get("damage", []) + doc.get("flags", []):
        f = entry.get("quantity", {}).get("formula")
        if f not in FORMULAS:
            raise ValueError(f"catalog entry {entry.get('id')}: unknown formula {f!r}")
        if entry.get("unit") not in ("SF", "LF", "EA"):
            raise ValueError(f"catalog entry {entry.get('id')}: unit must be SF, LF or EA")
    return doc


def at(m: Measurement | None, bound: str, default: float = 0.0) -> float:
    """Value of a measurement at "value", "lo" or "hi", falling back to the value when no interval is set."""
    if m is None:
        return default
    v = getattr(m, bound, None) if bound != "value" else m.value
    return float(m.value if v is None else v)


def flip(bound: str) -> str:
    return {"lo": "hi", "hi": "lo"}.get(bound, bound)


def surface_kind(surface_id: str) -> str:
    if surface_id.endswith("-FLOOR"):
        return "floor"
    if surface_id.endswith("-CEIL"):
        return "ceiling"
    return "wall"


@dataclass
class Surface:
    id: str
    kind: str
    room: Room
    wall: Wall | None

    def area(self, bound: str) -> float:
        if self.kind != "wall":
            return max(0.0, at(self.room.floor_area, bound))
        gross = at(self.wall.length, bound) * at(self.wall.height, bound)
        ob = flip(bound)
        holes = sum(at(o.width, ob) * at(o.height, ob) for o in self.room.openings if o.wall_id == self.wall.id)
        return max(0.0, gross - holes)

    def extent(self, bound: str):
        if self.kind == "wall":
            return box(0.0, 0.0, max(at(self.wall.length, bound), 1e-9), max(at(self.wall.height, bound), 1e-9))
        poly = Polygon(np.asarray(self.room.polygon, float))
        return poly if poly.is_valid else poly.buffer(0)


def _surfaces(plan: Plan) -> tuple[dict[str, Surface], dict[str, tuple[int, int]]]:
    surfaces, order = {}, {}
    for ri, room in enumerate(plan.rooms):
        for wi, wall in enumerate(room.walls):
            surfaces[wall.id] = Surface(wall.id, "wall", room, wall)
            order[wall.id] = (ri, wi)
        for k, (suffix, kind) in enumerate((("FLOOR", "floor"), ("CEIL", "ceiling"))):
            sid = f"{room.id}-{suffix}"
            surfaces[sid] = Surface(sid, kind, room, None)
            order[sid] = (ri, len(room.walls) + k)
    return surfaces, order


def _grown_boxes(regions: list[DamageRegion], margin: float, bound: str) -> list:
    out = []
    for d in regions:
        cu, cv = 0.5 * sum(d.u_range), 0.5 * sum(d.v_range)
        hw = 0.5 * max(at(d.width, bound), 0.0) + margin
        hh = 0.5 * max(at(d.height, bound), 0.0) + margin
        out.append(box(cu - hw, cv - hh, cu + hw, cv + hh))
    return out


def _u_union_length(intervals: list[tuple[float, float]]) -> float:
    total, end = 0.0, -np.inf
    for a, b in sorted(intervals):
        if b <= a:
            continue
        if a > end:
            total += b - a
            end = b
        elif b > end:
            total += b - end
            end = b
    return total


def quantity_m(formula: str, q: dict[str, Any], surface: Surface | None, regions: list[DamageRegion],
               n_flags: int, bound: str) -> float:
    """One formula at one bound, in m, m2 or a count."""
    margin = float(q.get("margin_m", 0.0))
    if formula == "each":
        count = q.get("count", "region")
        return float(len(regions) if count == "region" else (n_flags if count == "flag" else 1))
    if formula == "surface_area":
        return surface.area(bound) if surface is not None else 0.0
    if formula == "crack_length":
        return sum(max(at(d.length, bound) if d.length is not None else max(at(d.width, bound), at(d.height, bound)),
                       0.0) + margin for d in regions)
    if formula == "damage_area_margin":
        shapes = unary_union(_grown_boxes(regions, margin, bound))
        if surface is not None:
            shapes = shapes.intersection(surface.extent(bound))
        return float(shapes.area)
    if formula in ("flood_cut_length", "flood_cut_area"):
        lim = at(surface.wall.length, bound) if surface is not None and surface.wall is not None else np.inf
        iv = []
        for d in regions:
            cu = 0.5 * sum(d.u_range)
            half = 0.5 * max(at(d.width, bound), 0.0) + margin
            iv.append((max(0.0, cu - half), min(lim, cu + half)))
        length = _u_union_length(iv)
        return length * float(q.get("cut_height_m", 0.6)) if formula == "flood_cut_area" else length
    raise ValueError(f"unknown formula {formula}")


def _to_unit(x: float, unit: str) -> float:
    return x * M2_TO_SF if unit == "SF" else (x * M_TO_LF if unit == "LF" else x)


def _matches(entry: dict[str, Any], d: DamageRegion) -> bool:
    if d.cls not in entry.get("classes", []):
        return False
    if surface_kind(d.surface_id) not in entry.get("surfaces", ["wall", "floor", "ceiling"]):
        return False
    area = at(d.area, "value")
    if "max_area_m2" in entry and area > float(entry["max_area_m2"]):
        return False
    if "min_area_m2" in entry and area <= float(entry["min_area_m2"]):
        return False
    return True


def _item(entry: dict[str, Any], surface_id: str, room_id: str, surface: Surface | None,
          regions: list[DamageRegion], flags: list[ConcealedFlag]) -> LineItem | None:
    q = entry["quantity"]
    formula = q["formula"]
    if formula == "surface_area" and surface is None:
        return None
    unit = entry["unit"]
    vals = {b: _to_unit(quantity_m(formula, q, surface, regions, len(flags), b), unit) for b in BOUNDS}
    value = vals["value"]
    if value <= 0:
        return None
    lo, hi = min(vals["lo"], value), max(vals["hi"], value)
    rules = list(dict.fromkeys(f.rule_id for f in flags))
    return LineItem(
        id="", room_id=room_id, surface_id=surface_id, category=str(entry["category"]),
        selector=str(entry["selector"]), activity=str(entry["activity"]), description=str(entry["description"]),
        quantity=Measurement(value=value, lo=lo, hi=hi, unit=unit, kind="quantity",
                             evidence={"catalog_id": entry["id"], "formula": formula,
                                       **{k: v for k, v in q.items() if k != "formula"}}),
        unit=unit, damage_ids=[d.id for d in regions], flag_ids=[f.id for f in flags],
        rule_id=",".join(rules) if rules else None)


def generate(plan: Plan, damage: list[DamageRegion], flags: list[ConcealedFlag], *,
             catalog_path: str | Path | None = None) -> list[LineItem]:
    """Line items L1.. ordered by room, surface and catalog entry."""
    catalog = load_catalog(str(catalog_path) if catalog_path else None)
    surfaces, order = _surfaces(plan)
    by_id = {d.id: d for d in damage}
    raw: list[tuple[tuple, LineItem]] = []
    for ei, entry in enumerate(catalog.get("damage", [])):
        groups: dict[str, list[DamageRegion]] = {}
        for d in damage:
            if _matches(entry, d):
                groups.setdefault(d.surface_id, []).append(d)
        for sid, regions in groups.items():
            try:
                item = _item(entry, sid, regions[0].room_id, surfaces.get(sid), regions, [])
            except Exception as exc:  # odd geometry on one surface must not drop the whole scope
                log.warning("scope entry %s skipped on %s: %s", entry["id"], sid, exc)
                continue
            if item is not None:
                raw.append(((order.get(sid, (len(plan.rooms), 0)), 0, ei), item))
    for ei, entry in enumerate(catalog.get("flags", [])):
        groups_f: dict[str, tuple[list[ConcealedFlag], list[DamageRegion]]] = {}
        for f in flags:
            if f.rule_id not in entry.get("rules", []):
                continue
            per_flag = entry["quantity"].get("count") == "flag"
            for sid in (f.surface_ids[:1] if per_flag else f.surface_ids):
                fl, regs = groups_f.setdefault(sid, ([], []))
                fl.append(f)
                have = {d.id for d in regs}
                regs += [by_id[i] for i in dict.fromkeys(f.damage_ids)
                         if i in by_id and i not in have and (per_flag or by_id[i].surface_id == sid)]
        for sid, (fl, regs) in groups_f.items():
            room_id = regs[0].room_id if regs else fl[0].room_id
            try:
                item = _item(entry, sid, room_id, surfaces.get(sid), regs, fl)
            except Exception as exc:
                log.warning("scope entry %s skipped on %s: %s", entry["id"], sid, exc)
                continue
            if item is not None:
                raw.append(((order.get(sid, (len(plan.rooms), 0)), 1, ei), item))
    raw.sort(key=lambda x: x[0])
    items = []
    for i, (_, it) in enumerate(raw, 1):
        it.id = f"L{i}"
        items.append(it)
    return items
