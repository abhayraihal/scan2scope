"""Evaluate the concealed-damage rules in concealed_damage.yaml against the plan, damage and objects.

Conditions are plain Python predicates keyed by the names used in the YAML. Rules run before the uncertainty
stage, so they compare point values; each flag records the measured inputs that fired it.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from shapely.geometry import LineString, Point, box

from scan2scope.types import ConcealedFlag, DamageRegion, Opening, Plan, Room, Wall

log = logging.getLogger("scan2scope.rules")

RULES_PATH = Path(__file__).with_name("concealed_damage.yaml")
SQFT_PER_M2 = 10.7639
KNOWN_CONDITIONS = {"classes", "surfaces", "bottom_within_m", "near_fixture", "near_window", "near_opening_corner",
                    "min_length_m", "total_area_sqft"}


@functools.lru_cache(maxsize=4)
def load_rules(path: str | None = None) -> tuple[dict[str, Any], ...]:
    doc = yaml.safe_load(Path(path or RULES_PATH).read_text())
    rules = tuple(doc.get("rules", []))
    for r in rules:
        unknown = set(r.get("when", {})) - KNOWN_CONDITIONS
        if unknown:
            raise ValueError(f"rule {r.get('id')}: unknown conditions {sorted(unknown)}")
    return rules


@dataclass
class Prism:
    """A plan shape extruded over a height interval; distances between prisms are exact."""

    shape: Any  # shapely geometry in plan coordinates
    z0: float
    z1: float

    def distance(self, other: Prism) -> float:
        d_xy = float(self.shape.distance(other.shape))
        d_z = max(0.0, max(self.z0, other.z0) - min(self.z1, other.z1))
        return float(np.hypot(d_xy, d_z))


def _val(m: Any, default: float = 0.0) -> float:
    if m is None:
        return default
    return float(getattr(m, "value", m))


def _attr(o: Any, name: str, default: Any = None) -> Any:
    return o.get(name, default) if isinstance(o, dict) else getattr(o, name, default)


def surface_kind(surface_id: str) -> str:
    if surface_id.endswith("-FLOOR"):
        return "floor"
    if surface_id.endswith("-CEIL"):
        return "ceiling"
    return "wall"


class PlanIndex:
    def __init__(self, plan: Plan) -> None:
        self.plan = plan
        self.rooms: dict[str, Room] = {r.id: r for r in plan.rooms}
        self.room_order = {r.id: i for i, r in enumerate(plan.rooms)}
        self.walls: dict[str, tuple[Room, Wall]] = {w.id: (r, w) for r in plan.rooms for w in r.walls}

    def wall_axis(self, wall: Wall) -> tuple[np.ndarray, np.ndarray, float]:
        s = np.asarray(wall.start, float)
        e = np.asarray(wall.end, float)
        length = float(np.linalg.norm(e - s))
        return s, (e - s) / max(length, 1e-12), length

    def damage_prism(self, d: DamageRegion) -> Prism | None:
        u0, u1 = sorted(float(x) for x in d.u_range)
        v0, v1 = sorted(float(x) for x in d.v_range)
        kind = surface_kind(d.surface_id)
        if kind == "wall":
            hit = self.walls.get(d.surface_id)
            if hit is None:
                return None
            room, wall = hit
            s, t, _ = self.wall_axis(wall)
            a, b = s + t * u0, s + t * u1
            shape = LineString([a, b]) if u1 > u0 else Point(a)
            return Prism(shape, room.floor_z + v0, room.floor_z + v1)
        room = self.rooms.get(d.room_id)
        if room is None:
            return None
        z = room.floor_z if kind == "floor" else room.ceiling_z
        return Prism(box(u0, v0, max(u1, u0 + 1e-6), max(v1, v0 + 1e-6)), z, z)

    def opening_prism(self, o: Opening) -> Prism | None:
        hit = self.walls.get(o.wall_id)
        if hit is None:
            return None
        room, wall = hit
        s, t, _ = self.wall_axis(wall)
        u0 = _val(o.offset)
        u1 = u0 + _val(o.width)
        sill = _val(o.sill) if o.sill is not None else 0.0
        return Prism(LineString([s + t * u0, s + t * u1]), room.floor_z + sill, room.floor_z + sill + _val(o.height))

    def openings(self) -> list[tuple[Room, Opening]]:
        return [(r, o) for r in self.plan.rooms for o in r.openings]


def object_prism(o: Any) -> Prism | None:
    xy = _attr(o, "xy")
    if xy is None:
        return None
    xy = np.asarray(xy, float)
    xr = _attr(o, "x_range")
    yr = _attr(o, "y_range")
    if xr is not None and yr is not None and xr[1] > xr[0] and yr[1] > yr[0]:
        shape = box(float(xr[0]), float(yr[0]), float(xr[1]), float(yr[1]))
    else:
        shape = Point(float(xy[0]), float(xy[1]))
    z = _attr(o, "z_range") or (0.0, 0.0)
    return Prism(shape, float(min(z)), float(max(z)))


def _base_inputs(d: DamageRegion) -> dict[str, Any]:
    out = {"damage_id": d.id, "class": d.cls, "surface_id": d.surface_id, "surface_kind": surface_kind(d.surface_id),
           "area_m2": round(_val(d.area), 4), "score": round(float(d.score), 4)}
    if d.length is not None:
        out["length_m"] = round(_val(d.length), 4)
    return out


def _crack_ends(d: DamageRegion) -> list[np.ndarray]:
    ends: list[np.ndarray] = []
    for pair in d.evidence.get("endpoints_uv_all") or [d.evidence.get("endpoints_uv")] or []:
        if pair is None:
            continue
        for p in np.asarray(pair, float).reshape(-1, 2):
            ends.append(p)
    if not ends:  # no principal-axis ends recorded: use the corners of the uv box
        u0, u1 = d.u_range
        v0, v1 = d.v_range
        ends = [np.array(p, float) for p in ((u0, v0), (u1, v0), (u0, v1), (u1, v1))]
    return ends


def _region_conditions(when: dict[str, Any], d: DamageRegion, idx: PlanIndex, objects: list[Any]
                       ) -> dict[str, Any] | None:
    """Measured inputs when every region-level condition holds, else None."""
    if "classes" in when and d.cls not in when["classes"]:
        return None
    kind = surface_kind(d.surface_id)
    if "surfaces" in when and kind not in when["surfaces"]:
        return None
    inputs: dict[str, Any] = {}
    if "bottom_within_m" in when:
        thr = float(when["bottom_within_m"])
        bottom = float(min(d.v_range))
        if kind != "wall" or bottom > thr:
            return None
        inputs.update(bottom_above_floor_m=round(bottom, 4), bottom_threshold_m=thr)
    if "near_fixture" in when:
        spec = when["near_fixture"]
        thr = float(spec.get("within_m", 1.0))
        dp = idx.damage_prism(d)
        best = None
        for o in objects:
            if _attr(o, "cls") not in spec.get("classes", []):
                continue
            op = object_prism(o)
            if dp is None or op is None:
                continue
            dist = dp.distance(op)
            if best is None or dist < best[0]:
                best = (dist, o)
        if best is None or best[0] > thr:
            return None
        inputs.update(fixture=_attr(best[1], "id"), fixture_class=_attr(best[1], "cls"),
                      fixture_distance_m=round(best[0], 4), fixture_threshold_m=thr)
    if "near_window" in when:
        thr = float(when["near_window"].get("within_m", 0.5))
        dp = idx.damage_prism(d)
        cands = []
        if dp is not None:
            cands += [(idx.opening_prism(o), o.id, "plan_opening") for _, o in idx.openings() if o.type == "window"]
            cands += [(object_prism(o), _attr(o, "id"), "detected_object") for o in objects
                      if _attr(o, "cls") == "window"]
        dists = [(dp.distance(p), oid, src) for p, oid, src in cands if p is not None]
        best = min(dists, key=lambda x: x[0]) if dists else None
        if best is None or best[0] > thr:
            return None
        inputs.update(window=best[1], window_source=best[2], window_distance_m=round(best[0], 4),
                      window_threshold_m=thr)
    if "near_opening_corner" in when:
        spec = when["near_opening_corner"]
        thr = float(spec.get("within_m", 0.3))
        types = spec.get("types", ["door", "window", "opening"])
        if kind != "wall":
            return None
        hit = idx.walls.get(d.surface_id)
        best = None
        if hit is not None:
            ends = _crack_ends(d)
            for o in hit[0].openings:
                if o.wall_id != d.surface_id or o.type not in types:
                    continue
                u0 = _val(o.offset)
                u1 = u0 + _val(o.width)
                s0 = _val(o.sill) if o.sill is not None else 0.0
                s1 = s0 + _val(o.height)
                corners = {"bottom_start": (u0, s0), "bottom_end": (u1, s0), "top_start": (u0, s1),
                           "top_end": (u1, s1)}
                for name, c in corners.items():
                    dist = min(float(np.linalg.norm(e - np.asarray(c))) for e in ends)
                    if best is None or dist < best[0]:
                        best = (dist, o, name)
        if best is None or best[0] > thr:
            return None
        inputs.update(opening=best[1].id, opening_type=best[1].type, corner=best[2],
                      corner_distance_m=round(best[0], 4), corner_threshold_m=thr)
    if "min_length_m" in when:
        thr = float(when["min_length_m"])
        length = _val(d.length) if d.length is not None else max(_val(d.width), _val(d.height))
        if length <= thr:
            return None
        inputs.update(length_m=round(length, 4), length_threshold_m=thr)
    return inputs


def evaluate(plan: Plan, damage: list[DamageRegion], objects: list[Any] | None = None, *,
             rules_path: str | Path | None = None) -> list[ConcealedFlag]:
    """Concealed-damage flags F1.. for the damage regions, ordered by room, then rule, then damage id."""
    if not damage:
        return []
    rules = load_rules(str(rules_path) if rules_path else None)
    idx = PlanIndex(plan)
    objects = list(objects or [])
    damage_order = {d.id: i for i, d in enumerate(damage)}
    raw: list[tuple[tuple, ConcealedFlag]] = []
    for ri, rule in enumerate(rules):
        when = rule.get("when", {})
        per = rule.get("per", "region")
        hits = []
        for d in damage:
            try:
                inputs = _region_conditions(when, d, idx, objects)
            except Exception as exc:  # odd geometry on one region must not stop the other rules
                log.warning("rule %s skipped %s: %s", rule.get("id"), d.id, exc)
                continue
            if inputs is not None:
                hits.append((d, inputs))
        groups: dict[Any, list] = {}
        for d, inputs in hits:
            key = d.id if per == "region" else (d.surface_id if per == "surface" else d.room_id)
            groups.setdefault(key, []).append((d, inputs))
        for members in groups.values():
            regions = [d for d, _ in members]
            if per == "region":
                d, inputs = members[0]
                flag_inputs = {**_base_inputs(d), **inputs}
            else:
                total = sum(_val(d.area) for d in regions)
                flag_inputs = {"class": sorted({d.cls for d in regions}), "total_area_m2": round(total, 4),
                               "total_area_sqft": round(total * SQFT_PER_M2, 2),
                               "regions": {d.id: round(_val(d.area), 4) for d in regions}}
                if "total_area_sqft" in when:
                    spec = when["total_area_sqft"]
                    sqft = total * SQFT_PER_M2
                    if "gt" in spec and not sqft > float(spec["gt"]):
                        continue
                    if "lte" in spec and not sqft <= float(spec["lte"]):
                        continue
                    flag_inputs["threshold_sqft"] = {k: float(v) for k, v in spec.items()}
            room_id = regions[0].room_id
            flag = ConcealedFlag(
                id="", rule_id=rule["id"], title=rule["title"], basis=" ".join(str(rule["basis"]).split()),
                room_id=room_id, surface_ids=list(dict.fromkeys(d.surface_id for d in regions)),
                damage_ids=[d.id for d in regions], severity=rule["severity"],
                recommendation=" ".join(str(rule["recommendation"]).split()), inputs=flag_inputs)
            sort_key = (idx.room_order.get(room_id, len(plan.rooms)), ri, min(damage_order[d.id] for d in regions))
            raw.append((sort_key, flag))
    raw.sort(key=lambda x: x[0])
    flags = []
    for i, (_, f) in enumerate(raw, 1):
        f.id = f"F{i}"
        flags.append(f)
    return flags
