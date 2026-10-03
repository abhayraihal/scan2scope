"""Builds the result.json dict (schema/scan2scope.schema.json) from the pipeline objects.

The writer repairs what it can instead of failing: non-finite numbers, intervals that exclude their value,
unknown enum values and references to ids that do not exist are fixed or the record is dropped, and each
repair is recorded as a "writer:..." flag in property.flags.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from scan2scope import __version__
from scan2scope.types import (
    TIERS,
    Adjacency,
    CaptureInfo,
    ConcealedFlag,
    DamageRegion,
    LineItem,
    Measurement,
    Plan,
    Room,
)
from scan2scope.uncertainty.model import describe

log = logging.getLogger("scan2scope.output")

SCHEMA_VERSION = "1.0.0"
INTERVAL_LEVEL = 0.9
OPENING_TYPES = ("door", "window", "opening")
DAMAGE_CLASSES = ("water_stain", "mold", "crack", "hole", "peeling_paint")
SEVERITIES = ("low", "medium", "high")
ACTIVITIES = ("&", "-", "+", "R", "I")
SCOPE_UNITS = ("SF", "LF", "EA", "m2", "m")
ADJACENCY_SOURCES = ("shared_frame", "doorway_photo", "door_match")
CACHE_MODES = ("live", "replay", "mixed", "none")
UNIT_ALIASES = {"sf": "SF", "sqft": "SF", "sq ft": "SF", "lf": "LF", "ea": "EA", "each": "EA", "m2": "m2",
                "sqm": "m2", "m^2": "m2", "m²": "m2", "m": "m"}

UNITS = {
    "length": "m",
    "area": "m2",
    "plan": "x and y in metres in the capture's world frame, z up; room polygons counter-clockwise",
    "scope_quantity": "the unit named on each line item (SF, LF, EA, m2 or m)",
    "score": "detector score in [0, 1] after cross-view merge, not a calibrated probability",
}

DEFINITIONS = {
    "wall_length": "face to face at 1 m height between the two neighbouring walls, along the room polygon edge",
    "wall_height": "floor plane to ceiling plane distance at the wall",
    "ceiling_height": "floor plane to ceiling plane distance inside the room",
    "opening_width": "clear width between the jamb faces",
    "opening_height": "floor to the underside of the head for doors and passages; inside the frame for windows",
    "opening_offset": "along the wall, from the wall start to the near jamb of the opening",
    "sill": "floor to the window sill",
    "floor_area": "polygon area of the interior wall faces",
    "perimeter": "sum of the room's wall lengths",
    "wall_surface_area": "wall length times wall height minus the areas of the openings in that wall",
    "ceiling_surface_area": "equal to the floor area (flat ceiling assumed)",
    "footprint": "sum of the room floor areas; walls between rooms are not included",
    "extent": "size of the bounding box of all room polygons along plan x and plan y",
    "damage_extent": "width along the surface u axis and height along v of the damage region; area of its mask "
                     "projected onto the surface; walls use u from the wall start and v from the floor, floors "
                     "and ceilings use plan x and y",
}


def _f(x: Any) -> float | None:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _r(x: float) -> float:
    return round(float(x), 4) + 0.0


def _clip01(x: Any) -> float:
    return min(max(_f(x) or 0.0, 0.0), 1.0)


def _strs(x: Any) -> list[str]:
    if isinstance(x, (list, tuple)):
        return [str(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(str(v) for v in x)
    return []


def _plain(x: Any) -> Any:
    """JSON-safe copy of a free-form value: numpy to Python, non-finite floats to None, paths to str."""
    if x is None or isinstance(x, (str, bool)):
        return x
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return _f(x)
    if isinstance(x, Measurement):
        return _plain(x.to_dict())
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((_plain(v) for v in x), key=str)
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, Path):
        return str(x)
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        return _plain({f.name: getattr(x, f.name) for f in dataclasses.fields(x)})
    return str(x)


def _meta_obj(x: Any) -> dict[str, Any] | None:
    if x is None:
        return None
    if isinstance(x, dict):
        return _plain(x)
    if isinstance(x, bool):
        return {"enabled": x}
    return {"value": _plain(x)}


def _unit(u: Any) -> str | None:
    s = str(u).strip()
    return s if s in SCOPE_UNITS else UNIT_ALIASES.get(s.lower())


class _Writer:
    def __init__(self) -> None:
        self.repairs: list[str] = []
        self.no_interval = 0

    def repair(self, what: str) -> None:
        self.repairs.append(f"writer:{what}")
        log.warning("result repaired: %s", what)

    def unique(self, ident: Any, seen: set[str], what: str) -> str:
        base = str(ident)
        out, k = base, 2
        while out in seen:
            out, k = f"{base}~{k}", k + 1
        if out != base:
            self.repair(f"duplicate_{what}_id:{base}->{out}")
        seen.add(out)
        return out

    def meas(self, m: Any, unit: str, where: str) -> dict[str, Any]:
        if not isinstance(m, Measurement):
            self.repair(f"missing_measurement:{where}")
            return {"value": 0.0, "lo": 0.0, "hi": 0.0, "unit": unit}
        if m.lo is None or m.hi is None:
            self.no_interval += 1
        v = _f(m.value)
        lo = _f(m.value if m.lo is None else m.lo)
        hi = _f(m.value if m.hi is None else m.hi)
        if v is None:
            v = (lo + hi) / 2.0 if lo is not None and hi is not None else 0.0
            self.repair(f"nonfinite_value:{where}")
        if lo is None:
            lo = 0.0 if v >= 0 else v
            self.repair(f"nonfinite_lo:{where}")
        if hi is None:
            hi = v + max(abs(v), 1.0)
            self.repair(f"nonfinite_hi:{where}")
        if lo > v or hi < v:
            self.repair(f"interval_excludes_value:{where}")
            lo, hi = min(lo, v), max(hi, v)
        return {"value": _r(v), "lo": _r(lo), "hi": _r(hi), "unit": unit}

    def point(self, p: Any, where: str, required: bool = True) -> list[float] | None:
        try:
            x, y = _f(p[0]), _f(p[1])
        except (TypeError, IndexError, KeyError):
            x = y = None
        if x is None or y is None:
            if required:
                self.repair(f"bad_point:{where}")
                return [0.0, 0.0]
            return None
        return [_r(x), _r(y)]

    def range2(self, r: Any, where: str) -> list[float]:
        pt = self.point(r, where, required=False)
        if pt is None:
            self.repair(f"bad_range:{where}")
            return [0.0, 0.0]
        return sorted(pt)

    def polygon(self, room: Room, rid: str) -> list[list[float]]:
        try:
            raw = np.asarray(room.polygon, float)
            raw = raw[:, :2] if raw.ndim == 2 and raw.shape[1] >= 2 else raw.reshape(-1, 2)
        except (TypeError, ValueError):
            raw = np.zeros((0, 2))
        pts = [p for p in (self.point(q, f"{rid}.polygon", required=False) for q in raw) if p is not None]
        if len(pts) >= 3:
            return pts
        self.repair(f"degenerate_polygon:{rid}")
        starts = [p for p in (self.point(w.start, f"{rid}.polygon", required=False) for w in room.walls or [])
                  if p is not None]
        if len(starts) >= 3:
            return starts
        pts = pts or starts or [[0.0, 0.0]]
        return (pts * 3)[:3]

    def room(self, room: Room, rid: str, room_ids: set[str], wall_seen: set[str],
             opening_seen: set[str]) -> dict[str, Any]:
        walls = []
        for w in room.walls or []:
            wid = self.unique(w.id, wall_seen, "wall")
            walls.append({
                "id": wid, "start": self.point(w.start, f"{wid}.start"), "end": self.point(w.end, f"{wid}.end"),
                "length": self.meas(w.length, "m", f"{wid}.length"), "height": self.meas(w.height, "m", f"{wid}.height"),
                "observed_fraction": _r(_clip01(w.observed_fraction)), "flags": _strs(w.flags),
            })
        wall_ids = {w["id"] for w in walls}
        openings = []
        for op in room.openings or []:
            oid = self.unique(op.id, opening_seen, "opening")
            if str(op.wall_id) not in wall_ids:
                self.repair(f"dropped_opening:{oid}:unknown_wall:{op.wall_id}")
                continue
            kind = str(op.type).strip().lower()
            if kind not in OPENING_TYPES:
                self.repair(f"opening_type:{oid}:{op.type}->opening")
                kind = "opening"
            to = None if op.connects_to is None else str(op.connects_to)
            if to is not None and to not in room_ids:
                self.repair(f"opening_connects_to:{oid}:unknown_room:{to}")
                to = None
            openings.append({
                "id": oid, "type": kind, "wall_id": str(op.wall_id),
                "offset": self.meas(op.offset, "m", f"{oid}.offset"), "width": self.meas(op.width, "m", f"{oid}.width"),
                "height": self.meas(op.height, "m", f"{oid}.height"),
                "sill": None if op.sill is None else self.meas(op.sill, "m", f"{oid}.sill"),
                "center": None if op.center is None else self.point(op.center, f"{oid}.center", required=False),
                "connects_to": to, "confidence": _r(_clip01(op.confidence)), "flags": _strs(op.flags),
            })
        floor_area = self.meas(room.floor_area, "m2", f"{rid}.floor_area")
        return {
            "id": rid, "label": str(room.label if room.label is not None else rid),
            "source_hint": None if room.source_hint is None else str(room.source_hint),
            "polygon": self.polygon(room, rid), "floor_area": floor_area,
            "perimeter": self.meas(room.perimeter, "m", f"{rid}.perimeter"),
            "ceiling_height": self.meas(room.ceiling_height, "m", f"{rid}.ceiling_height"),
            "walls": walls, "openings": openings, "surfaces": self.surfaces(rid, walls, openings, floor_area),
            "flags": _strs(room.flags),
        }

    def surfaces(self, rid: str, walls: list[dict], openings: list[dict], floor_area: dict) -> list[dict[str, Any]]:
        """Wall net areas by interval arithmetic on length, height and the wall's openings; floor and ceiling."""
        out = []
        for w in walls:
            L, H = w["length"], w["height"]
            ops = [o for o in openings if o["wall_id"] == w["id"]]
            v = L["value"] * H["value"] - sum(o["width"]["value"] * o["height"]["value"] for o in ops)
            lo = L["lo"] * H["lo"] - sum(o["width"]["hi"] * o["height"]["hi"] for o in ops)
            hi = L["hi"] * H["hi"] - sum(o["width"]["lo"] * o["height"]["lo"] for o in ops)
            if v < 0:
                self.repair(f"openings_exceed_wall_area:{w['id']}")
            v = max(v, 0.0)
            lo, hi = max(0.0, min(lo, v)), max(hi, v)
            out.append({"id": w["id"], "kind": "wall", "wall_id": w["id"],
                        "area": {"value": _r(v), "lo": _r(lo), "hi": _r(hi), "unit": "m2"}})
        out.append({"id": f"{rid}-FLOOR", "kind": "floor", "wall_id": None, "area": dict(floor_area)})
        out.append({"id": f"{rid}-CEIL", "kind": "ceiling", "wall_id": None, "area": dict(floor_area)})
        return out

    def damage(self, d: DamageRegion, surfaces: dict[str, str], seen: set[str]) -> dict[str, Any] | None:
        did = self.unique(d.id, seen, "damage")
        cls = str(d.cls).strip().lower().replace(" ", "_").replace("-", "_")
        if cls not in DAMAGE_CLASSES:
            self.repair(f"dropped_damage:{did}:class:{d.cls}")
            return None
        sid = str(d.surface_id)
        if sid not in surfaces:
            self.repair(f"dropped_damage:{did}:unknown_surface:{sid}")
            return None
        room_id = str(d.room_id)
        if room_id != surfaces[sid]:
            self.repair(f"damage_room:{did}:{room_id}->{surfaces[sid]}")
            room_id = surfaces[sid]
        return {
            "id": did, "room_id": room_id, "surface_id": sid, "class": cls, "score": _r(_clip01(d.score)),
            "area": self.meas(d.area, "m2", f"{did}.area"), "width": self.meas(d.width, "m", f"{did}.width"),
            "height": self.meas(d.height, "m", f"{did}.height"),
            "length": None if d.length is None else self.meas(d.length, "m", f"{did}.length"),
            "u_range": self.range2(d.u_range, f"{did}.u_range"), "v_range": self.range2(d.v_range, f"{did}.v_range"),
            "view_ids": _strs(d.view_ids),
        }

    def refs(self, ids: Any, known: set[str] | dict[str, str], where: str) -> list[str]:
        vals = _strs(ids)
        keep = [v for v in vals if v in known]
        if len(keep) < len(vals):
            self.repair(f"dropped_refs:{where}:{','.join(v for v in vals if v not in known)}")
        return keep

    def room_for(self, room_id: Any, sids: list[str], surfaces: dict[str, str], room_ids: set[str],
                 where: str) -> str | None:
        rid = str(room_id)
        if rid in room_ids:
            return rid
        if sids:
            self.repair(f"room:{where}:{rid}->{surfaces[sids[0]]}")
            return surfaces[sids[0]]
        return None

    def concealed(self, f: ConcealedFlag, surfaces: dict[str, str], damage_ids: set[str], room_ids: set[str],
                  seen: set[str]) -> dict[str, Any] | None:
        fid = self.unique(f.id, seen, "flag")
        sids = self.refs(f.surface_ids, surfaces, f"{fid}.surface_ids")
        room_id = self.room_for(f.room_id, sids, surfaces, room_ids, fid)
        if room_id is None:
            self.repair(f"dropped_flag:{fid}:unknown_room:{f.room_id}")
            return None
        severity = str(f.severity).strip().lower()
        if severity not in SEVERITIES:
            self.repair(f"flag_severity:{fid}:{f.severity}->medium")
            severity = "medium"
        return {
            "id": fid, "rule_id": str(f.rule_id), "title": str(f.title), "basis": str(f.basis), "room_id": room_id,
            "surface_ids": sids, "damage_ids": self.refs(f.damage_ids, damage_ids, f"{fid}.damage_ids"),
            "severity": severity, "recommendation": str(f.recommendation),
            "inputs": _plain(f.inputs) if isinstance(f.inputs, dict) else {},
        }

    def item(self, it: LineItem, surfaces: dict[str, str], damage_ids: set[str], flag_ids: set[str],
             room_ids: set[str], seen: set[str]) -> dict[str, Any] | None:
        lid = self.unique(it.id, seen, "scope")
        sid = str(it.surface_id)
        unit = _unit(it.unit)
        if sid not in surfaces:
            self.repair(f"dropped_scope:{lid}:unknown_surface:{sid}")
            return None
        if str(it.activity) not in ACTIVITIES:
            self.repair(f"dropped_scope:{lid}:activity:{it.activity}")
            return None
        if unit is None:
            self.repair(f"dropped_scope:{lid}:unit:{it.unit}")
            return None
        room_id = self.room_for(it.room_id, [sid], surfaces, room_ids, lid)
        return {
            "id": lid, "room_id": room_id, "surface_id": sid, "category": str(it.category),
            "selector": str(it.selector), "activity": str(it.activity), "description": str(it.description),
            "quantity": self.meas(it.quantity, unit, f"{lid}.quantity"), "unit": unit,
            "damage_ids": self.refs(it.damage_ids, damage_ids, f"{lid}.damage_ids"),
            "flag_ids": self.refs(it.flag_ids, flag_ids, f"{lid}.flag_ids"),
            "rule_id": None if it.rule_id is None else str(it.rule_id),
        }

    def adjacency(self, a: Adjacency, room_ids: set[str], opening_ids: set[str]) -> dict[str, Any] | None:
        ra, rb, src = str(a.room_a), str(a.room_b), str(a.source)
        if ra not in room_ids or rb not in room_ids:
            self.repair(f"dropped_adjacency:{ra}-{rb}:unknown_room")
            return None
        if src not in ADJACENCY_SOURCES:
            self.repair(f"dropped_adjacency:{ra}-{rb}:source:{src}")
            return None
        ops = []
        for o in (a.opening_a, a.opening_b):
            if o is not None and str(o) not in opening_ids:
                self.repair(f"adjacency_opening:{ra}-{rb}:unknown_opening:{o}")
                o = None
            ops.append(None if o is None else str(o))
        return {"room_a": ra, "room_b": rb, "opening_a": ops[0], "opening_b": ops[1],
                "confidence": _r(_clip01(a.confidence)), "source": src}

    def capture(self, info: CaptureInfo | None, plan: Plan) -> dict[str, Any]:
        tier = str(getattr(info, "tier", "") or "")
        if tier not in TIERS:
            unc = plan.meta.get("uncertainty") if isinstance(plan.meta, dict) else None
            fixed = unc.get("tier") if isinstance(unc, dict) else None
            if fixed in TIERS:
                self.repair(f"capture_tier:{tier}->{fixed}")
                tier = str(fixed)
        return {
            "id": str(getattr(info, "id", "") or "capture"), "tier": tier, "path": str(getattr(info, "path", "")),
            "device": _plain(getattr(info, "device", None)) if isinstance(getattr(info, "device", None), dict) else {},
            "input_stats": (_plain(getattr(info, "input_stats", None))
                            if isinstance(getattr(info, "input_stats", None), dict) else {}),
            "flags": _strs(getattr(info, "flags", None)),
        }


def _conventions(plan: Plan) -> dict[str, Any]:
    unc = plan.meta.get("uncertainty") if isinstance(plan.meta, dict) else None
    interval: dict[str, Any] = {
        "level": INTERVAL_LEVEL,
        "method": describe(unc) + " Wall surface areas combine the length, height and opening intervals by "
                                  "interval arithmetic.",
    }
    if isinstance(unc, dict):
        interval["q"] = _r(_f(unc.get("q")) or 1.0)
        interval["calibration"] = str(unc.get("status") or "prior")
    return {"units": dict(UNITS), "interval": interval, "definitions": dict(DEFINITIONS)}


def _timing(timing: Any) -> dict[str, Any]:
    t = timing if isinstance(timing, dict) else {}
    stages = t.get("stages") if isinstance(t.get("stages"), dict) else {}
    out: dict[str, Any] = {"total_s": _r(max(_f(t.get("total_s")) or 0.0, 0.0)),
                           "stages": {str(k): _r(max(_f(v) or 0.0, 0.0)) for k, v in stages.items()}}
    out.update({str(k): _plain(v) for k, v in t.items() if k not in ("total_s", "stages")})
    return out


def _provenance(p: Any) -> dict[str, Any]:
    out = _plain(p) if isinstance(p, dict) else {}
    out["pipeline_version"] = str(out.get("pipeline_version") or __version__)
    out["git_commit"] = out.get("git_commit") if isinstance(out.get("git_commit"), str) else None
    models = out.get("models") if isinstance(out.get("models"), list) else []
    out["models"] = [{**m, **{k: str(m.get(k) or "unknown") for k in ("name", "revision", "license")}}
                     for m in models if isinstance(m, dict)]
    out["cache_mode"] = out.get("cache_mode") if out.get("cache_mode") in CACHE_MODES else "none"
    out["device"] = str(out.get("device") or "unknown")
    return out


def build_result(info: CaptureInfo, plan: Plan, damage: list[DamageRegion], flags: list[ConcealedFlag],
                 scope: list[LineItem], timing: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    """result.json content for one capture; every float is finite and every interval contains its value."""
    w = _Writer()
    capture = w.capture(info, plan)

    room_seen: set[str] = set()
    plan_rooms = list(plan.rooms or [])
    room_ids = [w.unique(r.id, room_seen, "room") for r in plan_rooms]
    wall_seen: set[str] = set()
    opening_seen: set[str] = set()
    rooms = [w.room(r, rid, room_seen, wall_seen, opening_seen) for r, rid in zip(plan_rooms, room_ids)]
    surfaces = {s["id"]: room["id"] for room in rooms for s in room["surfaces"]}
    opening_ids = {o["id"] for room in rooms for o in room["openings"]}

    damage_seen: set[str] = set()
    damage_out = [x for x in (w.damage(d, surfaces, damage_seen) for d in damage or []) if x is not None]
    damage_ids = {d["id"] for d in damage_out}
    flag_seen: set[str] = set()
    flags_out = [x for x in (w.concealed(f, surfaces, damage_ids, room_seen, flag_seen) for f in flags or [])
                 if x is not None]
    flag_ids = {f["id"] for f in flags_out}
    scope_seen: set[str] = set()
    scope_out = [x for x in (w.item(it, surfaces, damage_ids, flag_ids, room_seen, scope_seen) for it in scope or [])
                 if x is not None]
    adjacency = [x for x in (w.adjacency(a, room_seen, opening_ids) for a in plan.adjacency or []) if x is not None]

    meta = plan.meta if isinstance(plan.meta, dict) else {}
    footprint = w.meas(plan.footprint_area, "m2", "footprint_area")
    extent_x = w.meas(plan.extent_x, "m", "extent_x")
    extent_y = w.meas(plan.extent_y, "m", "extent_y")
    if w.no_interval:
        w.repair(f"no_interval:{w.no_interval}")
    return {
        "schema_version": SCHEMA_VERSION,
        "capture": capture,
        "conventions": _conventions(plan),
        "property": {
            "footprint_area": footprint, "extent_x": extent_x, "extent_y": extent_y, "adjacency": adjacency,
            "drift_correction": _meta_obj(meta.get("drift")), "stitch": _meta_obj(meta.get("stitch")),
            "flags": _strs(plan.flags) + w.repairs,
        },
        "rooms": rooms,
        "damage": damage_out,
        "concealed_damage_flags": flags_out,
        "scope": scope_out,
        "timing": _timing(timing),
        "provenance": _provenance(provenance),
    }
