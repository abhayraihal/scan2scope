"""Interval model: fills lo/hi of every Measurement in a Plan and its damage regions.

For a length v, sigma = sqrt((v s)^2 + a^2). s is the capture's log-scale sigma, shared by every measurement;
a is the tier's additive term for the measurement's role (priors.yaml), inflated by thin evidence. Heights add
a vertical log term (photo and video see heights across the image and lengths partly in depth). Areas use
sqrt((2 A s)^2 + (sum_i L_i a_i)^2) over the room's walls. The interval is value -/+ z q sigma with q per tier
from calibration.yaml, lo clipped at 0.

Evidence is read from the Measurement's evidence first, then from its wall, opening or room: observed_fraction
(walls also use Wall.observed_fraction), n_points (or n_inliers, support), and a fit residual in the
measurement's own unit (residual, residual_m, rms, fit_residual), added in quadrature. Room context comes from
quality["scenes"] and from flags: low_light, the photo count and a missing EXIF focal length.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from scan2scope.types import DamageRegion, Measurement, Plan, Room

log = logging.getLogger("scan2scope.uncertainty")

HERE = Path(__file__).resolve().parent
PRIORS_PATH = HERE / "priors.yaml"
CALIBRATION_PATH = HERE / "calibration.yaml"

OBS_KEYS = ("observed_fraction",)
NPTS_KEYS = ("n_points", "n_inliers", "support")
RESID_KEYS = ("residual", "residual_m", "rms", "fit_residual")
NPHOTO_KEYS = ("n_photos", "n_images", "num_images", "n_views")
HINT_KEYS = ("room_hint", "room", "room_id", "folder")
NOT_EXIF = ("default", "fallback", "estimated", "predicted", "model", "none", "missing")


def _num(x: Any) -> float | None:
    """x as a finite float, else None. Bools are not treated as numbers."""
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _first(sources: Iterable[Any], keys: tuple[str, ...]) -> float | None:
    for src in sources:
        if not isinstance(src, dict):
            continue
        for k in keys:
            v = _num(src.get(k))
            if v is not None:
                return v
    return None


def _step(table: list[dict] | None, x: float) -> float:
    for row in table or []:
        if x < float(row["below"]):
            return float(row["factor"])
    return 1.0


def _has_flag(flags: Iterable[str], name: str) -> bool:
    return any(f == name or f.startswith(name + ":") for f in flags)


def load_priors(path: str | Path | None = None) -> dict[str, Any]:
    with open(path or PRIORS_PATH) as f:
        return yaml.safe_load(f)


def load_calibration(path: str | Path | None = None) -> dict[str, Any]:
    """Calibration table; a missing or unreadable file means q = 1 with status prior for every tier."""
    p = Path(path) if path else CALIBRATION_PATH
    default: dict[str, Any] = {"level": 0.9, "min_rooms": 9, "tiers": {}}
    try:
        data = yaml.safe_load(p.read_text())
    except FileNotFoundError:
        return default
    except (OSError, yaml.YAMLError) as exc:
        log.warning("calibration file %s unreadable (%s); using q = 1 for every tier", p, exc)
        return default
    if not isinstance(data, dict):
        return default
    if not isinstance(data.get("tiers"), dict):
        data["tiers"] = {}
    return data


def tier_q(calibration: dict[str, Any], tier: str) -> tuple[float, str, dict[str, Any]]:
    entry = (calibration.get("tiers") or {}).get(tier) or {}
    q = _num(entry.get("q"))
    if q is None or q <= 0:
        return 1.0, "prior", entry
    return q, str(entry.get("status") or "prior"), entry


@dataclass
class _Model:
    s: float
    q: float
    zq: float
    infl: dict[str, Any]

    def additive(self, base: float, sources: list[Any]) -> float:
        a = base
        of = _first(sources, OBS_KEYS)
        if of is not None:
            a *= _step(self.infl.get("observed_fraction"), of)
        n = _first(sources, NPTS_KEYS)
        if n is not None:
            a *= _step(self.infl.get("n_points"), n)
        return math.hypot(a, abs(_first(sources, RESID_KEYS) or 0.0))

    def set(self, m: Measurement | None, sigma: float, parts: tuple[float, float] | dict[str, float]) -> None:
        if m is None:
            return
        ev = dict(m.evidence) if isinstance(m.evidence, dict) else {}
        v = _num(m.value)
        if v is None:
            m.lo = m.hi = None
            ev.update(sigma=None, q=self.q)
            m.evidence = ev
            return
        if not math.isfinite(sigma):
            sigma = abs(v)
        h = self.zq * sigma
        lo, hi = v - h, v + h
        m.lo, m.hi = (max(0.0, lo) if v >= 0 else lo), hi
        if not isinstance(parts, dict):
            parts = {"scale": parts[0], "additive": parts[1]}
        ev.update(sigma=sigma, q=self.q, sigma_parts=dict(parts))
        m.evidence = ev

    def length(self, m: Measurement | None, a: float, extra: dict[str, float] | None = None) -> None:
        if m is None:
            return
        sc = abs(_num(m.value) or 0.0) * self.s
        extra = {k: v for k, v in (extra or {}).items() if v > 0}
        sigma = math.sqrt(sc * sc + a * a + sum(v * v for v in extra.values()))
        self.set(m, sigma, {"scale": sc, "additive": a, **extra})

    def area(self, m: Measurement, t_add: float) -> None:
        sc = 2.0 * abs(_num(m.value) or 0.0) * self.s
        a = math.hypot(t_add, abs(_first([m.evidence], RESID_KEYS) or 0.0))
        self.set(m, math.hypot(sc, a), (sc, a))


def _hint(q: dict[str, Any]) -> str | None:
    for k in HINT_KEYS:
        if q.get(k):
            return str(q[k])
    return None


def _room_qualities(plan: Plan, quality: dict[str, Any] | None, tier: str) -> dict[str, list[dict[str, Any]]]:
    """Scene quality dicts that apply to each room; several candidates mean the worst one is used."""
    quality = quality if isinstance(quality, dict) else {}
    scenes = [q for q in (quality.get("scenes") or []) if isinstance(q, dict)]
    top = {k: v for k, v in quality.items() if k != "scenes"}
    names = [{str(x) for x in (r.id, r.label, r.source_hint) if x} for r in plan.rooms]
    known = set().union(*names) if names else set()
    # A scene whose hint names a room belongs to that room and is never a fallback for another one.
    free = [q for q in scenes if _hint(q) not in known]
    out: dict[str, list[dict[str, Any]]] = {}
    for i, room in enumerate(plan.rooms):
        match = next((q for q in scenes if _hint(q) in names[i]), None)
        if match is not None:
            cands = [match]
        elif tier == "photo" and len(scenes) == len(plan.rooms) and _hint(scenes[i]) not in known:
            cands = [scenes[i]]
        else:
            cands = free
        merged = []
        for c in cands or [{}]:
            q = {**top, **c, "flags": _flag_list(top) + _flag_list(c)}
            if _first([q], NPHOTO_KEYS) is None and room.view_ids:
                q["n_photos"] = len(set(room.view_ids))  # photo tier: the room's own views when no count is given
            merged.append(q)
        out[room.id] = merged
    return out


def _flag_list(q: dict[str, Any]) -> list[str]:
    fl = q.get("flags")
    return [str(f) for f in fl] if isinstance(fl, (list, tuple, set)) else []


def _quality_factor(q: dict[str, Any], flags: set[str], tier: str, infl: dict[str, Any]) -> tuple[float, list[str]]:
    flags = flags | set(_flag_list(q))
    f, why = 1.0, []
    if q.get("low_light") is True or _has_flag(flags, "low_light"):
        f *= float(infl.get("low_light", 1.0))
        why.append("low_light")
    few = infl.get("few_photos") or {}
    n = _first([q], NPHOTO_KEYS)
    if _has_flag(flags, "few_photos") or (tier == "photo" and few and n is not None and n < float(few["below"])):
        f *= float(few.get("factor", 1.0))
        why.append("few_photos")
    src = q.get("intrinsics_source")
    if (q.get("missing_exif_focal") is True or q.get("exif_focal") is False or _has_flag(flags, "missing_exif_focal")
            or (tier == "photo" and isinstance(src, str) and src.lower() in NOT_EXIF)):
        f *= float(infl.get("missing_exif_focal", 1.0))
        why.append("missing_exif_focal")
    return f, why


def _room_flags(room: Room, plan: Plan) -> set[str]:
    names = {str(x) for x in (room.id, room.label, room.source_hint) if x}
    flags = {str(f) for f in room.flags or []}
    for f in plan.flags or []:
        name, _, target = str(f).partition(":")
        if not target or target in names:
            flags.add(name)
    return flags


def _annotate_room(mdl: _Model, room: Room, add: dict[str, float], f_room: float, vertical: float) -> float:
    """Intervals for one room's walls, openings, ceiling, floor area and perimeter; returns its area term."""
    wall_terms: list[tuple[float, float]] = []

    def height(m: Measurement | None, a_h: float) -> None:
        if m is not None:
            mdl.length(m, a_h, {"vertical": abs(_num(m.value) or 0.0) * vertical})

    for wall in room.walls:
        src = [wall.evidence, {"observed_fraction": wall.observed_fraction}]
        a_len = mdl.additive(add["length"] * f_room, [wall.length.evidence, *src])
        mdl.length(wall.length, a_len)
        height(wall.height, mdl.additive(add["height"] * f_room, [wall.height.evidence, *src]))
        wall_terms.append((max(_num(wall.length.value) or 0.0, 0.0), a_len))
    for op in room.openings:
        for m in (op.offset, op.width, op.height, op.sill):
            if m is not None:
                mdl.length(m, mdl.additive(add["opening"] * f_room, [m.evidence, op.evidence]))
    height(room.ceiling_height,
           mdl.additive(add["height"] * f_room, [room.ceiling_height.evidence, room.evidence]))
    a0 = add["length"] * f_room
    if wall_terms:
        t_area = sum(L * a for L, a in wall_terms)
        t_perim = sum(a for _, a in wall_terms)
    else:
        n_edges = len(room.polygon) if room.polygon is not None else 4
        t_area = abs(_num(room.perimeter.value) or 0.0) * a0
        t_perim = max(n_edges, 3) * a0
    mdl.area(room.floor_area, t_area)
    mdl.length(room.perimeter, math.hypot(t_perim, abs(_first([room.perimeter.evidence], RESID_KEYS) or 0.0)))
    return math.hypot(t_area, abs(_first([room.floor_area.evidence], RESID_KEYS) or 0.0))


def _all_measurements(plan: Plan, damage: list[DamageRegion] | None) -> list[Measurement]:
    ms: list[Any] = [plan.footprint_area, plan.extent_x, plan.extent_y]
    for room in plan.rooms:
        ms += [getattr(room, k, None) for k in ("ceiling_height", "floor_area", "perimeter")]
        for w in getattr(room, "walls", None) or []:
            ms += [getattr(w, "length", None), getattr(w, "height", None)]
        for op in getattr(room, "openings", None) or []:
            ms += [getattr(op, k, None) for k in ("offset", "width", "height", "sill")]
    for d in damage or []:
        ms += [getattr(d, k, None) for k in ("area", "width", "height", "length")]
    return [m for m in ms if isinstance(m, Measurement)]


def _fallback(mdl: _Model, m: Measurement, a: float) -> bool:
    """Wide interval (50% plus one additive term) for a finite measurement the model did not reach."""
    v = _num(m.value)
    if v is None or (m.lo is not None and m.hi is not None):
        return False
    mdl.set(m, 0.5 * abs(v) + a, (0.5 * abs(v), a))
    m.evidence["fallback"] = True
    return True


def annotate(plan: Plan, damage: list[DamageRegion] | None, *, tier: str, quality: dict[str, Any] | None = None,
             priors: dict[str, Any] | None = None,
             calibration: dict[str, Any] | str | Path | None = None) -> dict[str, Any]:
    """Fill lo/hi and evidence["sigma"], evidence["q"] of every Measurement in plan and damage, in place.

    Returns the model record, also stored in plan.meta["uncertainty"] for the writer.
    """
    pri = priors if priors is not None else load_priors()
    cal = calibration if isinstance(calibration, dict) else load_calibration(calibration)
    tiers = pri["tiers"]
    model_tier = tier if tier in tiers else "photo"
    if model_tier != tier:
        log.warning("unknown tier %r; using the photo error model", tier)
        plan.flags.append(f"uncertainty_unknown_tier:{tier}")
    tp = tiers[model_tier]
    infl = pri.get("inflation") or {}
    z = float(pri.get("z", 1.645))
    q, status, cal_entry = tier_q(cal, model_tier)
    s_floor = float(tp["scale_floor"])
    vertical = float(tp.get("vertical", 0.0))
    s_cap = _num(quality.get("scale_log_sigma")) if isinstance(quality, dict) else None
    s = max(s_floor, abs(s_cap or 0.0))
    add = {k: float(v) for k, v in tp["additive"].items()}
    mdl = _Model(s=s, q=q, zq=z * q, infl=infl)

    room_quality = _room_qualities(plan, quality, tier)
    room_factors: dict[str, dict[str, Any]] = {}
    placement: list[str] = []
    area_terms: list[float] = []
    len_terms: list[float] = []
    for room in plan.rooms:
        flags = _room_flags(room, plan)
        f_room, why = max((_quality_factor(c, flags, tier, infl) for c in room_quality[room.id]),
                          key=lambda t: t[0])
        if _has_flag(flags, "placement_uncertain"):
            placement.append(room.id)
        room_factors[room.id] = {"factor": f_room, "reasons": why}
        a0 = add["length"] * f_room
        try:
            t_area = _annotate_room(mdl, room, add, f_room, vertical)
        except Exception as exc:  # noqa: BLE001 - one odd room must not cost the whole result its intervals
            log.warning("intervals for room %s failed (%s); using the fallback", room.id, exc)
            plan.flags.append(f"uncertainty_failed:{room.id}")
            t_area = abs(_num(getattr(room.perimeter, "value", 0.0)) or 0.0) * a0
        area_terms.append(t_area)
        len_terms.append(a0)

    # Footprint: the scale term is fully correlated across rooms, the per-room terms are independent.
    f_place = float(infl.get("placement_uncertain", 1.0)) if placement else 1.0
    if isinstance(plan.footprint_area, Measurement):
        sc = 2.0 * abs(_num(plan.footprint_area.value) or 0.0) * s
        t_fp = math.sqrt(sum(t * t for t in area_terms))
        mdl.set(plan.footprint_area, math.hypot(sc, t_fp) * f_place, (sc * f_place, t_fp * f_place))
    a_ext = math.sqrt(sum(a * a for a in len_terms)) if len_terms else add["length"]
    for m in (plan.extent_x, plan.extent_y):
        if isinstance(m, Measurement):
            sc = abs(_num(m.value) or 0.0) * s
            mdl.set(m, math.hypot(sc, a_ext) * f_place, (sc * f_place, a_ext * f_place))

    a_d = float(pri.get("damage_additive", 0.02))
    for d in damage or []:
        try:
            for m in (d.width, d.height, d.length):
                mdl.length(m, a_d)
            bb_perim = 2.0 * (abs(_num(d.width.value) or 0.0) + abs(_num(d.height.value) or 0.0))
            mdl.area(d.area, bb_perim * a_d)
        except Exception as exc:  # noqa: BLE001 - same for one odd damage region
            log.warning("intervals for damage %s failed (%s); using the fallback", getattr(d, "id", "?"), exc)
            plan.flags.append(f"uncertainty_failed:{getattr(d, 'id', '?')}")

    n_fallback = sum(_fallback(mdl, m, add["length"]) for m in _all_measurements(plan, damage))
    if n_fallback:
        plan.flags.append(f"uncertainty_fallback:{n_fallback}")

    record = {
        "tier": tier, "model_tier": model_tier, "level": float(pri.get("level", 0.9)), "z": z,
        "q": q, "status": status, "min_rooms": int(cal.get("min_rooms", 9)),
        "scale_sigma": s, "scale_floor": s_floor, "scale_capture": s_cap,
        "vertical": vertical,
        "additive": add, "damage_additive": a_d,
        "placement_factor": float(infl.get("placement_uncertain", 1.0)),
        "room_factors": room_factors, "placement_uncertain": placement,
        "calibration": {k: cal_entry[k] for k in ("n_rooms", "n_records", "empirical_quantile",
                                                  "conformal_quantile", "fitted") if k in cal_entry},
    }
    if isinstance(plan.meta, dict):
        plan.meta["uncertainty"] = record
    log.info("intervals: tier %s, s %.4f, q %.3f (%s), %d rooms, %d damage regions", tier, s, q, status,
             len(plan.rooms), len(damage or []))
    return record


def describe(record: dict[str, Any] | None) -> str:
    """Plain-text statement of the error model and calibration status, for result.json conventions."""
    if not record:
        return "No error model was applied; lo and hi equal the value."
    add = record.get("additive") or {}
    q = float(record.get("q", 1.0))
    if record.get("status") == "calibrated":
        n = (record.get("calibration") or {}).get("n_rooms", "?")
        cal = f"q = {q:.2f}, calibrated by split conformal on {n} ground-truth rooms of this tier (room as the unit)"
    else:
        cal = (f"q = {q:.2f}, prior value: not calibrated, this tier has fewer than "
               f"{record.get('min_rooms', 9)} ground-truth rooms")
    return (
        f"{record.get('model_tier', record.get('tier'))} tier error model. Lengths: sigma = sqrt((v*s)^2 + a^2) "
        f"with a scale term s = {float(record.get('scale_sigma', 0.0)):.3f} shared by every measurement and "
        f"additive terms a = {add.get('length', 0):.3f} m (walls), {add.get('height', 0):.3f} m (heights), "
        f"{add.get('opening', 0):.3f} m (openings), {float(record.get('damage_additive', 0.0)):.3f} m (damage), "
        "inflated for thin evidence (low observed wall fraction, few supporting points, fit residuals in "
        "quadrature, low light, fewer than 4 photos, missing EXIF focal length). Heights add a vertical term of "
        f"{float(record.get('vertical', 0.0)):.2f} (log). Areas: "
        "sqrt((2*A*s)^2 + (sum of L_i*a_i)^2). Footprint: scale term fully correlated across rooms plus "
        f"independent per-room terms; unplaced rooms widen footprint and extents "
        f"{float(record.get('placement_factor', 1.5)):.1f}x. Interval = value -/+ "
        f"{float(record.get('z', 1.645)):.3f}*q*sigma, lo clipped at 0; {cal}."
    )
