"""Per-capture scores against ground truth, and repeatability across captures of the same property and tier.

A record is one reported number scored against its GT value: kind (wall_length, ceiling_height,
opening_width, opening_height, floor_area, footprint, damage_area, damage_length), gt, pred, lo, hi,
err = pred - gt, rel_err, covered (gt inside [lo, hi]), half_width = (hi - lo) / 2, room (a "property/room"
unit shared by repeat captures, None for the footprint), tier and q (the interval multiplier in force, when
the result states it).
These are the fields scan2scope.uncertainty.calibrate reads.

GT items the capture should have reported but did not (room or wall not found, opening missed, capture failed)
go to `missing`, so gates count them as failures.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict
from typing import Any

import numpy as np

from scan2scope.bench.groundtruth import GroundTruth, GTCapture, GTRoom, num
from scan2scope.bench.match import CaptureMatch, PredRoom, match_capture, meas, pred_adjacency, pred_rooms

log = logging.getLogger("scan2scope.bench")

KINDS = ("wall_length", "ceiling_height", "opening_width", "opening_height", "floor_area", "footprint",
         "damage_area", "damage_length")
REPEAT_DEFAULT = {"abs": 0.01, "rel": 0.005, "rule": "max"}


def record(kind: str, item: str, gt_value: float, m: Any, *, tier: str, room: str | None, capture: str,
           prop: str, pred_id: str | None = None, q: float | None = None,
           **extra: Any) -> dict[str, Any] | None:
    t = meas(m)
    if t is None:
        return None
    v, lo, hi = t
    lo, hi = min(lo, v), max(hi, v)
    err = v - gt_value
    rec = {"kind": kind, "item": item, "property": prop, "capture": capture, "tier": tier, "room": room,
           "pred_id": pred_id, "gt": gt_value, "pred": v, "lo": lo, "hi": hi, "err": err,
           "abs_err": abs(err), "rel_err": err / gt_value if gt_value else None,
           "covered": bool(lo - 1e-9 <= gt_value <= hi + 1e-9), "half_width": (hi - lo) / 2.0, "q": q}
    rec.update(extra)
    return rec


def _missing(kind: str, item: str, gt_value: float | None, *, room: str | None,
             reason: str) -> dict[str, Any]:
    return {"kind": kind, "item": item, "gt": gt_value, "room": room, "reason": reason}


def _unit(prop: str, room_id: str) -> str:
    return f"{prop}/{room_id}"


def _room_missing(gt: GroundTruth, room: GTRoom, reason: str) -> list[dict[str, Any]]:
    u = _unit(gt.property, room.id)
    out = [_missing("wall_length", f"{room.id}/{w.id}", w.length, room=u, reason=reason)
           for w in room.walls if w.length is not None]
    if room.ceiling_height is not None:
        out.append(_missing("ceiling_height", room.id, room.ceiling_height, room=u, reason=reason))
    if room.floor_area is not None:
        out.append(_missing("floor_area", room.id, room.floor_area, room=u, reason=reason))
    out += [_missing("opening_width", f"{room.id}/{o.id}", o.width, room=u, reason=reason)
            for o in room.openings]
    out += [_missing("damage", f"{room.id}/{d.id}", None, room=u, reason=reason) for d in room.damage]
    return out


def _empty(gt: GroundTruth, capture: GTCapture) -> dict[str, Any]:
    rooms = gt.capture_rooms(capture)
    return {
        "property": gt.property, "capture": capture.id, "tier": capture.tier, "synthetic": gt.synthetic,
        "status": "ok", "error": None, "multi_room": len(rooms) >= 2, "gt_rooms": [r.id for r in rooms],
        "records": [], "missing": [], "flags": [],
        "rooms": {"gt": len(rooms), "pred": 0, "matched": 0, "missed": [r.id for r in rooms], "extra": []},
        "walls": {"gt": sum(len(r.walls) for r in rooms), "matched": 0, "missed": 0, "extra": 0},
        "openings": {"gt": sum(len(r.openings) for r in rooms), "matched": 0, "missed": 0, "phantom": 0,
                     "phantom_unscored": 0, "type_swaps": 0, "items": []},
        "damage": {"gt": sum(len(r.damage) for r in rooms), "matched": 0, "class_ok": 0, "missed": 0,
                   "phantom": 0, "phantom_unscored": 0, "items": []},
        "adjacency": None, "overlap": None, "drift": None, "timing": None, "q": None,
        "calibration_status": None, "by_gt": {}, "match": None,
    }


def _fail_all(out: dict[str, Any], gt: GroundTruth, capture: GTCapture, reason: str) -> dict[str, Any]:
    rooms = gt.capture_rooms(capture)
    for room in rooms:
        out["missing"] += _room_missing(gt, room, reason)
    fp = gt.footprint(capture)
    if fp is not None:
        out["missing"].append(_missing("footprint", "footprint", fp, room=None, reason=reason))
    out["openings"]["missed"] = out["openings"]["gt"]
    out["damage"]["missed"] = out["damage"]["gt"]
    out["walls"]["missed"] = out["walls"]["gt"]
    return out


def adjacency_check(gt: GroundTruth, capture: GTCapture, result: dict[str, Any], match: CaptureMatch
                    ) -> dict[str, Any]:
    """Predicted adjacency mapped through the room match, against GT adjacency among the capture's rooms."""
    ids = [r.id for r in gt.capture_rooms(capture)]
    gt_pairs = {tuple(sorted(p)) for p in gt.adjacency_pairs(ids)}
    mapped: set[tuple[str, str]] = set()
    unmatched = []
    for pair in sorted(pred_adjacency(result), key=lambda p: tuple(sorted(p))):
        a, b = sorted(pair)
        ga, gb = match.room_map.get(a), match.room_map.get(b)
        if ga is None or gb is None:
            unmatched.append([a, b])
        else:
            mapped.add(tuple(sorted((ga, gb))))
    missing = sorted(gt_pairs - mapped)
    wrong = sorted(mapped - gt_pairs)
    known = bool(gt_pairs) or len(ids) < 2
    return {"gt": [list(p) for p in sorted(gt_pairs)], "pred": [list(p) for p in sorted(mapped)],
            "correct": [list(p) for p in sorted(gt_pairs & mapped)], "missing": [list(p) for p in missing],
            "extra": [list(p) for p in wrong], "unmatched_rooms": unmatched,
            "exact": (not missing and not wrong and not unmatched) if known else None}


def room_overlaps(result: dict[str, Any]) -> dict[str, Any]:
    """Pairwise intersection areas of the predicted room polygons."""
    from shapely.errors import ShapelyError
    from shapely.geometry import Polygon

    polys = []
    skipped = []
    for p in pred_rooms(result):
        if len(p.polygon) < 3 or not np.isfinite(p.polygon).all():
            continue
        try:
            poly = Polygon(p.polygon)
            if not poly.is_valid:
                poly = poly.buffer(0)
        except (ShapelyError, ValueError):
            skipped.append(p.id)
            continue
        polys.append((p.id, poly))
    pairs = []
    for (ia, a), (ib, b) in itertools.combinations(polys, 2):
        try:
            area = float(a.intersection(b).area)
        except (ShapelyError, ValueError):
            skipped.append(f"{ia}|{ib}")
            continue
        if area > 1e-6:
            pairs.append({"rooms": [ia, ib], "area_m2": area})
    if skipped:
        log.warning("overlap check skipped invalid geometry: %s", ", ".join(skipped))
    return {"max_m2": max((p["area_m2"] for p in pairs), default=0.0),
            "total_m2": float(sum(p["area_m2"] for p in pairs)), "pairs": pairs}


def _interval_info(result: dict[str, Any]) -> tuple[float | None, str | None]:
    conv = result.get("conventions") if isinstance(result.get("conventions"), dict) else {}
    iv = conv.get("interval") if isinstance(conv.get("interval"), dict) else {}
    status = iv.get("calibration")
    return num(iv.get("q")), None if status is None else str(status)


def _mdict(m: Any) -> dict[str, float] | None:
    t = meas(m)
    return None if t is None else {"pred": t[0], "lo": t[1], "hi": t[2]}


def capture_metrics(gt: GroundTruth, capture: GTCapture, result: dict[str, Any] | None, *, status: str = "ok",
                    error: str | None = None, run_s: float | None = None) -> dict[str, Any]:
    """Every scored number, every miss and the detection counts of one capture."""
    out = _empty(gt, capture)
    out["timing"] = {"run_s": run_s}
    if result is None or status != "ok":
        out["status"] = "missing" if status == "ok" else status
        out["error"] = error or ("no result" if result is None else None)
        return _fail_all(out, gt, capture, f"capture_{out['status']}")
    try:
        return _score(out, gt, capture, result, run_s)
    except Exception as exc:  # a malformed result must not stop the benchmark
        log.exception("scoring %s/%s failed", gt.property, capture.id)
        fresh = _empty(gt, capture)
        fresh.update(status="failed", error=f"scoring failed: {type(exc).__name__}: {exc}",
                     timing={"run_s": run_s})
        return _fail_all(fresh, gt, capture, "scoring_failed")


def _score(out: dict[str, Any], gt: GroundTruth, capture: GTCapture, result: dict[str, Any],
           run_s: float | None) -> dict[str, Any]:
    tier, prop, cid = capture.tier, gt.property, capture.id
    cap = result.get("capture") if isinstance(result.get("capture"), dict) else {}
    if cap.get("tier") not in (None, tier):
        out["flags"].append(f"result_tier:{cap.get('tier')}")
    q, cal_status = _interval_info(result)
    out["q"], out["calibration_status"] = q, cal_status
    match = match_capture(gt, capture, result, tier)
    out["match"] = match.to_dict()
    out["flags"] += match.flags + [f"{r.gt_room}:{f}" for r in match.rooms for f in r.flags]
    preds = {p.id: p for p in pred_rooms(result)}
    rooms = {r.id: r for r in gt.capture_rooms(capture)}
    recs, missing = out["records"], out["missing"]
    kw = {"tier": tier, "capture": cid, "prop": prop, "q": q}

    def add(rec: dict[str, Any] | None, miss: dict[str, Any]) -> None:
        if rec is None:
            missing.append({**miss, "reason": "no_value"})
        else:
            recs.append(rec)

    for rm in match.rooms:
        g = rooms[rm.gt_room]
        unit = _unit(prop, g.id)
        if rm.pred_room is None:
            missing.extend(_room_missing(gt, g, "room_not_found"))
            out["walls"]["missed"] += len(g.walls)
            out["openings"]["missed"] += len(g.openings)
            out["damage"]["missed"] += len(g.damage)
            continue
        p: PredRoom = preds[rm.pred_room]
        idx: dict[str, Any] = {"pred_room": p.id, "walls": {}, "openings": {}, "n_walls": len(p.walls),
                               "n_openings": len(p.openings),
                               "gt_wall_lengths": {w.id: w.length for w in g.walls}}
        out["rooms"]["matched"] += 1
        for gi, pj in rm.walls.pairs:
            w, pw = g.walls[gi], p.walls[pj]
            idx["walls"][w.id] = {**(_mdict(pw.get("length")) or {}), "pred_id": str(pw.get("id"))}
            if w.length is not None:
                add(record("wall_length", f"{g.id}/{w.id}", w.length, pw.get("length"), room=unit,
                           pred_id=str(pw.get("id")), **kw),
                    _missing("wall_length", f"{g.id}/{w.id}", w.length, room=unit, reason=""))
        for gi in rm.walls.missed:
            w = g.walls[gi]
            if w.length is not None:
                missing.append(_missing("wall_length", f"{g.id}/{w.id}", w.length, room=unit,
                                        reason="wall_not_found"))
        out["walls"]["matched"] += len(rm.walls.pairs)
        out["walls"]["missed"] += len(rm.walls.missed)
        out["walls"]["extra"] += len(rm.walls.extra)
        idx["matched_walls"] = sorted(g.walls[gi].id for gi, _ in rm.walls.pairs)
        for key, gval, kind in (("ceiling_height", g.ceiling_height, "ceiling_height"),
                                ("floor_area", g.floor_area, "floor_area")):
            idx[key] = _mdict(p.raw.get(key))
            if gval is not None:
                add(record(kind, g.id, gval, p.raw.get(key), room=unit, pred_id=p.id, **kw),
                    _missing(kind, g.id, gval, room=unit, reason=""))
        idx["perimeter"] = _mdict(p.raw.get("perimeter"))
        ops_by_id = {str(o.get("id")): o for o in p.openings}
        gt_ops = {o.id: o for o in g.openings}
        for om in rm.openings:
            keys = ("status", "gt", "pred", "gt_wall", "pred_wall", "gt_type", "pred_type", "center_err")
            item = {"room": g.id, **{k: getattr(om, k) for k in keys}}
            out["openings"]["items"].append(item)
            if om.status == "missed":
                out["openings"]["missed"] += 1
                o = gt_ops[om.gt]
                missing.append(_missing("opening_width", f"{g.id}/{o.id}", o.width, room=unit,
                                        reason="opening_missed"))
                continue
            if om.status == "phantom":
                out["openings"]["phantom"] += 1
                continue
            out["openings"]["matched"] += 1
            if om.gt_type != om.pred_type:
                out["openings"]["type_swaps"] += 1
            o, po = gt_ops[om.gt], ops_by_id[om.pred]
            idx["openings"][o.id] = {"pred_id": om.pred, "width": _mdict(po.get("width")),
                                     "height": _mdict(po.get("height"))}
            for kind, gval, key in (("opening_width", o.width, "width"),
                                    ("opening_height", o.height, "height")):
                if gval is not None:
                    add(record(kind, f"{g.id}/{o.id}", gval, po.get(key), room=unit, pred_id=om.pred,
                               opening_type=o.type, **kw),
                        _missing(kind, f"{g.id}/{o.id}", gval, room=unit, reason=""))
        idx["matched_openings"] = sorted(om.gt for om in rm.openings if om.status == "matched")
        _score_damage(out, g, rm, result, unit, kw)
        out["by_gt"][g.id] = idx
    out["rooms"]["pred"] = len(preds)
    out["rooms"]["missed"] = [r.gt_room for r in match.rooms if r.pred_room is None]
    out["rooms"]["extra"] = list(match.extra_rooms)
    full = capture.rooms is None
    for om in match.extra_openings:
        out["openings"]["phantom" if full else "phantom_unscored"] += 1
        out["openings"]["items"].append({"room": None, "status": "phantom" if full else "phantom_unscored",
                                         "gt": None, "pred": om.pred, "gt_wall": None,
                                         "pred_wall": om.pred_wall, "gt_type": None,
                                         "pred_type": om.pred_type, "center_err": None})
    for dm in match.extra_damage:
        out["damage"]["phantom" if full else "phantom_unscored"] += 1
        out["damage"]["items"].append({"room": None, "status": "phantom" if full else "phantom_unscored",
                                       "gt": None, "pred": dm.pred, "gt_class": None,
                                       "pred_class": dm.pred_class})
    fp = gt.footprint(capture)
    prop_d = result.get("property") if isinstance(result.get("property"), dict) else {}
    if fp is not None:
        add(record("footprint", "footprint", fp, prop_d.get("footprint_area"), room=None, **kw),
            _missing("footprint", "footprint", fp, room=None, reason=""))
    out["adjacency"] = adjacency_check(gt, capture, result, match)
    out["overlap"] = room_overlaps(result)
    out["drift"] = prop_d.get("drift_correction")
    timing = result.get("timing") if isinstance(result.get("timing"), dict) else {}
    stages = timing.get("stages") if isinstance(timing.get("stages"), dict) else {}
    out["timing"] = {"run_s": run_s, "total_s": num(timing.get("total_s")),
                     "stages": {str(k): num(v) for k, v in stages.items()}}
    out["flags"] += [str(f) for f in cap.get("flags") or []]
    return out


def _score_damage(out: dict[str, Any], g: GTRoom, rm: Any, result: dict[str, Any], unit: str,
                  kw: dict[str, Any]) -> None:
    preds = {str(d.get("id")): d for d in result.get("damage") or [] if isinstance(d, dict)}
    gts = {d.id: d for d in g.damage}
    for dm in rm.damage:
        out["damage"]["items"].append({"room": g.id, "status": dm.status, "gt": dm.gt, "pred": dm.pred,
                                       "gt_class": dm.gt_class, "pred_class": dm.pred_class,
                                       "surface": dm.gt_surface, "pred_surface": dm.pred_surface})
        if dm.status != "matched":
            out["damage"][dm.status] += 1
            continue
        out["damage"]["matched"] += 1
        out["damage"]["class_ok"] += dm.class_ok
        d, pd = gts[dm.gt], preds[dm.pred]
        extra = {"damage_class": d.cls, "class_ok": dm.class_ok}
        if d.cls == "crack":
            if d.length is not None and isinstance(pd.get("length"), dict):
                rec = record("damage_length", f"{g.id}/{d.id}", d.length, pd["length"], room=unit,
                             pred_id=dm.pred, **kw, **extra)
                if rec is not None:
                    out["records"].append(rec)
            continue
        if d.area is not None:
            rec = record("damage_area", f"{g.id}/{d.id}", d.area, pd.get("area"), room=unit, pred_id=dm.pred,
                         basis="mask", **kw, **extra)
        else:
            w, h = meas(pd.get("width")), meas(pd.get("height"))
            if d.bbox_area is None or w is None or h is None:
                continue
            bbox = {"value": w[0] * h[0], "lo": max(w[1], 0.0) * max(h[1], 0.0), "hi": w[2] * h[2]}
            rec = record("damage_area", f"{g.id}/{d.id}", d.bbox_area, bbox, room=unit, pred_id=dm.pred,
                         basis="bbox", **kw, **extra)
        if rec is not None:
            out["records"].append(rec)


def _repeat_threshold(spec: dict[str, Any], length: float, strict: bool = False) -> float:
    a = num(spec.get("abs"))
    r = num(spec.get("rel"))
    vals = [v for v in (a, None if r is None else r * length) if v is not None]
    if not vals:
        return math.inf
    if strict:
        return min(vals)
    return max(vals) if str(spec.get("rule", "max")) == "max" else min(vals)


def repeatability(metrics: list[dict[str, Any]], cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Wall-by-wall agreement, ceiling spread and structural sameness across captures of one property and
    tier.

    Rooms and walls are matched through the GT, so W2 of a room in one capture is compared with W2 of the same
    room in the other. Allowed |delta| is max(1 cm, 0.5% of the GT length) per gates.yaml, and the strict
    reading min(1 cm, 0.5%) is reported alongside.
    """
    tiers = (cfg or {}).get("tiers") or {}
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for m in metrics:
        if m.get("status") == "ok":
            groups[(m["property"], m["tier"])].append(m)
    walls, ceilings, structure = [], [], []
    for (prop, tier), ms in sorted(groups.items()):
        spec = {**REPEAT_DEFAULT, **((tiers.get(tier) or {}).get("repeatability") or {})}
        ms = sorted(ms, key=lambda m: m["capture"])
        for rid in sorted({r for m in ms for r in m["by_gt"]}):
            have = [m for m in ms if rid in m["by_gt"]]
            if len(have) < 2:
                continue
            for a, b in itertools.combinations(have, 2):
                A, B = a["by_gt"][rid], b["by_gt"][rid]
                order = list(A["gt_wall_lengths"])
                for wid in sorted(set(A["walls"]) & set(B["walls"]), key=order.index):
                    va, vb = A["walls"][wid].get("pred"), B["walls"][wid].get("pred")
                    if va is None or vb is None:
                        continue
                    L = A["gt_wall_lengths"].get(wid) or (va + vb) / 2.0
                    delta = abs(va - vb)
                    allowed, strict = _repeat_threshold(spec, L), _repeat_threshold(spec, L, strict=True)
                    walls.append({"property": prop, "tier": tier, "room": rid, "wall": wid,
                                  "captures": [a["capture"], b["capture"]], "values": [va, vb], "gt": L,
                                  "delta": delta, "allowed": allowed, "pass": delta <= allowed + 1e-9,
                                  "strict_allowed": strict, "strict_pass": delta <= strict + 1e-9})
                same_walls = A["n_walls"] == B["n_walls"] and A["matched_walls"] == B["matched_walls"]
                same_ops = (A["n_openings"] == B["n_openings"]
                            and A["matched_openings"] == B["matched_openings"])
                structure.append({"property": prop, "tier": tier, "room": rid,
                                  "captures": [a["capture"], b["capture"]],
                                  "n_walls": [A["n_walls"], B["n_walls"]],
                                  "n_openings": [A["n_openings"], B["n_openings"]], "same_walls": same_walls,
                                  "same_openings": same_ops})
            vals = [(m["capture"], m["by_gt"][rid]["ceiling_height"]["pred"]) for m in have
                    if m["by_gt"][rid].get("ceiling_height")]
            if len(vals) >= 2:
                hs = [v for _, v in vals]
                ceilings.append({"property": prop, "tier": tier, "room": rid,
                                 "captures": [c for c, _ in vals], "values": hs, "spread": max(hs) - min(hs)})
    by_tier: dict[str, dict[str, Any]] = {}
    for tier in sorted({w["tier"] for w in walls} | {c["tier"] for c in ceilings}):
        tw = [w for w in walls if w["tier"] == tier]
        tc = [c for c in ceilings if c["tier"] == tier]
        ts = [s for s in structure if s["tier"] == tier]
        by_tier[tier] = {"wall_pairs": len(tw), "pass": sum(w["pass"] for w in tw),
                         "strict_pass": sum(w["strict_pass"] for w in tw),
                         "max_delta": max((w["delta"] for w in tw), default=None),
                         "ceiling_rooms": len(tc),
                         "max_ceiling_spread": max((c["spread"] for c in tc), default=None),
                         "structure_pairs": len(ts),
                         "structure_same": sum(s["same_walls"] and s["same_openings"] for s in ts)}
    return {"walls": walls, "ceiling": ceilings, "structure": structure, "by_tier": by_tier}
