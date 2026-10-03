"""Drift-correction ablation: each multi-room capture scored with drift correction on and off.

The runner writes the off run to <capture>__nodrift. Both runs are scored against the same ground truth;
positive improvements mean the correction reduced the error.
"""

from __future__ import annotations

import math
from typing import Any

from scan2scope.bench.gates import item_threshold

NODRIFT_SUFFIX = "__nodrift"


def _mean(xs: list[float]) -> float | None:
    return sum(xs) / len(xs) if xs else None


def wall_stats(m: dict[str, Any], spec: dict[str, Any] | None) -> dict[str, Any]:
    walls = [r for r in m["records"] if r["kind"] == "wall_length"]
    errs = [abs(r["err"]) for r in walls]
    rel = [abs(r["rel_err"]) for r in walls if r["rel_err"] is not None]
    n_missing = sum(x["kind"] == "wall_length" for x in m["missing"])
    passed = None
    if spec:
        passed = sum(abs(r["err"]) <= (item_threshold(spec, r["gt"]) or math.inf) + 1e-12 for r in walls)
    n = len(walls) + n_missing
    return {"n": len(walls), "missing": n_missing, "mean_abs_err": _mean(errs), "max_abs_err": max(errs, default=None),
            "rms_err": math.sqrt(_mean([e * e for e in errs])) if errs else None, "mean_abs_rel_err": _mean(rel),
            "pass": passed, "pass_share": None if passed is None or n == 0 else passed / n}


def _side(m: dict[str, Any], spec: dict[str, Any] | None) -> dict[str, Any]:
    fp = next((r for r in m["records"] if r["kind"] == "footprint"), None)
    drift = m.get("drift") if isinstance(m.get("drift"), dict) else None
    return {"status": m["status"], "error": m.get("error"), "footprint_gt": None if fp is None else fp["gt"],
            "footprint_pred": None if fp is None else fp["pred"], "footprint_err": None if fp is None else fp["err"],
            "footprint_rel_err": None if fp is None else fp["rel_err"], "walls": wall_stats(m, spec),
            "drift_enabled": None if drift is None else bool(drift.get("enabled")), "drift": drift}


def _gain(on: float | None, off: float | None) -> float | None:
    if on is None or off is None:
        return None
    return abs(off) - abs(on)


def drift_ablation(main: list[dict[str, Any]], nodrift: list[dict[str, Any]], cfg: dict[str, Any] | None = None
                   ) -> dict[str, Any]:
    """Pairs every main capture with its __nodrift run: footprint and wall errors with the correction on and off."""
    tiers = (cfg or {}).get("tiers") or {}
    off_by = {(m["property"], m["capture"]): m for m in nodrift}
    entries = []
    for m in main:
        off = off_by.get((m["property"], m["capture"]))
        if off is None:
            continue
        spec = (tiers.get(m["tier"]) or {}).get("wall_length")
        on_s, off_s = _side(m, spec), _side(off, spec)
        entries.append({
            "property": m["property"], "capture": m["capture"], "tier": m["tier"], "synthetic": m.get("synthetic"),
            "on": on_s, "off": off_s,
            "footprint_gain": _gain(on_s["footprint_rel_err"], off_s["footprint_rel_err"]),
            "wall_mean_gain": _gain(on_s["walls"]["mean_abs_err"], off_s["walls"]["mean_abs_err"]),
            "wall_max_gain": _gain(on_s["walls"]["max_abs_err"], off_s["walls"]["max_abs_err"]),
            "on_has_drift_enabled": bool(on_s["drift_enabled"]),
            "off_has_drift_disabled": off_s["drift_enabled"] is False,
        })
    return {"entries": entries}
