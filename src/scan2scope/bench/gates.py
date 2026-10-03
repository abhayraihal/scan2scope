"""Evaluates bench/gates.yaml per tier from capture metrics, repeatability, the drift ablation and the h2h.

Each row: gate, tier, threshold (text), measured (number in the gate's unit) and measured_text, n, status
(pass | fail | n.a.), assumed, pass_share, kind (error | rate | calibration | count | check), shortfall and
the worst items. Error gates need every item within its allowance; a missing value (room, wall or capture not
found) is a failing item. The opening gate is a rate: misses and phantoms are failures against min_pass_rate.

Failing rows are ranked for the fix loop by a normalised shortfall `score`:
  error gates        worst |err| / allowance - 1 (a missing item counts as twice its allowance)
  rate gates         (target - measured) / target
  calibration        miss rate at the near end of the coverage CI over the nominal miss rate, minus 1
                     (under-coverage), or the CI's distance above nominal over the nominal miss rate
                     (over-coverage)
  confident garbage  the number of confident misses
  checks             share of failing items
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from scan2scope.bench.groundtruth import num
from scan2scope.types import TIERS

log = logging.getLogger("scan2scope.bench")

GATES_PATH = Path(__file__).resolve().parents[3] / "bench" / "gates.yaml"
MISSING_RATIO = 2.0
WORST_N = 5
H2H_MIN_DEFAULT = 0.70
FOOTPRINT_DEFAULT = 0.08
MIN_ROOMS = 9  # below this many independent rooms the interval multiplier stays at its prior (design.md)


def load_gates(path: str | Path | None = None) -> dict[str, Any]:
    """gates.yaml from the argument, $SCAN2SCOPE_GATES, the repo's bench/gates.yaml or ./bench/gates.yaml."""
    candidates = [path, os.environ.get("SCAN2SCOPE_GATES"), GATES_PATH, Path.cwd() / "bench" / "gates.yaml"]
    for c in candidates:
        if c and Path(c).is_file():
            cfg = yaml.safe_load(Path(c).read_text()) or {}
            cfg.setdefault("tiers", {})
            cfg.setdefault("calibration", {})
            cfg["_path"] = str(c)
            return cfg
    raise FileNotFoundError("bench/gates.yaml not found (pass a path or set SCAN2SCOPE_GATES)")


def item_threshold(spec: dict[str, Any], gt: float | None) -> float | None:
    """Allowed |error| for one item: abs, rel x |gt|, or max/min of the two by the spec's rule."""
    a = num(spec.get("abs"))
    r = num(spec.get("rel"))
    vals = [v for v in (a, None if r is None or gt is None else r * abs(gt)) if v is not None]
    if not vals:
        return None
    return min(vals) if str(spec.get("rule", "max")) == "min" else max(vals)


def _cm(x: float) -> str:
    return f"{100 * x:.1f} cm"


def _pct(x: float | None) -> str:
    return "n/a" if x is None or not math.isfinite(x) else f"{100 * x:.1f}%"


def threshold_text(spec: dict[str, Any]) -> str:
    a, r = num(spec.get("abs")), num(spec.get("rel"))
    if a is not None and r is not None:
        return f"|err| <= {spec.get('rule', 'max')}({_cm(a)}, {_pct(r)} of GT)"
    if r is not None:
        return f"|err| <= {_pct(r)} of GT"
    if a is not None:
        return f"|err| <= {_cm(a)}"
    return "no threshold"


def clopper_pearson(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    from scipy.stats import beta

    a = (1.0 - ci) / 2.0
    lo = 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def is_garbage(rec: dict[str, Any], ratio: float) -> bool:
    """A miss by more than ratio x the interval half-width."""
    if rec.get("covered"):
        return False
    hw = rec.get("half_width") or 0.0
    return abs(rec["err"]) > ratio * hw


def calibration(records: list[dict[str, Any]], *, level: float = 0.9, ci: float = 0.95,
                ratio: float = 2.0) -> dict[str, dict[str, Any]]:
    """Per tier: coverage of the 90% intervals, its Clopper-Pearson CI, half-widths and confident garbage."""
    by_tier: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in records:
        by_tier[r["tier"]].append(r)
    out = {}
    for tier, rs in by_tier.items():
        k = sum(bool(r["covered"]) for r in rs)
        lo, hi = clopper_pearson(k, len(rs), ci)
        kinds: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for r in rs:
            kinds[r["kind"]].append(r)
        garbage = [r for r in rs if is_garbage(r, ratio)]
        rel_hw = [r["half_width"] / abs(r["gt"]) for r in rs if r["gt"]]
        out[tier] = {
            "n": len(rs), "covered": k, "coverage": k / len(rs) if rs else None, "ci": [lo, hi],
            "ci_level": ci, "mean_rel_half_width": sum(rel_hw) / len(rel_hw) if rel_hw else None,
            "level": level, "contains_nominal": lo <= level <= hi,
            "rooms": len({r["room"] for r in rs if r["room"]}),
            "confident_garbage": len(garbage), "garbage_items": [_rec_ref(r) for r in garbage],
            "by_kind": {kd: {"n": len(v), "covered": sum(bool(r["covered"]) for r in v),
                             "coverage": sum(bool(r["covered"]) for r in v) / len(v),
                             "mean_half_width": sum(r["half_width"] for r in v) / len(v),
                             "mean_abs_err": sum(abs(r["err"]) for r in v) / len(v)}
                        for kd, v in sorted(kinds.items())},
        }
    return out


def _rec_ref(r: dict[str, Any]) -> dict[str, Any]:
    hw = r.get("half_width") or 0.0
    return {"item": r["item"], "capture": r["capture"], "property": r["property"], "kind": r["kind"],
            "gt": r["gt"], "pred": r["pred"], "lo": r["lo"], "hi": r["hi"], "err": r["err"],
            "miss_ratio": abs(r["err"]) / hw if hw > 0 else (math.inf if r["err"] else 0.0)}


def _ref(x: dict[str, Any], **extra: Any) -> dict[str, Any]:
    """Item reference for worst-item lists: item, capture and property of x, plus extra fields."""
    return {"item": x.get("item"), "capture": x.get("capture"), "property": x.get("property"), **extra}


def _row(gate: str, tier: str, threshold: str, *, kind: str, assumed: bool = False) -> dict[str, Any]:
    return {"gate": gate, "tier": tier, "threshold": threshold, "measured": None, "measured_text": "no data",
            "n": 0, "status": "n.a.", "assumed": bool(assumed), "pass_share": None, "kind": kind,
            "shortfall": None, "score": 0.0, "worst": []}


def _where(x: dict[str, Any]) -> str:
    return f"{x.get('property', '?')}/{x.get('capture', '?')}/{x['item']}"


def error_gate(gate: str, tier: str, spec: dict[str, Any], records: list[dict[str, Any]],
               missing: list[dict[str, Any]], scope: str) -> dict[str, Any]:
    """Every item within its allowance; missing values fail."""
    row = _row(gate, tier, f"{threshold_text(spec)} {scope}", kind="error", assumed=bool(spec.get("assumed")))
    items = []
    for r in records:
        thr = item_threshold(spec, r["gt"])
        if thr is None:
            continue
        ratio = abs(r["err"]) / thr if thr > 0 else (0.0 if r["err"] == 0 else math.inf)
        items.append(_ref(r, gt=r["gt"], err=r["err"], rel_err=r["rel_err"], allowed=thr, ratio=ratio,
                          **{"pass": ratio <= 1.0}))
    for m in missing:
        items.append(_ref(m, gt=m.get("gt"), err=None, rel_err=None,
                          allowed=item_threshold(spec, m.get("gt")), ratio=None, reason=m.get("reason"),
                          **{"pass": False}))
    if not items:
        return row
    n_pass = sum(i["pass"] for i in items)
    measured_items = [i for i in items if i["ratio"] is not None]
    n_missing = len(items) - len(measured_items)
    worst_ratio = max((i["ratio"] for i in measured_items), default=None)
    a, r = num(spec.get("abs")), num(spec.get("rel"))
    if r is not None and a is None:
        measured = max((abs(i["rel_err"]) for i in measured_items if i["rel_err"] is not None), default=None)
        unit_text = _pct
    elif a is not None and r is None:
        measured = max((abs(i["err"]) for i in measured_items), default=None)
        unit_text = _cm
    else:
        measured = worst_ratio
        unit_text = (lambda x: f"{x:.2f}x allowed")
    status = "pass" if n_pass == len(items) else "fail"
    worst = sorted(items, key=lambda i: (i["ratio"] is not None, -(i["ratio"] or 0.0)))
    worst = [i for i in worst if not i["pass"]][:WORST_N] or worst[:1]
    text = []
    if measured is not None:
        top = max(measured_items, key=lambda i: i["ratio"])
        text.append(f"worst {unit_text(measured)} ({_where(top)})")
    text.append(f"{n_pass}/{len(items)} pass")
    if n_missing:
        text.append(f"{n_missing} missing")
    eff = max([worst_ratio or 0.0] + ([MISSING_RATIO] if n_missing else []))
    score = max(eff - 1.0, 0.0) if status == "fail" else 0.0
    row.update(measured=measured, measured_text=", ".join(text), n=len(items), status=status,
               pass_share=n_pass / len(items), shortfall=worst_ratio, score=score, worst=worst)
    return row


def opening_gate(tier: str, spec: dict[str, Any], records: list[dict[str, Any]],
                 missing: list[dict[str, Any]], phantoms: list[dict[str, Any]]) -> dict[str, Any]:
    target = num(spec.get("min_pass_rate")) or 0.85
    width = threshold_text(spec)
    row = _row("opening_width", tier, f"{width} on >= {_pct(target)} of openings (misses and phantoms fail)",
               kind="rate", assumed=bool(spec.get("assumed")))
    passed = failed = 0
    worst = []
    for r in records:
        thr = item_threshold(spec, r["gt"])
        ok = thr is not None and abs(r["err"]) <= thr + 1e-12
        passed += ok
        if not ok:
            failed += 1
            worst.append(_ref(r, gt=r["gt"], err=r["err"], allowed=thr,
                              ratio=abs(r["err"]) / thr if thr else None, **{"pass": False}))
    for m in missing:
        failed += 1
        worst.append(_ref(m, gt=m.get("gt"), err=None, allowed=None, ratio=None, reason=m.get("reason"),
                          **{"pass": False}))
    for p in phantoms:
        failed += 1
        worst.append(_ref(p, gt=None, err=None, allowed=None, ratio=None, reason="phantom",
                          **{"pass": False}))
    n = passed + failed
    if n == 0:
        return row
    rate = passed / n
    status = "pass" if rate >= target - 1e-12 else "fail"
    n_missed, n_phantom = len(missing), len(phantoms)
    row.update(measured=rate, n=n, status=status, pass_share=rate, shortfall=max(target - rate, 0.0),
               score=(target - rate) / target if status == "fail" else 0.0, worst=worst[:WORST_N],
               measured_text=f"{passed}/{n} pass ({_pct(rate)}); {n_missed} missed, {n_phantom} phantom, "
                             f"{failed - n_missed - n_phantom} out of tolerance")
    return row


def calibration_rows(tier: str, cal: dict[str, Any] | None, cfg: dict[str, Any],
                     level: float) -> list[dict[str, Any]]:
    ci = num(cfg.get("ci")) or 0.95
    ratio = num(cfg.get("confident_garbage_ratio")) or 2.0
    assumed = bool(cfg.get("assumed"))
    cov = _row("calibration", tier, f"{_pct(ci)} Clopper-Pearson CI of coverage contains {level:.2f}",
               kind="calibration", assumed=assumed)
    gar = _row("confident_garbage", tier, f"no miss larger than {ratio:g} x half-width", kind="count",
               assumed=assumed)
    if not cal or not cal.get("n"):
        return [cov, gar]
    lo, hi = cal["ci"]
    ok = bool(cal["contains_nominal"])
    if hi < level:
        gap, score = level - hi, ((1.0 - hi) / max(1.0 - level, 1e-9)) - 1.0
    elif lo > level:
        gap, score = lo - level, (lo - level) / max(1.0 - level, 1e-9)
    else:
        gap, score = 0.0, 0.0
    thin = cal["rooms"] < MIN_ROOMS
    cov.update(measured=cal["coverage"], n=cal["n"], status="pass" if ok else "fail",
               pass_share=cal["coverage"], shortfall=gap, score=score if not ok else 0.0, thin_evidence=thin,
               measured_text=f"{cal['covered']}/{cal['n']} covered ({_pct(cal['coverage'])}), CI "
                             f"[{_pct(lo)}, {_pct(hi)}], {cal['rooms']} rooms"
                             + (f" (fewer than {MIN_ROOMS}: too few to show calibration)" if thin else ""))
    g = cal["confident_garbage"]
    items = sorted(cal["garbage_items"], key=lambda x: -x["miss_ratio"])
    worst = f", worst {items[0]['miss_ratio']:.1f}x half-width ({_where(items[0])})" if items else ""
    gar.update(measured=g, n=cal["n"], status="pass" if g == 0 else "fail", pass_share=1.0 - g / cal["n"],
               shortfall=float(g), score=float(g), worst=items[:WORST_N],
               measured_text=f"{g} of {cal['n']} values{worst}")
    return [cov, gar]


def check_gate(gate: str, tier: str, threshold: str, items: list[dict[str, Any]], *, assumed: bool = False,
               kind: str = "check", pass_details: bool = True) -> dict[str, Any]:
    """Pass when every item passes; items carry `pass` and a `detail` string."""
    row = _row(gate, tier, threshold, kind=kind, assumed=assumed)
    if not items:
        return row
    n_pass = sum(bool(i["pass"]) for i in items)
    status = "pass" if n_pass == len(items) else "fail"
    fails = [i for i in items if not i["pass"]]
    shown = fails or (items if pass_details else [])
    text = "; ".join(i["detail"] for i in shown[:3])
    if len(shown) > 3:
        text += f" (+{len(shown) - 3} more)"
    share = n_pass / len(items)
    row.update(measured=share, n=len(items), status=status, pass_share=share, shortfall=1.0 - share,
               score=1.0 - share, worst=fails[:WORST_N],
               measured_text=f"{n_pass}/{len(items)} pass" + (f"; {text}" if text else ""))
    return row


def _tier_metrics(metrics: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    return [m for m in metrics if m["tier"] == tier]


def _collect(ms: list[dict[str, Any]], kind: str, multi_only: bool = False) -> tuple[list, list]:
    recs, miss = [], []
    for m in ms:
        if multi_only and not m.get("multi_room"):
            continue
        recs += [r for r in m["records"] if r["kind"] == kind]
        miss += [{**x, "capture": m["capture"], "property": m["property"]} for x in m["missing"]
                 if x["kind"] == kind]
    return recs, miss


def _ceiling_rows(tier: str, spec: dict[str, Any], ms: list[dict[str, Any]], repeat: dict[str, Any] | None
                  ) -> tuple[dict[str, Any], dict[str, Any]]:
    """Ceiling accuracy (every room) and spread across repeat captures (every repeated room)."""
    acc = error_gate("ceiling_height", tier, spec, *_collect(ms, "ceiling_height"), "in every room")
    lim = num(spec.get("repeat_spread_abs"))
    items = [_ref({"item": c["room"], "capture": ",".join(c["captures"]), "property": c["property"]},
                  spread=c["spread"], detail=f"{c['property']}/{c['room']}: spread {_cm(c['spread'])} over "
                                             f"{len(c['values'])} captures",
                  **{"pass": lim is None or c["spread"] <= lim + 1e-12})
             for c in (repeat or {}).get("ceiling", []) if c["tier"] == tier]
    sp = check_gate("ceiling_spread", tier, f"max - min across captures <= {_cm(lim or 0.0)} per room", items)
    if items:
        worst = max(items, key=lambda i: i["spread"])
        sp["measured"] = worst["spread"]
        if sp["status"] == "fail" and lim:
            sp["shortfall"] = worst["spread"] / lim
            sp["score"] = worst["spread"] / lim - 1.0
    acc["mode"] = ceiling_mode(acc["status"], sp["status"])
    return acc, sp


def _repeat_rows(tier: str, spec: dict[str, Any], repeat: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Wall agreement between repeat captures, and whether they found the same walls and openings."""
    items = []
    for w in (repeat or {}).get("walls", []):
        if w["tier"] != tier:
            continue
        pair = " vs ".join(w["captures"])
        items.append(_ref({"item": f"{w['room']}/{w['wall']}", "capture": pair, "property": w["property"]},
                          delta=w["delta"], allowed=w["allowed"], strict_pass=w["strict_pass"],
                          detail=f"{w['property']}/{w['room']}/{w['wall']} {pair}: |delta| {_cm(w['delta'])} "
                                 f"(allowed {_cm(w['allowed'])})", **{"pass": w["pass"]}))
    limit = f"{spec.get('rule', 'max')}({_cm(num(spec.get('abs')) or 0.0)}, {_pct(num(spec.get('rel')))})"
    row = check_gate("repeatability", tier, f"|delta| <= {limit} per wall between captures", items,
                     assumed=bool(spec.get("assumed")))
    if items:
        worst = max(items, key=lambda i: i["delta"] / i["allowed"] if i["allowed"] else math.inf)
        strict = sum(i["strict_pass"] for i in items)
        row["measured"] = worst["delta"] / worst["allowed"] if worst["allowed"] else None
        row["measured_text"] = (f"{sum(i['pass'] for i in items)}/{len(items)} walls pass, strict reading "
                                f"{strict}/{len(items)}; worst {worst['detail']}")
        row["strict_pass_share"] = strict / len(items)
        if row["status"] == "fail" and row["measured"] is not None:
            row["shortfall"] = row["measured"]
            row["score"] = row["measured"] - 1.0
    same = []
    for st in (repeat or {}).get("structure", []):
        if st["tier"] != tier:
            continue
        pair = " vs ".join(st["captures"])
        counts = f"walls {'/'.join(map(str, st['n_walls']))}, openings {'/'.join(map(str, st['n_openings']))}"
        same.append(_ref({"item": st["room"], "capture": pair, "property": st["property"]},
                         detail=f"{st['property']}/{st['room']} {pair}: {counts}",
                         **{"pass": st["same_walls"] and st["same_openings"]}))
    structure = check_gate("repeat_structure", tier, "repeat captures find the same walls and openings "
                           "in each room", same, pass_details=False)
    return [row, structure]


def _produced_row(tier: str, ms: list[dict[str, Any]]) -> dict[str, Any]:
    items = [_ref({"item": m["capture"], "capture": m["capture"], "property": m["property"]},
                  detail=f"{m['property']}/{m['capture']}: {m['status']}"
                         + (f" ({m['error']})" if m.get("error") else ""), **{"pass": m["status"] == "ok"})
             for m in ms]
    return check_gate("result_produced", tier, "every capture returns a scored result", items,
                      pass_details=False)


def evaluate(metrics: list[dict[str, Any]], repeat: dict[str, Any] | None, ablation: dict[str, Any] | None,
             h2h: dict[str, Any] | None, cfg: dict[str, Any]) -> dict[str, Any]:
    """All gate rows per tier, the calibration table, the ceiling failure mode and the ranked failures."""
    level = num(cfg.get("interval_level")) or 0.9
    cal_cfg = cfg.get("calibration") or {}
    cal = calibration([r for m in metrics for r in m["records"]], level=level,
                      ci=num(cal_cfg.get("ci")) or 0.95,
                      ratio=num(cal_cfg.get("confident_garbage_ratio")) or 2.0)
    rows: list[dict[str, Any]] = []
    modes: dict[str, str] = {}
    tier_cfg = cfg.get("tiers") or {}
    for tier in [t for t in TIERS if t in tier_cfg] + sorted(t for t in tier_cfg if t not in TIERS):
        tcfg = tier_cfg.get(tier) or {}
        ms = _tier_metrics(metrics, tier)
        rows.append(_produced_row(tier, ms))
        if "wall_length" in tcfg:
            rows.append(error_gate("wall_length", tier, tcfg["wall_length"], *_collect(ms, "wall_length"),
                                   "on every wall"))
        if "ceiling_height" in tcfg:
            acc, spread = _ceiling_rows(tier, tcfg["ceiling_height"], ms, repeat)
            rows += [acc, spread]
            modes[tier] = acc["mode"]
        if "opening_width" in tcfg:
            phantoms = [{"item": f"{i['pred']}", "capture": m["capture"], "property": m["property"]}
                        for m in ms for i in m["openings"]["items"] if i["status"] == "phantom"]
            rows.append(opening_gate(tier, tcfg["opening_width"], *_collect(ms, "opening_width"), phantoms))
        if "floor_area" in tcfg:
            rows.append(error_gate("floor_area", tier, tcfg["floor_area"], *_collect(ms, "floor_area"),
                                   "in every room"))
        if "footprint" in tcfg:
            rows.append(error_gate("footprint", tier, tcfg["footprint"],
                                   *_collect(ms, "footprint", multi_only=True), "per multi-room capture"))
        if "stitch" in tcfg:
            rows.append(stitch_gate(tier, tcfg["stitch"], (tcfg.get("footprint") or {}), ms))
        if "repeatability" in tcfg:
            rows += _repeat_rows(tier, tcfg["repeatability"], repeat)
        if "drift_ablation" in tcfg:
            rows.append(drift_gate(tier, tcfg["drift_ablation"], ms, ablation))
        rows += calibration_rows(tier, cal.get(tier), cal_cfg, level)
        if h2h:
            row = h2h_gate(tier, h2h, cfg)
            if row is not None:
                rows.append(row)
    for row in rows:
        ms = [m for m in metrics if m["tier"] == row["tier"]]
        row["synthetic"] = _synthetic_label(ms)
        if not ms and row["status"] == "n.a.":
            row["measured_text"] = "no captures in this tier"
    ranked = sorted((r for r in rows if r["status"] == "fail"), key=lambda r: -r["score"])
    keys = ("gate", "tier", "score", "shortfall", "measured_text", "threshold")
    return {"rows": rows, "ranked_failures": [{k: r[k] for k in keys} for r in ranked], "calibration": cal,
            "ceiling_mode": modes, "config": cfg.get("_path")}


def _synthetic_label(ms: list[dict[str, Any]]) -> str:
    if not ms:
        return "none"
    s = [bool(m.get("synthetic")) for m in ms]
    return "all" if all(s) else ("some" if any(s) else "none")


def ceiling_mode(accuracy: str, spread: str) -> str:
    """bias, spread, both or none (GATE-09); n.a. without ceiling data."""
    if accuracy == "n.a.":
        return "n.a."
    bad_bias, bad_spread = accuracy == "fail", spread == "fail"
    if bad_bias and bad_spread:
        return "both"
    return "bias" if bad_bias else ("spread" if bad_spread else "none")


def stitch_gate(tier: str, spec: dict[str, Any], fp_spec: dict[str, Any],
                ms: list[dict[str, Any]]) -> dict[str, Any]:
    max_ov = num(spec.get("max_overlap_m2"))
    fp_rel = num(fp_spec.get("rel")) or FOOTPRINT_DEFAULT
    items = []
    for m in ms:
        if not m.get("multi_room"):
            continue
        name = f"{m['property']}/{m['capture']}"
        if m["status"] != "ok":
            items.append(_ref({"item": m["capture"], **m}, detail=f"{name}: {m['status']}",
                              **{"pass": False}))
            continue
        adj = m.get("adjacency") or {}
        ov = (m.get("overlap") or {}).get("max_m2", 0.0)
        fp = next((r for r in m["records"] if r["kind"] == "footprint"), None)
        parts, ok = [], True
        if spec.get("adjacency_exact", True):
            if adj.get("exact") is None:
                ok = False
                parts.append("GT adjacency unknown")
            elif not adj["exact"]:
                ok = False
                unmatched = f" unmatched {adj['unmatched_rooms']}" if adj.get("unmatched_rooms") else ""
                parts.append(f"adjacency missing {adj.get('missing')} extra {adj.get('extra')}{unmatched}")
            else:
                k = len(adj.get("gt") or [])
                parts.append(f"adjacency exact ({k} pair{'' if k == 1 else 's'})")
        if max_ov is not None:
            ok_ov = ov <= max_ov + 1e-12
            ok &= ok_ov
            parts.append(f"max overlap {ov:.3f} m2")
        if fp is None or fp["rel_err"] is None:
            ok = False
            parts.append("footprint missing")
        else:
            ok &= abs(fp["rel_err"]) <= fp_rel + 1e-12
            parts.append(f"footprint {_pct(fp['rel_err'])}")
        items.append({"item": m["capture"], "capture": m["capture"], "property": m["property"], "pass": ok,
                      "detail": f"{name}: " + ", ".join(parts)})
    thr = (f"adjacency exact, max pairwise overlap <= {max_ov:g} m2, footprint within {_pct(fp_rel)}"
           if max_ov is not None else f"adjacency exact, footprint within {_pct(fp_rel)}")
    return check_gate("stitch", tier, thr, items, assumed=bool(spec.get("assumed")))


def drift_gate(tier: str, spec: dict[str, Any], ms: list[dict[str, Any]], ablation: dict[str, Any] | None
               ) -> dict[str, Any]:
    entries = {(e["property"], e["capture"]): e for e in (ablation or {}).get("entries", [])}
    items = []
    for m in ms:
        if not m.get("multi_room"):
            continue
        name = f"{m['property']}/{m['capture']}"
        drift = m.get("drift") if isinstance(m.get("drift"), dict) else {}
        enabled = bool(drift.get("enabled")) if drift else False
        e = entries.get((m["property"], m["capture"]))
        ok = enabled and e is not None and m["status"] == "ok"
        if m["status"] != "ok":
            detail = f"{name}: {m['status']}"
        elif not enabled:
            detail = f"{name}: drift correction off in the main run (poses used as-is)"
        elif e is None:
            detail = f"{name}: no __nodrift run"
        else:
            on, off = e["on"]["footprint_rel_err"], e["off"]["footprint_rel_err"]
            detail = f"{name}: footprint err on {_pct(on)}, off {_pct(off)}"
        items.append(_ref({"item": m["capture"], **m}, detail=detail, **{"pass": ok}))
    row = check_gate("drift_ablation", tier, "drift correction on for every multi-room capture, with an off "
                     "run for the ablation", items, assumed=bool(spec.get("assumed")))
    if not spec.get("required", True) and row["status"] == "fail":
        row["status"] = "n.a."
    return row


def h2h_gate(tier: str, h2h: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any] | None:
    comps = [c for c in h2h.get("comparisons", []) if c.get("tier") == tier]
    if not comps:
        return None
    target = num((cfg.get("head_to_head") or {}).get("min_beat_or_tie")) or H2H_MIN_DEFAULT
    dims = [d for c in comps for d in c["dimensions"] if d.get("beat_or_tie") is not None]
    row = _row("head_to_head", tier, f"beat or tie magicplan on >= {_pct(target)} of shared dimensions",
               kind="rate", assumed=bool((cfg.get("head_to_head") or {}).get("assumed")))
    if not dims:
        return row
    k = sum(bool(d["beat_or_tie"]) for d in dims)
    share = k / len(dims)
    status = "pass" if share >= target - 1e-12 else "fail"
    losses = [{"item": f"{d['room']}/{d['dimension']}", "capture": d["capture"], "property": d["property"],
               "pass": False, "ours_err": d["ours_err"], "theirs_err": d["theirs_err"],
               **({"reason": "not reported by our capture"} if d["ours_err"] is None else {})}
              for d in dims if not d["beat_or_tie"]]
    row.update(measured=share, n=len(dims), status=status, pass_share=share,
               shortfall=max(target - share, 0.0),
               score=(target - share) / target if status == "fail" else 0.0, worst=losses[:WORST_N],
               measured_text=f"{k}/{len(dims)} dimensions beat or tie ({_pct(share)}) over "
                             f"{', '.join(sorted({c['capture'] for c in comps}))}")
    return row
