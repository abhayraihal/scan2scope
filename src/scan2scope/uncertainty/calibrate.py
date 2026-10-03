"""Split-conformal fit of the per-tier interval multiplier q, with the room as the exchangeable unit.

A record is one measurement scored against ground truth: tier, kind, room (a unit id shared by repeat captures
of the same physical room), err (prediction minus truth) and half_width (the reported half-width, (hi - lo) / 2).
The score is |err| / half_width. Scores are pooled with equal weight per room and the quantile is taken at
ceil((n + 1) * level) / n for n rooms, which is above 1 (no finite quantile) for fewer than 9 rooms at level 0.9.
Below min_rooms, q keeps its current value with status prior and the plain quantile is kept as a diagnostic.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scan2scope.types import TIERS
from scan2scope.uncertainty.model import CALIBRATION_PATH, _num, load_calibration, tier_q

log = logging.getLogger("scan2scope.uncertainty")

LEVEL = 0.9
MIN_ROOMS = 9
MIN_FACTOR = 0.2  # a fitted multiplier never shrinks intervals below a fifth of their reported width

HEADER = ("# Interval multiplier q per tier, written by scan2scope.uncertainty.calibrate.fit_q.\n"
          "# status prior: not fitted (fewer than min_rooms ground-truth rooms in the tier); "
          "calibrated: split-conformal fit.\n")


@dataclass(frozen=True)
class CalRecord:
    tier: str
    kind: str
    room: str
    err: float
    half_width: float
    q: float | None = None  # q in force when the interval was reported, when known

    @property
    def score(self) -> float:
        e = abs(self.err)
        if self.half_width > 0:
            return e / self.half_width
        return 0.0 if e == 0 else math.inf


def as_record(r: Any) -> CalRecord | None:
    """CalRecord from a dict or object with tier, kind, room (or room_id, unit), err, half_width and optional q."""
    if isinstance(r, CalRecord):
        return r

    def get(k: str) -> Any:
        return r.get(k) if isinstance(r, dict) else getattr(r, k, None)

    room = next((get(k) for k in ("room", "room_id", "unit") if get(k) is not None), None)
    err, hw, tier = _num(get("err")), _num(get("half_width")), get("tier")
    if tier is None or room is None or err is None or hw is None:
        return None
    return CalRecord(str(tier), str(get("kind") or ""), str(room), err, hw, _num(get("q")))


def conformal_level(n_units: int, level: float = LEVEL) -> float:
    return math.inf if n_units <= 0 else math.ceil((n_units + 1) * level - 1e-9) / n_units


def room_quantile(records: list[CalRecord], p: float) -> float:
    """Quantile p of the scores, each room weighted equally and its weight split evenly over its records."""
    if not records or p > 1:
        return math.inf
    by_room: dict[str, list[float]] = defaultdict(list)
    for r in records:
        by_room[r.room].append(r.score)
    scores: list[float] = []
    weights: list[float] = []
    for vals in by_room.values():
        scores += vals
        weights += [1.0 / (len(by_room) * len(vals))] * len(vals)
    order = np.argsort(scores, kind="stable")
    s = np.asarray(scores)[order]
    cum = np.cumsum(np.asarray(weights)[order])
    i = int(np.searchsorted(cum, p - 1e-9, side="left"))
    return float(s[min(i, len(s) - 1)])


def _r4(x: float) -> float | None:
    return round(float(x), 4) if math.isfinite(x) else None


def _group(records: Iterable[Any]) -> dict[str, list[CalRecord]]:
    by_tier: dict[str, list[CalRecord]] = defaultdict(list)
    for r in records:
        rec = as_record(r)
        if rec is not None:
            by_tier[rec.tier].append(rec)
    return dict(sorted(by_tier.items()))


def fit_q(records: Iterable[Any], *, path: str | Path = CALIBRATION_PATH, write: bool = True,
          level: float = LEVEL, min_rooms: int = MIN_ROOMS) -> dict[str, Any]:
    """Fit q per tier from scored records, starting from the table at path, and write it back by default.

    New q = q in force when the intervals were reported (record q, else the table's) times the conformal
    quantile. Tiers without records keep their entry.
    """
    current = load_calibration(path)
    table: dict[str, Any] = {"level": level, "min_rooms": min_rooms,
                             "tiers": {t: dict(e) for t, e in current["tiers"].items() if isinstance(e, dict)}}
    for tier in TIERS:
        table["tiers"].setdefault(tier, {"q": 1.0, "status": "prior"})
    for tier, rs in _group(records).items():
        q_now, _, _ = tier_q(current, tier)
        used = [r.q for r in rs if r.q is not None and r.q > 0]
        q_reported = float(np.median(used)) if len(used) == len(rs) else q_now
        n_rooms = len({r.room for r in rs})
        kinds: dict[str, list[CalRecord]] = defaultdict(list)
        for r in rs:
            kinds[r.kind].append(r)
        entry: dict[str, Any] = {
            "q": q_now, "status": "prior", "n_rooms": n_rooms, "n_records": len(rs), "q_reported": q_reported,
            "empirical_quantile": _r4(room_quantile(rs, level)), "conformal_quantile": None,
            "coverage_as_reported": round(sum(r.score <= 1.0 for r in rs) / len(rs), 4),
            "by_kind": {k: {"n": len(v), "empirical_quantile": _r4(room_quantile(v, level))}
                        for k, v in sorted(kinds.items())},
            "fitted": dt.datetime.now(dt.UTC).date().isoformat(),
        }
        if n_rooms >= min_rooms:
            quant = room_quantile(rs, conformal_level(n_rooms, level))
            if math.isfinite(quant):
                entry.update(q=round(q_reported * max(quant, MIN_FACTOR), 4), status="calibrated",
                             conformal_quantile=round(quant, 4))
            else:
                log.warning("tier %s: conformal quantile is not finite (zero-width intervals that missed); "
                            "q stays at %.3f", tier, q_now)
        else:
            log.info("tier %s: %d rooms (< %d); q stays at %.3f with status prior, empirical quantile %s",
                     tier, n_rooms, min_rooms, q_now, entry["empirical_quantile"])
        table["tiers"][tier] = entry
    if write:
        write_calibration(table, path)
    return table


def write_calibration(table: dict[str, Any], path: str | Path = CALIBRATION_PATH) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(HEADER + yaml.safe_dump(table, sort_keys=False, default_flow_style=None))
    return p


def clopper_pearson(k: int, n: int, ci: float = 0.95) -> tuple[float, float]:
    """Exact binomial interval for k successes in n trials."""
    if n <= 0:
        return 0.0, 1.0
    from scipy.stats import beta

    a = (1.0 - ci) / 2.0
    lo = 0.0 if k <= 0 else float(beta.ppf(a, k, n - k + 1))
    hi = 1.0 if k >= n else float(beta.ppf(1.0 - a, k + 1, n - k))
    return lo, hi


def loro_coverage(records: Iterable[Any], *, level: float = LEVEL, min_rooms: int = MIN_ROOMS,
                  ci: float = 0.95, garbage_ratio: float = 2.0) -> dict[str, dict[str, Any]]:
    """Leave-one-room-out coverage per tier, for reports.

    Each room is scored with q refitted on the other rooms. When the other rooms are fewer than min_rooms the
    fit would keep the prior q, so rooms are scored with their intervals as reported (mode as_reported).
    A miss by more than garbage_ratio times the half-width counts as confident garbage.
    """
    out: dict[str, dict[str, Any]] = {}
    for tier, rs in _group(records).items():
        rooms = sorted({r.room for r in rs})
        n_train = len(rooms) - 1
        mode = "loro_conformal" if n_train >= min_rooms else "as_reported"
        covered = garbage = 0
        by_kind: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        for room in rooms:
            factor = 1.0
            if mode == "loro_conformal":
                train = [r for r in rs if r.room != room]
                factor = max(room_quantile(train, conformal_level(n_train, level)), MIN_FACTOR)
            for r in (r for r in rs if r.room == room):
                ok = r.score <= factor
                covered += ok
                garbage += r.score > garbage_ratio * factor
                by_kind[r.kind][0] += ok
                by_kind[r.kind][1] += 1
        lo, hi = clopper_pearson(covered, len(rs), ci)
        out[tier] = {
            "mode": mode, "n_rooms": len(rooms), "n_records": len(rs), "covered": covered,
            "coverage": covered / len(rs), "ci": [lo, hi], "ci_level": ci, "contains_nominal": lo <= level <= hi,
            "confident_garbage": garbage,
            "by_kind": {k: {"n": t, "covered": c, "coverage": c / t} for k, (c, t) in sorted(by_kind.items())},
        }
    return out
