"""Plain-text summary of a result: every wall, opening and ceiling height with its interval, then the footprint,
damage, concealed-damage flags, scope and timing. Ids are the ones drawn on plan.svg."""

from __future__ import annotations

import math
import sys
from collections import defaultdict
from typing import Any, TextIO


def _f(x: Any, default: float | None = None) -> float | None:
    if x is None or isinstance(x, bool):
        return default
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default


def mvals(m: Any) -> tuple[float, float, float] | None:
    """(value, lo, hi) of a measurement dict, or None."""
    if not isinstance(m, dict):
        return None
    v = _f(m.get("value"))
    if v is None:
        return None
    return v, _f(m.get("lo"), v), _f(m.get("hi"), v)


def half_width(m: Any) -> float | None:
    vals = mvals(m)
    return None if vals is None else max(vals[0] - vals[1], vals[2] - vals[0], 0.0)


def fmt_iv(m: Any, nd: int = 3) -> str:
    vals = mvals(m)
    if vals is None:
        return "-"
    v, lo, hi = vals
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


def fmt_v(m: Any, nd: int = 3) -> str:
    vals = mvals(m)
    return "-" if vals is None else f"{vals[0]:.{nd}f}"


def fmt_pm(m: Any, nd: int = 2, unit: str = "m") -> str:
    """'3.41 m ±0.03' with the larger side of the interval, so the label never understates it."""
    vals = mvals(m)
    if vals is None:
        return "?"
    return f"{vals[0]:.{nd}f} {unit} ±{half_width(m):.{nd}f}"


def short_id(ident: Any, room_id: Any) -> str:
    s, prefix = str(ident), f"{room_id}-"
    return s.removeprefix(prefix)


def drift_state(result: dict[str, Any]) -> str:
    prop = result.get("property") or {}
    d = prop.get("drift_correction")
    tier = (result.get("capture") or {}).get("tier")
    if not isinstance(d, dict):
        return "not applicable (rooms reconstructed separately)" if tier == "photo" else "not recorded"
    if d.get("enabled") is False:
        return "off"
    parts = [k.replace("_", " ") for k, v in d.items() if k != "enabled" and v is True]
    nums = [f"{k.replace('_', ' ')} {float(v):.3g}" for k, v in d.items()
            if not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v)][:2]
    state = "on" if d.get("enabled") is True else "recorded"
    extra = ", ".join(parts + nums)
    return f"{state} ({extra})" if extra else state


def _table(headers: list[str], rows: list[list[str]], indent: str = "  ") -> list[str]:
    widths = [max(len(h), *(len(r[i]) for r in rows)) if rows else len(h) for i, h in enumerate(headers)]
    out = [indent + "  ".join(h.ljust(w) for h, w in zip(headers, widths)).rstrip(),
           indent + "  ".join("-" * w for w in widths)]
    out += [indent + "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip() for r in rows]
    return out


def _join(x: Any) -> str:
    return ",".join(str(v) for v in x) if isinstance(x, list) and x else "-"


def _room_lines(room: dict[str, Any]) -> list[str]:
    rid, label, hint = room.get("id"), room.get("label"), room.get("source_hint")
    head = f"ROOM {rid}  {label}" + (f'  (folder "{hint}")' if hint and hint != label else "")
    lines = ["", head,
             (f"  floor area {fmt_iv(room.get('floor_area'), 2)} m2   perimeter {fmt_iv(room.get('perimeter'))} m   "
              f"ceiling height {fmt_iv(room.get('ceiling_height'))} m")]
    walls = [w for w in room.get("walls") or [] if isinstance(w, dict)]
    rows = [[str(w.get("id")), fmt_iv(w.get("length")), fmt_iv(w.get("height")),
             f"{_f(w.get('observed_fraction'), 0.0):.2f}", _join(w.get("flags"))] for w in walls]
    lines += _table(["wall", "length m [lo, hi]", "height m [lo, hi]", "observed", "flags"], rows)
    ops = [o for o in room.get("openings") or [] if isinstance(o, dict)]
    if ops:
        rows = [[str(o.get("id")), str(o.get("type")), str(o.get("wall_id")), fmt_iv(o.get("width")),
                 fmt_iv(o.get("height")), fmt_v(o.get("offset")), fmt_v(o.get("sill")),
                 str(o.get("connects_to") or "-"), _join(o.get("flags"))] for o in ops]
        lines.append("")
        lines += _table(["opening", "type", "wall", "width m [lo, hi]", "height m [lo, hi]", "offset m", "sill m",
                         "to", "flags"], rows)
    if room.get("flags"):
        lines.append(f"  room flags: {_join(room.get('flags'))}")
    return lines


def format_summary(result: dict[str, Any]) -> str:
    cap = result.get("capture") or {}
    interval = (result.get("conventions") or {}).get("interval") or {}
    prop = result.get("property") or {}
    q = _f(interval.get("q"))
    qtxt = f", q {q:.2f} ({interval.get('calibration', 'prior')})" if q is not None else ""
    lines = [f"scan2scope  capture {cap.get('id')}  tier {cap.get('tier')}  schema {result.get('schema_version')}",
             (f"every number is value [lo, hi] at {(_f(interval.get('level'), 0.9) or 0.9):.0%} nominal{qtxt}; "
              "lengths m, areas m2")]
    if cap.get("flags"):
        lines.append(f"capture flags: {_join(cap.get('flags'))}")

    lines += ["", "PROPERTY",
              (f"  footprint {fmt_iv(prop.get('footprint_area'), 2)} m2   extent x {fmt_iv(prop.get('extent_x'))} m   "
               f"extent y {fmt_iv(prop.get('extent_y'))} m"),
              f"  drift correction: {drift_state(result)}"]
    stitch = prop.get("stitch")
    if isinstance(stitch, dict) and stitch:
        simple = [f"{k}={v}" for k, v in stitch.items() if isinstance(v, (str, int, float, bool)) or v is None]
        lines.append(f"  stitch: {', '.join(simple)[:300] or 'recorded'}")
    adj = [a for a in prop.get("adjacency") or [] if isinstance(a, dict)]
    if adj:
        rows = [[f"{a.get('room_a')} - {a.get('room_b')}", f"{a.get('opening_a') or '-'} / {a.get('opening_b') or '-'}",
                 str(a.get("source")), f"{_f(a.get('confidence'), 0.0):.2f}"] for a in adj]
        lines.append("")
        lines += _table(["rooms", "via openings", "source", "confidence"], rows)
    if prop.get("flags"):
        lines.append(f"  property flags: {_join(prop.get('flags'))}")

    for room in result.get("rooms") or []:
        if isinstance(room, dict):
            lines += _room_lines(room)

    damage = [d for d in result.get("damage") or [] if isinstance(d, dict)]
    lines += ["", "DAMAGE"]
    if damage:
        rows = [[str(d.get("id")), str(d.get("room_id")), str(d.get("surface_id")), str(d.get("class")),
                 f"{_f(d.get('score'), 0.0):.2f}", fmt_iv(d.get("area"), 3), fmt_iv(d.get("width"), 2),
                 fmt_iv(d.get("height"), 2), fmt_iv(d.get("length"), 2)] for d in damage]
        lines += _table(["id", "room", "surface", "class", "score", "area m2 [lo, hi]", "width m", "height m",
                         "length m"], rows)
    else:
        lines.append("  none detected")

    flags = [f for f in result.get("concealed_damage_flags") or [] if isinstance(f, dict)]
    lines += ["", "CONCEALED-DAMAGE FLAGS"]
    if flags:
        rows = [[str(f.get("id")), str(f.get("rule_id")), str(f.get("severity")), str(f.get("room_id")),
                 _join(f.get("surface_ids")), _join(f.get("damage_ids")), str(f.get("title"))] for f in flags]
        lines += _table(["id", "rule", "severity", "room", "surfaces", "damage", "title"], rows)
        lines += [f"  {f.get('id')}: {f.get('recommendation')}" for f in flags if f.get("recommendation")]
    else:
        lines.append("  none")

    scope = [s for s in result.get("scope") or [] if isinstance(s, dict)]
    lines += ["", "SCOPE (Xactimate-style codes, not an official price list)"]
    if scope:
        by_cat: dict[str, list[dict]] = defaultdict(list)
        for s in scope:
            by_cat[str(s.get("category"))].append(s)
        for cat in sorted(by_cat):
            rows = [[str(s.get("id")), str(s.get("room_id")), str(s.get("surface_id")), str(s.get("activity")),
                     str(s.get("selector")), f"{fmt_iv(s.get('quantity'), 2)} {s.get('unit')}",
                     str(s.get("description")), _join(s.get("damage_ids")), _join(s.get("flag_ids")),
                     str(s.get("rule_id") or "-")] for s in by_cat[cat]]
            lines.append(f"  {cat}")
            lines += _table(["id", "room", "surface", "act", "sel", "quantity [lo, hi]", "description", "damage",
                             "flags", "rule"], rows, indent="    ")
    else:
        lines.append("  none")

    timing = result.get("timing") or {}
    stages = timing.get("stages") if isinstance(timing.get("stages"), dict) else {}
    lines += ["", "TIMING"]
    rows = [[str(k), f"{_f(v, 0.0):.2f}"] for k, v in stages.items()]
    rows.append(["total", f"{_f(timing.get('total_s'), 0.0):.2f}"])
    lines += _table(["stage", "seconds"], rows)
    return "\n".join(lines) + "\n"


def print_summary(result: dict[str, Any], file: TextIO | None = None) -> None:
    print(format_summary(result), file=file or sys.stdout, end="")
