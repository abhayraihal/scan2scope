"""Head-to-head against magicplan on the same rooms, dimension by dimension against the same ground truth.

Inputs under bench/data/<property>/magicplan/:

  statistics.csv   the magicplan Statistics export: one row per room with floor area, perimeter and wall height.
                   Parsed defensively: delimiter sniffed, header row searched, units read from the header or the
                   cells (m, cm, m2, ft, sq ft, feet-inches), decimal commas accepted; the columns found are reported.
  dimensions.yaml  wall lengths and opening widths transcribed from the magicplan Sketch PDF, keyed to GT ids:

      app: magicplan
      version: "9.4.1"
      mode: "AR camera, no LiDAR"
      rooms:
        "02 kitchen":
          name: Kitchen                 # the room's name in statistics.csv
          walls: {W1: 3.41, W2: 2.95}
          openings: {D1: 0.80}
          ceiling_height: 2.40          # optional; otherwise the CSV wall height is used

Tie rule: ours beats or ties when |ours - gt| <= |theirs - gt| + 0.003 m for lengths, or when our relative error
is within 0.2 percentage points of theirs for areas.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from pathlib import Path
from typing import Any

import yaml

from scan2scope.bench.groundtruth import GroundTruth, norm_name, num, room_label

log = logging.getLogger("scan2scope.bench")

LENGTH_TIE_M = 0.003
AREA_TIE_PP = 0.2
FT = 0.3048
INCH = 0.0254

_SQFT = re.compile(r"(sq\.?\s*ft|ft\s*(²|2|\^2)|square\s*feet|sf\b)", re.IGNORECASE)
_SQM = re.compile(r"(m\s*(²|2|\^2)|sq\.?\s*m\b|square\s*met)", re.IGNORECASE)
_FT = re.compile(r"(\bft\b|feet|foot|')", re.IGNORECASE)
_CM = re.compile(r"\bcm\b", re.IGNORECASE)
_MM = re.compile(r"\bmm\b", re.IGNORECASE)
_FEET_INCHES = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*'\s*(?:(\d+(?:[.,]\d+)?)\s*(?:\"|''|in)?)?\s*$")


def _unit_of(text: str, kind: str) -> str | None:
    if kind == "area":
        if _SQFT.search(text):
            return "sqft"
        if _SQM.search(text):
            return "m2"
        return None
    if _CM.search(text):
        return "cm"
    if _MM.search(text):
        return "mm"
    if _FT.search(text):
        return "ft"
    if re.search(r"\bm\b|metre|meter", text, re.IGNORECASE):
        return "m"
    return None


def parse_quantity(cell: Any, kind: str, header_unit: str | None = None) -> float | None:
    """Metres (kind length) or square metres (kind area) from a number or a string with units."""
    if cell is None:
        return None
    if not isinstance(cell, str):
        v = num(cell)
        return None if v is None else _convert(v, header_unit, kind)
    s = cell.strip()
    if not s:
        return None
    if kind == "length":
        fi = _FEET_INCHES.match(s)
        if fi:
            feet = float(fi.group(1).replace(",", "."))
            inches = float(fi.group(2).replace(",", ".")) if fi.group(2) else 0.0
            return feet * FT + inches * INCH
    m = re.search(r"-?\d+(?:[.,]\d+)*", s.replace("\u00a0", "").replace("\u202f", "").replace(" ", ""))
    if not m:
        return None
    raw = m.group(0)
    if raw.count(",") == 1 and raw.count(".") == 0:
        raw = raw.replace(",", ".")
    elif raw.count(",") and raw.count("."):
        raw = raw.replace(",", "") if raw.rfind(".") > raw.rfind(",") else raw.replace(".", "").replace(",", ".")
    v = num(raw)
    if v is None:
        return None
    return _convert(v, _unit_of(s, kind) or header_unit, kind)


def _convert(v: float, unit: str | None, kind: str) -> float:
    if kind == "area":
        return v * FT * FT if unit == "sqft" else v
    return {"cm": v / 100.0, "mm": v / 1000.0, "ft": v * FT}.get(unit or "m", v)


def _clean_header(h: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[\(\[].*?[\)\]]", " ", h)).strip().lower()


def _pick(headers: list[str], include: list[str], exclude: list[str], prefer: list[str]) -> int | None:
    cands = []
    for k, h in enumerate(headers):
        c = _clean_header(h)
        if any(x in c for x in include) and not any(x in c for x in exclude):
            cands.append((-sum(p in c for p in prefer), k))
    return min(cands)[1] if cands else None


def parse_statistics(path: str | Path) -> dict[str, Any]:
    """Rows {name, floor_area, perimeter, wall_height} from a magicplan Statistics CSV, with what was found."""
    path = Path(path)
    out: dict[str, Any] = {"path": str(path), "columns": [], "mapping": {}, "units": {}, "rows": [], "flags": []}
    try:
        text = path.read_text(encoding="utf-8-sig", errors="replace")
    except OSError as exc:
        out["flags"].append(f"unreadable:{type(exc).__name__}")
        return out
    # The header line decides the delimiter: cells may hold decimal commas, header names rarely do.
    header_line = next((ln for ln in text.splitlines() if re.search(r"room|name|space", ln, re.IGNORECASE)
                        and re.search(r"area|surface", ln, re.IGNORECASE)), "")
    counts = {d: header_line.count(d) for d in (";", "\t", ",")}
    delim = max(counts, key=lambda d: (counts[d], d != ",")) if any(counts.values()) else ","
    rows = list(csv.reader(io.StringIO(text), delimiter=delim))
    header_at = None
    for k, r in enumerate(rows):
        cells = [_clean_header(c) for c in r]
        if any(c in ("room", "name", "room name", "space", "label") or c.startswith("room") for c in cells) and \
                any("area" in c or "surface" in c for c in cells):
            header_at = k
            break
    if header_at is None:
        out["flags"].append("no_header_row")
        return out
    headers = [h.strip() for h in rows[header_at]]
    out["columns"] = headers
    cols = {
        "name": _pick(headers, ["room", "name", "space", "label"], ["area", "perim", "height", "floor"],
                      ["room name", "room", "name"]),
        "floor_area": _pick(headers, ["area", "surface"], ["wall", "window", "door", "opening", "ceiling"],
                            ["floor", "net", "ground"]),
        "perimeter": _pick(headers, ["perimeter"], ["wall area"], ["floor", "room"]),
        "wall_height": _pick(headers, ["height"], ["door", "window", "sill", "opening"], ["wall", "ceiling"]),
    }
    out["mapping"] = {k: (None if v is None else headers[v]) for k, v in cols.items()}
    kinds = {"floor_area": "area", "perimeter": "length", "wall_height": "length"}
    units = {k: (None if cols[k] is None else _unit_of(headers[cols[k]], kinds[k])) for k in kinds}
    out["units"] = units
    if cols["name"] is None:
        out["flags"].append("no_room_name_column")
        return out
    for r in rows[header_at + 1:]:
        if cols["name"] >= len(r):
            continue
        name = r[cols["name"]].strip()
        if not name or norm_name(name) in ("total", "sum", "totals", "all rooms"):
            continue
        row = {"name": name, "raw": {headers[k]: r[k] for k in range(min(len(headers), len(r)))}}
        for k, kind in kinds.items():
            c = cols[k]
            row[k] = None if c is None or c >= len(r) else parse_quantity(r[c], kind, units[k])
        out["rows"].append(row)
    return out


def _value_map(x: Any, kind: str = "length") -> dict[str, float]:
    if isinstance(x, dict):
        items = x.items()
    elif isinstance(x, list):
        items = [(str(e.get("id")), e.get("length", e.get("width", e.get("value")))) for e in x if isinstance(e, dict)]
    else:
        return {}
    out = {}
    for k, v in items:
        q = parse_quantity(v, kind)
        if q is not None and q > 0:
            out[str(k)] = q
    return out


def load_dimensions(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    out: dict[str, Any] = {"path": str(path), "app": "magicplan", "version": None, "mode": None, "rooms": {},
                           "flags": []}
    try:
        doc = yaml.safe_load(path.read_text()) or {}
    except (OSError, yaml.YAMLError) as exc:
        out["flags"].append(f"unreadable:{type(exc).__name__}")
        return out
    if not isinstance(doc, dict):
        out["flags"].append("not_a_mapping")
        return out
    out.update(app=str(doc.get("app") or "magicplan"), version=None if doc.get("version") is None
               else str(doc["version"]), mode=None if doc.get("mode") is None else str(doc["mode"]))
    rooms = doc.get("rooms") or {}
    if isinstance(rooms, list):
        rooms = {str(r.get("id") or r.get("gt_room")): r for r in rooms if isinstance(r, dict)}
    for rid, r in rooms.items() if isinstance(rooms, dict) else []:
        if not isinstance(r, dict):
            continue
        name = r.get("name", r.get("magicplan_name"))
        out["rooms"][str(rid)] = {
            "name": None if name is None else str(name),
            "walls": _value_map(r.get("walls")), "openings": _value_map(r.get("openings")),
            "ceiling_height": parse_quantity(r.get("ceiling_height"), "length"),
            "floor_area": parse_quantity(r.get("floor_area"), "area"),
            "perimeter": parse_quantity(r.get("perimeter"), "length"),
        }
    return out


def beat_or_tie(ours: float, theirs: float, gt: float, kind: str) -> bool:
    if kind == "area":
        return abs(ours - gt) / gt * 100.0 <= abs(theirs - gt) / gt * 100.0 + AREA_TIE_PP + 1e-9
    return abs(ours - gt) <= abs(theirs - gt) + LENGTH_TIE_M + 1e-9


def _their_rooms(gt: GroundTruth, stats: dict[str, Any] | None, dims: dict[str, Any] | None
                 ) -> tuple[dict[str, dict[str, Any]], list[str]]:
    """GT room id -> magicplan numbers, from dimensions.yaml and the CSV rows it names (or that match by name)."""
    flags: list[str] = []
    rooms: dict[str, dict[str, Any]] = {}
    gt_ids = {r.id for r in gt.rooms}
    for rid, r in (dims or {}).get("rooms", {}).items():
        if rid not in gt_ids:
            flags.append(f"dimensions_unknown_room:{rid}")
            continue
        rooms[rid] = dict(r)
    rows = (stats or {}).get("rows", [])
    used: set[int] = set()
    for rid in sorted(gt_ids):
        entry = rooms.get(rid, {})
        wanted = entry.get("name")
        hit = None
        for k, row in enumerate(rows):
            if k in used:
                continue
            n = norm_name(row["name"])
            if wanted is not None and n == norm_name(wanted):
                hit = k
                break
            if wanted is None and n in (norm_name(rid), norm_name(room_label(rid))):
                hit = k
                break
        if hit is None:
            if wanted is not None and rows:
                flags.append(f"statistics_row_not_found:{rid}:{wanted}")
            continue
        used.add(hit)
        row = rows[hit]
        e = rooms.setdefault(rid, {"name": row["name"], "walls": {}, "openings": {}, "ceiling_height": None,
                                   "floor_area": None, "perimeter": None})
        e["csv_row"] = row["name"]
        for key, src in (("floor_area", "floor_area"), ("perimeter", "perimeter"), ("ceiling_height", "wall_height")):
            if e.get(key) is None and row.get(src) is not None:
                e[key] = row[src]
    return rooms, flags


def compare_capture(gt: GroundTruth, m: dict[str, Any], theirs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Dimension-by-dimension table for one of our captures against magicplan.

    A dimension counts when the GT and magicplan both have it. If our capture covers the room but did not report
    the dimension (room, wall or opening not found), it counts as a loss.
    """
    dims = []

    def add(room: str, dimension: str, kind: str, g: float | None, ours: float | None, th: float | None) -> None:
        if g is None or th is None or g <= 0:
            return
        dims.append({"property": gt.property, "capture": m["capture"], "tier": m["tier"], "room": room,
                     "dimension": dimension, "kind": kind, "gt": g, "ours": ours, "theirs": th,
                     "ours_err": None if ours is None else ours - g, "theirs_err": th - g,
                     "beat_or_tie": ours is not None and beat_or_tie(ours, th, g, kind)})

    covered = set(m.get("gt_rooms") or [])
    for rid in sorted(theirs):
        room = gt.room(rid)
        if room is None or rid not in covered:
            continue
        idx = m.get("by_gt", {}).get(rid) or {"walls": {}, "openings": {}}
        t = theirs[rid]
        for w in room.walls:
            add(rid, f"{w.id} length", "length", w.length, (idx["walls"].get(w.id) or {}).get("pred"),
                t["walls"].get(w.id))
        for o in room.openings:
            ours_w = ((idx["openings"].get(o.id) or {}).get("width") or {}).get("pred")
            add(rid, f"{o.id} width", "length", o.width, ours_w, t["openings"].get(o.id))
        add(rid, "floor area", "area", room.floor_area, (idx.get("floor_area") or {}).get("pred"), t.get("floor_area"))
        add(rid, "perimeter", "length", room.perimeter, (idx.get("perimeter") or {}).get("pred"), t.get("perimeter"))
        add(rid, "ceiling height", "length", room.ceiling_height, (idx.get("ceiling_height") or {}).get("pred"),
            t.get("ceiling_height"))
    k = sum(d["beat_or_tie"] for d in dims)
    return {"property": gt.property, "capture": m["capture"], "tier": m["tier"], "synthetic": m.get("synthetic"),
            "dimensions": dims, "n": len(dims), "beat_or_tie": k, "share": k / len(dims) if dims else None}


def head_to_head(gt: GroundTruth, metrics: list[dict[str, Any]], folder: str | Path | None = None
                 ) -> dict[str, Any] | None:
    """Comparisons of every scored capture of this property that covers a magicplan room, or None without data."""
    folder = Path(folder) if folder is not None else gt.root / "magicplan"
    csv_path, dims_path = folder / "statistics.csv", folder / "dimensions.yaml"
    if not csv_path.is_file() and not dims_path.is_file():
        return None
    stats = parse_statistics(csv_path) if csv_path.is_file() else None
    dims = load_dimensions(dims_path) if dims_path.is_file() else None
    theirs, flags = _their_rooms(gt, stats, dims)
    flags += (stats or {}).get("flags", []) + (dims or {}).get("flags", [])
    if stats is None:
        flags.append("no_statistics_csv")
    if dims is None:
        flags.append("no_dimensions_yaml")
    comps = [compare_capture(gt, m, theirs) for m in metrics
             if m["property"] == gt.property and m["status"] == "ok"]
    comps = [c for c in comps if c["n"]]
    return {"property": gt.property, "app": (dims or {}).get("app", "magicplan"), "version": (dims or {}).get("version"),
            "mode": (dims or {}).get("mode"), "columns_found": (stats or {}).get("columns", []),
            "column_mapping": (stats or {}).get("mapping", {}), "units": (stats or {}).get("units", {}),
            "rooms": theirs, "comparisons": comps, "flags": flags}


def merge(results: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """One h2h record over properties: comparisons concatenated, per-property details kept."""
    results = [r for r in results if r]
    if not results:
        return None
    return {"comparisons": [c for r in results for c in r["comparisons"]],
            "by_property": {r["property"]: {k: v for k, v in r.items() if k != "comparisons"} for r in results}}
