"""Validation of a result dict: JSON Schema 2020-12 (schema/scan2scope.schema.json) plus semantic checks."""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

SCHEMA_ENV = "SCAN2SCOPE_SCHEMA"
REPO_SCHEMA = Path(__file__).resolve().parents[3] / "schema" / "scan2scope.schema.json"
TOL = 1e-9


def schema_path() -> Path:
    candidates = [Path(os.environ[SCHEMA_ENV])] if os.environ.get(SCHEMA_ENV) else []
    candidates.append(REPO_SCHEMA)
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"scan2scope schema not found at {', '.join(map(str, candidates))}; "
                            f"set {SCHEMA_ENV} to its path")


@functools.lru_cache(maxsize=4)
def _validator(path: str) -> Draft202012Validator:
    schema = json.loads(Path(path).read_text())
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def _jpath(parts: Any) -> str:
    return "$" + "".join(f"[{p}]" if isinstance(p, int) else f".{p}" for p in parts)


def _short(msg: str, n: int = 240) -> str:
    return msg if len(msg) <= n else msg[: n - 3] + "..."


def _num(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _walk(x: Any, path: str, out: list[str]) -> None:
    """Non-finite numbers anywhere, and lo <= value <= hi for every measurement object."""
    if isinstance(x, dict):
        if {"value", "lo", "hi", "unit"} <= set(x) and all(_num(x[k]) for k in ("value", "lo", "hi")):
            v, lo, hi = x["value"], x["lo"], x["hi"]
            if lo > v + TOL:
                out.append(f"{path}: lo {lo} > value {v}")
            if v > hi + TOL:
                out.append(f"{path}: value {v} > hi {hi}")
        for k, v in x.items():
            _walk(v, f"{path}.{k}", out)
    elif isinstance(x, list):
        for i, v in enumerate(x):
            _walk(v, f"{path}[{i}]", out)
    elif isinstance(x, float) and not math.isfinite(x):
        out.append(f"{path}: non-finite number {x}")


def _items(x: Any, path: str) -> list[tuple[str, dict]]:
    return [(f"{path}[{i}]", v) for i, v in enumerate(x)] if isinstance(x, list) else []


def _in(x: Any, known: set[Any]) -> bool:
    return isinstance(x, (str, int)) and x in known


def _ids(items: list[tuple[str, Any]], what: str, out: list[str]) -> set[Any]:
    seen: dict[Any, str] = {}
    for p, it in items:
        if not isinstance(it, dict) or not isinstance(it.get("id"), (str, int)):
            continue
        i = it["id"]
        if i in seen:
            out.append(f"{p}.id: duplicate {what} id {i!r} (first at {seen[i]})")
        else:
            seen[i] = p
    return set(seen)


def _refs(it: dict, key: str, known: set[Any], p: str, what: str, out: list[str]) -> None:
    vals = it.get(key)
    for v in vals if isinstance(vals, list) else []:
        if not _in(v, known):
            out.append(f"{p}.{key}: unknown {what} {v!r}")


def _semantic(r: Any) -> list[str]:
    out: list[str] = []
    _walk(r, "$", out)
    if not isinstance(r, dict):
        return out
    rooms = [(p, x) for p, x in _items(r.get("rooms"), "$.rooms") if isinstance(x, dict)]
    room_ids = _ids(rooms, "room", out)
    walls, openings, surfaces = [], [], []
    for rp, room in rooms:
        ws = [(p, w) for p, w in _items(room.get("walls"), f"{rp}.walls") if isinstance(w, dict)]
        walls += ws
        own_walls = {w.get("id") for _, w in ws if isinstance(w.get("id"), (str, int))}
        for p, o in _items(room.get("openings"), f"{rp}.openings"):
            if not isinstance(o, dict):
                continue
            openings.append((p, o))
            if not _in(o.get("wall_id"), own_walls):
                out.append(f"{p}.wall_id: {o.get('wall_id')!r} is not a wall of room {room.get('id')!r}")
            if o.get("connects_to") is not None and not _in(o.get("connects_to"), room_ids):
                out.append(f"{p}.connects_to: unknown room {o.get('connects_to')!r}")
        surfaces += [(p, s) for p, s in _items(room.get("surfaces"), f"{rp}.surfaces") if isinstance(s, dict)]
    _ids(walls, "wall", out)
    opening_ids = _ids(openings, "opening", out)
    surface_ids = _ids(surfaces, "surface", out)

    damage = [(p, d) for p, d in _items(r.get("damage"), "$.damage") if isinstance(d, dict)]
    damage_ids = _ids(damage, "damage", out)
    for p, d in damage:
        if not _in(d.get("surface_id"), surface_ids):
            out.append(f"{p}.surface_id: unknown surface {d.get('surface_id')!r}")
        if not _in(d.get("room_id"), room_ids):
            out.append(f"{p}.room_id: unknown room {d.get('room_id')!r}")

    flags = [(p, f) for p, f in _items(r.get("concealed_damage_flags"), "$.concealed_damage_flags")
             if isinstance(f, dict)]
    flag_ids = _ids(flags, "flag", out)
    for p, f in flags:
        if not _in(f.get("room_id"), room_ids):
            out.append(f"{p}.room_id: unknown room {f.get('room_id')!r}")
        _refs(f, "surface_ids", surface_ids, p, "surface", out)
        _refs(f, "damage_ids", damage_ids, p, "damage region", out)

    scope = [(p, s) for p, s in _items(r.get("scope"), "$.scope") if isinstance(s, dict)]
    _ids(scope, "scope item", out)
    for p, s in scope:
        if not _in(s.get("surface_id"), surface_ids):
            out.append(f"{p}.surface_id: unknown surface {s.get('surface_id')!r}")
        if not _in(s.get("room_id"), room_ids):
            out.append(f"{p}.room_id: unknown room {s.get('room_id')!r}")
        _refs(s, "damage_ids", damage_ids, p, "damage region", out)
        _refs(s, "flag_ids", flag_ids, p, "flag", out)

    prop = r.get("property") if isinstance(r.get("property"), dict) else {}
    for p, a in _items(prop.get("adjacency"), "$.property.adjacency"):
        if not isinstance(a, dict):
            continue
        for k in ("room_a", "room_b"):
            if not _in(a.get(k), room_ids):
                out.append(f"{p}.{k}: unknown room {a.get(k)!r}")
        for k in ("opening_a", "opening_b"):
            if a.get(k) is not None and not _in(a.get(k), opening_ids):
                out.append(f"{p}.{k}: unknown opening {a.get(k)!r}")
    return out


def problems(result: Any) -> list[str]:
    """Every schema and semantic problem, as 'json.path: message' strings."""
    validator = _validator(str(schema_path()))
    errors = sorted(validator.iter_errors(result), key=lambda e: [str(p) for p in e.absolute_path])
    return [f"{_jpath(e.absolute_path)}: {_short(e.message)}" for e in errors] + _semantic(result)


def validate(result: Any) -> None:
    """Raise ValueError listing every problem when result does not satisfy the schema and the semantic checks."""
    errs = problems(result)
    if errs:
        raise ValueError(f"result failed validation with {len(errs)} problem(s):\n" + "\n".join(f"  {e}" for e in errs))
