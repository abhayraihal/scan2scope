"""Photo-tier stitching: place single-room plans and scenes into one property frame.

Placement evidence, strongest first: doorway photos registered into a neighbouring room (stitch.doorway), then
door matching (stitch.doors). stitch.solver merges and scores the hypotheses and places the rooms with no
overlaps. Rooms keep their own metric scale; relative scales from doorway photos are recorded, not applied.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from shapely.geometry import Polygon

from scan2scope.stitch.doors import door_match_hypotheses, room_doors, room_manhattan
from scan2scope.stitch.doorway import RunnerFn, mapanything_runner, register_doorways
from scan2scope.stitch.solver import (
    Hypothesis,
    Params,
    Solution,
    StitchRoom,
    apply2,
    rigid2,
    rotate2,
    solve,
    yaw2,
)
from scan2scope.types import Adjacency, Measurement, Plan, Room, Scene

log = logging.getLogger("scan2scope.stitch")

__all__ = ["stitch_rooms"]

SCALE_TOL = 0.10
GAP = 1.0


def _natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def _label_from_hint(hint: str) -> str:
    label = re.sub(r"^\s*\d+[\s_.-]*", "", hint).strip()
    return label or hint


def _rename(child_id: str | None, old_room: str, new_room: str, kind: str, k: int) -> str:
    if child_id and child_id.startswith(f"{old_room}-"):
        return f"{new_room}-{child_id[len(old_room) + 1:]}"
    return f"{new_room}-{kind}{k}"


def _renamed_room(room: Room, new_id: str, hint: str) -> Room:
    r = copy.deepcopy(room)
    old = room.id
    wall_ids: dict[str, str] = {}
    for k, w in enumerate(r.walls, 1):
        new = _rename(w.id, old, new_id, "W", k)
        wall_ids[w.id] = new
        w.id, w.room_id = new, new_id
    for k, o in enumerate(r.openings, 1):
        o.id, o.room_id = _rename(o.id, old, new_id, "O", k), new_id
        o.wall_id = wall_ids.get(o.wall_id, _rename(o.wall_id, old, new_id, "W", 0))
        o.connects_to = None
    r.id = new_id
    r.source_hint = r.source_hint or hint
    r.label = r.label or _label_from_hint(hint)
    return r


def _centroid(room: Room) -> np.ndarray:
    P = np.asarray(room.polygon, float)
    if P.ndim == 2 and len(P) >= 3 and np.isfinite(P).all():
        poly = Polygon(P[:, :2])
        if poly.is_valid and poly.area > 1e-6:
            return np.array(poly.centroid.coords[0])
        return P[:, :2].mean(0)
    return np.zeros(2)


def _pair_inputs(scenes: list[Scene], plans: list[Plan],
                 flags: list[str]) -> list[tuple[int | None, Plan | None]]:
    if len(scenes) == len(plans):
        return list(zip(range(len(scenes)), plans))
    flags.append("stitch_input_mismatch")
    log.warning("stitch: %d scenes but %d room plans, pairing by folder name", len(scenes), len(plans))
    free = list(range(len(scenes)))
    pairs: list[tuple[int | None, Plan | None]] = []
    for plan in plans:
        hint = plan.rooms[0].source_hint if plan.rooms else None
        match = next((si for si in free if hint and scenes[si].room_hint == hint), None)
        if match is None and free and hint is None:
            match = free[0]
        if match is not None:
            free.remove(match)
        pairs.append((match, plan))
    pairs += [(si, None) for si in free]
    return pairs


def _prepare(scenes: list[Scene], plans: list[Plan],
             flags: list[str]) -> tuple[list[StitchRoom], list[int | None]]:
    entries = []
    for k, (si, plan) in enumerate(_pair_inputs(scenes, plans, flags)):
        scene = scenes[si] if si is not None else None
        rooms = list(plan.rooms) if plan is not None else []
        room = max(rooms, key=lambda r: getattr(r.floor_area, "value", 0.0) or 0.0) if rooms else None
        hint = (scene.room_hint if scene is not None else None) or (room.source_hint if room else None) \
            or f"room_{k + 1:02d}"
        entries.append((hint, k, si, scene, room, plan))
    entries.sort(key=lambda e: (_natural_key(e[0]), e[1]))
    out: list[StitchRoom] = []
    scene_room: list[int | None] = [None] * len(scenes)
    for hint, _, si, scene, room, plan in entries:
        if room is None:
            flags.append(f"room_missing:{hint}")
            continue
        if plan is not None and len(plan.rooms) > 1:
            flags.append(f"room_plan_multiple_rooms:{hint}")
        idx = len(out)
        r = _renamed_room(room, f"R{idx + 1}", hint)
        for f in plan.flags if plan is not None else []:
            if f not in r.flags:
                r.flags.append(f)
        out.append(StitchRoom(idx, r.id, hint, r, scene, room_manhattan(r), room_doors(r), _centroid(r),
                              aux={"layout_meta": plan.meta if plan is not None else {}}))
        if si is not None:
            scene_room[si] = idx
    return out, scene_room


def _lift(T: np.ndarray, z_shift: float) -> np.ndarray:
    T4 = np.eye(4)
    T4[:2, :2] = T[:2, :2]
    T4[:2, 3] = T[:2, 2]
    T4[2, 3] = z_shift
    return T4


def _z_shift(room: Room) -> float:
    return -float(room.floor_z) if room.floor_z is not None and np.isfinite(room.floor_z) else 0.0


def _transform_room(r: Room, T: np.ndarray, z: float) -> None:
    r.polygon = apply2(T, np.asarray(r.polygon, float)) if len(r.polygon) else np.asarray(r.polygon, float)
    for w in r.walls:
        w.start, w.end = apply2(T, w.start), apply2(T, w.end)
        w.normal_in = rotate2(T, w.normal_in)
    for o in r.openings:
        if o.center is not None:
            o.center = apply2(T, o.center)
    r.floor_z = float(r.floor_z) + z
    r.ceiling_z = float(r.ceiling_z) + z


def _transform_scene(scene: Scene, T4: np.ndarray, info: dict) -> Scene:
    R, t = T4[:3, :3], T4[:3, 3]
    pts = np.asarray(scene.points)
    nrm = np.asarray(scene.normals)
    points = (pts @ R.T + t).astype(pts.dtype, copy=False) if pts.size else pts.copy()
    normals = (nrm @ R.T).astype(nrm.dtype, copy=False) if nrm.size else nrm.copy()
    views = []
    for v in scene.views:
        pm = v.pointmap
        if pm is not None:
            pm_arr = np.asarray(pm)
            pm = (pm_arr @ R.T + t).astype(pm_arr.dtype, copy=False)
        views.append(replace(v, T_wc=T4 @ np.asarray(v.T_wc, float), pointmap=pm))
    meta = dict(scene.meta)
    meta["stitch"] = {**info, "T_property_from_room": np.round(T4, 6).tolist()}
    return replace(scene, points=points, normals=normals, views=views, meta=meta)


def _path_edges(sol: Solution, a: int, b: int) -> int | None:
    def chain(r: int) -> list[int]:
        out = [r]
        while out[-1] in sol.parent:
            out.append(sol.parent[out[-1]][0])
        return out

    ca, cb = chain(a), chain(b)
    common = set(ca) & set(cb)
    if not common:
        return None
    return min(ca.index(c) + cb.index(c) for c in common)


def _relative_scales(rooms: list[StitchRoom], sol: Solution) -> dict[int, float | None]:
    scale: dict[int, float | None] = {}

    def get(r: int) -> float | None:
        if r in scale:
            return scale[r]
        if r not in sol.parent:
            scale[r] = 1.0 if r == sol.root else None
            return scale[r]
        p, k = sol.parent[r]
        e = sol.edges[k]
        sp = get(p)
        if sp is None or not e.scales:
            scale[r] = sp
        else:
            s_ji = float(np.median(e.scales))
            scale[r] = sp * (s_ji if r == e.j else 1.0 / s_ji)
        return scale[r]

    for sr in rooms:
        get(sr.index)
    return scale


def _facing_adjacency(rooms: list[StitchRoom], sol: Solution, out: dict[int, Room],
                      used: set[tuple[int, str]]) -> list[Adjacency]:
    """Door pairs that face each other after placement but were not tree edges (loops in the room graph)."""
    comp = {r: ci for ci, members in enumerate(sol.components) for r in members}
    adj = []
    for p in range(len(rooms)):
        for q in range(p + 1, len(rooms)):
            if comp.get(p) != comp.get(q) or p not in sol.T or q not in sol.T:
                continue
            for dp in rooms[p].doors:
                for dq in rooms[q].doors:
                    if (p, dp.id) in used or (q, dq.id) in used:
                        continue
                    cp, cq = apply2(sol.T[p], dp.center), apply2(sol.T[q], dq.center)
                    n_p, n_q = rotate2(sol.T[p], dp.normal), rotate2(sol.T[q], dq.normal)
                    dist = float(np.linalg.norm(cp - cq))
                    ang = float(np.degrees(np.arccos(np.clip(-(n_p @ n_q), -1.0, 1.0))))
                    ratio = min(dp.width, dq.width) / max(dp.width, dq.width)
                    if dist > 0.4 or ang > 25.0 or ratio < 0.7:
                        continue
                    conf = 0.5 * np.exp(-0.5 * (ang / 15.0) ** 2) * (1.0 if dist <= 0.3 else 0.5)
                    used.update({(p, dp.id), (q, dq.id)})
                    _connect(out[p], dp.id, rooms[q].id)
                    _connect(out[q], dq.id, rooms[p].id)
                    adj.append(Adjacency(rooms[p].id, rooms[q].id, dp.id, dq.id, round(float(conf), 3),
                                         "door_match"))
    return adj


def _connect(room: Room, opening_id: str | None, other: str) -> None:
    for o in room.openings:
        if o.id == opening_id and o.connects_to is None:
            o.connects_to = other


def _assemble(rooms: list[StitchRoom], sol: Solution, hyps: list[Hypothesis], records: list[dict],
              stats: dict, door_pairs: int, flags: list[str]) -> Plan:
    out: dict[int, Room] = {}
    for sr in rooms:
        r = copy.deepcopy(sr.room)
        _transform_room(r, sol.T.get(sr.index, np.eye(3)), _z_shift(sr.room))
        out[sr.index] = r

    scales = _relative_scales(rooms, sol)
    adjacency: list[Adjacency] = []
    edges_meta: list[dict] = []
    used: set[tuple[int, str]] = set()
    for child in sorted(sol.parent):
        parent, k = sol.parent[child]
        e = sol.edges[k]
        ambiguous = any(x.startswith("ambiguous") for x in sol.uncertain.get(child, []))
        conf = float(np.clip(sol.eff_score[child] * (0.5 if ambiguous else 1.0), 0.0, 1.0))
        a, b = rooms[e.i], rooms[e.j]
        for room_k, oid, other in ((e.i, e.opening_i, b.id), (e.j, e.opening_j, a.id)):
            if oid is not None:
                _connect(out[room_k], oid, other)
                used.add((room_k, oid))
        adjacency.append(Adjacency(a.id, b.id, e.opening_i, e.opening_j, round(conf, 3), e.source))
        s_med = float(np.median(e.scales)) if e.scales else None
        if e.scales and (max(abs(np.log(e.scales))) > np.log1p(SCALE_TOL)
                         or max(e.scales) / min(e.scales) > 1.0 + SCALE_TOL):
            flags.append(f"scale_disagreement:{a.id}-{b.id}")
        edges_meta.append({
            "room_a": a.id, "room_b": b.id, "opening_a": e.opening_i, "opening_b": e.opening_j,
            "source": e.source, "score": round(e.score, 4), "confidence": round(conf, 3),
            "parent": rooms[parent].id, "child": rooms[child].id, "n_hypotheses": len(e.members),
            "relative_scale": None if s_med is None else round(s_med, 4),
            "nudge_m": round(float(np.linalg.norm(sol.shift[child])), 3) if child in sol.shift else 0.0,
        })
    adjacency += _facing_adjacency(rooms, sol, out, used)

    for i in sorted(sol.uncertain):
        flag = f"placement_uncertain:{rooms[i].id}"
        flags.append(flag)
        if flag not in out[i].flags:
            out[i].flags.append(flag)
    for i in sol.shift:
        flags.append(f"placement_adjusted:{rooms[i].id}")
    for p, q in sol.overlaps:
        flags.append(f"overlap_unresolved:{rooms[min(p, q)].id}-{rooms[max(p, q)].id}")

    for sr in rooms:
        T = sol.T.get(sr.index, np.eye(3))
        parent = sol.parent.get(sr.index)
        out[sr.index].evidence = dict(out[sr.index].evidence)
        out[sr.index].evidence["stitch"] = {
            "method": sol.method.get(sr.index, "unplaced"),
            "parent": rooms[parent[0]].id if parent else None,
            "edge_score": round(sol.eff_score[sr.index], 4) if sr.index in sol.eff_score else None,
            "uncertain": list(sol.uncertain.get(sr.index, [])),
            "relative_scale": scales.get(sr.index),
            "yaw_deg": round(float(np.degrees(yaw2(T))), 3),
        }

    room_list = [out[sr.index] for sr in rooms]
    areas = [float(r.floor_area.value) for r in room_list
             if r.floor_area is not None and np.isfinite(r.floor_area.value)]
    footprint = Measurement(float(sum(areas)), unit="m2", kind="area", evidence={
        "method": "sum_of_room_floor_areas", "rooms": [r.id for r in room_list]})
    extent_x, extent_y = _extents(rooms, room_list, sol)

    pairs = {frozenset((rooms[h.a].id, rooms[h.b].id)) for h in hyps}
    pairs |= {frozenset((rec["room_a"], rec["room_b"])) for rec in records}
    pairs |= {frozenset((a.id, b.id)) for a in rooms for b in rooms
              if a.index < b.index and a.doors and b.doors}
    status = {}
    for k, e in enumerate(sol.edges):
        for m in e.members:
            status[m] = sol.edge_status.get(k, "unused")
    hyp_meta = []
    for m, h in enumerate(hyps):
        hyp_meta.append({"room_a": rooms[h.a].id, "room_b": rooms[h.b].id, "source": h.source,
                         "opening_a": h.opening_a, "opening_b": h.opening_b,
                         "score": round(float(h.score), 4), "status": status.get(m, "discarded"),
                         "relative_scale": None if h.scale_ba is None else round(float(h.scale_ba), 4)})
    best: dict[str, float] = {}
    for e in sol.edges:
        key = f"{rooms[e.i].id}-{rooms[e.j].id}"
        best[key] = max(best.get(key, 0.0), round(e.score, 4))
    meta = {
        "method": "photo_rooms",
        "root": rooms[sol.root].id if sol.root is not None else None,
        "pairs_tested": len(pairs),
        "door_pairs_tested": door_pairs,
        "doorway": stats,
        "edges": edges_meta,
        "method_per_room": {sr.id: sol.method.get(sr.index, "unplaced") for sr in rooms},
        "scores": best,
        "uncertain": [{"room": rooms[i].id, "reasons": list(r)} for i, r in sorted(sol.uncertain.items())],
        "hypotheses": hyp_meta,
        "doorway_registrations": records,
        "relative_scale": {sr.id: None if scales.get(sr.index) is None else round(scales[sr.index], 4)
                           for sr in rooms},
        "adjusted": {rooms[i].id: np.round(s, 3).tolist() for i, s in sol.shift.items()},
        "separated": {rooms[i].id: np.round(s, 3).tolist() for i, s in sol.separated.items()},
        "components": [[rooms[i].id for i in c] for c in sol.components],
        "transforms": {sr.id: {"yaw_deg": round(float(np.degrees(yaw2(sol.T[sr.index]))), 4),
                               "t": np.round(sol.T[sr.index][:2, 2], 4).tolist(),
                               "z_shift": round(_z_shift(sr.room), 4)} for sr in rooms if sr.index in sol.T},
        "z_alignment": "floors_to_zero",
    }
    plan_flags = list(dict.fromkeys(flags))
    room_meta = {sr.id: sr.aux.get("layout_meta", {}) for sr in rooms}
    return Plan(room_list, adjacency, footprint, extent_x, extent_y, flags=plan_flags,
                meta={"stitch": _plain(meta), "layout_per_room": room_meta})


def _extents(rooms: list[StitchRoom], room_list: list[Room],
             sol: Solution) -> tuple[Measurement, Measurement]:
    spans = []
    for sr, r in zip(rooms, room_list):
        P = np.asarray(r.polygon, float)
        if P.ndim == 2 and len(P) and np.isfinite(P).all():
            spans.append((sr.index, P[:, 0].min(), P[:, 1].min(), P[:, 0].max(), P[:, 1].max()))
    out = []
    for axis in (0, 1):
        if not spans:
            out.append(Measurement(0.0, unit="m", kind="length",
                                   evidence={"method": "bbox_of_stitched_rooms"}))
            continue
        lo = min(spans, key=lambda s: s[1 + axis])
        hi = max(spans, key=lambda s: s[3 + axis])
        uncertain = sorted(rooms[i].id for i in sol.uncertain)
        out.append(Measurement(float(hi[3 + axis] - lo[1 + axis]), unit="m", kind="length", evidence={
            "method": "bbox_of_stitched_rooms", "axis": "xy"[axis],
            "rooms_at_bounds": [rooms[lo[0]].id, rooms[hi[0]].id],
            "chain_edges": _path_edges(sol, lo[0], hi[0]), "uncertain_rooms": uncertain}))
    return out[0], out[1]


def _door_hints(rooms: list[StitchRoom], hyps: list[Hypothesis],
                params: Params | None) -> dict[tuple[int, str], int]:
    """The room that door matching alone puts behind each door; doorway photos try that room first."""
    sol = solve(rooms, hyps, params)
    hints: dict[tuple[int, str], int] = {}
    for _, k in sol.parent.values():
        e = sol.edges[k]
        if e.opening_i is not None:
            hints[(e.i, e.opening_i)] = e.j
        if e.opening_j is not None:
            hints[(e.j, e.opening_j)] = e.i
    return hints


def _plain(x: Any) -> Any:
    """Numpy scalars and arrays inside the stitch record become plain JSON types."""
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    if isinstance(x, np.ndarray):
        return _plain(x.tolist())
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


def _write_debug(work_dir: str | Path | None, meta: dict) -> None:
    if work_dir is None:
        return
    try:
        d = Path(work_dir) / "stitch"
        d.mkdir(parents=True, exist_ok=True)
        (d / "stitch.json").write_text(json.dumps(meta, indent=1, default=str))
    except OSError as exc:
        log.debug("stitch: could not write debug record: %s", exc)


def stitch_rooms(room_scenes: list[Scene], room_plans: list[Plan], work_dir: str | Path | None = None, *,
                 cache: Any = None, runner: RunnerFn | None = None, use_doorway_photos: bool = True,
                 max_registrations: int = 24, params: Params | None = None) -> tuple[Plan, list[Scene]]:
    """Place photo-tier rooms in one property frame and transform their scenes into it.

    runner(views, key, cache) replaces the MapAnything call for doorway-photo registration; tests pass a
    fake one.
    """
    flags: list[str] = []
    scenes = list(room_scenes or [])
    rooms, scene_room = _prepare(scenes, list(room_plans or []), flags)
    if not rooms:
        flags.append("no_rooms")

    dm: list[Hypothesis] = []
    door_pairs = 0
    try:
        dm, door_pairs = door_match_hypotheses(rooms)
    except Exception as exc:
        log.warning("stitch: door matching failed: %s", exc, exc_info=True)
        flags.append(f"door_match_failed:{type(exc).__name__}")
    dw: list[Hypothesis] = []
    records: list[dict] = []
    stats: dict = {"photos": 0, "runs": 0, "skipped_runs": 0, "hypotheses": 0}
    if use_doorway_photos and len(rooms) > 1:
        try:
            dw, records, dw_flags, stats = register_doorways(
                rooms, runner or mapanything_runner, cache, max_runs=max_registrations,
                hints=_door_hints(rooms, dm, params))
            flags += dw_flags
        except Exception as exc:  # fall back to door matching
            log.warning("stitch: doorway registration failed: %s", exc, exc_info=True)
            flags.append(f"doorway_registration_failed:{type(exc).__name__}")
    hyps = dw + dm
    try:
        sol = solve(rooms, hyps, params)
    except Exception as exc:  # lay rooms out side by side rather than fail the run
        log.warning("stitch: placement failed: %s", exc, exc_info=True)
        flags.append(f"stitch_failed:{type(exc).__name__}")
        hyps = []
        sol = solve(rooms, [], params)

    plan = _assemble(rooms, sol, hyps, records, stats, door_pairs, flags)

    out_scenes = []
    right = max((float(np.max(np.asarray(r.polygon, float)[:, 0])) for r in plan.rooms
                 if len(r.polygon) and np.isfinite(r.polygon).all()), default=0.0)
    for si, scene in enumerate(scenes):
        k = scene_room[si]
        if k is not None:
            sr = rooms[k]
            T4 = _lift(sol.T.get(k, np.eye(3)), _z_shift(sr.room))
            out_scenes.append(_transform_scene(scene, T4, {"room_id": sr.id, "method": sol.method.get(k)}))
            continue
        # a scene without a room still goes into the property frame, clear of every room
        P = np.asarray(scene.points, float)
        P = P[np.isfinite(P).all(1)] if P.ndim == 2 and P.shape[1:] == (3,) else np.zeros((0, 3))
        if len(P) == 0:
            P = np.zeros((1, 3))
        T = rigid2(0.0, (right + GAP - P[:, 0].min(), -P[:, 1].min()))
        right += GAP + float(np.ptp(P[:, 0]))
        out_scenes.append(_transform_scene(scene, _lift(T, 0.0), {"room_id": None, "method": "unplaced"}))

    _write_debug(work_dir, plan.meta["stitch"])
    n_unc = len(plan.meta["stitch"]["uncertain"])
    log.info("stitch: %d rooms, %d adjacencies, %d uncertain, footprint %.2f m2", len(plan.rooms),
             len(plan.adjacency), n_unc, plan.footprint_area.value)
    return plan, out_scenes
