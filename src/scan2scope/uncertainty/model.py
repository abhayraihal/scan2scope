"""Interval model: fills lo/hi of every Measurement in a Plan and its damage regions.

An interval runs from value - z q sigma to value + z q sigma_hi, with q per tier from calibration.yaml and lo
clipped at 0. sigma is the symmetric part. A one-sided extent U (how far the evidence lets the truth lie on
one side, at q = 1) widens only that side: sigma_hi = sqrt(sigma^2 + (U / z)^2).

Lengths: sigma^2 = (v s)^2 + a^2 + the end-wall terms below. s is the capture's log-scale sigma, shared by
every measurement: the larger of the tier floor and the geometry's estimate, combined with the scale
inconsistency the capture measured itself (video: the RMS deviation of the chunks' world scales from their
median, and half the scale error of a loop closure that registered well but was rejected; the larger of the
two counts). a is the tier's additive term for the measurement's role (priors.yaml), inflated by thin evidence
on the measurement (observed_fraction, n_points, a fit residual in its own unit in quadrature) and by room
context (low light, few photos, missing focal length).

A wall's length is the distance between its two end walls, so their evidence counts as well: a thinly observed
end wall inflates the additive term the same way, the part of an end wall's face rms above the capture's
surface noise (a doubled or cluttered face) adds in quadrature, and an end wall with no face of its own
(wall_unobserved, wall_face_missing, wall_face_mismatch, fewer than min_face_points face points) leaves that
end where camera free space ran out and adds a share of the room's extent along the wall. Half the translation
error of a credible rejected loop closure adds to every length.

Structure: a wall that ends at an unobserved, sparse or step-like wall (shorter than short_wall_m, or shorter
than step_wall_m with a face covering less than step_observed of the floor-to-ceiling plane) may be a piece of
a longer wall. Its upper side reaches the far end of the pieces that continue in the same direction past such
walls; a wall that is not itself a step may also span the room's extent along it. A room's area reaches up to
its bounding box when any wall may be a fragment, and widens by each unobserved wall's length times that
wall's position term.

Heights add a vertical log term (photo and video see heights across the image and lengths partly in depth).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from scan2scope.types import DamageRegion, Measurement, Plan, Room, Wall

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
UNOBSERVED_WALL = ("wall_unobserved", "wall_face_missing", "wall_face_mismatch")
SPARSE_WALL = ("wall_face_sparse",)

# Defaults for the priors.yaml sections that older files do not have.
STRUCTURE = {"min_face_points": 30, "sparse_observed": 0.1, "unobserved_end": 0.15, "short_wall_m": 0.5,
             "step_wall_m": 1.0, "step_observed": 0.6}
CAPTURE = {"loop_min_overlap": 0.2, "loop_min_inliers": 0.3, "loop_max_rot_deg": 20.0, "report_ratio": 1.1}


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


def _ev(m: Any) -> dict[str, Any]:
    ev = getattr(m, "evidence", None)
    return ev if isinstance(ev, dict) else {}


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
    z: float
    infl: dict[str, Any]

    @property
    def zq(self) -> float:
        return self.z * self.q

    def inflation(self, sources: list[Any]) -> float:
        f = 1.0
        of = _first(sources, OBS_KEYS)
        if of is not None:
            f *= _step(self.infl.get("observed_fraction"), of)
        n = _first(sources, NPTS_KEYS)
        if n is not None:
            f *= _step(self.infl.get("n_points"), n)
        return f

    def additive(self, base: float, sources: list[Any]) -> float:
        return math.hypot(base * self.inflation(sources), abs(_first(sources, RESID_KEYS) or 0.0))

    def set(self, m: Measurement | None, sigma: float, parts: dict[str, float], up: float = 0.0,
            reasons: Iterable[str] = ()) -> None:
        """Interval from the symmetric sigma and the one-sided upward extent up (at q = 1)."""
        if m is None:
            return
        ev = dict(_ev(m))
        v = _num(m.value)
        if v is None:
            m.lo = m.hi = None
            ev.update(sigma=None, q=self.q)
            m.evidence = ev
            return
        if not math.isfinite(sigma):
            sigma = abs(v)
        up = up if math.isfinite(up) else abs(v)
        s_hi = math.hypot(sigma, up / self.z) if up > 0 else sigma
        lo, hi = v - self.zq * sigma, v + self.zq * s_hi
        m.lo, m.hi = (max(0.0, lo) if v >= 0 else lo), hi
        ev.update(sigma=sigma, q=self.q, sigma_parts=dict(parts))
        for k in ("sigma_hi", "widened"):
            ev.pop(k, None)
        if up > 0:
            ev["sigma_hi"] = s_hi
        why = sorted(set(reasons))
        if why:
            ev["widened"] = why
        m.evidence = ev

    def length(self, m: Measurement | None, a: float, extra: dict[str, float] | None = None, up: float = 0.0,
               reasons: Iterable[str] = ()) -> None:
        if m is None:
            return
        sc = abs(_num(m.value) or 0.0) * self.s
        extra = {k: v for k, v in (extra or {}).items() if v > 0}
        sigma = math.sqrt(sc * sc + a * a + sum(v * v for v in extra.values()))
        parts = {"scale": sc, "additive": a, **extra}
        if up > 0:
            parts["upper"] = up
        self.set(m, sigma, parts, up, reasons)

    def area(self, m: Measurement, t_add: float, extra: dict[str, float] | None = None, up: float = 0.0,
             reasons: Iterable[str] = ()) -> None:
        sc = 2.0 * abs(_num(m.value) or 0.0) * self.s
        a = math.hypot(t_add, abs(_first([_ev(m)], RESID_KEYS) or 0.0))
        extra = {k: v for k, v in (extra or {}).items() if v > 0}
        sigma = math.sqrt(sc * sc + a * a + sum(v * v for v in extra.values()))
        parts = {"scale": sc, "additive": a, **extra}
        if up > 0:
            parts["upper"] = up
        self.set(m, sigma, parts, up, reasons)


def _reach(sigma: float, delta: float, sigma_far: float, z: float) -> float:
    """One-sided extent that puts the bound at value + delta + z sigma_far (the far hypothesis)."""
    return math.sqrt(max(0.0, (delta + z * sigma_far) ** 2 - (z * sigma) ** 2))


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


def _quality_factor(q: dict[str, Any], flags: set[str], tier: str,
                    infl: dict[str, Any]) -> tuple[float, list[str]]:
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
    if (q.get("missing_exif_focal") is True or q.get("exif_focal") is False
            or _has_flag(flags, "missing_exif_focal")
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


# capture ----------------------------------------------------------------------------------------------------


@dataclass
class _Capture:
    s: float  # log-scale sigma shared by every measurement
    s_base: float  # max(tier floor, geometry estimate)
    s_chunks: float  # RMS deviation of the chunks' world scales from their median (log)
    s_loop: float  # half the scale error of a credible rejected loop closure (log)
    drift_m: float  # half the translation error of a credible rejected loop closure (m)
    reasons: list[str]

    def record(self) -> dict[str, Any]:
        return {"scale_sigma": self.s, "scale_base": self.s_base, "scale_chunks": self.s_chunks,
                "scale_loop": self.s_loop, "drift_m": self.drift_m, "reasons": list(self.reasons)}


def _capture(plan: Plan, quality: dict[str, Any] | None, s_floor: float, cfg: dict[str, Any]) -> _Capture:
    """Scale and drift terms from what the capture measured about its own consistency."""
    quality = quality if isinstance(quality, dict) else {}
    s_cap = _num(quality.get("scale_log_sigma"))
    s_base = max(s_floor, abs(s_cap or 0.0))
    flags = set(_flag_list(quality)) | {str(f) for f in plan.flags or []}
    meta = plan.meta if isinstance(plan.meta, dict) else {}
    drift = meta.get("drift") if isinstance(meta.get("drift"), dict) else {}
    raw = drift.get("chunks")
    chunks = [c for c in raw if isinstance(c, dict)] if isinstance(raw, list) else []
    scales = [_num(c.get("world_scale")) for c in chunks]
    logs = np.array([math.log(w) for w in scales if w is not None and w > 0])
    s_chunks = 0.0
    if len(logs) >= 2:  # chunks scaled apart: a measurement taken mostly from one of them carries its offset
        s_chunks = float(np.sqrt(np.mean((logs - np.median(logs)) ** 2)))
    fallback = (any(c.get("align_method") == "poses" for c in chunks)
                or _has_flag(flags, "chunk_align_fallback"))
    loop = drift.get("loop_closure") if isinstance(drift.get("loop_closure"), dict) else {}
    s_loop = drift_m = 0.0
    if loop.get("attempted") and loop.get("accepted") is False:
        overlap, inliers = _num(loop.get("overlap")), _num(loop.get("inlier_frac"))
        rot = _num(loop.get("error_rot_deg"))
        credible = (overlap is not None and overlap >= float(cfg["loop_min_overlap"])
                    and (inliers is None or inliers >= float(cfg["loop_min_inliers"]))
                    and rot is not None and abs(rot) <= float(cfg["loop_max_rot_deg"]))
        if credible:  # a good registration that was refused: its error measures the chain's drift
            s_loop = 0.5 * abs(_num(loop.get("error_log_scale")) or 0.0)
            drift_m = 0.5 * abs(_num(loop.get("error_trans_m")) or 0.0)
    s = math.hypot(s_base, max(s_chunks, s_loop))
    ratio = float(cfg["report_ratio"])
    reasons = []
    if math.hypot(s_base, s_chunks) > ratio * s_base:
        reasons += ["chunk_scale_spread"] + (["chunk_align_fallback"] if fallback else [])
    if math.hypot(s_base, s_loop) > ratio * s_base or drift_m > 0:
        reasons.append("loop_closure_rejected")
    return _Capture(s, s_base, s_chunks, s_loop, drift_m, reasons)


# rooms ------------------------------------------------------------------------------------------------------


@dataclass
class _WallEv:
    unobserved: bool  # no face of its own: the line sits where free space ran out
    sparse: bool
    step: bool  # short, or short and low: a furniture face or a misregistration step
    inflation: float  # factor on the additive term from its observed fraction and face points
    smear: float  # face rms above the capture's surface noise: a doubled or cluttered face


def _wall_ev(mdl: _Model, w: Wall, st: dict[str, Any]) -> _WallEv:
    flags = [str(f) for f in w.flags or []]
    src = [_ev(w.length), w.evidence, {"observed_fraction": w.observed_fraction}]
    n = _first(src, NPTS_KEYS)
    unobserved = (any(f.startswith(UNOBSERVED_WALL) for f in flags)
                  or (n is not None and n < st["min_face_points"]))
    of = _num(w.observed_fraction)
    sparse = not unobserved and (any(f.startswith(SPARSE_WALL) for f in flags)
                                 or (of is not None and of < st["sparse_observed"]))
    length = abs(_num(getattr(w.length, "value", None)) or 0.0)
    # a counter or wardrobe side covers only the lower part of a floor-to-ceiling face
    low = of is not None and of < st["step_observed"]
    step = length < st["short_wall_m"] or (length < st["step_wall_m"] and low)
    rms, noise = _first(src, ("fit_rms",)), _first(src, ("noise_sigma",))
    smear = max(0.0, rms - noise) if rms is not None and noise is not None and not unobserved else 0.0
    return _WallEv(unobserved, sparse, step, mdl.inflation(src), smear)


@dataclass
class _Frame:
    extent: np.ndarray  # (2,) room extent along its two dominant wall directions
    axes: list[int]  # which of the two directions each wall runs along
    dirs: np.ndarray  # (K, 2) unit direction of each wall


def _room_frame(room: Room) -> _Frame | None:
    try:
        S = np.array([np.asarray(w.start, float).reshape(-1)[:2] for w in room.walls])
        E = np.array([np.asarray(w.end, float).reshape(-1)[:2] for w in room.walls])
    except (TypeError, ValueError):
        return None
    if len(S) < 3 or S.shape != E.shape or S.shape[1] != 2 or not np.isfinite(np.r_[S, E]).all():
        return None
    d = E - S
    L = np.hypot(d[:, 0], d[:, 1])
    if L.sum() <= 0:
        return None
    th = float(np.angle((L * np.exp(4j * np.arctan2(d[:, 1], d[:, 0]))).sum()) / 4.0)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, s], [-s, c]])  # rotation by -th
    P = np.vstack([S, E]) @ R.T
    dr = d @ R.T
    return _Frame(P.max(0) - P.min(0), [0 if abs(x) >= abs(y) else 1 for x, y in dr],
                  d / np.maximum(L, 1e-12)[:, None])


def _continuation(k: int, step: int, frame: _Frame, doubt: list[float], lengths: list[float]) -> float:
    """Length wall k would gain if the doubtful walls past one of its ends were artefacts: every piece
    beyond a doubtful end wall that runs on in the same direction (the far side of a notch or a step)."""
    K = len(lengths)
    gain, i = 0.0, k
    for _ in range(K // 2):
        j, nxt = (i + step) % K, (i + 2 * step) % K
        if nxt == k or doubt[j] <= 0 or float(frame.dirs[k] @ frame.dirs[nxt]) < 0.95:
            break
        gain += lengths[nxt]
        i = nxt
    return gain


@dataclass
class _WallOut:
    a_own: float  # additive term from the wall's own evidence
    end_noise: float  # extra additive term from thinly observed or smeared end walls
    end_pos: float  # position terms of unobserved or sparse end walls
    up: float  # one-sided upward extent if the wall may be a fragment
    p_fragment: float
    pos: float  # position term of this wall when it is unobserved, used by its room's area
    reasons: list[str]


def _walls(mdl: _Model, room: Room, a: float, st: dict[str, Any], drift_m: float) -> list[_WallOut]:
    walls = room.walls or []
    K = len(walls)
    evs = [_wall_ev(mdl, w, st) for w in walls]
    frame = _room_frame(room) if K >= 3 else None
    lengths = [abs(_num(getattr(w.length, "value", None)) or 0.0) for w in walls]
    doubt = [1.0 if (e.unobserved or e.step) else (0.5 if e.sparse else 0.0) for e in evs]
    out = []
    for k, w in enumerate(walls):
        a_own = mdl.additive(a, [_ev(w.length), w.evidence, {"observed_fraction": w.observed_fraction}])
        L = lengths[k]
        noise2 = pos2 = 0.0
        real = 1.0  # chance-like product that the walls where this one stops are real walls
        reasons: list[str] = []
        E = E_perp = 0.0
        if frame is not None:
            E, E_perp = float(frame.extent[frame.axes[k]]), float(frame.extent[1 - frame.axes[k]])
            for j in ((k - 1) % K, (k + 1) % K):
                e = evs[j]
                noise2 += 0.5 * a * a * max(e.inflation ** 2 - 1.0, 0.0) + e.smear ** 2
                if e.unobserved:
                    pos2 += (st["unobserved_end"] * E) ** 2
                    real = 0.0
                    reasons.append("wall_end_unobserved")
                elif e.sparse:
                    pos2 += (0.5 * st["unobserved_end"] * E) ** 2
                    real *= 0.5
                    reasons.append("wall_end_sparse")
                if e.step and not e.unobserved:
                    real = 0.0
                    reasons.append("wall_end_step")
            if evs[k].unobserved:  # no face of its own: a short one is most likely a piece of a longer wall
                real = 0.0 if evs[k].step else 0.5 * real
                reasons.append("wall_unobserved")
        p = 1.0 - real
        deficit = 0.0
        if frame is not None and p > 0:
            gain = _continuation(k, 1, frame, doubt, lengths) + _continuation(k, -1, frame, doubt, lengths)
            # a step that is an artefact vanishes rather than grows, so only longer walls may span the extent
            reach = L + gain if evs[k].step else max(L + gain, E)
            deficit = max(0.0, min(E, reach) - L)
        delta = deficit * min(1.0, 2.0 * p)
        up = 0.0
        if delta > a_own:  # the far end of the pieces, measured at the same scale as the rest
            rest = a_own ** 2 + noise2 + pos2 + drift_m ** 2
            up = _reach(math.sqrt((L * mdl.s) ** 2 + rest), delta,
                        math.sqrt(((L + delta) * mdl.s) ** 2 + rest), mdl.z)
            reasons.append("wall_fragment")
        pos = st["unobserved_end"] * E_perp if evs[k].unobserved else 0.0
        out.append(_WallOut(a_own, math.sqrt(noise2), math.sqrt(pos2), up, p, pos, reasons))
    return out


def _annotate_room(mdl: _Model, room: Room, add: dict[str, float], f_room: float, flags: set[str],
                   st: dict[str, Any], cap: _Capture,
                   vertical: float) -> tuple[float, float, float, list[str]]:
    """Intervals for one room's walls, openings, ceiling, floor area and perimeter.

    Returns its area term without the scale part, its upward area extent, its length term for the extents and
    the reasons its intervals were widened."""
    a0 = add["length"] * f_room
    outs = _walls(mdl, room, a0, st, cap.drift_m)
    reasons: list[str] = []
    for wall, o in zip(room.walls, outs):
        mdl.length(wall.length, o.a_own, {"end_noise": o.end_noise, "end_position": o.end_pos,
                                          "drift": cap.drift_m}, o.up, o.reasons)
        reasons += o.reasons

    def height(m: Measurement | None, sources: list[Any]) -> None:
        if m is None:
            return
        a_h = mdl.additive(add["height"] * f_room, sources)
        vert = abs(_num(m.value) or 0.0) * vertical
        mdl.length(m, a_h, {"vertical": vert})

    for wall in room.walls:
        height(wall.height, [_ev(wall.height), wall.evidence, {"observed_fraction": wall.observed_fraction}])
    for op in room.openings:
        for m in (op.offset, op.width, op.height, op.sill):
            if m is None:
                continue
            mdl.length(m, mdl.additive(add["opening"] * f_room, [_ev(m), op.evidence]))
    height(room.ceiling_height, [_ev(room.ceiling_height), room.evidence])

    perim = abs(_num(room.perimeter.value) or 0.0)
    lengths = [max(_num(w.length.value) or 0.0, 0.0) for w in room.walls]
    if outs:
        t_area = sum(L * o.a_own for L, o in zip(lengths, outs))
        t_perim = sum(o.a_own for o in outs)
    else:
        n_edges = len(room.polygon) if room.polygon is not None else 4
        t_area = perim * a0
        t_perim = max(n_edges, 3) * a0
    t_unobs = math.sqrt(sum((L * o.pos) ** 2 for L, o in zip(lengths, outs)))
    t_drift = 0.5 * perim * cap.drift_m
    t_room = math.sqrt(math.hypot(t_area, abs(_first([_ev(room.floor_area)], RESID_KEYS) or 0.0)) ** 2
                       + t_unobs ** 2 + t_drift ** 2)
    area_v = abs(_num(room.floor_area.value) or 0.0)
    frame = _room_frame(room) if len(room.walls) >= 3 else None
    p_room = max((o.p_fragment for o in outs if o.up > 0), default=0.0)
    up_area = 0.0
    if frame is not None and p_room > 0:  # if the notches are not real, the room fills its bounding box
        box = float(frame.extent[0] * frame.extent[1])
        delta = max(0.0, box - area_v) * min(1.0, 2.0 * p_room)
        if delta > 0:
            up_area = _reach(math.hypot(2 * area_v * mdl.s, t_room), delta,
                             math.hypot(2 * box * mdl.s, t_room), mdl.z)
    mdl.area(room.floor_area, t_area, {"unobserved_walls": t_unobs, "drift": t_drift}, up_area,
             ["room_fragment"] if up_area > 0 else [])
    t_end = math.sqrt(sum(o.end_pos ** 2 for o in outs))
    up_perim = math.sqrt(sum(o.up ** 2 for o in outs))
    p_add = math.hypot(t_perim, abs(_first([_ev(room.perimeter)], RESID_KEYS) or 0.0))
    mdl.length(room.perimeter, p_add,
               {"end_position": t_end, "drift": cap.drift_m * math.sqrt(max(len(outs), 1))}, up_perim)
    return t_room, up_area, math.hypot(a0, cap.drift_m), sorted(set(reasons))


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
    mdl.set(m, 0.5 * abs(v) + a, {"scale": 0.5 * abs(v), "additive": a})
    m.evidence["fallback"] = True
    return True


def annotate(plan: Plan, damage: list[DamageRegion] | None, *, tier: str,
             quality: dict[str, Any] | None = None, priors: dict[str, Any] | None = None,
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
    st = {**STRUCTURE, **(pri.get("structure") or {})}
    cfg = {**CAPTURE, **(pri.get("capture") or {})}
    z = float(pri.get("z", 1.645))
    q, status, cal_entry = tier_q(cal, model_tier)
    s_floor = float(tp["scale_floor"])
    vertical = float(tp.get("vertical", 0.0))
    cap = _capture(plan, quality, s_floor, cfg)
    add = {k: float(v) for k, v in tp["additive"].items()}
    mdl = _Model(s=cap.s, q=q, z=z, infl=infl)

    room_quality = _room_qualities(plan, quality, tier)
    room_factors: dict[str, dict[str, Any]] = {}
    placement: list[str] = []
    area_terms: list[float] = []
    area_up: list[float] = []
    len_terms: list[float] = []
    for room in plan.rooms:
        flags = _room_flags(room, plan)
        f_room, why = max((_quality_factor(c, flags, tier, infl) for c in room_quality[room.id]),
                          key=lambda t: t[0])
        if _has_flag(flags, "placement_uncertain"):
            placement.append(room.id)
        a0 = add["length"] * f_room
        structure: list[str] = []
        try:
            t_area, up, t_len, structure = _annotate_room(mdl, room, add, f_room, flags, st, cap, vertical)
        except Exception as exc:  # noqa: BLE001 - one odd room must not cost the whole result its intervals
            log.warning("intervals for room %s failed (%s); using the fallback", room.id, exc)
            plan.flags.append(f"uncertainty_failed:{room.id}")
            t_area, up, t_len = abs(_num(getattr(room.perimeter, "value", 0.0)) or 0.0) * a0, 0.0, a0
        room_factors[room.id] = {"factor": f_room, "reasons": why, "structure": structure}
        area_terms.append(t_area)
        area_up.append(up)
        len_terms.append(t_len)

    # Footprint: the scale term is fully correlated across rooms, the per-room terms are independent.
    f_place = float(infl.get("placement_uncertain", 1.0)) if placement else 1.0
    s = cap.s
    if isinstance(plan.footprint_area, Measurement):
        sc = 2.0 * abs(_num(plan.footprint_area.value) or 0.0) * s
        t_fp = math.sqrt(sum(t * t for t in area_terms))
        up_fp = math.sqrt(sum(u * u for u in area_up))
        parts = {"scale": sc * f_place, "additive": t_fp * f_place, **({"upper": up_fp} if up_fp else {})}
        mdl.set(plan.footprint_area, math.hypot(sc, t_fp) * f_place, parts, up_fp)
    a_ext = math.sqrt(sum(a * a for a in len_terms)) if len_terms else math.hypot(add["length"], cap.drift_m)
    for m in (plan.extent_x, plan.extent_y):
        if isinstance(m, Measurement):
            sc = abs(_num(m.value) or 0.0) * s
            mdl.set(m, math.hypot(sc, a_ext) * f_place, {"scale": sc * f_place, "additive": a_ext * f_place})

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
        "scale_sigma": s, "scale_floor": s_floor,
        "scale_capture": _num(quality.get("scale_log_sigma")) if isinstance(quality, dict) else None,
        "capture": cap.record(), "capture_reasons": list(cap.reasons), "vertical": vertical,
        "additive": add, "damage_additive": a_d, "structure": {k: st[k] for k in sorted(st)},
        "placement_factor": float(infl.get("placement_uncertain", 1.0)),
        "room_factors": room_factors, "placement_uncertain": placement,
        "calibration": {k: cal_entry[k] for k in ("n_rooms", "n_records", "empirical_quantile",
                                                  "conformal_quantile", "fitted") if k in cal_entry},
    }
    if isinstance(plan.meta, dict):
        plan.meta["uncertainty"] = record
    log.info("intervals: tier %s, s %.4f (%s), q %.3f (%s), %d rooms, %d damage regions", tier, s,
             ", ".join(cap.reasons) or "no capture evidence", q, status, len(plan.rooms), len(damage or []))
    return record


def describe(record: dict[str, Any] | None) -> str:
    """Plain-text statement of the error model and calibration status, for result.json conventions."""
    if not record:
        return "No error model was applied; lo and hi equal the value."
    add = record.get("additive") or {}
    q = float(record.get("q", 1.0))
    if record.get("status") == "calibrated":
        n = (record.get("calibration") or {}).get("n_rooms", "?")
        cal = (f"q = {q:.2f}, calibrated by split conformal on {n} ground-truth rooms of this tier "
               "(room as the unit)")
    else:
        cal = (f"q = {q:.2f}, prior value: not calibrated, this tier has fewer than "
               f"{record.get('min_rooms', 9)} ground-truth rooms")
    st = {**STRUCTURE, **(record.get("structure") or {})}
    reasons = record.get("capture_reasons") or []
    widened = f" This capture's scale term was widened for: {', '.join(reasons)}." if reasons else ""
    tier = record.get("model_tier", record.get("tier"))
    s = float(record.get("scale_sigma", 0.0))
    vert = float(record.get("vertical", 0.0))
    return (
        f"{tier} tier error model. Lengths: sigma = sqrt((v*s)^2 + a^2) with a scale term s = {s:.3f} shared "
        "by every measurement (the larger of the tier floor and the geometry's estimate, with the scale "
        "spread across video chunks or the scale error of a rejected loop closure) and additive terms "
        f"a = {add.get('length', 0):.3f} m (walls), {add.get('height', 0):.3f} m (heights), "
        f"{add.get('opening', 0):.3f} m (openings), {float(record.get('damage_additive', 0.0)):.3f} m "
        "(damage), inflated for thin evidence on the measurement and on a wall's two end walls (low observed "
        "fraction, few face points, fit residuals in quadrature, low light, fewer than 4 photos, "
        "missing focal length). An end wall with no face of its own adds "
        f"{float(st['unobserved_end']):.2f} of the room's extent along the wall. A wall ending at an "
        f"unobserved wall or a step (shorter than {float(st['short_wall_m']):.2f} m, or shorter than "
        f"{float(st['step_wall_m']):.2f} m and low) may be a fragment: its upper bound reaches the pieces "
        f"beyond the step or the room's extent along the wall. Heights add a vertical term of {vert:.2f} "
        "(log). Areas: "
        "sqrt((2*A*s)^2 + (sum of L_i*a_i)^2) plus unobserved wall positions, and upward to the bounding box "
        "when a wall may be a fragment. Footprint: scale term "
        "fully correlated across rooms plus independent per-room terms; unplaced rooms widen footprint and "
        f"extents {float(record.get('placement_factor', 1.5)):.1f}x. Interval = value - z*q*sigma to value + "
        f"z*q*sigma_hi with z = {float(record.get('z', 1.645)):.3f}, the one-sided terms widening only their "
        f"side, lo clipped at 0; {cal}.{widened}"
    )
