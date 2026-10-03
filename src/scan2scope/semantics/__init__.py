"""Damage regions and context objects (fixtures, mirrors, doors, windows) on room surfaces.

analyze() runs Grounding DINO boxes and SAM 2.1 masks over the views of every scene, lifts the masks through
the view point maps onto the nearest room surface, merges repeated detections across views and numbers the
damage regions D1.. and the objects O1... Model outputs go through the OutputCache, so a replay run needs no
weights. Torch and transformers are imported only when a model is actually needed.
"""

from __future__ import annotations

import functools
import hashlib
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from scan2scope.semantics.detector import (
    Detection,
    DetectorConfig,
    GroundingDinoDetector,
    ModelUnavailable,
    box_iou_matrix,
    decode,
    working_size,
)
from scan2scope.semantics.lift import (
    LiftConfig,
    assign_surface,
    lift_mask,
    measure_on_surface,
    place_object,
    pointmap_jacobian,
    see_through_fraction,
    view_valid,
)
from scan2scope.semantics.merge import (
    DamageObservation,
    MergeConfig,
    ObjectObservation,
    merge_damage,
    merge_objects,
)
from scan2scope.semantics.segmenter import Sam2Segmenter, SegmenterConfig
from scan2scope.types import CameraView, DamageRegion, Measurement, Plan, Scene

log = logging.getLogger("scan2scope.semantics")

DAMAGE_CLASSES = ("water_stain", "mold", "crack", "hole", "peeling_paint")
OBJECT_CLASSES = ("door", "window", "mirror", "sink", "toilet", "bathtub", "shower", "stove", "refrigerator",
                  "washing_machine")
WET_FIXTURES = ("sink", "toilet", "bathtub", "shower", "washing_machine")
RING_CLASSES = ("door", "window", "mirror")  # placed from the band around the mask: their pixels see through


@dataclass
class SceneObject:
    """A detected fixture, mirror, door or window, placed in plan coordinates."""

    id: str  # "O1", ...
    cls: str
    room_id: str | None
    xy: np.ndarray  # (2,) plan position
    z_range: tuple[float, float]
    score: float  # combined detector score, not a calibrated probability
    x_range: tuple[float, float] = (0.0, 0.0)
    y_range: tuple[float, float] = (0.0, 0.0)
    view_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticsResult:
    damage: list[DamageRegion]
    objects: list[SceneObject]
    flags: list[str] = field(default_factory=list)
    dropped: list[dict[str, Any]] = field(default_factory=list)  # detections not reported, with the reason
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticsConfig:
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    lift: LiftConfig = field(default_factory=LiftConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    max_views_per_scene: int = 40  # long video and LiDAR scenes are subsampled evenly in capture order
    suppress_inside: tuple[str, ...] = ("door", "window", "mirror")  # damage boxes inside these are dropped
    suppress_overlap: float = 0.6  # share of the damage box inside the object box
    ring_px: int = 3
    wall_object_max_nz: float = 0.5  # doors, windows and mirrors whose surround is tilted more are dropped
    min_see_through: float = 0.3  # ... and those whose inside is mostly coplanar with the surround
    see_through_behind_m: float = 0.15
    write_debug: bool = True


@functools.lru_cache(maxsize=512)
def _sha256_cached(path: str, size: int, mtime_ns: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def file_sha256(path: str | Path) -> str:
    st = os.stat(path)
    return _sha256_cached(str(path), st.st_size, st.st_mtime_ns)


def _cached(cache: Any, key: dict, fn: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    return fn() if cache is None else cache.compute(key, fn)


def load_view_image(view: CameraView, long_side: int) -> tuple[np.ndarray, list[str]]:
    """Upright RGB uint8 image of a view, resized to the working size; also returns notes on mismatches."""
    notes: list[str] = []
    rgb = None
    try:
        from scan2scope.ingest.images import load_image

        rgb = np.asarray(load_image(Path(view.image_path))[0])
    except Exception:  # ingest not available or failed: decode here
        rgb = None
    if rgb is None:
        from PIL import Image, ImageOps

        try:
            import pillow_heif

            pillow_heif.register_heif_opener()
        except ImportError:
            log.debug("pillow_heif not installed; HEIC images cannot be read")
        with Image.open(view.image_path) as im:
            rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[..., None], 3, axis=2)
    rgb = rgb[..., :3]
    h, w = rgb.shape[:2]
    if abs(w / h - view.width / view.height) > 0.01 * view.width / view.height:
        notes.append(f"semantics_image_aspect_mismatch:{view.id}")
    import cv2

    ww, hh = working_size(view.width, view.height, long_side)
    if (w, h) != (ww, hh):
        interp = cv2.INTER_AREA if ww < w else cv2.INTER_LINEAR
        rgb = cv2.resize(np.ascontiguousarray(rgb), (ww, hh), interpolation=interp)
    return np.ascontiguousarray(rgb, dtype=np.uint8), notes


def select_views(views: list[CameraView], max_views: int) -> tuple[list[CameraView], dict[str, int]]:
    """Views that have an image and a usable point map, evenly subsampled to at most max_views."""
    skipped = {"no_image": 0, "no_pointmap": 0}
    usable = []
    for v in views:
        if v.image_path is None or not Path(v.image_path).exists():
            skipped["no_image"] += 1
        elif view_valid(v) is None:
            skipped["no_pointmap"] += 1
        else:
            usable.append(v)
    if max_views > 0 and len(usable) > max_views:
        idx = np.unique(np.round(np.linspace(0, len(usable) - 1, max_views)).astype(int))
        skipped["subsampled"] = len(usable) - len(idx)
        usable = [usable[i] for i in idx]
    return usable, skipped


def suppress_inside_objects(dets: list[Detection], classes: tuple[str, ...], overlap: float
                            ) -> tuple[list[Detection], list[tuple[Detection, str]]]:
    """Drop damage boxes that sit mostly inside a door, window or mirror box (views through or reflections)."""
    objs = [d for d in dets if d.kind == "object" and d.cls in classes]
    if not objs:
        return dets, []
    ob = np.stack([d.box for d in objs])
    kept, dropped = [], []
    for d in dets:
        if d.kind == "damage":
            _, inside = box_iou_matrix(d.box[None], ob)
            k = int(np.argmax(inside[0]))
            if inside[0, k] >= overlap:
                dropped.append((d, f"inside_{objs[k].cls}"))
                continue
        kept.append(d)
    return kept, dropped


def _m(value: float, unit: str, kind: str, evidence: dict[str, Any]) -> Measurement:
    return Measurement(value=float(value), unit=unit, kind=kind, evidence=dict(evidence))


def _surface_order(plan: Plan) -> dict[str, tuple[int, int]]:
    order = {}
    for ri, room in enumerate(plan.rooms):
        for wi, wall in enumerate(room.walls):
            order[wall.id] = (ri, wi)
        order[f"{room.id}-FLOOR"] = (ri, len(room.walls))
        order[f"{room.id}-CEIL"] = (ri, len(room.walls) + 1)
    return order


def analyze(scenes: list[Scene], plan: Plan, work_dir: str | Path | None, *, cache: Any = None,
            config: SemanticsConfig | None = None, detector: Any = None, segmenter: Any = None) -> SemanticsResult:
    """Detect, segment, lift and merge damage and context objects for all scenes of one capture.

    Raises ModelUnavailable (a RuntimeError) when a model is needed and its weights are missing, re-raises a
    replay CacheMiss, and raises when every view fails; the pipeline records these as a flag. Odd views (no
    image, no point map, unreadable file, a model error on one image) are skipped and flagged instead.
    """
    cfg = config or SemanticsConfig()
    res = SemanticsResult([], [])
    if not plan.rooms:
        res.flags.append("semantics_no_rooms")
        return res
    views: list[CameraView] = []
    skipped_total: dict[str, int] = {}
    seen: set[str] = set()
    for scene in scenes:
        vs, skipped = select_views(scene.views, cfg.max_views_per_scene)
        for v in vs:  # one photo registered into two rooms is still one observation
            try:
                sha = file_sha256(v.image_path)
            except OSError:
                skipped["no_image"] = skipped.get("no_image", 0) + 1
                continue
            if sha in seen:
                skipped["duplicate_image"] = skipped.get("duplicate_image", 0) + 1
                continue
            seen.add(sha)
            views.append(v)
        for k, n in skipped.items():
            skipped_total[k] = skipped_total.get(k, 0) + n
    res.stats["views_used"] = len(views)
    res.stats["views_skipped"] = skipped_total
    for k in ("no_image", "no_pointmap"):
        if skipped_total.get(k):
            res.flags.append(f"semantics_views_{k}:{skipped_total[k]}")
    if not views:
        res.flags.append("semantics_no_views")
        return res
    det = detector or GroundingDinoDetector(cfg.detector)
    seg = segmenter or Sam2Segmenter(cfg.segmenter)
    damage_obs: list[DamageObservation] = []
    object_obs: list[ObjectObservation] = []
    records: list[dict[str, Any]] = []
    failed: list[Exception] = []
    for view in views:
        try:
            _process_view(view, plan, cfg, det, seg, cache, damage_obs, object_obs, records, res)
        except ModelUnavailable:
            raise
        except Exception as exc:
            if any(c.__name__ == "CacheMiss" for c in type(exc).__mro__):
                raise  # replay must reproduce the full result or fail
            log.warning("semantics skipped view %s: %s", view.id, exc)
            res.flags.append(f"semantics_view_failed:{view.id}:{type(exc).__name__}")
            failed.append(exc)
    if failed and len(failed) == len(views):
        raise RuntimeError(f"semantics failed on all {len(views)} views: {failed[-1]}") from failed[-1]
    merged, dropped = merge_damage(damage_obs, cfg.merge)
    res.dropped += [r for r in records if r.get("status") == "dropped"] + dropped
    order = _surface_order(plan)
    merged.sort(key=lambda m: (order.get(m.surface_id, (len(plan.rooms), 0)), m.u_range[0], m.v_range[0], m.cls))
    for i, m in enumerate(merged, 1):
        ev = {"source": "semantics", "n_views": m.evidence["n_views"],
              "rel_spread": m.evidence["area_rel_spread"], "pixel_m": m.evidence["pixel_m"],
              "valid_fraction": m.evidence["valid_fraction"]}
        per_view = m.evidence["per_view"]
        res.damage.append(DamageRegion(
            id=f"D{i}", room_id=m.room_id, surface_id=m.surface_id, cls=m.cls, score=round(m.score, 4),
            area=_m(m.area, "m2", "area", {**ev, "per_view": [p["area_m2"] for p in per_view]}),
            width=_m(m.width, "m", "width", {**ev, "per_view": [p["width_m"] for p in per_view]}),
            height=_m(m.height, "m", "height", {**ev, "per_view": [p["height_m"] for p in per_view]}),
            u_range=(float(m.u_range[0]), float(m.u_range[1])), v_range=(float(m.v_range[0]), float(m.v_range[1])),
            length=_m(m.length, "m", "length", ev) if m.cls == "crack" else None,
            view_ids=list(m.view_ids), evidence={**m.evidence, "surface_kind": m.kind}))
    objs, odropped = merge_objects(object_obs, cfg.merge)
    res.dropped += odropped
    room_index = {r.id: i for i, r in enumerate(plan.rooms)}
    objs.sort(key=lambda o: (room_index.get(o.room_id, len(plan.rooms)), o.cls, float(o.xy[0]), float(o.xy[1])))
    for i, o in enumerate(objs, 1):
        res.objects.append(SceneObject(
            id=f"O{i}", cls=o.cls, room_id=o.room_id, xy=np.asarray(o.xy, float), z_range=o.z_range,
            score=round(o.score, 4), x_range=o.x_range, y_range=o.y_range, view_ids=o.view_ids,
            evidence=o.evidence))
    single = sum(1 for d in dropped if d["reason"].startswith("single_view"))
    if single:
        res.flags.append(f"semantics_single_view_dropped:{single}")
    res.stats.update({"damage_observations": len(damage_obs), "object_observations": len(object_obs),
                      "damage_regions": len(res.damage), "objects": len(res.objects)})
    if cfg.write_debug and work_dir is not None:
        _write_debug(Path(work_dir), records, res)
    log.info("semantics: %d views, %d damage regions, %d objects", len(views), len(res.damage), len(res.objects))
    return res


def _process_view(view: CameraView, plan: Plan, cfg: SemanticsConfig, det: Any, seg: Any, cache: Any,
                  damage_obs: list[DamageObservation], object_obs: list[ObjectObservation],
                  records: list[dict[str, Any]], res: SemanticsResult) -> None:
    sha = file_sha256(view.image_path)
    size = working_size(view.width, view.height, cfg.detector.long_side)
    image: np.ndarray | None = None

    def get_image() -> np.ndarray:
        nonlocal image
        if image is None:
            image, notes = load_view_image(view, cfg.detector.long_side)
            res.flags.extend(n for n in notes if n not in res.flags)
        return image

    dets: list[Detection] = []
    for prompt in (cfg.detector.damage, cfg.detector.objects):
        raw = _cached(cache, det.cache_key(sha, size, prompt), lambda p=prompt: det.predict(get_image(), p))
        dets += decode(raw, prompt, size[0], size[1], cfg.detector)
    if not dets:
        return
    boxes = np.stack([d.box for d in dets])
    raw = _cached(cache, seg.cache_key(sha, size, boxes), lambda: seg.predict(get_image(), boxes))
    masks = seg.masks(raw, boxes)
    jac = pointmap_jacobian(view.pointmap, view_valid(view))
    # objects first: doors, windows and mirrors only count (and only suppress damage) when they sit in a wall
    # and their inside shows depth past it, so a sheet of paper taped to the wall is not a window
    placed: list[Detection] = []
    for d, (mask, _, _) in zip(dets, masks):
        if d.kind != "object":
            continue
        ring = cfg.ring_px if d.cls in RING_CLASSES else 0
        lifted = lift_mask(view, mask, jac=jac, ring=ring)
        place = place_object(lifted, plan, cfg.lift) if lifted is not None else None
        if place is None:
            records.append(_record(view, d, "dropped", "no_geometry_under_mask"))
            continue
        if d.cls in RING_CLASSES:
            if place.normal_z > cfg.wall_object_max_nz:
                records.append(_record(view, d, "dropped", "not_in_a_wall", normal_z=round(place.normal_z, 3)))
                continue
            through = see_through_fraction(view, mask, lifted, cfg.see_through_behind_m)
            if through < cfg.min_see_through:
                records.append(_record(view, d, "dropped", "not_see_through", see_through=round(through, 3)))
                continue
        placed.append(d)
        object_obs.append(ObjectObservation(view.id, d.cls, d.score, place.room_id, place.xy, place.x_range,
                                            place.y_range, place.z_range, place.n_pixels))
        records.append(_record(view, d, "kept", "", room_id=place.room_id, xy=[round(float(x), 3) for x in place.xy]))
    damage = [(d, m) for d, m in zip(dets, masks) if d.kind == "damage"]
    kept, suppressed = suppress_inside_objects([d for d, _ in damage] + placed, cfg.suppress_inside,
                                               cfg.suppress_overlap)
    for d, reason in suppressed:
        records.append(_record(view, d, "dropped", reason))
    keep_ids = {id(d) for d in kept}
    for d, (mask, iou, fallback) in damage:
        if id(d) not in keep_ids:
            continue
        lifted = lift_mask(view, mask, jac=jac)
        if lifted is None or len(lifted.points) == 0:
            records.append(_record(view, d, "dropped", "no_geometry_under_mask"))
            continue
        assignment, reason = assign_surface(lifted, plan, cfg.lift)
        if assignment is None:
            records.append(_record(view, d, "dropped", reason))
            continue
        measure, reason = measure_on_surface(lifted, assignment, cfg.lift)
        if measure is None:
            records.append(_record(view, d, "dropped", reason, surface_id=assignment.surface_id))
            continue
        damage_obs.append(DamageObservation(
            view_id=view.id, cls=d.cls, score=d.score, room_id=assignment.room.id,
            surface_id=assignment.surface_id, kind=assignment.kind, area=measure.area, width=measure.width,
            height=measure.height, length=measure.length, u_range=measure.u_range, v_range=measure.v_range,
            endpoints=measure.endpoints, pixel_m=measure.pixel_m, valid_fraction=measure.valid_fraction,
            inlier_fraction=measure.inlier_fraction, match=assignment.match, mask_fallback=fallback,
            phrase_scores=[float(x) for x in d.phrase_scores]))
        records.append(_record(view, d, "kept", "", surface_id=assignment.surface_id, match=assignment.match,
                               area_m2=round(measure.area, 5), mask_iou=round(iou, 3), mask_fallback=fallback))


def _record(view: CameraView, d: Detection, status: str, reason: str, **extra: Any) -> dict[str, Any]:
    r = {"view_id": view.id, "kind": d.kind, "class": d.cls, "score": round(float(d.score), 4),
         "box": [round(float(x), 1) for x in d.box], "status": status}
    if reason:
        r["reason"] = reason
    r.update(extra)
    return r


def _write_debug(work_dir: Path, records: list[dict[str, Any]], res: SemanticsResult) -> None:
    try:
        out = work_dir / "semantics"
        out.mkdir(parents=True, exist_ok=True)
        doc = {"detections": records, "dropped": res.dropped, "flags": res.flags, "stats": res.stats,
               "damage": [{"id": d.id, "class": d.cls, "surface_id": d.surface_id, "score": d.score,
                           "area_m2": round(d.area.value, 5), "view_ids": d.view_ids} for d in res.damage],
               "objects": [{"id": o.id, "class": o.cls, "room_id": o.room_id, "score": o.score,
                            "xy": [round(float(x), 3) for x in o.xy],
                            "z_range": [round(float(x), 3) for x in o.z_range]} for o in res.objects]}
        (out / "detections.json").write_text(json.dumps(doc, indent=1, default=str))
    except OSError as exc:
        log.warning("could not write semantics debug file: %s", exc)


__all__ = ["DAMAGE_CLASSES", "OBJECT_CLASSES", "WET_FIXTURES", "ModelUnavailable", "SceneObject",
           "SemanticsConfig", "SemanticsResult", "analyze"]
