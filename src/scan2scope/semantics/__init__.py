"""Damage regions and context objects (fixtures, mirrors, doors, windows) on room surfaces.

analyze() runs Grounding DINO boxes and SAM 2.1 masks over the views of every scene, lifts the masks through
the view point maps onto the nearest room surface, merges repeated detections across views and numbers the
damage regions D1.. and the objects O1... Model outputs go through the OutputCache, so a replay run needs no
weights. Torch and transformers are imported only when a model is actually needed.

Grounding DINO scores damage weakly and confuses look-alikes, so a damage box has to survive these checks:
it is not a box the object or distractor prompts explain at least as well (a lamp socket, a curtain, a mirror);
it does not lie on a distractor or inside a door, window or mirror; its class fits the image (a crack needs a
thin dark line that wanders like a crack, and a weak area-class box that holds one is that crack); the class
is plausible on the surface it lands on (no peeling paint or mold on a floor, more evidence for floor stains,
cracks and holes). Damage boxes found only on the image tiles count only as cracks.
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
    concat_raw,
    decode,
    merge_tile_detections,
    tile_rects,
    tile_to_image,
    working_size,
)
from scan2scope.semantics.evidence import (
    LineConfig,
    LineEvidence,
    darker_than_surround,
    lightness,
    mask_shape,
    thin_dark_line,
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
AREA_CLASSES = ("water_stain", "mold", "hole", "peeling_paint")

# Minimum Grounding DINO score per class, from a sweep on the held-out set (Commons damage photos as positives,
# ARKitScenes rooms and kitchen frames without damage as negatives): lowering the area classes below 0.35 found
# no more held-out damage and added false detections, so they stay there. Cracks drop to 0.2 because the
# thin-line test removes most of the extra false detections and a crack still needs two views.
CLASS_MIN_SCORE = {"water_stain": 0.35, "mold": 0.35, "crack": 0.2, "hole": 0.35, "peeling_paint": 0.35}

# Evidence a merged region needs by class and surface kind: None means the class is not reported there at all,
# a pair is (minimum combined score, minimum number of views). Missing entries need nothing extra. Floors are
# tile, wood or carpet, not painted plaster: peeling paint and mold there are look-alikes (rugs, mats, grout).
SURFACE_EVIDENCE: dict[str, dict[str, tuple[float, int] | None]] = {
    "peeling_paint": {"floor": None},
    "mold": {"floor": None},
    "water_stain": {"floor": (0.6, 2)},
    "crack": {"floor": (0.6, 2)},
    "hole": {"floor": (0.75, 3)},  # dark gaps under cabinets and doors look like holes in a floor
}


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
    # ... unless the inside is this much darker (L*, 0..100) than the wall band around it: glass at night, a
    # mirror and a closed door are (by 20 to 80 in the benchmark room); a sheet of paper on the wall is not
    dark_glass_dl: float = 15.0
    glass_ring_px: int = 8  # band width for that comparison, in working-image pixels
    wall_mounted: tuple[str, ...] = ("window", "mirror")
    min_mount_height: float = 0.15  # m: a window or mirror whose band reaches the floor is a door or a passage
    # a damage box that matches an object or distractor box (IoU) whose score is within the margin of the damage
    # score is that object
    lookalike_iou: float = 0.5
    lookalike_margin: float = 0.1
    # damage boxes mostly inside a confident distractor box lie on that object (a stain on a rug or a curtain)
    covering: tuple[str, ...] = ("curtain", "rug", "doormat", "clothes", "picture_frame", "ceiling_fan",
                                 "light_fixture", "cabinet", "wardrobe")
    covering_inside: float = 0.8
    covering_min_score: float = 0.35
    line: LineConfig = field(default_factory=LineConfig)
    thin_elongation: float = 5.0  # an area-class mask this elongated with a thin dark line in it is a crack
    dark_area_dl: float = 3.0  # L*: an area-class region less this much darker than its surround is checked for a line
    # minimum detector score to report each class as itself; the damage prompt decodes everything from the crack
    # minimum up, because a weaker box of any class can still turn out to be a crack (see resolve_class)
    class_min_score: dict[str, float] = field(default_factory=lambda: dict(CLASS_MIN_SCORE))
    surface_evidence: dict[str, dict[str, tuple[float, int] | None]] = field(
        default_factory=lambda: {k: dict(v) for k, v in SURFACE_EVIDENCE.items()})
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


def suppress_inside_objects(dets: list[Detection], classes: tuple[str, ...], overlap: float,
                            kinds: tuple[str, ...] = ("object",), prefix: str = "inside",
                            min_score: float = 0.0) -> tuple[list[Detection], list[tuple[Detection, str]]]:
    """Drop damage boxes that sit mostly inside a door, window or mirror box (views through or reflections), or
    with other kinds and classes inside a distractor box such as a curtain or a rug."""
    objs = [d for d in dets if d.kind in kinds and d.cls in classes and d.score >= min_score]
    if not objs:
        return dets, []
    ob = np.stack([d.box for d in objs])
    kept, dropped = [], []
    for d in dets:
        if d.kind == "damage":
            _, inside = box_iou_matrix(d.box[None], ob)
            k = int(np.argmax(inside[0]))
            if inside[0, k] >= overlap:
                dropped.append((d, f"{prefix}_{objs[k].cls}"))
                continue
        kept.append(d)
    return kept, dropped


def drop_lookalikes(damage: list[Detection], others: list[Detection], iou: float, margin: float
                    ) -> tuple[list[Detection], list[tuple[Detection, str]]]:
    """Drop damage boxes that coincide with an object or distractor box scoring at least the damage score minus
    margin: the same thing in the image under two labels, and the non-damage label explains it as well."""
    if not damage or not others:
        return damage, []
    ob = np.stack([o.box for o in others])
    kept, dropped = [], []
    for d in damage:
        ious, _ = box_iou_matrix(d.box[None], ob)
        hits = [k for k in np.flatnonzero(ious[0] >= iou) if others[k].score >= d.score - margin]
        if hits:
            k = max(hits, key=lambda j: others[j].score)
            dropped.append((d, f"lookalike_{others[k].cls}"))
        else:
            kept.append(d)
    return kept, dropped


def _swap(scores: dict[str, float], a: str, b: str) -> dict[str, float]:
    out = dict(scores)
    out[a], out[b] = scores.get(b, 0.0), scores.get(a, 0.0)
    return out


def resolve_class(d: Detection, mask: np.ndarray, get_lightness: Callable[[], np.ndarray], prompt: Any,
                  cfg: SemanticsConfig) -> tuple[str | None, dict[str, float], np.ndarray, dict[str, Any]]:
    """Class a damage detection keeps, its per-class scores, the mask to lift and the evidence behind it.

    A crack is a thin dark line: a crack box without one takes its best area class when that clears the class
    minimum and is dropped otherwise, and the line (not the SAM mask, which often covers the wall around it) is
    what gets measured. Holes, stains, mold and peeling paint are areas, and except peeling paint they are darker
    than the surface around them. So an area-class box whose mask is long and thin, or which is not darker than
    its surround, and which holds a thin dark line across it, is a crack: Grounding DINO names the paper with a
    drawn line, or the band of wall around a hairline crack, by an area class at a low score. Any other area
    detection needs the class minimum of its own class, and must come from the full view: the tiles are there
    to find thin cracks (on the held-out set they found no more area damage, only more false areas).
    """
    scores = dict(d.class_scores) if d.class_scores else {d.cls: d.score}
    floor = cfg.class_min_score
    shape = mask_shape(mask)
    info: dict[str, Any] = {}
    if shape is not None:
        info.update(mask_elongation=round(shape.elongation, 2), mask_width_px=round(shape.width_px, 2))
    look_for_line = d.cls == "crack" or (shape is not None and shape.elongation >= cfg.thin_elongation)
    if not look_for_line and d.cls in AREA_CLASSES:
        dl = darker_than_surround(get_lightness(), mask, cfg.glass_ring_px)
        info["darker_by"] = round(dl, 1)
        look_for_line = dl < cfg.dark_area_dl
    line: LineEvidence | None = None
    if look_for_line and d.score >= floor.get("crack", 0.0):
        line = thin_dark_line(get_lightness(), d.box, cfg.line)
    if line is not None:
        info.update(line_length_px=round(line.length_px, 1), line_width_px=round(line.width_px, 2),
                    line_contrast=round(line.contrast, 2), line_span=round(line.span, 3))
    if d.cls == "crack":
        if line is not None:
            return "crack", scores, line.mask, info
        alt = max((c for c in AREA_CLASSES if c in scores), key=lambda c: scores[c], default=None)
        if alt is not None and scores[alt] >= floor.get(alt, prompt.threshold(alt)) and d.source != "tile":
            return alt, _swap(scores, "crack", alt), mask, {**info, "relabel": f"crack->{alt}:no_thin_line"}
        return None, scores, mask, {**info, "reason": "crack_without_thin_line"}
    if line is not None:
        return "crack", _swap(scores, d.cls, "crack"), line.mask, {**info, "relabel": f"{d.cls}->crack:thin_line"}
    if d.source == "tile":  # tiles are there for thin cracks; areas are large enough for the full view
        return None, scores, mask, {**info, "reason": "tile_area_without_line"}
    if d.score < floor.get(d.cls, prompt.threshold(d.cls)):
        return None, scores, mask, {**info, "reason": f"{d.cls}_below_{floor.get(d.cls)}"}
    return d.cls, scores, mask, info


def same_region_in_view(view_obs: list[DamageObservation], ob: DamageObservation, overlap: float = 0.5,
                        grow: float = 0.02) -> int | None:
    """Index of an observation of the same view, class and surface whose uv box (grown by `grow` m, since crack
    boxes are thin) covers at least `overlap` of the smaller of the two boxes, or None."""
    g = np.array([-grow, -grow, grow, grow])
    for i, o in enumerate(view_obs):
        if o.cls != ob.cls or o.surface_id != ob.surface_id:
            continue
        a, b = o.box + g, ob.box + g
        iw = min(a[2], b[2]) - max(a[0], b[0])
        ih = min(a[3], b[3]) - max(a[1], b[1])
        if iw <= 0 or ih <= 0:
            continue
        small = min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))
        if iw * ih >= overlap * small:
            return i
    return None


def surface_rule(rules: dict[str, dict[str, Any]], cls: str, kind: str) -> tuple[bool, tuple[float, int] | None]:
    """(class allowed on this surface kind, extra (min score, min views) evidence or None)."""
    per = rules.get(cls, {})
    if kind not in per:
        return True, None
    rule = per[kind]
    return (False, None) if rule is None else (True, (float(rule[0]), int(rule[1])))


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
    strong = []
    for m in merged:
        allowed, need = surface_rule(cfg.surface_evidence, m.cls, m.kind)
        n_views = int(m.evidence.get("n_views", len(m.view_ids)))
        if allowed and (need is None or (m.score >= need[0] and n_views >= need[1])):
            strong.append(m)
            continue
        why = f"{m.cls}_implausible_on_{m.kind}" if not allowed else \
            f"{m.cls}_on_{m.kind}_needs_score_{need[0]}_and_{need[1]}_views"
        dropped.append({"kind": "damage", "class": m.cls, "surface_id": m.surface_id, "view_ids": m.view_ids,
                        "score": round(m.score, 4), "n_views": n_views, "reason": why})
    merged = strong
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
    weak = sum(1 for d in dropped if "_on_" in d["reason"] or "_implausible_on_" in d["reason"])
    if weak:
        res.flags.append(f"semantics_surface_evidence_dropped:{weak}")
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
    light: np.ndarray | None = None

    def get_image() -> np.ndarray:
        nonlocal image
        if image is None:
            image, notes = load_view_image(view, cfg.detector.long_side)
            res.flags.extend(n for n in notes if n not in res.flags)
        return image

    def get_lightness() -> np.ndarray:
        nonlocal light
        if light is None:
            light = lightness(get_image())
        return light

    full: np.ndarray | None = None

    def tile_image(rect: tuple[int, int, int, int]) -> np.ndarray:
        import cv2

        nonlocal full
        if full is None:
            full = load_view_image(view, max(view.width, view.height))[0]
        x0, y0, x1, y1 = rect
        crop = np.ascontiguousarray(full[y0:y1, x0:x1])
        tw, th = working_size(x1 - x0, y1 - y0, cfg.detector.long_side)
        interp = cv2.INTER_AREA if tw < crop.shape[1] else cv2.INTER_LINEAR
        return np.ascontiguousarray(cv2.resize(crop, (tw, th), interpolation=interp), dtype=np.uint8)

    dets: list[Detection] = []
    for prompt in (cfg.detector.damage, cfg.detector.objects, cfg.detector.distractors):
        if prompt is None:
            continue
        raw = _cached(cache, det.cache_key(sha, size, prompt), lambda p=prompt: det.predict(get_image(), p))
        found = decode(raw, prompt, size[0], size[1], cfg.detector)
        if prompt is cfg.detector.damage and getattr(det, "supports_tiles", False):
            parts = []
            for rect in tile_rects(view.width, view.height, cfg.detector.damage_tiles, cfg.detector.tile_overlap):
                r = _cached(cache, det.cache_key(sha, size, prompt, tile=rect),
                            lambda p=prompt, rc=rect: det.predict(tile_image(rc), p))
                parts.append(tile_to_image(r, rect, view.width, view.height))
            if parts:
                tiled = decode(concat_raw(parts, len(prompt.phrases)), prompt, size[0], size[1], cfg.detector)
                found = merge_tile_detections(found, tiled, cfg.detector.nms_iou, cfg.detector.nms_containment)
        dets += found
    damage_dets = [d for d in dets if d.kind == "damage"]
    objects = [d for d in dets if d.kind == "object"]
    distractors = [d for d in dets if d.kind == "distractor"]
    damage_dets, gone = drop_lookalikes(damage_dets, objects + distractors, cfg.lookalike_iou, cfg.lookalike_margin)
    covered = suppress_inside_objects(damage_dets + distractors, cfg.covering, cfg.covering_inside,
                                      kinds=("distractor",), prefix="on", min_score=cfg.covering_min_score)
    damage_dets = [d for d in covered[0] if d.kind == "damage"]
    for d, reason in gone + covered[1]:
        records.append(_record(view, d, "dropped", reason))
    seg_dets = damage_dets + objects
    if not seg_dets:
        return
    boxes = np.stack([d.box for d in seg_dets])
    raw = _cached(cache, seg.cache_key(sha, size, boxes), lambda: seg.predict(get_image(), boxes))
    masks = seg.masks(raw, boxes)
    jac = pointmap_jacobian(view.pointmap, view_valid(view))
    rooms = {r.id: r for r in plan.rooms}
    # objects first: doors, windows and mirrors only count (and only suppress damage) when they sit in a wall
    # and their inside shows depth past it or is much darker than the wall, so a sheet of paper taped to the
    # wall is not a window; windows and mirrors also hang above the floor
    placed: list[Detection] = []
    for d, (mask, _, _) in zip(seg_dets, masks):
        if d.kind != "object":
            continue
        ring = cfg.ring_px if d.cls in RING_CLASSES else 0
        lifted = lift_mask(view, mask, jac=jac, ring=ring)
        place = place_object(lifted, plan, cfg.lift) if lifted is not None else None
        if place is None:
            records.append(_record(view, d, "dropped", "no_geometry_under_mask"))
            continue
        extra: dict[str, Any] = {}
        if d.cls in RING_CLASSES:
            if place.normal_z > cfg.wall_object_max_nz:
                records.append(_record(view, d, "dropped", "not_in_a_wall", normal_z=round(place.normal_z, 3)))
                continue
            room = rooms.get(place.room_id)
            if d.cls in cfg.wall_mounted and room is not None:
                if d.box[3] >= size[1] - 2:  # its lower edge is out of the view: hanging above the floor is unknown
                    records.append(_record(view, d, "dropped", "bottom_not_seen"))
                    continue
                lift_off = float(place.z_range[0]) - room.floor_z
                if lift_off < cfg.min_mount_height:
                    records.append(_record(view, d, "dropped", "reaches_floor", bottom_m=round(lift_off, 3)))
                    continue
            through = see_through_fraction(view, mask, lifted, cfg.see_through_behind_m)
            extra["see_through"] = round(through, 3)
            if through < cfg.min_see_through:
                dl = darker_than_surround(get_lightness(), mask, cfg.glass_ring_px)
                extra["darker_by"] = round(dl, 1)
                if dl < cfg.dark_glass_dl:
                    records.append(_record(view, d, "dropped", "not_see_through", **extra))
                    continue
        placed.append(d)
        object_obs.append(ObjectObservation(view.id, d.cls, d.score, place.room_id, place.xy, place.x_range,
                                            place.y_range, place.z_range, place.n_pixels))
        records.append(_record(view, d, "kept", "", room_id=place.room_id, xy=[round(float(x), 3) for x in place.xy],
                               **extra))
    damage = [(d, m) for d, m in zip(seg_dets, masks) if d.kind == "damage"]
    kept, suppressed = suppress_inside_objects([d for d, _ in damage] + placed, cfg.suppress_inside,
                                               cfg.suppress_overlap)
    for d, reason in suppressed:
        records.append(_record(view, d, "dropped", reason))
    keep_ids = {id(d) for d in kept}
    view_obs: list[DamageObservation] = []
    for d, (mask, iou, fallback) in damage:
        if id(d) not in keep_ids:
            continue
        cls, scores, lift_m, info = resolve_class(d, mask, get_lightness, cfg.detector.damage, cfg)
        if cls is None:
            records.append(_record(view, d, "dropped", info.pop("reason"), **info))
            continue
        lifted = lift_mask(view, lift_m, jac=jac)
        if lifted is None or len(lifted.points) == 0:
            records.append(_record(view, d, "dropped", "no_geometry_under_mask"))
            continue
        assignment, reason = assign_surface(lifted, plan, cfg.lift)
        if assignment is None:
            records.append(_record(view, d, "dropped", reason))
            continue
        if not surface_rule(cfg.surface_evidence, cls, assignment.kind)[0]:
            records.append(_record(view, d, "dropped", f"{cls}_implausible_on_{assignment.kind}",
                                   surface_id=assignment.surface_id))
            continue
        measure, reason = measure_on_surface(lifted, assignment, cfg.lift)
        if measure is None:
            records.append(_record(view, d, "dropped", reason, surface_id=assignment.surface_id))
            continue
        ob = DamageObservation(
            view_id=view.id, cls=cls, score=scores.get(cls, d.score), room_id=assignment.room.id,
            surface_id=assignment.surface_id, kind=assignment.kind, area=measure.area, width=measure.width,
            height=measure.height, length=measure.length, u_range=measure.u_range, v_range=measure.v_range,
            endpoints=measure.endpoints, pixel_m=measure.pixel_m, valid_fraction=measure.valid_fraction,
            inlier_fraction=measure.inlier_fraction, match=assignment.match, mask_fallback=fallback,
            phrase_scores=[float(x) for x in d.phrase_scores], class_scores=scores, shape=info)
        dup = same_region_in_view(view_obs, ob)
        if dup is not None:  # two boxes of this view found the same region (the line and the paper around it)
            records.append(_record(view, d, "dropped", "same_region_in_view", surface_id=assignment.surface_id,
                                   **({"as_class": cls} if cls != d.cls else {})))
            if ob.score > view_obs[dup].score:
                view_obs[dup] = ob
            continue
        view_obs.append(ob)
        records.append(_record(view, d, "kept", "", surface_id=assignment.surface_id, match=assignment.match,
                               area_m2=round(measure.area, 5), mask_iou=round(iou, 3), mask_fallback=fallback,
                               **({"as_class": cls} if cls != d.cls else {}), **info))
    damage_obs.extend(view_obs)


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
