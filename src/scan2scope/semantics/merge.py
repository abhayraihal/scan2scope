"""Merge per-view observations of the same damage region or object across views.

Damage observations merge when class and surface match and their uv boxes overlap (IoU above min_iou) or their
centres are within max_center_dist. A cluster holds at most one observation per view. The merged area, width,
height and length are medians of the per-view estimates; the uv box is the union. The score is a noisy-OR of
the best top_k per-view scores: a ranking signal, not a calibrated probability.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

log = logging.getLogger("scan2scope.semantics")


@dataclass
class MergeConfig:
    min_iou: float = 0.2
    max_center_dist: float = 0.3  # m on the surface
    min_single_view_score: float = 0.5  # single-view detections below this are dropped
    top_k: int = 3  # views that count towards the combined score
    object_max_dist: float = 0.5  # m in plan between observations of the same object
    drop_single_view_objects: bool = True


@dataclass
class DamageObservation:
    view_id: str
    cls: str
    score: float
    room_id: str
    surface_id: str
    kind: str  # wall | floor | ceiling
    area: float
    width: float
    height: float
    length: float
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    endpoints: np.ndarray  # (2, 2) uv
    pixel_m: float = 0.0
    valid_fraction: float = 1.0
    inlier_fraction: float = 1.0
    match: str = "wall"
    mask_fallback: bool = False
    phrase_scores: list[float] = field(default_factory=list)

    @property
    def box(self) -> np.ndarray:
        return np.array([self.u_range[0], self.v_range[0], self.u_range[1], self.v_range[1]], float)

    @property
    def center(self) -> np.ndarray:
        b = self.box
        return np.array([(b[0] + b[2]) / 2, (b[1] + b[3]) / 2])


@dataclass
class MergedDamage:
    cls: str
    room_id: str
    surface_id: str
    kind: str
    score: float
    area: float
    width: float
    height: float
    length: float
    u_range: tuple[float, float]
    v_range: tuple[float, float]
    view_ids: list[str]
    evidence: dict[str, Any]


@dataclass
class ObjectObservation:
    view_id: str
    cls: str
    score: float
    room_id: str | None
    xy: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    n_pixels: int = 0


@dataclass
class MergedObject:
    cls: str
    room_id: str | None
    xy: np.ndarray
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    score: float
    view_ids: list[str]
    evidence: dict[str, Any]


def combine_scores(scores: list[float], top_k: int = 3) -> float:
    s = sorted((float(np.clip(x, 0.0, 1.0)) for x in scores), reverse=True)[:max(1, top_k)]
    return float(1.0 - np.prod([1.0 - x for x in s]))


def box_iou(a: np.ndarray, b: np.ndarray) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def _match(a: DamageObservation, b: DamageObservation, cfg: MergeConfig) -> float | None:
    """Merge affinity (higher is better) or None when the pair should not merge."""
    iou = box_iou(a.box, b.box)
    dist = float(np.linalg.norm(a.center - b.center))
    if iou > cfg.min_iou or dist <= cfg.max_center_dist:
        return iou - 0.01 * dist
    return None


def _clusters(items: list, views: list[str], affinity, keys: list) -> list[list[int]]:
    """Greedy single-linkage clustering by score order, at most one item per view in a cluster."""
    clusters: list[list[int]] = []
    for i in range(len(items)):
        best, best_aff = None, None
        for c, members in enumerate(clusters):
            if keys[members[0]] != keys[i] or any(views[m] == views[i] for m in members):
                continue
            affs = [affinity(items[m], items[i]) for m in members]
            affs = [x for x in affs if x is not None]
            if affs and (best_aff is None or max(affs) > best_aff):
                best, best_aff = c, max(affs)
        if best is None:
            clusters.append([i])
        else:
            clusters[best].append(i)
    changed = True
    while changed:  # join clusters that ended up overlapping
        changed = False
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                ca, cb = clusters[a], clusters[b]
                if keys[ca[0]] != keys[cb[0]] or {views[m] for m in ca} & {views[m] for m in cb}:
                    continue
                if any(affinity(items[x], items[y]) is not None for x in ca for y in cb):
                    clusters[a] = ca + cb
                    del clusters[b]
                    changed = True
                    break
            if changed:
                break
    return clusters


def _rel_spread(values: list[float]) -> float:
    v = np.asarray(values, float)
    med = float(np.median(v))
    if len(v) < 2 or med <= 0:
        return 0.0
    return float(1.4826 * np.median(np.abs(v - med)) / med)


def merge_damage(obs: list[DamageObservation], cfg: MergeConfig | None = None
                 ) -> tuple[list[MergedDamage], list[dict[str, Any]]]:
    """Merged damage regions and the dropped clusters with the reason they were dropped."""
    cfg = cfg or MergeConfig()
    order = sorted(range(len(obs)), key=lambda i: -obs[i].score)
    items = [obs[i] for i in order]
    clusters = _clusters(items, [o.view_id for o in items], lambda a, b: _match(a, b, cfg),
                         [(o.cls, o.surface_id) for o in items])
    merged, dropped = [], []
    for members in clusters:
        ms = sorted((items[m] for m in members), key=lambda o: -o.score)
        score = combine_scores([o.score for o in ms], cfg.top_k)
        best = ms[0]
        if len(ms) == 1 and best.score < cfg.min_single_view_score:
            dropped.append({"kind": "damage", "class": best.cls, "surface_id": best.surface_id,
                            "view_id": best.view_id, "score": round(best.score, 4),
                            "reason": f"single_view_score_below_{cfg.min_single_view_score}"})
            continue
        areas = [o.area for o in ms]
        u0 = min(o.u_range[0] for o in ms)
        u1 = max(o.u_range[1] for o in ms)
        v0 = min(o.v_range[0] for o in ms)
        v1 = max(o.v_range[1] for o in ms)
        evidence = {
            "n_views": len(ms),
            "single_view": len(ms) == 1,
            "score_method": f"noisy_or_top{cfg.top_k}",
            "per_view": [{"view_id": o.view_id, "score": round(o.score, 4), "area_m2": round(o.area, 5),
                          "width_m": round(o.width, 4), "height_m": round(o.height, 4),
                          "u_range": [round(x, 4) for x in o.u_range], "v_range": [round(x, 4) for x in o.v_range],
                          "match": o.match, "mask_fallback": o.mask_fallback} for o in ms],
            "area_rel_spread": round(_rel_spread(areas), 4),
            "pixel_m": round(float(np.median([o.pixel_m for o in ms])), 5),
            "valid_fraction": round(float(min(o.valid_fraction for o in ms)), 4),
            "inlier_fraction": round(float(np.median([o.inlier_fraction for o in ms])), 4),
            "endpoints_uv": np.round(best.endpoints, 4).tolist(),
            "endpoints_uv_all": [np.round(o.endpoints, 4).tolist() for o in ms],
        }
        if best.phrase_scores:
            evidence["phrase_scores"] = [round(float(x), 4) for x in np.mean([o.phrase_scores for o in ms], 0)]
        merged.append(MergedDamage(
            cls=best.cls, room_id=best.room_id, surface_id=best.surface_id, kind=best.kind, score=score,
            area=float(np.median(areas)), width=float(np.median([o.width for o in ms])),
            height=float(np.median([o.height for o in ms])), length=float(np.median([o.length for o in ms])),
            u_range=(u0, u1), v_range=(v0, v1), view_ids=[o.view_id for o in ms], evidence=evidence))
    return merged, dropped


def _object_affinity(a: ObjectObservation, b: ObjectObservation, cfg: MergeConfig) -> float | None:
    if a.room_id is not None and b.room_id is not None and a.room_id != b.room_id:
        return None
    d = float(np.linalg.norm(np.asarray(a.xy) - np.asarray(b.xy)))
    return -d if d <= cfg.object_max_dist else None


def merge_objects(obs: list[ObjectObservation], cfg: MergeConfig | None = None
                  ) -> tuple[list[MergedObject], list[dict[str, Any]]]:
    cfg = cfg or MergeConfig()
    order = sorted(range(len(obs)), key=lambda i: -obs[i].score)
    items = [obs[i] for i in order]
    clusters = _clusters(items, [o.view_id for o in items], lambda a, b: _object_affinity(a, b, cfg),
                         [o.cls for o in items])
    merged, dropped = [], []
    for members in clusters:
        ms = sorted((items[m] for m in members), key=lambda o: -o.score)
        score = combine_scores([o.score for o in ms], cfg.top_k)
        if cfg.drop_single_view_objects and len(ms) == 1 and ms[0].score < cfg.min_single_view_score:
            dropped.append({"kind": "object", "class": ms[0].cls, "room_id": ms[0].room_id,
                            "view_id": ms[0].view_id, "score": round(ms[0].score, 4),
                            "reason": f"single_view_score_below_{cfg.min_single_view_score}"})
            continue
        rooms = [o.room_id for o in ms if o.room_id is not None]
        room_id = max(set(rooms), key=rooms.count) if rooms else None
        rng = {a: (float(np.median([getattr(o, a)[0] for o in ms])), float(np.median([getattr(o, a)[1] for o in ms])))
               for a in ("x_range", "y_range", "z_range")}
        merged.append(MergedObject(
            cls=ms[0].cls, room_id=room_id, xy=np.median(np.stack([np.asarray(o.xy, float) for o in ms]), 0),
            x_range=rng["x_range"], y_range=rng["y_range"], z_range=rng["z_range"], score=score,
            view_ids=[o.view_id for o in ms],
            evidence={"n_views": len(ms), "single_view": len(ms) == 1, "score_method": f"noisy_or_top{cfg.top_k}",
                      "per_view_scores": [round(o.score, 4) for o in ms]}))
    return merged, dropped
