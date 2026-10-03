"""End-to-end run for one capture: ingest, geometry, layout, stitch, semantics, rules, scope, intervals, output."""

from __future__ import annotations

import json
import logging
import subprocess
import time
import traceback
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from scan2scope import __version__
from scan2scope.config import MODELS, OUTPUT_CACHE_DIR, setup_env

log = logging.getLogger("scan2scope")


class Timer:
    def __init__(self) -> None:
        self.stages: dict[str, float] = {}
        self.t0 = time.perf_counter()

    @contextmanager
    def stage(self, name: str):
        t = time.perf_counter()
        log.info("stage %s ...", name)
        try:
            yield
        finally:
            self.stages[name] = round(self.stages.get(name, 0.0) + time.perf_counter() - t, 3)
            log.info("stage %s done in %.1fs", name, self.stages[name])

    @property
    def total(self) -> float:
        return round(time.perf_counter() - self.t0, 3)


def _git_commit() -> str | None:
    try:
        root = Path(__file__).resolve().parents[2]
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def _reject_mirror_openings(plan, objects) -> list[str]:
    """Drop windows and openings that coincide with a detected mirror: the reflected room behind a mirror
    looks like see-through evidence to the layout stage."""
    import numpy as np

    mirrors = [o for o in objects if getattr(o, "cls", None) == "mirror"]
    flags = []
    for room in plan.rooms:
        keep = []
        for op in room.openings:
            hit = None
            if op.connects_to is None and op.center is not None:
                for m in mirrors:
                    d = float(np.linalg.norm(np.asarray(m.xy, float) - np.asarray(op.center, float)))
                    if d < max(0.5 * op.width.value, 0.4):
                        hit = m
                        break
            if hit is None:
                keep.append(op)
            else:
                flags.append(f"opening_rejected_mirror:{op.id}")
        room.openings = keep
    removed = {f.split(":", 1)[1] for f in flags}
    plan.adjacency = [a for a in plan.adjacency if a.opening_a not in removed and a.opening_b not in removed]
    return flags


def _empty_plan():
    from scan2scope.types import Measurement, Plan

    return Plan(rooms=[], adjacency=[], footprint_area=Measurement(0.0, unit="m2", kind="area"),
                extent_x=Measurement(0.0), extent_y=Measurement(0.0), flags=["no_geometry"])


def run_capture(
    path: str | Path,
    out_dir: str | Path,
    *,
    tier: str | None = None,
    cache_mode: str = "live",
    drift_correction: bool = True,
    semantics: bool = True,
    quiet: bool = False,
) -> dict[str, Any]:
    """Run the full pipeline on one capture and write result.json, plan.svg/png and room sheets to out_dir."""
    setup_env()
    from scan2scope.cache import OutputCache
    from scan2scope.ingest.detect import detect_capture
    from scan2scope.output import console, render, schema, writer
    from scan2scope.rules import evaluate as evaluate_rules
    from scan2scope.scope import generate as generate_scope
    from scan2scope.uncertainty import annotate

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "work"
    work_dir.mkdir(exist_ok=True)
    timer = Timer()
    cache = OutputCache(mode=cache_mode, root=OUTPUT_CACHE_DIR)
    stage_errors: list[str] = []

    with timer.stage("ingest"):
        tier, root, info = detect_capture(Path(path), tier=tier, work_dir=work_dir)

    scenes = []
    plan = None
    try:
        with timer.stage("geometry"):
            if tier == "lidar":
                from scan2scope.geometry.lidar import build_scene

                scenes = [build_scene(root, work_dir, drift_correction=drift_correction)]
            elif tier == "video":
                from scan2scope.geometry.video import build_scene

                scenes = [build_scene(root, work_dir, drift_correction=drift_correction, cache=cache)]
            else:
                from scan2scope.geometry.photo import build_room_scenes

                scenes = build_room_scenes(root, work_dir, cache=cache)
        for s in scenes:
            stage_errors.extend(f for f in s.meta.get("flags", []) if f not in stage_errors)

        with timer.stage("layout"):
            from scan2scope.layout import build_plan

            if tier == "photo":
                room_plans = [build_plan(s, single_room=True) for s in scenes]
            else:
                plan = build_plan(scenes[0])
                plan.meta["drift"] = scenes[0].meta.get("drift")

        if tier == "photo":
            with timer.stage("stitch"):
                from scan2scope.stitch import stitch_rooms

                plan, scenes = stitch_rooms(scenes, room_plans, work_dir, cache=cache)
    except Exception as exc:  # any input must still produce a schema-valid result
        if cache_mode == "replay" and type(exc).__name__ == "CacheMiss":
            raise
        log.error("geometry stage failed: %s", exc)
        log.debug(traceback.format_exc())
        stage_errors.append(f"geometry_failed:{type(exc).__name__}:{str(exc)[:120]}")
        plan = _empty_plan()

    damage, objects = [], []
    if semantics:
        with timer.stage("semantics"):
            try:
                from scan2scope.semantics import analyze

                sem = analyze(scenes, plan, work_dir, cache=cache)
                damage, objects = sem.damage, sem.objects
                stage_errors.extend(getattr(sem, "flags", []))
                plan.flags.extend(_reject_mirror_openings(plan, objects))
            except Exception as exc:  # semantics must never block the geometric result
                log.warning("semantics failed: %s", exc)
                log.debug(traceback.format_exc())
                stage_errors.append(f"semantics_failed:{type(exc).__name__}")

    with timer.stage("rules"):
        flags = evaluate_rules(plan, damage, objects)

    with timer.stage("uncertainty"):
        quality = {"tier": tier, "scale_log_sigma": max((s.scale_log_sigma for s in scenes), default=0.0),
                   "scenes": [{"room_hint": s.room_hint, **s.meta.get("quality", {})} for s in scenes]}
        annotate(plan, damage, tier=tier, quality=quality)

    with timer.stage("scope"):
        scope_items = generate_scope(plan, damage, flags)

    info.flags.extend(stage_errors)
    provenance = {
        "pipeline_version": __version__,
        "git_commit": _git_commit(),
        "models": [{"name": m.repo, "revision": m.revision, "license": m.license} for m in MODELS.values()],
        "cache_mode": cache.effective_mode,
        "device": cache.device_label(),
    }
    result = writer.build_result(info, plan, damage, flags, scope_items,
                                 timing={"total_s": timer.total, "stages": timer.stages}, provenance=provenance)
    schema.validate(result)
    (out_dir / "result.json").write_text(json.dumps(result, indent=1))
    render.render_all(result, out_dir)
    if not quiet:
        console.print_summary(result)
    return result
