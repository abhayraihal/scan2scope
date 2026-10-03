"""Runs every capture listed in each property's ground_truth.yaml, scores the results and writes the report.

Layout: <data_root>/<property>/ground_truth.yaml (or ground_truth.yaml directly in data_root), results in
<out_dir>/<property>/<capture>/result.json, and for multi-room video and LiDAR captures a second run with
drift correction off in <out_dir>/<property>/<capture>__nodrift (semantics skipped there, it does not change
the plan).
Each run also writes bench_run.json (status, error, runner seconds) and, on a crash, error.txt with the
traceback, so a later skip_run rescoring keeps failures as failures.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from scan2scope.bench import h2h as h2h_mod
from scan2scope.bench.ablation import NODRIFT_SUFFIX, drift_ablation
from scan2scope.bench.gates import evaluate, load_gates
from scan2scope.bench.groundtruth import GroundTruth, GTCapture, load_ground_truth
from scan2scope.bench.metrics import capture_metrics, repeatability
from scan2scope.bench.report import write_report

log = logging.getLogger("scan2scope.bench")

RunFn = Callable[..., dict[str, Any]]
RESULT = "result.json"
RUN_RECORD = "bench_run.json"  # status, error and runner seconds of the last run, read back by skip_run


def find_properties(data_root: str | Path) -> list[Path]:
    """ground_truth.yaml in data_root itself and in its immediate subfolders, sorted."""
    root = Path(data_root)
    out = [root / "ground_truth.yaml"] if (root / "ground_truth.yaml").is_file() else []
    return out + sorted(p for p in root.glob("*/ground_truth.yaml") if p.is_file())


def _selected(only: list[str] | None, prop: str, cid: str) -> bool:
    if not only:
        return True
    return cid in only or prop in only or f"{prop}/{cid}" in only


def _load_result(path: Path) -> dict[str, Any] | None:
    try:
        doc = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def needs_ablation(gt: GroundTruth, cap: GTCapture) -> bool:
    return cap.tier in ("video", "lidar") and len(gt.capture_rooms(cap)) >= 2


def _reload(run: dict[str, Any], dest: Path) -> dict[str, Any]:
    """skip_run: the stored result, with the status, error and runner time of the run that wrote it."""
    res = _load_result(dest / RESULT)
    rec = _load_result(dest / RUN_RECORD) or {}
    run["run_s"] = rec.get("run_s")
    if res is not None:
        run["result"] = res
    elif rec.get("status") == "failed":
        run.update(status="failed", error=rec.get("error"))
    else:
        run.update(status="missing", error=f"no {RESULT} in {dest} (skip_run)")
    return run


def run_one(gt: GroundTruth, cap: GTCapture, dest: Path, *, cache_mode: str, skip_run: bool, drift: bool,
            run_fn: RunFn | None = None) -> dict[str, Any]:
    """One pipeline run (or a reload with skip_run); failures are recorded, never raised."""
    run: dict[str, Any] = {"property": gt.property, "capture": cap.id, "tier": cap.tier,
                           "variant": "main" if drift else "nodrift", "out_dir": str(dest), "status": "ok",
                           "error": None, "run_s": None, "result": None}
    if skip_run:
        return _reload(run, dest)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (RESULT, RUN_RECORD, "error.txt"):
        (dest / name).unlink(missing_ok=True)
    t0 = time.perf_counter()
    if not cap.path.exists():
        run.update(status="failed", error=f"capture path not found: {cap.path}")
    else:
        if run_fn is None:
            from scan2scope.pipeline import run_capture

            run_fn = run_capture
        try:
            run["result"] = run_fn(cap.path, dest, tier=cap.tier, cache_mode=cache_mode,
                                   drift_correction=drift, semantics=drift, quiet=True)
        except Exception as exc:  # noqa: BLE001  one broken capture must not stop the benchmark
            tb = traceback.format_exc(limit=12)
            log.error("%s/%s%s failed: %s", gt.property, cap.id, "" if drift else NODRIFT_SUFFIX, exc)
            log.debug(tb)
            run.update(status="failed", error=f"{type(exc).__name__}: {exc}", traceback=tb)
            (dest / "error.txt").write_text(tb)
    run["run_s"] = round(time.perf_counter() - t0, 3)
    record = {k: run[k] for k in ("status", "error", "run_s", "variant")}
    (dest / RUN_RECORD).write_text(json.dumps(record, indent=1))
    return run


def score_runs(gts: list[GroundTruth], runs: list[dict[str, Any]], cfg: dict[str, Any]) -> dict[str, Any]:
    """Metrics, repeatability, ablation, head-to-head and gates from finished (or failed) runs."""
    by_prop = {gt.property: gt for gt in gts}
    main, nodrift = [], []
    commits: set[str] = set()
    cache_modes: set[str] = set()
    for run in runs:
        gt = by_prop[run["property"]]
        cap = gt.capture(run["capture"])
        res = run.get("result")
        m = capture_metrics(gt, cap, res, status=run["status"], error=run.get("error"),
                            run_s=run.get("run_s"))
        m["variant"], m["out_dir"] = run["variant"], run.get("out_dir")
        (main if run["variant"] == "main" else nodrift).append(m)
        prov = (res or {}).get("provenance") if isinstance((res or {}).get("provenance"), dict) else {}
        if prov.get("git_commit"):
            commits.add(str(prov["git_commit"]))
        if prov.get("cache_mode"):
            cache_modes.add(str(prov["cache_mode"]))
    rep = repeatability(main, cfg)
    abl = drift_ablation(main, nodrift, cfg)
    h2h = h2h_mod.merge([h2h_mod.head_to_head(gt, main) for gt in gts])
    gates = evaluate(main, rep, abl, h2h, cfg)
    return {
        "generated": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "properties": [{"property": gt.property, "path": str(gt.path), "synthetic": gt.synthetic,
                        "measured_by": gt.measured_by, "instrument": gt.instrument, "date": gt.date,
                        "rooms": [r.id for r in gt.rooms], "footprint": gt.footprint(),
                        "captures": [c.id for c in gt.captures], "flags": gt.flags} for gt in gts],
        "runs": [{k: v for k, v in r.items() if k != "result"} for r in runs],
        "metrics": main, "nodrift_metrics": nodrift, "repeatability": rep, "ablation": abl, "h2h": h2h,
        "gates": gates, "commits": sorted(commits), "cache_mode": ", ".join(sorted(cache_modes)) or "n/a",
    }


def run_benchmark(data_root: str | Path, out_dir: str | Path, *, cache_mode: str = "live",
                  only: list[str] | None = None, skip_run: bool = False, gates_path: str | Path | None = None,
                  run_fn: RunFn | None = None) -> dict[str, Any]:
    """Run (or with skip_run reuse) every listed capture, then score and write benchmark_report.md,
    metrics.json and gates.json to out_dir. Returns the scored benchmark dict."""
    data_root, out_dir = Path(data_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_gates(gates_path)
    gts: list[GroundTruth] = []
    load_errors = []
    for path in find_properties(data_root):
        try:
            gts.append(load_ground_truth(path))
        except (OSError, ValueError, TypeError, yaml.YAMLError) as exc:
            log.error("cannot read %s: %s", path, exc)
            load_errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    if not gts:
        log.warning("no ground_truth.yaml under %s", data_root)
    runs = []
    for gt in gts:
        for cap in gt.captures:
            if not _selected(only, gt.property, cap.id):
                continue
            log.info("benchmark: %s/%s (%s)", gt.property, cap.id, cap.tier)
            runs.append(run_one(gt, cap, out_dir / gt.property / cap.id, cache_mode=cache_mode,
                                skip_run=skip_run, drift=True, run_fn=run_fn))
            if needs_ablation(gt, cap):
                runs.append(run_one(gt, cap, out_dir / gt.property / f"{cap.id}{NODRIFT_SUFFIX}",
                                    cache_mode=cache_mode, skip_run=skip_run, drift=False, run_fn=run_fn))
    if only and not runs:
        log.warning("no capture matched --only %s", " ".join(only))
    bench = score_runs(gts, runs, cfg)
    bench["data_root"] = str(data_root)
    bench["load_errors"] = load_errors
    write_report(out_dir, bench)
    fails = bench["gates"]["ranked_failures"]
    log.info("benchmark: %d captures, %d failing gates%s", len(bench["metrics"]), len(fails),
             f", worst {fails[0]['tier']} {fails[0]['gate']}" if fails else "")
    return bench
