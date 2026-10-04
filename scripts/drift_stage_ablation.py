"""Per-stage drift ablation on five synthetic LiDAR captures: mean wall-length error with each correction stage
switched off. Output: runs/drift_stages/rows.json and one line per run (docs/benchmark/drift_stages.txt).

    uv run python scripts/drift_stage_ablation.py
"""

import functools
import json
import logging
from pathlib import Path

import scan2scope.geometry.lidar as L
from scan2scope.bench.groundtruth import load_ground_truth
from scan2scope.bench.metrics import capture_metrics
from scan2scope.pipeline import run_capture

logging.basicConfig(level=logging.WARNING)
orig = L.build_scene
configs = {"all": {}, "no_manhattan": {"manhattan_anchoring": False}, "no_plane": {"plane_anchoring": False},
           "loop_only": {"manhattan_anchoring": False, "plane_anchoring": False}, "off": None}
caps = [("synth_3", "lidar_drift"), ("synth_2", "lidar_1"), ("synth_1", "lidar_2"), ("synth_0", "lidar_drift"), ("synth_3", "lidar_1")]
ROOT = Path(__file__).resolve().parents[1]
root = ROOT / "bench" / "synthetic"
OUT = ROOT / "runs" / "drift_stages"
rows = []
for prop, cid in caps:
    gt = load_ground_truth(root / prop / "ground_truth.yaml")
    cap = next(c for c in gt.captures if c.id == cid)
    for name, opts in configs.items():
        L.build_scene = functools.partial(orig, drift_options=opts) if opts is not None else orig
        out = OUT / f"{prop}_{cid}_{name}"
        res = run_capture(cap.path, out, tier="lidar", cache_mode="off", drift_correction=opts is not None,
                          semantics=False, quiet=True)
        m = capture_metrics(gt, cap, res)
        walls = [r for r in m["records"] if r["kind"] == "wall_length"]
        mean = sum(abs(r["err"]) for r in walls) / max(1, len(walls))
        ok = sum(abs(r["err"]) <= max(0.02, 0.01 * r["gt"]) for r in walls)
        rows.append((prop, cid, name, round(100 * mean, 2), f"{ok}/{len(walls)}", len(m.get("missing", []))))
        print(*rows[-1], flush=True)
(OUT / "rows.json").write_text(json.dumps(rows))
