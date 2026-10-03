# Fix declaration

Declared on 2026-10-03 at about 18:55 IST, before any fix code, in the commit tagged `fixloop-declared`. The code being fixed is tag `fixloop-before` (commit 2d6f3b4). At this point the benchmark holds the synthetic LiDAR set only (4 properties, 12 captures, exact ground truth); the real photo and video captures had not been taken yet. The before-run report is in `docs/fixloop/before/`.

## 1. Worst gate and the failing number

LiDAR repeatability, rank 1 under the ranking rule committed with the harness (`src/scan2scope/bench/gates.py`, score 204.4). 82 of 236 wall pairs from repeat captures of the same room agree within max(1 cm, 0.5%): **34.7%** (strict reading min(1 cm, 0.5%): 44 of 236). The worst pair is the synth_0 hallway W1, which differs by 205 cm between `lidar_1` and `lidar_2`.

## 2. Root-cause hypothesis and evidence

Hypothesis: the layout stage, not sensor noise or poses, makes repeat captures disagree. Three mechanisms:

1. Wall lines are fitted once per property from a global histogram. Two parallel faces of different rooms that lie within about 10 cm of each other collapse into one line placed between them. Evidence: in synth_0/lidar_1 the bedroom (true width 3.519 m) and the bathroom (true width 3.661 m) were both reported as 3.593 m, the mean of the two.
2. Room polygons take steps along both faces of interior walls and along furniture faces. Evidence: 43 of 112 room instances have the wrong number of walls (truth: 18 rooms with 4 walls and 2 with 6; predicted: 69 with 4, 24 with 6, 15 with 8, 4 with 12 or 22). The synth_0 hallway, a 4-wall rectangle, came out with 8 walls including two 0.14 and 0.16 m steps, the thickness of the walls on its far side. The synth_0 kitchen got a 0.60 x 1.04 m notch at a furniture box.
3. When the wall structure differs between captures, the matcher can only pair walls that do not correspond, which produces the metre-scale deltas.

Check against a noise explanation: in the 28 room pairs where both captures gave a 4-wall rectangle, the per-wall |delta| still has a median of 2.4 cm and a 90th percentile of 9.0 cm. Synthetic depth noise (sigma 0.4 cm + 0.6% of range, thousands of points per wall) predicts a few millimetres per wall face. Ceiling heights, fitted per room from the same depth and poses, already agree within 0.3 cm across captures, which points at the wall-line step rather than at depth or poses.

## 3. Fix to ship, and the predicted number

Changes in `src/scan2scope/layout` only:

1. Per-room wall refinement: after rooms are segmented, refit each room's walls from the points within 15 cm of that edge whose normals face into that room, then rebuild the polygon corners from the refined lines.
2. Polygon regularisation: remove steps shorter than 0.25 m between parallel edges by snapping to the better-supported face.
3. Furniture filter: an edge whose points never reach the band 0.5 m below the ceiling is treated as furniture, and its notch is filled.

Nothing else changes in this fix. Improvements to opening detection come afterwards in separate commits, so they do not mix into the before/after comparison.

Prediction for the after run on the same data:

- Repeatability per-wall pass share: 34.7% → **85%** (I expect 75 to 95%).
- Median |delta| in rectangle pairs: 2.4 cm → under 0.6 cm.
- Room instances with the wrong wall count: 43 of 112 → under 10.
- The every-wall gate may still fail on a few walls if some room is still split or merged differently between captures. The post-mortem will list those walls.
- Side effects I expect: wall-length pass share 58% → at least 85%, floor-area pass share 57% → at least 80%.

## Regenerating both runs

```sh
uv run scan2scope synth --out bench/synthetic --properties 4 --seed 0
BENCH_ARGS=--no-semantics CACHE=off scripts/fixloop.sh fixloop-before fixloop-after bench/synthetic
```

The script benchmarks both refs from clean worktrees on the same data and writes `runs/fixloop/before`, `runs/fixloop/after` and `runs/fixloop/diff.md` (gate table and code diff).
