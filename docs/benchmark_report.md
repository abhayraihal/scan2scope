# Benchmark report

This report is curated from two generated reports, copied unchanged into `docs/benchmark/`: [real_report.md](benchmark/real_report.md) scores the photo and video tiers on a real room, and [synthetic_report.md](benchmark/synthetic_report.md) scores the LiDAR tier on synthetic captures. Their gate results are `real_gates.json` and `synthetic_gates.json` in the same folder.

- The real-room report was generated on 2026-10-03 at 19:20 UTC by commit 7d26d19 from the committed `bench/data`, in cache mode `mixed`. Its MapAnything outputs came from the cache written by the live run on the same files at commit 226bcff, whose report was generated at 18:37 UTC, and its damage detection was recomputed with the changes merged at 8df5f97. Compared value by value, every wall, ceiling, floor area and interval is identical in the two runs. The live run's own report is kept on the build machine in `runs/final_real_pre_damage_fix/` (not committed); it is the source of the live stage times under "Timing".
- The synthetic report was generated on 2026-10-03 at 18:30 UTC by commit 226bcff. It calls no model: LiDAR geometry needs none, and damage detection was skipped with `--no-semantics`.

A number that is in neither report says where it comes from.

## What the benchmark contains

The real room is one furnished bedroom, captured with an iPhone 17 (the only phone available) and entered as two properties made from the same captures.

- `bedroom` holds the phone's files, sent as documents. photo_1 is 5 photos at 5712x4284 with EXIF (1x lens, 26 mm equivalent). video_1 is 57 s at 1080p and video_2 is 49 s at 4K, both 30 fps, H.264, SDR, filmed in portrait. Laptop screens are blurred in the published copies by `scripts/blur_screens.py`: photos are re-saved as JPEG at quality 95 with EXIF kept, and videos are re-encoded to H.264 at the same size and frame rate. Every number here is computed from the published copies.
- `bedroom_whatsapp` holds the same captures after WhatsApp's ordinary media send, which left photos at 1280x960 with no EXIF and videos at 464x832. It is a degraded copy of the `bedroom` files.
- The ground truth is the owner's tape readings in inches, converted to metres: walls 162 in (4.115 m) and 142 in (3.607 m), ceiling 130 in (3.302 m), floor area 14.84 m2 from the two wall lengths. The geometry analysis suggested the ceiling might be about 120 in, and the owner confirmed 130 in a second time. Openings and damage extents were not measured, so they are not scored. The staged damage is one class, a paper sheet with a drawn crack.
- `bench/data/bedroom/` and `bench/data/bedroom_whatsapp/` hold `ground_truth.yaml` and `raw_manifest.json`. The raw files are GitHub release assets under tag `benchmark-data-v1` (6 zips, 331 MB), each pinned by size and SHA-256.

The synthetic LiDAR set has 4 generated properties of 4 to 6 rooms, each with a hallway, and 3 captures per property in the Stray Scanner 1.4 format with exact ground truth: 12 captures, 20 rooms, 252 walls and 192 openings. lidar_1 has ordinary pose drift (1 to 3 degrees of yaw over 3 minutes, 1 to 2 cm per minute), lidar_2 repeats the property with a different route and different noise and drift seeds, and lidar_drift is lidar_1 with strong drift (4 degrees and 10 cm by the end) for the drift ablation. Each capture is 200 to 206 s at 10 fps, with depth noise of 0.4 cm plus 0.6% of range. No phone produced them, and every LiDAR number below is synthetic.

Not in the benchmark: the public Stray Scanner office recording (vslamlab/strayscanner 4e41d0a7da, real iPhone LiDAR, no tape ground truth; see "Real LiDAR without ground truth") and three ARKitScenes development rooms (iPad Pro LiDAR, Apple's ARKitScenes licence), which were used to find failure causes during the build. Neither is redistributed.

## How to regenerate

```sh
uv sync --all-extras && uv run scan2scope fetch-weights
uv run python scripts/benchmark_data.py fetch bench/data/bedroom
uv run python scripts/benchmark_data.py fetch bench/data/bedroom_whatsapp
uv run scan2scope bench bench/data --out runs/final_real
uv run scan2scope synth --out bench/synthetic --properties 4 --seed 0
uv run scan2scope bench bench/synthetic --out runs/final_synth --no-semantics
```

On the M4 (16 GB) the live real-room bench at 226bcff took 12.5 minutes (the sum of its six runner times), and the rerun at 7d26d19 took 7.0 minutes with the geometry read from the cache. A clean machine has no cache, so its first run computes everything; at 7d26d19 that comes to about 16 minutes (live geometry from the first run plus damage detection from the rerun), which was not measured as one run. The synthetic bench took 5.5 minutes, including the 12 runs with drift correction off.

The real-room runs call MapAnything, Grounding DINO and SAM 2.1 on Apple MPS. Live reruns of bedroom/photo_1 and bedroom/video_1 at 7d26d19 on 2026-10-04, with the cache off so every model output was recomputed, gave the same walls, ceilings, floor areas and intervals to the 0.1 mm stored in result.json, and the same damage regions; the other four captures were not rerun for that check. The synthetic runs call no model, and the per-stage drift runs below reproduced the on and off rows of the generated report.

## Gate summary

Thresholds are in `bench/gates.yaml`. No Round 1 gate list was provided, so all thresholds are ours. In the Source column, "brief" means the brief states the number, "assumed" means it does not and we chose one, and "own rule" means the brief asks for the property and the test is ours. The brief's 1.5 cm ceiling bound and 2 cm opening bound are applied to LiDAR as written; photo and video use the tier's wall-length bound for ceilings and openings instead (assumed). A room, wall or opening that is not found counts as a failing item. Calibration passes when the 95% Clopper-Pearson confidence interval (CI) of the coverage contains 90%. n.a. means the benchmark has no item for that gate, and the Measured column gives the reason.

### Photo tier (real room, 2 captures)

| Gate | Status | Measured | Threshold | Source |
|---|---|---|---|---|
| result_produced | PASS | 2 of 2 captures | every capture returns a scored result with geometry | own rule |
| wall_length | PASS | 8 of 8 walls; worst -7.5% (bedroom_whatsapp W1) | within 8% of tape on every wall | brief |
| ceiling_height | FAIL | 0 of 2 rooms; worst -19.8% (bedroom) | within 8% in every room | assumed |
| ceiling_spread | n.a. | one photo capture per property | spread across captures at most 1 cm | brief |
| opening_width | n.a. | openings not measured | within 8% on at least 85% of openings | assumed |
| floor_area | PASS | 2 of 2 rooms; worst -11.5% (bedroom) | within 16% in every room | assumed |
| footprint | n.a. | no multi-room capture | within 8% per multi-room capture | brief |
| stitch | n.a. | one room, nothing to stitch | adjacency exact, overlap at most 0.05 m2, footprint within 8% | brief |
| repeatability | n.a. | one photo capture per property | delta at most max(1 cm, 0.5%) per wall | brief |
| repeat_structure | n.a. | one photo capture per property | repeat captures find the same walls and openings | brief |
| calibration | PASS | 14 of 14 values covered (100%), CI [76.8%, 100%], 2 rooms | CI of coverage contains 90% | own rule |
| confident_garbage | PASS | 0 of 14 values | no miss larger than 2 x the interval half-width | own rule |

### Video tier (real room, 4 captures)

| Gate | Status | Measured | Threshold | Source |
|---|---|---|---|---|
| result_produced | PASS | 4 of 4 captures | every capture returns a scored result with geometry | own rule |
| wall_length | FAIL | 1 of 16 walls; worst -58.1% (bedroom/video_2 W4) | within 3% of tape on every wall | brief |
| ceiling_height | FAIL | 0 of 4 rooms; worst -31.0% (bedroom/video_2) | within 3% in every room | assumed |
| ceiling_spread | FAIL | 0 of 2 rooms; 58.5 cm (bedroom) and 39.3 cm (bedroom_whatsapp) between video_1 and video_2 | spread across captures at most 1 cm | brief |
| opening_width | n.a. | openings not measured | within 3% on at least 85% of openings | assumed |
| floor_area | FAIL | 0 of 4 rooms; worst -25.5% (bedroom_whatsapp/video_2) | within 6% in every room | assumed |
| repeatability | FAIL | 1 of 8 wall pairs (strict reading 0 of 8); worst 193.5 cm (bedroom W4) | delta at most max(1 cm, 0.5%) per wall | brief |
| repeat_structure | FAIL | 0 of 2 rooms: 4 vs 8 walls (bedroom), 4 vs 6 walls (bedroom_whatsapp); 2 openings in every clip | repeat captures find the same walls and openings | brief |
| drift_ablation | n.a. | single-room captures; the harness runs the drift-off variant only on multi-room captures | drift correction on for every multi-room capture, with an off run | brief |
| calibration | PASS | 25 of 28 values covered (89.3%), CI [71.8%, 97.7%], 2 rooms | CI of coverage contains 90% | own rule |
| confident_garbage | PASS | 0 of 28 values | no miss larger than 2 x the interval half-width | own rule |

### LiDAR tier (synthetic, 12 captures)

| Gate | Status | Measured | Threshold | Source |
|---|---|---|---|---|
| result_produced | PASS | 12 of 12 captures | every capture returns a scored result with geometry | own rule |
| wall_length | FAIL | 209 of 252 walls (82.9%), 10 not found; worst 98x the allowance (synth_0/lidar_drift kitchen W4, 3.89 m too long) | within max(2 cm, 1%) on every wall | assumed |
| ceiling_height | FAIL | 58 of 60 rooms; both failures are rooms not found, and the worst ceiling found is 0.5 cm off | within 1.5 cm in every room | brief |
| ceiling_spread | PASS | 20 of 20 rooms; largest spread 0.6 cm | spread across captures at most 1 cm | brief |
| opening_width | FAIL | 61 of 202 (30.2%): 97 of 192 openings missed, 10 phantoms, 34 found but more than 2 cm off | within 2 cm on at least 85% of openings | brief |
| floor_area | FAIL | 52 of 60 rooms, 2 not found; worst +62.9% (synth_0/lidar_drift kitchen) | within 2% in every room | assumed |
| repeatability | FAIL | 155 of 232 wall pairs (66.8%), strict reading 92 of 232; worst 389.1 cm | delta at most max(1 cm, 0.5%) per wall | brief |
| repeat_structure | FAIL | 36 of 56 room pairs | repeat captures find the same walls and openings | brief |
| drift_ablation | PASS | 12 of 12 captures have an on and an off run | drift correction on for every multi-room capture, with an off run | brief |
| calibration | FAIL | 479 of 560 values covered (85.5%), CI [82.3%, 88.3%], 20 rooms | CI of coverage contains 90% | own rule |
| confident_garbage | FAIL | 38 of 560 values; worst 95x the half-width (synth_0/lidar_drift kitchen W4) | no miss larger than 2 x the interval half-width | own rule |

Room adjacency on the synthetic captures is exact on 9 of 12; the 3 others are the strong-drift captures of synth_0, synth_1 and synth_3 (rooms table in the synthetic report). No synthetic room overlaps another.

The harness also ranks the failing gates for the fix loop ("Failing gates ranked" in each generated report). Repeatability is the worst gate in both: video on the real room, LiDAR on the synthetic set.

## Real room: every scored value

Predicted value in metres (m2 for area), signed error against the tape in brackets. Every capture found the one room. The room has two pairs of equal walls, and a rectangular plan gives opposite walls the same length, so "x2" marks a value that scored twice.

| Capture | Input | Walls, 4.115 m pair | Walls, 3.607 m pair | Ceiling, 3.302 m | Floor area, 14.84 m2 | Walls in plan |
|---|---|---|---|---|---|---|
| bedroom/photo_1 | 5 photos, 5712x4284, EXIF | 3.851 (-6.4%) x2 | 3.409 (-5.5%) x2 | 2.648 (-19.8%) | 13.13 (-11.5%) | 4 |
| bedroom_whatsapp/photo_1 | 5 photos, 1280x960, no EXIF | 3.808 (-7.5%) x2 | 3.595 (-0.3%) x2 | 2.713 (-17.8%) | 13.69 (-7.8%) | 4 |
| bedroom/video_1 | 57 s, 1080p | 3.855 (-6.3%) x2 | 3.445 (-4.5%) x2 | 2.862 (-13.3%) | 13.28 (-10.5%) | 4 |
| bedroom/video_2 | 49 s, 4K | 4.900 (+19.1%), 3.370 (-18.1%) | 3.434 (-4.8%), 1.510 (-58.1%) | 2.277 (-31.0%) | 13.76 (-7.3%) | 8 |
| bedroom_whatsapp/video_1 | 57 s, 464x832 | 3.758 (-8.7%) x2 | 3.363 (-6.8%) x2 | 2.825 (-14.4%) | 12.64 (-14.9%) | 4 |
| bedroom_whatsapp/video_2 | 49 s, 464x832 | 3.227 (-21.6%), 2.361 (-42.6%) | 3.222 (-10.7%), 3.504 (-2.9%) | 2.431 (-26.4%) | 11.06 (-25.5%) | 6 |

23 of the 24 wall values, all 6 ceilings and all 6 floor areas are short of the tape.

Damage is not scored, because its extents were not measured. In the rerun at 7d26d19, 4 of the 6 captures report a crack on a wall; the room's staged damage is a crack drawn on paper, but no measurement ties each detection to that sheet. The live run at 226bcff reported no crack, and instead a hole on a wall in both video_1 copies and peeling paint on the floor in bedroom/video_2.

## Ceiling height: bias or spread

The brief fails a ceiling that is repeatable but biased and one that is unrepeatable, and asks which of the two applies.

| Tier | Data | Mean signed error | Spread across repeat captures | Mode |
|---|---|---|---|---|
| Photo | real room, 2 captures | -62.2 cm | no repeat capture | bias |
| Video | real room, 4 captures | -70.3 cm | 58.5 cm (bedroom), 39.3 cm (bedroom_whatsapp) | both |
| LiDAR | synthetic, 58 of 60 rooms found | -0.0 cm; worst 0.5 cm | at most 0.6 cm over 20 rooms | neither; the gate fails on the 2 rooms not found |

## Repeatability

Allowed |delta| is max(1 cm, 0.5% of the tape length): 1.8 cm on the 3.607 m walls and 2.1 cm on the 4.115 m walls. The strict reading, min(1 cm, 0.5%), is 1.0 cm on both.

Real room, video_1 against video_2. The photo tier has one capture per property, so it has no pairs.

| Property | Wall (tape) | video_1 (m) | video_2 (m) | \|delta\| | Allowed | Pass | Strict pass |
|---|---|---|---|---|---|---|---|
| bedroom | W1 (4.115) | 3.855 | 4.900 | 104.5 cm | 2.1 cm | no | no |
| bedroom | W2 (3.607) | 3.445 | 3.434 | 1.0 cm | 1.8 cm | yes | no |
| bedroom | W3 (4.115) | 3.855 | 3.370 | 48.4 cm | 2.1 cm | no | no |
| bedroom | W4 (3.607) | 3.445 | 1.510 | 193.5 cm | 1.8 cm | no | no |
| bedroom_whatsapp | W1 (4.115) | 3.758 | 3.227 | 53.1 cm | 2.1 cm | no | no |
| bedroom_whatsapp | W2 (3.607) | 3.363 | 3.222 | 14.1 cm | 1.8 cm | no | no |
| bedroom_whatsapp | W3 (4.115) | 3.758 | 2.361 | 139.7 cm | 2.1 cm | no | no |
| bedroom_whatsapp | W4 (3.607) | 3.363 | 3.504 | 14.1 cm | 1.8 cm | no | no |

The ceiling came out at 2.862 and 2.277 m in bedroom (58.5 cm apart) and 2.825 and 2.431 m in bedroom_whatsapp (39.3 cm). Neither room gives the same plan twice: bedroom has 4 walls in video_1 and 8 in video_2, and bedroom_whatsapp has 4 and 6; every clip finds 2 openings. In the live run at 226bcff a mirror detection removed one of the two openings in bedroom_whatsapp/video_1; after the damage-detection changes it no longer does. The openings are not scored, since none were measured.

Synthetic LiDAR, by capture pair. The 232 rows are in [synthetic_report.md](benchmark/synthetic_report.md); the medians are computed from the same run's `metrics.json`.

| Pair | Wall pairs | Pass | Strict pass | Median \|delta\| |
|---|---|---|---|---|
| lidar_1 vs lidar_2 | 84 | 60 (71%) | 41 | 0.94 cm |
| lidar_1 vs lidar_drift | 74 | 48 (65%) | 30 | 1.33 cm |
| lidar_2 vs lidar_drift | 74 | 47 (64%) | 21 | 1.65 cm |
| all | 232 | 155 (66.8%) | 92 | 1.24 cm |

Ceiling heights agree within 1 cm in all 20 synthetic rooms (largest spread 0.6 cm). 36 of the 56 room pairs found the same walls and openings in both captures.

## Calibration

Coverage of the 90% intervals over every scored value, with an exact (Clopper-Pearson) 95% CI on that coverage. Confident garbage is a miss by more than twice the interval half-width. The interval multiplier q is still its prior, 1.0, at every tier: no tier has the 9 real ground-truth rooms the conformal fit needs, and q was not fitted on the synthetic rooms.

| Tier | Data | Values | Rooms | Covered | Coverage | 95% CI | Contains 90% | Mean half-width (% of truth) | Confident garbage | q |
|---|---|---|---|---|---|---|---|---|---|---|
| Photo | real room | 14 | 1 physical (counted as 2) | 14 | 100.0% | [76.8%, 100.0%] | yes | 33.0% | 0 | prior 1.0 |
| Video | real room | 28 | 1 physical (counted as 2) | 25 | 89.3% | [71.8%, 97.7%] | yes | 27.2% | 0 | prior 1.0 |
| LiDAR | synthetic | 560 | 20 | 479 | 85.5% | [82.3%, 88.3%] | no | 3.1% | 38 | prior 1.0 |

By kind (metres, or m2 for areas):

| Tier | Kind | n | Coverage | Mean half-width | Mean \|error\| |
|---|---|---|---|---|---|
| Photo | wall_length | 8 | 100.0% | 1.008 | 0.195 |
| Photo | ceiling_height | 2 | 100.0% | 0.861 | 0.622 |
| Photo | floor_area | 2 | 100.0% | 7.442 | 1.434 |
| Video | wall_length | 16 | 81.2% | 0.809 | 0.561 |
| Video | ceiling_height | 4 | 100.0% | 0.971 | 0.703 |
| Video | floor_area | 4 | 100.0% | 5.706 | 2.159 |
| LiDAR | wall_length | 242 | 82.2% | 0.049 | 0.101 |
| LiDAR | ceiling_height | 58 | 100.0% | 0.017 | 0.001 |
| LiDAR | floor_area | 58 | 94.8% | 0.583 | 0.415 |
| LiDAR | opening_width | 95 | 64.2% | 0.021 | 0.039 |
| LiDAR | opening_height | 95 | 98.9% | 0.197 | 0.261 |
| LiDAR | footprint | 12 | 100.0% | 2.210 | 0.703 |

What these numbers support:

- Photo and video pass on one room because their intervals are wide. The harness counts 2 rooms because the two quality versions are separate properties; physically it is one bedroom, and each one-room capture also scores its footprint, which repeats its floor area. The photo interval on a 4.115 m wall is about ±1 m (2.90 to 4.80 m for bedroom/photo_1). These intervals contain the tape, but a CI that spans 23 to 26 points does not show calibration. The 3 video misses are all in video_2: bedroom W3 (-18.1%) and bedroom_whatsapp W1 (-21.6%) and W3 (-42.6%), none by more than twice the half-width.
- The LiDAR intervals are too narrow. Coverage is 85.5% and its CI excludes 90%. The 81 misses are 43 wall lengths, 34 opening widths, 3 floor areas and 1 opening height. The 38 confident misses are 21 wall lengths, 16 opening widths and 1 floor area, spread over all three capture types (16 in lidar_drift, 15 in lidar_1, 7 in lidar_2). Opening widths get a mean half-width of 2.1 cm against a mean error of 3.9 cm.

## Drift ablation

The brief asks for the stitched footprint with drift handling on and off, and fails a pipeline that uses poses as they come. Every LiDAR and video capture runs with drift correction on (method in `docs/technical_report.md` section 4). The real room has single-room captures only, so the ablation is on the synthetic LiDAR set.

### On and off (synthetic LiDAR)

Each capture is run with drift correction on and with it off (`<capture>__nodrift`). Walls in tolerance count the walls within max(2 cm, 1%) out of all walls in the truth, so walls not found count against a run.

| Capture | Footprint truth (m2) | Footprint error on / off | Wall mean \|error\| on / off | Walls in tolerance on / off | Loop closures accepted |
|---|---|---|---|---|---|
| synth_0/lidar_1 | 78.25 | -1.1% / -1.0% | 9.4 / 18.6 cm | 85.0% / 90.0% | 55 of 120 |
| synth_0/lidar_2 | 78.25 | -0.2% / -0.1% | 0.5 / 0.5 cm | 100.0% / 100.0% | 54 of 120 |
| synth_0/lidar_drift | 78.25 | -2.8% / -2.4% | 47.7 / 39.1 cm | 65.0% / 35.0% | 60 of 120 |
| synth_1/lidar_1 | 76.61 | -3.3% / -3.6% | 18.7 / 20.8 cm | 72.7% / 68.2% | 59 of 120 |
| synth_1/lidar_2 | 76.61 | -0.1% / -0.4% | 0.9 / 8.5 cm | 100.0% / 77.3% | 57 of 120 |
| synth_1/lidar_drift | 76.61 | -1.6% / -1.0% | 22.2 / 17.8 cm | 63.6% / 63.6% | 49 of 120 |
| synth_2/lidar_1 | 79.44 | -0.5% / -0.0% | 1.5 / 0.4 cm | 87.5% / 100.0% | 71 of 120 |
| synth_2/lidar_2 | 79.44 | -0.5% / +1.4% | 2.2 / 3.9 cm | 100.0% / 87.5% | 61 of 120 |
| synth_2/lidar_drift | 79.44 | +0.0% / -0.1% | 1.6 / 0.9 cm | 100.0% / 100.0% | 64 of 120 |
| synth_3/lidar_1 | 65.40 | -0.1% / -0.1% | 1.4 / 1.2 cm | 84.6% / 92.3% | 19 of 120 |
| synth_3/lidar_2 | 65.40 | -0.7% / -0.2% | 1.7 / 0.7 cm | 84.6% / 92.3% | 39 of 120 |
| synth_3/lidar_drift | 65.40 | -0.2% / -0.0% | 19.9 / 2.1 cm | 65.4% / 88.5% | 17 of 120 |

Counted from the unrounded values in the run's `metrics.json`: with correction on, the footprint error is smaller on 5 of the 12 captures and larger on 7 (mean absolute footprint error 0.92% on, 0.87% off). The mean wall error is lower on 4 and higher on 8 (all 4 strong-drift captures among them; on synth_0/lidar_2 by 0.04 cm). The count of walls in tolerance is higher on 4, lower on 5 and equal on 3, and summed over the 12 captures it is 209 with correction and 208 without. The drift_ablation gate passes because it checks that correction runs and that an off run exists. It does not check that correction helps, and on this set it does not help reliably. On synth_3/lidar_drift the correction loses the bathroom, which the off run finds.

### Per stage (synthetic LiDAR, 5 captures)

Drift correction has three stages: loop closure, plane anchoring and Manhattan yaw anchoring (`docs/technical_report.md` section 4). Each was switched off on its own on 5 captures. Each cell gives the mean |wall length error| over the walls found, then the walls within max(2 cm, 1%) out of the walls found; the number after the capture is its wall count in the truth.

| Capture (walls) | All stages | No Manhattan yaw | No plane anchoring | Loop closure only | Off |
|---|---|---|---|---|---|
| synth_3/lidar_drift (26) | 19.93 cm, 17 of 20 | 2.17 cm, 22 of 26 | 19.92 cm, 17 of 20 | 2.17 cm, 22 of 26 | 2.15 cm, 23 of 26 |
| synth_2/lidar_1 (16) | 1.48 cm, 14 of 16 | 1.48 cm, 14 of 16 | 1.48 cm, 14 of 16 | 1.39 cm, 14 of 16 | 0.37 cm, 16 of 16 |
| synth_1/lidar_2 (22) | 0.93 cm, 22 of 22 | 6.52 cm, 21 of 22 | 0.94 cm, 22 of 22 | 5.79 cm, 20 of 22 | 8.45 cm, 17 of 22 |
| synth_0/lidar_drift (20) | 47.69 cm, 13 of 16 | 36.45 cm, 13 of 16 | 36.07 cm, 13 of 16 | 36.50 cm, 13 of 16 | 39.09 cm, 7 of 12 |
| synth_3/lidar_1 (26) | 1.43 cm, 22 of 26 | 0.96 cm, 24 of 26 | 1.33 cm, 22 of 26 | 0.93 cm, 24 of 26 | 1.20 cm, 24 of 26 |

- Manhattan yaw anchoring moves the result most, in both directions. Switching it off takes synth_3/lidar_drift from 19.93 to 2.17 cm and synth_3/lidar_1 from 1.43 to 0.96 cm, and takes synth_1/lidar_2 from 0.93 to 6.52 cm.
- Plane anchoring changes little, except on synth_0/lidar_drift, where switching it off takes the mean error from 47.69 to 36.07 cm.
- Loop closure alone beats no correction on synth_1/lidar_2 (5.79 against 8.45 cm) and synth_3/lidar_1 (0.93 against 1.20 cm), and loses on synth_2/lidar_1 (1.39 against 0.37 cm).
- On synth_0/lidar_drift every corrected variant finds 16 of the 20 walls and the off run finds 12, so the off mean is over fewer walls.

A self-check was tried to choose between corrected and raw poses: keep whichever set gives sharper walls (fewer occupied voxels for wall points). It preferred the corrected poses on all 6 captures it was run on, including synth_3/lidar_drift, synth_2/lidar_1 and synth_3/lidar_1, where the corrected poses give the larger mean wall error and fewer walls in tolerance, so it was not shipped.

No CLI flag switches single stages, so the per-stage runs came from a short script. It wraps `scan2scope.geometry.lidar.build_scene` with `drift_options` (`{"manhattan_anchoring": False}`, `{"plane_anchoring": False}`, or both for loop closure only), runs `scan2scope.pipeline.run_capture(..., cache_mode="off", semantics=False)` and scores walls with `scan2scope.bench.metrics.capture_metrics`; "off" is `drift_correction=False`.

### Real LiDAR without ground truth

The public Stray Scanner office recording is 232 s, 3,481 frames at 15 fps; README, "Reproduce the reported numbers", has the pinned download. Drift correction accepted 71 of 122 loop closures, and the spread of floor height across its 76 segments fell from 47.9 cm to 5.5 cm; the largest correction was 0.54 m and 2.15 degrees. Rerun on 2026-10-04 at d766d21, whose pipeline source is the same as 226bcff, the open-plan office came out as one room with 16 walls, 17 openings and 189.6 m2 [177.4, 258.0]. A run at 4b544c2, before the fix loop's layout changes, gave one room with 32 walls and 191.5 m2 (log on the build machine). With `--no-drift` the same recording gives one room with 10 walls, 19 openings and 184.8 m2 [176.7, 215.4]. There are no tape readings for it, so neither plan can be scored.

## Head-to-head against magicplan: not done

The brief asks for our LiDAR-tier output against one consumer app on 2 benchmark rooms, in one table of both errors per dimension, beating or tying on at least 70% of shared dimensions. None of this was done:

- No LiDAR phone was available, so there is no LiDAR-tier output of a real room. The fallback in `docs/design.md` was magicplan on its free Starter plan, in its camera mode without LiDAR on the iPhone 17, against our photo and video tiers.
- The one magicplan export, `bench/data/bedroom/magicplan/room.ifc`, is 1,978 bytes and has no room geometry to score: it holds a project and a building and no IfcSpace, IfcWall, IfcSlab, IfcDoor or IfcWindow. The Statistics CSV and Sketch PDF that the harness reads (`bench/data/<property>/magicplan/statistics.csv` and `dimensions.yaml`) were not exported, and the app version was not recorded.
- No data from any other app was captured.

The comparison code exists (`src/scan2scope/bench/h2h.py`; ours ties when it is within 3 mm of the app's error on a length, or 0.2 percentage points on an area), and both generated reports say "No magicplan export found". The compliance matrix marks the head-to-head as failed.

## Timing

M4 MacBook, 16 GB, Apple MPS; seconds per stage from each result.json. The real room needs two runs. The live run at 226bcff (report in `runs/final_real_pre_damage_fix/` on the build machine) gives the live geometry times. The rerun at 7d26d19 (`docs/benchmark/real_report.md`) gives the damage-detection times of the current code; its own geometry column shows cache reads of 0.5 to 11.3 s, which are not live times.

MapAnything loads once per process, on the first capture that needs it (bedroom/photo_1 here), while Grounding DINO and SAM 2.1 load again in every capture's damage-detection stage. Ingest, rules, intervals and scope take under 0.5 s together on every capture.

| Capture | Input | Geometry, live | Layout | Damage detection at 226bcff | Damage detection at 7d26d19 | Runner, live at 226bcff | Live estimate at 7d26d19 |
|---|---|---|---|---|---|---|---|
| bedroom/photo_1 | 5 photos, 5712x4284 | 36.1 | 1.6 | 17.3 | 26.1 | 55.6 | 64 |
| bedroom_whatsapp/photo_1 | 5 photos, 1280x960 | 6.9 | 0.9 | 10.8 | 20.9 | 18.8 | 29 |
| bedroom/video_1 | 57 s, 1080p | 139.4 | 1.3 | 38.1 | 89.0 | 179.3 | 230 |
| bedroom/video_2 | 49 s, 4K | 133.2 | 1.3 | 34.4 | 76.4 | 169.2 | 211 |
| bedroom_whatsapp/video_1 | 57 s, 464x832 | 121.5 | 1.0 | 39.6 | 84.7 | 162.3 | 207 |
| bedroom_whatsapp/video_2 | 49 s, 464x832 | 128.5 | 1.7 | 34.3 | 80.0 | 164.7 | 210 |

The last column adds the live geometry, layout and ingest of the first run to the damage detection of the rerun. Two captures were later run live at 7d26d19 with the cache off and other jobs running: bedroom/photo_1 took 65 s (32 s geometry including the model load, 32 s damage detection) and bedroom/video_1 264 s (142 s geometry, 120 s damage detection). Video geometry took 2.1 to 2.7 minutes per minute of clip, so at 7d26d19 a clip of about a minute takes about 3.5 to 4.5 minutes. The damage-detection changes merged at 8df5f97 made that stage 1.5 to 2.3 times slower.

LiDAR, with no damage detection on any capture (it was not timed for this tier):

| Captures | Input | Geometry | Layout | Total | Runner |
|---|---|---|---|---|---|
| 12 synthetic captures, 226bcff | 200 to 206 s at 10 fps | 11.3 to 17.0 | 2.9 to 5.0 | 14.5 to 22.3 | 14.8 to 22.6 |
| the same 12, drift correction off | as above | 4.0 to 6.1 | 3.2 to 4.6 | 7.7 to 10.4 | 8.1 to 10.9 |
| office recording, rerun at d766d21 | 232 s at 15 fps | 21.3 | 2.7 | 24.8 | 25.8 (whole process) |
| the same, drift correction off | as above | 11.8 | 3.2 | 15.8 | 16.7 (whole process) |

Clean-machine setup, measured on 2026-10-03 from a fresh clone with empty caches at about 11 MB/s: clone 6 s, `uv sync --all-extras` 75 s, `fetch-weights` 579 s (5.8 GB of weights), `doctor` 19 s. With the first photo run (about 50 s, README; 65 s at 7d26d19) that is 12 to 12.5 minutes from clone to a result, most of it the weight download.

## What fails and why

- The real room came out small. 23 of 24 wall values, every ceiling and every floor area are short of the tape. Photo and video take their metric scale from MapAnything alone, which measured 7 to 10% off per room on real captures during the fixes, roughly constant across the chunks of one clip (`docs/design.md`, decision log). Correct intrinsics did not fix it on this room, and a door-height scale cue was not shipped because the evidence did not support it. For the videos and the WhatsApp photos, which carry no focal length, MapAnything estimated a focal of 0.54 to 0.59 of the long image side, below 0.6, the shortest the pipeline treats as plausible for the protocol's 1x lens. Technical report sections 5 and 9.
- Ceilings came out lower than walls: 13 to 31% low, against walls 0.3 to 8.7% short in the four captures that found a 4-wall rectangle. The original photos passed their EXIF focal and still lost 19.8%. The cause inside MapAnything was not isolated, and the interval model carries a separate vertical term for it (`src/scan2scope/uncertainty/priors.yaml`). Both video_2 clips flag `ceiling_not_observed`, and the WhatsApp copy fell back to an assumed ceiling; the clips were filmed in portrait, while the protocol asks for the phone held sideways with the ceiling line in view. The ceiling check (a level counts only if its points spread over the room) rejected 1 to 6 candidate levels per capture, and the level that remained is still low. Technical report section 9.
- video_2 broke the plan into 8 walls (original) and 6 walls (WhatsApp copy) for a 4-wall room, with errors up to -58.1%. In the WhatsApp copy one chunk link was refused, because the two runs placed the shared cameras 14.4% of scene depth apart (the bound is 10%), and the loop closure bridged it. Technical report section 4.
- Video repeatability cannot pass with scale from MapAnything alone: the allowance is 0.5% of the wall (1.8 to 2.1 cm), while MapAnything's scale is 7 to 10% off per room and is not the same from run to run (video_1 W1 came out 3.855 m from the original file and 3.758 m from the WhatsApp copy, 2.5% apart).
- LiDAR misses half the openings: 97 of the 192 synthetic openings, all 84 windows among them, since an opening needs depth seen through it and the synthetic windows have nothing behind them; 10 are phantoms. These are the same numbers as in the fix-loop after run.
- LiDAR repeatability and wall length fail on drift that correction leaves behind and on lost rooms. After correction the synthetic camera centres are still 0.9 to 5.9 cm RMS from the truth and yaw is off by 0.11 to 0.73 degrees (post-mortem), against a 1 cm allowance. In 2 of the 4 strong-drift captures a room is not found (the synth_0 hallway, the synth_3 bathroom) and a neighbouring room runs on through its space; these two captures hold all 10 missing walls and the worst wall error (3.89 m). The post-mortem also finds 15 of 116 room instances with the wrong wall count, mostly where a doorway opens onto free space and the room continues through it (`docs/fixloop/postmortem.md`). Technical report section 7.
- The LiDAR intervals are too narrow: 85.5% coverage with 38 confident misses, opening-width intervals covering 64.2% and wall-length intervals 82.2%. Technical report section 5.
- Drift correction helps on some synthetic captures and hurts on others, and Manhattan yaw anchoring causes the largest regression (synth_3/lidar_drift, 2.15 cm with correction off against 19.93 cm with it on). Technical report section 4.
- Not measured or not shown: openings and damage extents in the real room; photo stitching, which no benchmark capture exercises (no real or synthetic capture has more than one photo room, so it is covered by unit tests only); adjacency and a video drift ablation on real data, which need a real multi-room capture; the LiDAR tier on a phone with tape ground truth; the head-to-head; a second staged damage class.

Known failure modes (mirrors, glass, wet floors, low light, furniture against walls) are in `docs/technical_report.md` section 9. The fix loop is in `docs/fixloop/`.
