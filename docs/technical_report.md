# scan2scope technical report

Numbers come from the generated reports in `docs/benchmark/`, curated in `docs/benchmark_report.md`: the real room at commit 7d26d19, which reused the MapAnything outputs of a live run at 226bcff and recomputed damage detection, and synthetic LiDAR at 226bcff. Other sources: the runs' `metrics.json` and `result.json`, `docs/fixloop/postmortem.md`, `docs/design.md`, code comments and ARKitScenes development runs.

## 1. What it does

`scan2scope run <capture>` turns one iPhone capture (photo folders, a video, or a Stray Scanner LiDAR export) into a dimensioned property plan, damage regions on named surfaces, concealed-damage flags naming the rule that fired, and scope line items keyed to surfaces, with a 90% interval on every measurement, in a `result.json` validated against `schema/scan2scope.schema.json`.

The benchmark is one real bedroom shot on an iPhone 17 (5 photos, a 57 s 1080p clip, a 49 s 4K clip), scored as the original files (laptop screens blurred) and as WhatsApp's recompressed copies (photos 1280x960, video 464x832, no EXIF), plus 12 synthetic Stray Scanner captures of 4 properties (20 rooms) with exact ground truth. Used in development and not redistributed: three ARKitScenes validation scans (iPad Pro LiDAR, Apple's ARKitScenes licence) and the public office recording `4e41d0a7da` from `vslamlab/strayscanner`, which has no tape readings.

## 2. Architecture

Stages run in the order ingest, geometry, layout, stitch (photo only), semantics, rules, intervals, scope and output. The tiers differ only in how geometry builds the `Scene` (gravity-aligned points with normals and confidence weights, and camera views with per-pixel world point maps), so a layout fix such as the fix loop's (section 7) reaches all three tiers.

| Stage | Method |
|----------|--------------------------------------------------|
| LiDAR geometry | Stray Scanner depth (256x192, 0.2 to 4.5 m) with per-frame ARKit poses and intrinsics, weighted by confidence; drift correction (section 4) |
| Video geometry | MapAnything on 24-frame chunks of frames sampled at 1.5 fps, chained by Sim(3); loop closure; floor and Manhattan anchoring |
| Photo geometry, stitch | MapAnything per room folder with the EXIF focal; rooms placed by doorway photos or door matching, without overlap |
| Layout | floor and ceiling from a height histogram, a ceiling kept only if it spreads over the room; Manhattan wall lines refitted per room; cells labelled by free-space rays; rooms split at door-sized gaps; openings where rays pass through a wall |
| Semantics | Grounding DINO tiny boxes and SAM 2.1 masks lifted onto surfaces and merged across views; look-alikes and single-view detections under 0.5 dropped; cracks need a thin dark line; openings at detected mirrors removed |
| Rules, scope | 11 rules (8 citing EPA or IICRC guidance, 3 heuristics); Xactimate-style line items |

## 3. Tiers and devices

| Tier | Input | Phones | Capture tool | Benchmarked on |
|------|------------|------------|------------|------------|
| Photo | a folder per room, 2 to 8 photos | iPhone 15 or newer, including 16e, 17e, Air | Camera app, 1x | iPhone 17, 1 room, 2 captures |
| Video | one walkthrough ending where it started | iPhone 15 or newer | Camera app, 1x, HDR off, Lock Camera on | iPhone 17, 1 room, 4 captures |
| LiDAR | Stray Scanner folder or zip | 15 Pro to 18 Pro Max, iOS 18.6+ | Stray Scanner 1.4 (free) | 12 synthetic captures; 4 real, no ground truth |

`docs/device_matrix.md` has the per-model matrix and accuracy. LiDAR has metric depth and ARKit poses, so its errors are drift and layout; photo and video take scale, intrinsics and poses from MapAnything, whose metric scale nothing downstream corrects. Photo rooms are reconstructed one at a time and stitched, hence a photo through every doorway; video runs in 24-frame chunks (about 32 views at 518x336 fit in 16 GB), so its scale drift enters at the chunk links.

## 4. Drift handling

LiDAR: point-to-plane ICP registers 3 s trajectory segments more than 20 s apart whose clouds overlap, and the first segment against the last when both have depth. A loop edge needs an RMS residual under 2 cm, an overlap over 30%, a correction under 15 degrees and 1.5 m and at least 4 constrained degrees of freedom, and constrains only the directions its geometry fixes. Loop and odometry edges form a 4-DoF pose graph; a second solve adds per-segment floor-height priors (plane anchoring) and snaps segment yaw to the global wall direction within 5 degrees (Manhattan anchoring).

Video: chunks are linked by Sim(3) on shared frames, so scale drift is modelled, and a link is refused when the two runs place the shared cameras more than 10% of the scene depth apart (consistent real links 0.7 to 7%, broken ones 13% or more). A loop registration on the first and last 6 frames can replace one refused link, and unconnected chunks are dropped and flagged; the old camera-pose fallback had chained a 23% scale jump into the real room's second clip. On bedroom/video_2 the accepted loop cut the end-to-start error from 0.38 m and 6.0 degrees to 0.04 m and 0.64 degrees.

The on and off ablation covers the 12 synthetic captures, the only scored multi-room ones (`lidar_1` and `lidar_2` drift 1 to 3 degrees per 3 minutes and 1 to 2 cm per minute, `lidar_drift` 4 degrees and 10 cm). Walls count within max(2 cm, 1%), and a missing wall fails.

| Captures (4 each) | Mean footprint error, on / off | Worst footprint error, on / off | Walls in tolerance, on / off |
|---|---|---|---|
| `lidar_1` | 1.25% / 1.18% | 3.3% / 3.6% | 69 / 73 of 84 |
| `lidar_2` | 0.36% / 0.54% | 0.7% / 1.4% | 80 / 75 of 84 |
| `lidar_drift` | 1.15% / 0.87% | 2.8% / 2.4% | 60 / 60 of 84 |
| all 12 | 0.92% / 0.87% | 3.3% / 3.6% | 209 / 208 of 252 |

The correction is neutral overall (mean wall error lower on 4 captures and higher on 8): it fixes synth_1/lidar_2 (8.5 to 0.9 cm), finds 4 more walls on synth_0/lidar_drift and breaks synth_3/lidar_drift (2.1 to 19.9 cm, one room lost). The ablation gate passes because it checks only that correction ran and an off run exists. Single stages switched off on 5 captures (mean wall error in cm, walls in tolerance over walls found):

| Capture | All stages | No Manhattan yaw | No plane anchoring | Loop closure only | Off |
|--------------|--------|--------|--------|--------|--------|
| synth_3/lidar_drift | 19.9 (17/20) | 2.2 (22/26) | 19.9 (17/20) | 2.2 (22/26) | 2.1 (23/26) |
| synth_2/lidar_1 | 1.5 (14/16) | 1.5 (14/16) | 1.5 (14/16) | 1.4 (14/16) | 0.4 (16/16) |
| synth_1/lidar_2 | 0.9 (22/22) | 6.5 (21/22) | 0.9 (22/22) | 5.8 (20/22) | 8.5 (17/22) |
| synth_0/lidar_drift | 47.7 (13/16) | 36.5 (13/16) | 36.1 (13/16) | 36.5 (13/16) | 39.1 (7/12) |
| synth_3/lidar_1 | 1.4 (22/26) | 1.0 (24/26) | 1.3 (22/26) | 0.9 (24/26) | 1.2 (24/26) |

Manhattan yaw anchoring moves the result most, both ways; plane anchoring matters only on synth_0/lidar_drift.

On the real office recording (232 s, no tape readings) the correction accepted 71 of 122 loop closures, cut the floor-height spread across segments from 47.9 to 5.5 cm and moved poses by up to 0.54 m (at most 0.19 m on the synthetic captures). The plan is one room with 16 walls and 189.6 m2 [177.4, 258.0], against 10 walls and 184.8 m2 with `--no-drift`.

## 5. Error budget and calibration

On a 4.1 m wall the gates allow 33 cm (photo, 8%), 12 cm (video, 3%) and 4.1 cm (LiDAR); repeatability allows 2.1 cm.

| Source | Photo (real room) | Video (real room) | LiDAR (synthetic) |
|--------|------------|------------|------------|
| Metric scale | walls 0.3 to 7.5% short; prior 0.15 (log) | video_1 walls 4.5 to 8.7% short, chunk scales within 4.3% (7.4% WhatsApp); prior 0.08 | metric depth; prior 0.003 |
| Heights beyond scale | ceilings about 14 points below walls | video_1 ceilings 7 to 8 points below walls | within 0.5 cm |
| Ceiling selection | 1 and 2 candidate levels rejected | 3 to 6 rejected; video_2 never saw the ceiling (87 and 103 cm low) | 1 to 3 rejected on 9 of 12 captures, none wrongly kept |
| Wall-face fit | single views fit LiDAR to 1.5 to 14 cm (ARKitScenes) | as photo | 0.03 cm median repeat delta, exact poses |
| Opening edges | not measured | not measured | 52 of 83 matched doors within 2 cm; 0 of 84 windows found |
| Pose drift, registration | isolated views 0.36 to 5.8 m off (ARKitScenes) | 23% scale jump, old pose fallback | camera centres 0.9 to 5.9 cm RMS, yaw 0.11 to 0.73 degrees off |
| Ground truth | whole inches (up to 1.3 cm rounding), one ceiling reading | same | exact |

MapAnything's metric scale measured 7 to 10% off per room on real captures (`docs/design.md`), 29 to 41 cm on that wall, and heights lose 7 to 14 points more, so photo passes walls but not ceilings and video fails both. For LiDAR the pose residual, about 1 cm per wall between captures, fills the 1 cm repeatability floor on its own; the face fit is about 30 times smaller, and openings are the weakest output.

### Interval model

sigma^2 = (v s)^2 + a^2 plus structure terms, v being the value. The capture's log-scale s (LiDAR 0.003, video 0.08, photo 0.15) widens with video chunk scale spread, link disagreement and an implausible MapAnything focal; the additive a (lengths 0.8, 2.5 and 4 cm) grows 1.2 to 2 times per kind of thin evidence; photo and video heights add a 0.1 log term. One-sided terms widen only the open side: a wall ending at a wall with no face may run on by 0.15 of the room extent, a possible fragment reaches the far end of the pieces in line with it, and an unobserved ceiling may reach 4 m. Intervals are value -/+ 1.645 q sigma per side, with the per-tier multiplier q at 1.

### Calibration

| Tier | Data | Values | Physical rooms | Covered | 95% CI (Clopper-Pearson) | Mean half-width | Misses beyond 2 half-widths |
|---|---|---|---|---|---|---|---|
| Photo | real | 14 | 1 | 14 (100%) | 76.8 to 100% | 33.0% of truth | 0 |
| Video | real | 28 | 1 | 25 (89.3%) | 71.8 to 97.7% | 27.2% | 0 |
| LiDAR | synthetic | 560 | 20 | 479 (85.5%) | 82.3 to 88.3% | 3.1% | 38 |

Photo and video pass the gate and show little: the 14 photo values are one rectangle in two copies, with equal opposite walls and a footprint equal to the floor area, so they hold 8 distinct numbers, and they cover because they are wide (a third of the true value). The 3 video misses are walls of the two video_2 copies. Before the real-room fixes (a94ee18, unpublished copy) 6 of the 14 WhatsApp video values missed by more than 2 half-widths.

LiDAR fails: opening widths cover 64.2% (half-width 2.1 cm, mean error 3.9 cm), wall lengths 82.2%. The worst miss, synth_0/lidar_drift kitchen W4 (3.964 m, reported [7.810, 7.892] m), ran on through a hallway the run did not find, which a per-measurement model cannot see. q stays at 1.0: split conformal with the room as the unit needs the ceil((n+1) x 0.9)/n quantile, which exists only for n >= 9 rooms, and q is not fitted on synthetic rooms. A fit on the 20 synthetic rooms (section 8, not applied) gives q = 2.52 and leave-one-room-out coverage of 94.0% (515 of 548; CI 91.6 to 95.8%), conservative on the simulator.

## 6. Results

| Gate (photo, video, LiDAR threshold) | Photo, real | Video, real | LiDAR, synthetic |
|----------------|--------|--------|--------|
| Wall length (8%, 3%, max(2 cm, 1%)*) | pass, 8 of 8 | fail, 1 of 16 | fail, 209 of 252 |
| Ceiling height (8%*, 3%*, 1.5 cm) | fail, 0 of 2 | fail, 0 of 4 | fail, 58 of 60 |
| Ceiling spread across captures (1 cm) | no repeat | fail, 58.5 and 39.3 cm | pass, 20 of 20 |
| Opening width on 85% of openings (8%*, 3%*, 2 cm) | not measured | not measured | fail, 61 of 202 |
| Floor area (16%*, 6%*, 2%*) | pass, 2 of 2 | fail, 0 of 4 | fail, 52 of 60 |
| Repeat walls within max(1 cm, 0.5%) | no repeat | fail, 1 of 8 | fail, 155 of 232 |
| Same plan from repeat captures | no repeat | fail, 0 of 2 | fail, 36 of 56 |

`*` marks thresholds the brief does not state (`assumed: true` in `bench/gates.yaml`). Ceiling failure mode: photo is biased (mean -62.2 cm), video biased and unrepeatable (mean -70.3 cm), and LiDAR neither (worst 0.5 cm; the gate fails on 2 rooms not found).

The three LiDAR recordings provided with the problem statement have no ground truth (`docs/testdata_validation.md`): `c7d28f72c6` gave 7 rooms with 3.07 to 3.09 m ceilings; the two filmed looking down each merged into one room, never seeing the ceiling.

## 7. Fix loop

Declared before any fix code or real capture (tag `fixloop-declared`): the worst gate was LiDAR repeatability, 82 of 236 wall pairs (34.7%) within max(1 cm, 0.5%). The hypothesis was that layout made repeats match walls that do not correspond: property-wide wall lines merged parallel faces of neighbouring rooms under about 10 cm apart, polygons stepped along interior-wall and furniture faces (a 4-wall hallway got 8 walls), and rectangle pairs differed by a median 2.4 cm where depth noise predicts millimetres. The fix (tag `fixloop-after`, 4 commits in `layout/` and its tests) refits each room's walls from the points within 15 cm facing into it, removes steps under 0.25 m and fills furniture notches.

| Measure | Before | After | Predicted |
|----------------|--------|--------|--------|
| Repeat wall pairs in tolerance | 82/236 (34.7%) | 155/232 (66.8%) | 85% (75 to 95%) |
| Median repeat delta, rectangle pairs | 2.43 cm | 1.04 cm | under 0.6 cm |
| Room instances with the wrong wall count | 34/116 | 15/116 | under 10 |
| Wall-length pass share | 58.3% | 82.9% | at least 85% |

The prediction missed (`docs/fixloop/postmortem.md`). With exact synthetic poses the same captures reach 96.3% (86.9% before), so the layout mechanisms were real; the declaration underweighted pose error left after drift correction (camera centres 0.9 to 5.9 cm RMS, yaw 0.11 to 0.73 degrees), about 1 cm per wall and the whole allowance, and the exact-pose run should have come first. 15 room instances keep the wrong wall count, mostly where a doorway opens onto free space. Later fixes left every LiDAR value unchanged and raised interval coverage from 73.2% to 85.5%.

## 8. Reproduction

The README ("Reproduce the reported numbers") has every command: the release-asset fetch (6 zips, 331 MB, SHA-256 checked), both benches, the seeded synthetic set, the fix loop from clean worktrees, the office recording with and without `--no-drift`, and the synthetic conformal fit of section 5. Each `bench` run repeats its multi-room captures with drift correction off. The per-stage drift rows come from `scripts/drift_stage_ablation.py`, since no CLI flag switches single stages. The ARKitScenes analysis is not scripted.

## 9. Known failure modes

ARKitScenes numbers come from development runs before the real-room fixes.

- Photo and video come out small: 23 of 24 real wall values and every ceiling and floor area were short. The original photos passed their EXIF focal and still lost 19.8% of the ceiling; the cause inside MapAnything was not isolated, and the owner re-confirmed the 130 in reading.
- Media without a focal length get a short one: MapAnything predicted 0.54 to 0.59 of the long side on both clips and the WhatsApp photos, where the 1x lens has at least 0.69; those captures are flagged and widened. Recompression made video_1 smaller (walls 8.7 and 6.8% short against 6.3 and 4.5%) but not the photos, and quality-70 re-saves cut depth by 35% on ARKitScenes.
- Mirrors have one real data point: at 226bcff a detected mirror removed an opening from the WhatsApp video_1; at 7d26d19 mirror boxes reaching the floor are dropped and it keeps 2 doors. No opening was taped.
- Windows with nothing in depth range behind them are missed, since an opening needs rays ending 25 cm past the wall face: the LiDAR tier found none of the 84 synthetic windows. Only the video_2 copies reported the real night window.
- No wet floor or dim room was captured, and only the photo tier detects low light (EXIF ISO and exposure). Joints of the glossy tile floor score up to 0.56 as cracks, so a floor crack needs 3 views and a combined 0.75 (set on this room); on ARKitScenes a lamp highlight on glossy tile became a water stain that fired R-FIXTURE-LEAK.
- Close-up frames get the wrong depth: ARKitScenes frames of a poster 0.35 to 0.58 m away came out at 1.6 to 2.5 times the LiDAR depth and split a bedroom into 2 rooms; no filter is shipped.
- Open plans merge, since rooms split only at door-sized gaps: the office is one 189.6 m2 room with 16 walls, after 75.9 m2 of holes in its polygon were filled.
- Walls up to 8 degrees off square are fitted as slanted lines (unit tests at 3 and 6 degrees keep 4 walls within 3 cm); larger angles are not modelled or benchmarked, and Manhattan yaw anchoring assumes one wall grid.
- A wall no view sees is closed where the rays end, or on furniture in front of it: bedroom/video_2, aimed at furniture, gave 8 walls for 4 and W4 at 1.51 m of 3.607 m, inside a widened [0.62, 4.01] m; 3 walls of the two video_2 copies still missed.
- The staged sheet's drawn line was not reported as damage. bedroom/photo_1 and both video_1 copies report a 1.0 to 1.4 m crack along a plaster line above the tube light (R-CRACK-LONG fires), the video_1 copies a 16 cm crack just above the sheet and bedroom/video_2 a 22 cm one at floor level; none is scored.

## 10. Deviations from the brief

- One real room and no LiDAR phone: photo stitching (unit tests in `tests/test_stitch.py`), adjacency and the drift ablation are tested on synthetic data only, the LiDAR tier's four real recordings have no tape readings, and the photo tier has no repeat capture.
- One staged damage class (a drawn crack) where the brief asks for two; the real room's openings and damage extents were not measured, so neither is scored.
- No head-to-head: the magicplan free-plan IFC export had no room geometry, and no other app data was captured.
- No Round 1 schema or gates were provided, so both are ours; 8 thresholds in `bench/gates.yaml` are marked `assumed: true`.
- The ground truth is whole-inch tape readings with one ceiling reading (our protocol asks for millimetres, two readings each and three ceiling points), and the capture departs from our protocol: portrait clips, video_2 at 4K and aimed at furniture, 5 photos (6 to 8 recommended), nothing wet or dim.
