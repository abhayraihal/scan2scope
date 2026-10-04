# Benchmark report

Generated 2026-10-03 19:20 UTC from `bench/data` with gates from `bench/gates.yaml`. Pipeline commit(s) in the results: 7d26d19657. Cache mode: mixed.

Properties: bedroom, bedroom_whatsapp; 6 captures scored.

## Gate summary

Status per tier against `bench/gates.yaml`. `assumed` marks thresholds the brief does not state. A missing value (room, wall or opening not found, capture failed) counts as a failing item.

### photo

| Gate | Status | Measured | Threshold | n | Pass share | Assumed |
|---|---|---|---|---|---|---|
| result_produced | PASS | 2/2 pass | every capture returns a scored result with geometry | 2 | 100.0% |  |
| wall_length | PASS | worst 7.5% (bedroom_whatsapp/photo_1/01 bedroom/W1), 8/8 pass | \|err\| <= 8.0% of GT on every wall | 8 | 100.0% |  |
| ceiling_height | FAIL | worst 19.8% (bedroom/photo_1/01 bedroom), 0/2 pass | \|err\| <= 8.0% of GT in every room | 2 | 0.0% | assumed |
| ceiling_spread | n.a. | no data | max - min across captures <= 1.0 cm per room | 0 |  |  |
| opening_width | n.a. | no data | \|err\| <= 8.0% of GT on >= 85.0% of openings (misses and phantoms fail) | 0 |  | assumed |
| floor_area | PASS | worst 11.5% (bedroom/photo_1/01 bedroom), 2/2 pass | \|err\| <= 16.0% of GT in every room | 2 | 100.0% | assumed |
| footprint | n.a. | no data | \|err\| <= 8.0% of GT per multi-room capture | 0 |  |  |
| stitch | n.a. | no data | adjacency exact, max pairwise overlap <= 0.05 m2, footprint within 8.0% | 0 |  |  |
| repeatability | n.a. | no data | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures | 0 |  |  |
| repeat_structure | n.a. | no data | repeat captures find the same walls and openings in each room | 0 |  |  |
| calibration | PASS | 14/14 covered (100.0%), CI [76.8%, 100.0%], 2 rooms (fewer than 9: too few to show calibration) | 95.0% Clopper-Pearson CI of coverage contains 0.90 | 14 | 100.0% |  |
| confident_garbage | PASS | 0 of 14 values | no miss larger than 2 x half-width | 14 | 100.0% |  |

Worst items:

- ceiling_height: bedroom/photo_1/01 bedroom err -65.4 cm (allowed 26.4 cm); bedroom_whatsapp/photo_1/01 bedroom err -58.9 cm (allowed 26.4 cm)

### video

| Gate | Status | Measured | Threshold | n | Pass share | Assumed |
|---|---|---|---|---|---|---|
| result_produced | PASS | 4/4 pass | every capture returns a scored result with geometry | 4 | 100.0% |  |
| wall_length | FAIL | worst 58.1% (bedroom/video_2/01 bedroom/W4), 1/16 pass | \|err\| <= 3.0% of GT on every wall | 16 | 6.2% |  |
| ceiling_height | FAIL | worst 31.0% (bedroom/video_2/01 bedroom), 0/4 pass | \|err\| <= 3.0% of GT in every room | 4 | 0.0% | assumed |
| ceiling_spread | FAIL | 0/2 pass; bedroom/01 bedroom: spread 58.5 cm over 2 captures; bedroom_whatsapp/01 bedroom: spread 39.3 cm over 2 captures | max - min across captures <= 1.0 cm per room | 2 | 0.0% |  |
| opening_width | n.a. | no data | \|err\| <= 3.0% of GT on >= 85.0% of openings (misses and phantoms fail) | 0 |  | assumed |
| floor_area | FAIL | worst 25.5% (bedroom_whatsapp/video_2/01 bedroom), 0/4 pass | \|err\| <= 6.0% of GT in every room | 4 | 0.0% | assumed |
| repeatability | FAIL | 1/8 walls pass, strict reading 0/8; worst bedroom/01 bedroom/W4 video_1 vs video_2: \|delta\| 193.5 cm (allowed 1.8 cm) | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures | 8 | 12.5% |  |
| repeat_structure | FAIL | 0/2 pass; bedroom/01 bedroom video_1 vs video_2: walls 4/8, openings 2/2; bedroom_whatsapp/01 bedroom video_1 vs video_2: walls 4/6, openings 2/2 | repeat captures find the same walls and openings in each room | 2 | 0.0% |  |
| drift_ablation | n.a. | no data | drift correction on for every multi-room capture, with an off run for the ablation | 0 |  |  |
| calibration | PASS | 25/28 covered (89.3%), CI [71.8%, 97.7%], 2 rooms (fewer than 9: too few to show calibration) | 95.0% Clopper-Pearson CI of coverage contains 0.90 | 28 | 89.3% |  |
| confident_garbage | PASS | 0 of 28 values | no miss larger than 2 x half-width | 28 | 100.0% |  |

Worst items:

- wall_length: bedroom/video_2/01 bedroom/W4 err -209.7 cm (allowed 10.8 cm); bedroom_whatsapp/video_2/01 bedroom/W3 err -175.4 cm (allowed 12.3 cm); bedroom_whatsapp/video_2/01 bedroom/W1 err -88.8 cm (allowed 12.3 cm); bedroom/video_2/01 bedroom/W1 err +78.5 cm (allowed 12.3 cm); bedroom/video_2/01 bedroom/W3 err -74.5 cm (allowed 12.3 cm)
- ceiling_height: bedroom/video_2/01 bedroom err -102.5 cm (allowed 9.9 cm); bedroom_whatsapp/video_2/01 bedroom err -87.1 cm (allowed 9.9 cm); bedroom_whatsapp/video_1/01 bedroom err -47.7 cm (allowed 9.9 cm); bedroom/video_1/01 bedroom err -44.0 cm (allowed 9.9 cm)
- ceiling_spread: bedroom/01 bedroom: spread 58.5 cm over 2 captures; bedroom_whatsapp/01 bedroom: spread 39.3 cm over 2 captures
- floor_area: bedroom_whatsapp/video_2/01 bedroom err -378.0 cm (allowed 89.1 cm); bedroom_whatsapp/video_1/01 bedroom err -220.5 cm (allowed 89.1 cm); bedroom/video_1/01 bedroom err -156.5 cm (allowed 89.1 cm); bedroom/video_2/01 bedroom err -108.6 cm (allowed 89.1 cm)
- repeatability: bedroom/01 bedroom/W1 video_1 vs video_2: |delta| 104.5 cm (allowed 2.1 cm); bedroom/01 bedroom/W3 video_1 vs video_2: |delta| 48.4 cm (allowed 2.1 cm); bedroom/01 bedroom/W4 video_1 vs video_2: |delta| 193.5 cm (allowed 1.8 cm); bedroom_whatsapp/01 bedroom/W1 video_1 vs video_2: |delta| 53.1 cm (allowed 2.1 cm); bedroom_whatsapp/01 bedroom/W2 video_1 vs video_2: |delta| 14.1 cm (allowed 1.8 cm)
- repeat_structure: bedroom/01 bedroom video_1 vs video_2: walls 4/8, openings 2/2; bedroom_whatsapp/01 bedroom video_1 vs video_2: walls 4/6, openings 2/2

### lidar

| Gate | Status | Measured | Threshold | n | Pass share | Assumed |
|---|---|---|---|---|---|---|
| result_produced | n.a. | no captures in this tier | every capture returns a scored result with geometry | 0 |  |  |
| wall_length | n.a. | no captures in this tier | \|err\| <= max(2.0 cm, 1.0% of GT) on every wall | 0 |  | assumed |
| ceiling_height | n.a. | no captures in this tier | \|err\| <= 1.5 cm in every room | 0 |  |  |
| ceiling_spread | n.a. | no captures in this tier | max - min across captures <= 1.0 cm per room | 0 |  |  |
| opening_width | n.a. | no captures in this tier | \|err\| <= 2.0 cm on >= 85.0% of openings (misses and phantoms fail) | 0 |  |  |
| floor_area | n.a. | no captures in this tier | \|err\| <= 2.0% of GT in every room | 0 |  | assumed |
| repeatability | n.a. | no captures in this tier | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures | 0 |  |  |
| repeat_structure | n.a. | no captures in this tier | repeat captures find the same walls and openings in each room | 0 |  |  |
| drift_ablation | n.a. | no captures in this tier | drift correction on for every multi-room capture, with an off run for the ablation | 0 |  |  |
| calibration | n.a. | no captures in this tier | 95.0% Clopper-Pearson CI of coverage contains 0.90 | 0 |  |  |
| confident_garbage | n.a. | no captures in this tier | no miss larger than 2 x half-width | 0 |  |  |

## Failing gates ranked for the fix loop

Score: worst error over its allowance minus 1 for error gates, relative gap to the target for rate gates, miss rate over the nominal miss rate minus 1 for calibration, the count for confident garbage, failing share for checks.

| Rank | Tier | Gate | Score | Measured | Threshold |
|---|---|---|---|---|---|
| 1 | video | repeatability | 106.26 | 1/8 walls pass, strict reading 0/8; worst bedroom/01 bedroom/W4 video_1 vs video_2: \|delta\| 193.5 cm (allowed 1.8 cm) | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures |
| 2 | video | ceiling_spread | 57.45 | 0/2 pass; bedroom/01 bedroom: spread 58.5 cm over 2 captures; bedroom_whatsapp/01 bedroom: spread 39.3 cm over 2 captures | max - min across captures <= 1.0 cm per room |
| 3 | video | wall_length | 18.38 | worst 58.1% (bedroom/video_2/01 bedroom/W4), 1/16 pass | \|err\| <= 3.0% of GT on every wall |
| 4 | video | ceiling_height | 9.34 | worst 31.0% (bedroom/video_2/01 bedroom), 0/4 pass | \|err\| <= 3.0% of GT in every room |
| 5 | video | floor_area | 3.24 | worst 25.5% (bedroom_whatsapp/video_2/01 bedroom), 0/4 pass | \|err\| <= 6.0% of GT in every room |
| 6 | photo | ceiling_height | 1.48 | worst 19.8% (bedroom/photo_1/01 bedroom), 0/2 pass | \|err\| <= 8.0% of GT in every room |
| 7 | video | repeat_structure | 1.00 | 0/2 pass; bedroom/01 bedroom video_1 vs video_2: walls 4/8, openings 2/2; bedroom_whatsapp/01 bedroom video_1 vs video_2: walls 4/6, openings 2/2 | repeat captures find the same walls and openings in each room |

## Ceiling height: bias or spread

Accuracy against the tape or laser reading and spread across repeat captures are scored separately; the mode names which of the two fails.

| Tier | Accuracy | Accuracy detail | Mean signed error | Spread | Spread detail | Mode |
|---|---|---|---|---|---|---|
| photo | FAIL | worst 19.8% (bedroom/photo_1/01 bedroom), 0/2 pass | -62.2 cm | n.a. | no data | bias |
| video | FAIL | worst 31.0% (bedroom/video_2/01 bedroom), 0/4 pass | -70.3 cm | FAIL | 0/2 pass; bedroom/01 bedroom: spread 58.5 cm over 2 captures; bedroom_whatsapp/01 bedroom: spread 39.3 cm over 2 captures | both |
| lidar | n.a. | no captures in this tier |  | n.a. | no captures in this tier | n.a. |

## Repeatability

Rooms and walls are matched through the ground truth. Allowed |delta| is max(1 cm, 0.5% of the GT length); the strict reading is min(1 cm, 0.5%).

| Property | Tier | Room | Wall | Captures | Values (m) | \|delta\| | Allowed | Pass | Strict allowed | Strict pass |
|---|---|---|---|---|---|---|---|---|---|---|
| bedroom | video | 01 bedroom | W1 | video_1 vs video_2 | 3.855 / 4.900 | 104.5 cm | 2.1 cm | no | 1.0 cm | no |
| bedroom | video | 01 bedroom | W2 | video_1 vs video_2 | 3.445 / 3.434 | 1.0 cm | 1.8 cm | yes | 1.0 cm | no |
| bedroom | video | 01 bedroom | W3 | video_1 vs video_2 | 3.855 / 3.370 | 48.4 cm | 2.1 cm | no | 1.0 cm | no |
| bedroom | video | 01 bedroom | W4 | video_1 vs video_2 | 3.445 / 1.510 | 193.5 cm | 1.8 cm | no | 1.0 cm | no |
| bedroom_whatsapp | video | 01 bedroom | W1 | video_1 vs video_2 | 3.758 / 3.227 | 53.1 cm | 2.1 cm | no | 1.0 cm | no |
| bedroom_whatsapp | video | 01 bedroom | W2 | video_1 vs video_2 | 3.363 / 3.222 | 14.1 cm | 1.8 cm | no | 1.0 cm | no |
| bedroom_whatsapp | video | 01 bedroom | W3 | video_1 vs video_2 | 3.758 / 2.361 | 139.7 cm | 2.1 cm | no | 1.0 cm | no |
| bedroom_whatsapp | video | 01 bedroom | W4 | video_1 vs video_2 | 3.363 / 3.504 | 14.1 cm | 1.8 cm | no | 1.0 cm | no |

Ceiling height across captures:

| Property | Tier | Room | Captures | Values (m) | Spread |
|---|---|---|---|---|---|
| bedroom | video | 01 bedroom | video_1, video_2 | 2.862 / 2.277 | 58.5 cm |
| bedroom_whatsapp | video | 01 bedroom | video_1, video_2 | 2.825 / 2.431 | 39.3 cm |

Same plan structure (walls and openings found in both captures):

| Property | Tier | Room | Captures | Walls | Openings | Same walls | Same openings |
|---|---|---|---|---|---|---|---|
| bedroom | video | 01 bedroom | video_1 vs video_2 | 4 / 8 | 2 / 2 | no | yes |
| bedroom_whatsapp | video | 01 bedroom | video_1 vs video_2 | 4 / 6 | 2 / 2 | no | yes |

## Calibration

Coverage of the 90% intervals over every scored value, with an exact (Clopper-Pearson) 95% interval. Rooms counts distinct physical rooms; repeat captures of a room share one unit. With fewer than 9 rooms a tier keeps its prior interval multiplier and its coverage interval is too wide to show calibration. Confident garbage is a miss by more than the configured multiple of the half-width.

| Tier | Values | Rooms | Covered | Coverage | 95% CI | Contains 0.90 | Mean half-width (% of GT) | Confident garbage | Interval multiplier |
|---|---|---|---|---|---|---|---|---|---|
| photo | 14 | 2 | 14 | 100.0% | [76.8%, 100.0%] | yes | 33.0% | 0 | prior q=1 |
| video | 28 | 2 | 25 | 89.3% | [71.8%, 97.7%] | yes | 27.2% | 0 | prior q=1 |

By kind (half-widths and errors in metres, or square metres for areas):

| Tier | Kind | n | Coverage | Mean half-width | Mean \|err\| |
|---|---|---|---|---|---|
| photo | ceiling_height | 2 | 100.0% | 0.861 | 0.622 |
| photo | floor_area | 2 | 100.0% | 7.442 | 1.434 |
| photo | footprint | 2 | 100.0% | 7.442 | 1.434 |
| photo | wall_length | 8 | 100.0% | 1.008 | 0.195 |
| video | ceiling_height | 4 | 100.0% | 0.971 | 0.703 |
| video | floor_area | 4 | 100.0% | 5.706 | 2.159 |
| video | footprint | 4 | 100.0% | 5.706 | 2.159 |
| video | wall_length | 16 | 81.2% | 0.809 | 0.561 |

## Rooms, openings and damage

Openings match on the same matched wall, compatible type (door and open passage are interchangeable) and centre within half the GT width. Phantoms in rooms that match no GT room are scored only when the capture covers the whole property.

| Capture | Tier | Status | GT rooms | Predicted | Matched | Missed rooms | Extra rooms | Adjacency | Max overlap (m2) |
|---|---|---|---|---|---|---|---|---|---|
| bedroom/photo_1 | photo | ok | 1 | 1 | 1 |  |  | exact (0 pairs) | 0.000 |
| bedroom/video_1 | video | ok | 1 | 1 | 1 |  |  | exact (0 pairs) | 0.000 |
| bedroom/video_2 | video | ok | 1 | 1 | 1 |  |  | exact (0 pairs) | 0.000 |
| bedroom_whatsapp/photo_1 | photo | ok | 1 | 1 | 1 |  |  | exact (0 pairs) | 0.000 |
| bedroom_whatsapp/video_1 | video | ok | 1 | 1 | 1 |  |  | exact (0 pairs) | 0.000 |
| bedroom_whatsapp/video_2 | video | ok | 1 | 1 | 1 |  |  | exact (0 pairs) | 0.000 |

Openings:

| Capture | Tier | GT | Matched | Missed | Phantom | Door/passage swaps | Unscored phantoms |
|---|---|---|---|---|---|---|---|
| bedroom/photo_1 | photo | 0 | 0 | 0 | 0 | 0 | 1 |
| bedroom/video_1 | video | 0 | 0 | 0 | 0 | 0 | 2 |
| bedroom/video_2 | video | 0 | 0 | 0 | 0 | 0 | 2 |
| bedroom_whatsapp/photo_1 | photo | 0 | 0 | 0 | 0 | 0 | 2 |
| bedroom_whatsapp/video_1 | video | 0 | 0 | 0 | 0 | 0 | 2 |
| bedroom_whatsapp/video_2 | video | 0 | 0 | 0 | 0 | 0 | 2 |

## Drift ablation

No multi-room video or LiDAR capture has both runs.

## Head-to-head against magicplan

No magicplan export found (bench/data/<property>/magicplan/statistics.csv and dimensions.yaml).

## Timing

Seconds per pipeline stage from result.json, and the runner's wall time per capture.

| Capture | Tier | Total | ingest | geometry | layout | stitch | semantics | rules | uncertainty | scope | Runner |
|---|---|---|---|---|---|---|---|---|---|---|---|
| bedroom/photo_1 | photo | 29.4 | 0.0 | 1.9 | 1.4 | 0.0 | 26.1 | 0.0 | 0.0 | 0.0 | 29.8 |
| bedroom/video_1 | video | 98.6 | 0.3 | 8.1 | 1.1 |  | 89.0 | 0.0 | 0.0 | 0.0 | 98.7 |
| bedroom/video_2 | video | 89.1 | 0.1 | 11.3 | 1.3 |  | 76.4 | 0.0 | 0.0 | 0.0 | 89.2 |
| bedroom_whatsapp/photo_1 | photo | 22.3 | 0.0 | 0.5 | 0.9 | 0.0 | 20.9 | 0.0 | 0.0 | 0.0 | 22.4 |
| bedroom_whatsapp/video_1 | video | 91.9 | 0.0 | 6.2 | 0.9 |  | 84.7 | 0.0 | 0.0 | 0.0 | 92.0 |
| bedroom_whatsapp/video_2 | video | 87.3 | 0.0 | 5.9 | 1.0 |  | 80.0 | 0.0 | 0.0 | 0.0 | 87.5 |

## Failed or missing captures

None.
