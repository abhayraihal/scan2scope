# Validation on the provided test recordings

Three Stray Scanner recordings came with the problem statement as test data: real LiDAR captures from a Pro iPhone of one apartment (living area, hallway, bathroom, office, rooms off a corridor). No measurements came with them, so they are not scored against ground truth. They show how the LiDAR tier behaves on real Pro-iPhone data, which the owner's own phone (an iPhone 17, no LiDAR) could not produce. The recordings are not redistributed here; the outputs are in `docs/testdata/<recording>/` (`result.json`, `plan.png`, `plan.svg`).

Each was run once, cold, with one command and the default settings (drift correction and damage detection on), at commit 5b09a11 on the M4 MacBook:

```sh
uv run scan2scope run <path to the recording folder>
```

## Results

| Recording | Length | Rooms | Footprint m2 [90% interval] | Ceiling m | Walls (well observed) | Openings | Drift correction | Time |
|---|---|---|---|---|---|---|---|---|
| c7d28f72c6 | 215 s, 9,745 frames | 7, linked by 7 doorways | 61.8 [58.6, 90.4] | 3.07 to 3.09 in every room, about +/-0.02 | 44 (39) | 16 | 38 of 122 loop closures accepted; floor-height spread 12.9 cm to 1.8 cm | 230 s |
| 1a8384c3f6 | 115 s, 5,251 frames | 1 (merged) | 70.3 [53.6, 152.6] | not observed (2.20, interval up to 4.00) | 48 (28) | 4 | 10 of 32 accepted; one 0.65 m pose jump flagged and corrected | 226 s |
| c00a170fe1 | 37 s, 1,715 frames | 1 (merged) | 24.4 [18.2, 53.6] | not observed (2.50, interval up to 4.00) | 16 (9) | 1 | 1 of 3 accepted | 203 s |

Geometry and layout take 8 to 25 s per recording; the rest is damage detection (about 200 s for 37 to 60 views).

## What the runs show

- `c7d28f72c6` gives a multi-room plan: seven rooms with door swings and adjacency, ceilings that agree across rooms within 2.3 cm, rectangular rooms such as 3.01 x 3.13 m (9.4 m2) and 3.47 x 3.08 m (10.7 m2), and door widths of 0.69 to 1.08 m. Two small spaces of about 2 m2 are corridor pieces split at doorways. The largest space comes out as one L-shaped room with short steps where its far walls were seen only partly, and those walls carry wide intervals. Two crack detections fired the crack-at-opening and long-crack rules; they were not checked against the space.
- In `1a8384c3f6` and `c00a170fe1` the phone points down at the floor for almost the whole walk: the sampled frames show floors, lower walls and furniture, and never the ceiling or the tops of the doors. Without the door heads and the upper walls the layout cannot separate the rooms, so each recording becomes one merged outline, and with no ceiling in view the ceiling height is flagged `ceiling_not_observed` and its interval reaches 4.0 m. The intervals and flags report this; no number in these two runs is stated more tightly than its evidence allows.
- The difference between the good run and the two poor ones is the capture, not the space: the capture protocol asks for the floor line and the ceiling line to stay in view (`docs/capture_protocol.md`), and only `c7d28f72c6` keeps the ceiling in view.

## How to rerun

Put the three folders anywhere and run the command above on each; the outputs land in `out/<recording>/`. Results match these files when the same commit and model revisions are used (`--cache replay` reuses stored model outputs on the machine that made them).
