# Validation on the provided test recordings

Three Stray Scanner recordings came with the problem statement as test data: real LiDAR captures of one apartment (living area, hallway, bathroom, office, rooms off a corridor), recorded within 7 minutes of each other on 2026-09-01 (`rgb.mp4` creation times). The files do not name the device; recording LiDAR depth needs a Pro iPhone or an iPad Pro. They are in the Stray Scanner 1.4 format at 60 fps by median frame spacing, 45 to 46 fps on average after dropped frames (the protocol sets 30 fps). No measurements came with them, so they are not scored against ground truth. They show how the LiDAR tier behaves on real LiDAR data, which the owner's own phone (an iPhone 17, no LiDAR) could not produce. The recordings are not redistributed here; the outputs are in `docs/testdata/<recording>/` (`result.json`, `plan.png`, `plan.svg`), with the local recording path in each `result.json` (`capture.path`) shortened to `testdata/<recording>`.

Each was run once, cold, with one command and the default settings (drift correction and damage detection on), at commit 5b09a11 (tag `v1.0`) on the M4 MacBook:

```sh
uv run scan2scope run <path to the recording folder>
```

## Results

| Recording | Length | Rooms | Footprint m2 [90% interval] | Ceiling m | Walls (face at least 30% seen) | Openings | Drift correction | Time |
|---|---|---|---|---|---|---|---|---|
| c7d28f72c6 | 215 s, 9,745 frames | 7, linked by 7 doorways | 61.8 [58.6, 90.4] | 3.07 to 3.09 in every room, ±0.02 to ±0.09 | 44 (40) | 16 | 38 of 122 loop closures accepted; floor-height spread across segments cut from 12.9 to 1.8 cm | 230 s |
| 1a8384c3f6 | 115 s, 5,251 frames | 1 (merged) | 70.3 [53.6, 152.6] | not observed (2.20, interval up to 4.00) | 48 (30) | 4 | 10 of 32 accepted; spread cut from 5.1 to 1.4 cm; one 0.6 m pose jump flagged, largest correction 0.65 m | 226 s |
| c00a170fe1 | 37 s, 1,715 frames | 1 (merged) | 24.4 [18.2, 53.6] | not observed (2.50 assumed, interval up to 4.00) | 16 (10) | 1 | 1 of 3 accepted; spread cut from 4.9 to 1.2 cm | 203 s |

Geometry and layout take 8 to 25 s per recording; damage detection takes the rest, 194 to 209 s on 37 to 40 views. The size of the pose jump comes from `odometry.csv`, because the drift record notes only that a jump was found.

## What the runs show

- `c7d28f72c6` gives a multi-room plan: seven rooms with door swings, all linked through doorways, ceilings within 2.4 cm of each other, rectangular rooms such as 3.01 x 3.13 m (9.4 m2) and 3.47 x 3.08 m (10.7 m2), and doors 0.69 to 1.08 m wide apart from one 1.45 m door in R1. Two small spaces of 1.7 and 2.0 m2 (R4 and R2) sit between doorways and are probably pieces of the corridor. The largest space, R3, comes out as one L-shaped room of 16 walls with short steps where its far walls were seen only partly, and those walls carry wide intervals.
- In `1a8384c3f6` and `c00a170fe1` the camera never points above horizontal. In `odometry.csv` its highest pitch is 8 and 6 degrees below horizontal and its median 27 and 31 degrees below; in `c7d28f72c6`, 42% of the frames look above horizontal. The ceiling and most door heads stay out of view, and each recording comes out as one merged outline (48 and 16 walls). The likely reason is the missing upper walls: the layout splits rooms only where it has wall evidence, and a gap with wall seen above it (a door head) counts as wall, so with only the lower walls in view, often behind furniture, the gaps between rooms stay open. This was not tested further, for example by rerunning `c7d28f72c6` without its upward views. With no ceiling in view the ceiling height is flagged `ceiling_not_observed` and its interval reaches 4.0 m. The flags and the wide intervals mark these values as weak; without measurements, whether each interval contains the true value cannot be checked.
- Damage detection reported cracks in all three, none checked against the space. In `c7d28f72c6` two cracks on R3 walls fired the crack-at-opening and long-crack rules and gave 4 scope line items. `1a8384c3f6` and `c00a170fe1` have one crack each (scores 0.43 and 0.41) that fired no rule and gave 2 scope line items each.
- All three are of the same apartment, so the clearest difference between the good run and the two poor ones is where the camera points: the capture protocol asks for the floor line and the ceiling line to stay in view (`docs/capture_protocol.md`), and only `c7d28f72c6` keeps the ceiling in view.

## How to rerun

Put the three folders in `testdata/` at the repository root (git ignores it) and run the command above on each; the outputs land in `out/<recording>/`. Each recording was run once, so run-to-run agreement on these files was not checked. `--no-semantics` skips damage detection; a run then takes about as long as geometry and layout, 8 to 25 s here.
