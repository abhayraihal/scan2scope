# scan2scope design

Status: approved 2026-10-03. This is the working design for the case study submission. Decisions that changed during the build are logged at the bottom with the reason.

## Constraints that shape everything

- About 30 hours of build time, one person plus AI coding tools, one machine (Apple M4, 16 GB, macOS 26).
- The only phone available for the benchmark is an iPhone 17 (no LiDAR). Photo and video tiers get real captures with tape or laser ground truth. The LiDAR tier is implemented against the documented Stray Scanner export format and benchmarked on synthetic captures with exact ground truth, labelled as such everywhere it is reported.
- Free tools only: Stray Scanner (free, no in-app purchases), the native Camera app, magicplan free Starter plan for the head-to-head.
- No Round 1 schema or gate list was provided. We publish our own schema (`schema/scan2scope.schema.json`) and our own gate thresholds (`bench/gates.yaml`), and the compliance matrix marks every assumed threshold as an assumption.

## Capture route

Route 2, a stock-capture protocol on one page (`docs/capture_protocol.md`):

| Tier | Tool | Devices | Input handed to the pipeline |
|---|---|---|---|
| Photo | Camera app, Photo mode, 1x | any iPhone 15 or newer | one folder per room, 2 to 8 photos each (6 to 8 recommended), folders numbered in walking order |
| Video | Camera app, Video mode, 1x, 1080p or 4K at 30 fps, HDR off, Lock Camera on | any iPhone 15 or newer | one walkthrough clip of the whole property that ends where it started |
| LiDAR | Stray Scanner 1.4, 30 fps | iPhone 15 Pro to 18 Pro Max on iOS 18.6+ | the Stray Scanner dataset folder or its share-sheet zip |

The 1x lens is used for every tier because the 16e, 17e and Air have no Ultra Wide camera. Doors stay fully open, lights on. Each room's photo set includes one photo through every doorway, because photo-tier stitching needs content shared between folders.

## Architecture

```
capture path ──> ingest (tier detection, decoding, EXIF intrinsics, keyframes)
             ──> geometry backend per tier ──> Scene (gravity-aligned points + normals + cameras + image refs)
             ──> layout core (floor/ceiling, Manhattan walls, cell complex, rooms, openings)
             ──> stitch (photo tier: per-room plans placed into one property; other tiers: single frame already)
             ──> semantics (doors/windows/mirrors/fixtures/damage detections lifted onto surfaces)
             ──> rules (concealed-damage flags) ──> scope (line items keyed to surfaces)
             ──> uncertainty (interval on every number) ──> result.json + plan.svg/png + console table
```

One command per capture: `scan2scope run <capture_path>`. The tier is detected from the files and can be forced with `--tier`.

Package layout (`src/scan2scope/`): `ingest/`, `geometry/` (lidar, mapanything wrapper, video chunking, photo rooms, drift), `layout/`, `stitch/`, `semantics/`, `rules/`, `scope/`, `uncertainty/`, `output/` (schema, json, render), `bench/` (ground truth, matching, metrics, gates, reports), `synth/` (synthetic apartments and LiDAR captures), `cli.py`.

## Geometry backends

LiDAR (Stray Scanner): per-frame pose, intrinsics, 256x192 depth in mm and confidence. Keep confidence 2 and depth under 4 m, back-project with per-frame intrinsics scaled from RGB to depth resolution, convert ARKit's y-up world to z-up. Drift correction runs before layout (see below).

Video: decode with PyAV, sample about 1-2 fps, drop blurred frames by Laplacian variance, run MapAnything (Apache-2.0 weights) on overlapping chunks of at most 24 frames (measured ceiling on 16 GB is about 32 views at 518x336), align consecutive chunks with Sim(3) on shared frames, close the loop between the first and last frames, then gravity-align and anchor to planes.

Photo: per room, MapAnything on the room's 2 to 8 photos with intrinsics from EXIF (`FocalLengthIn35mmFormat`, converted on the image diagonal). Each room gets its own metric frame and its own Scene.

Metric scale for photo and video comes from MapAnything. A door-height prior can refine it when a full-height door opening is detected; the scale estimate and its spread feed the intervals.

## Layout core (shared by all tiers)

1. Gravity: LiDAR from ARKit; photo and video from the floor plane normal nearest the mean camera down vector.
2. Floor and ceiling: height histogram of near-horizontal surfaces, refined by least squares on inliers.
3. Wall directions: histogram of horizontal normal angles modulo 90 degrees gives the Manhattan frame; walls within a few degrees snap to it, others keep their angle.
4. Wall lines: 1-D density peaks of wall points along each axis, refined to the inlier mean.
5. Cell complex: the wall lines cut the plan into cells. A cell is inside when 2-D free-space evidence (camera-to-point rays) covers it. Inside cells separated by strong wall evidence form different rooms; a wall gap of door width between two inside regions is an opening that connects them.
6. Openings: per wall, an occupancy grid in (along-wall, height). A door is a gap from the floor to about 2 m, a window is a gap with wall below it, and points seen through the gap confirm it. Width is measured between the two jamb edges.
7. Dimensions: wall lengths are face to face along the room polygon edges; floor area is the polygon area; ceiling height is the floor-to-ceiling plane distance inside the room.

## Photo-tier stitching

Each room is reconstructed on its own, so rooms must be placed relative to each other. Two sources of placement evidence:

1. Doorway photos: a photo taken in room A through an open door shows part of room B. Running MapAnything on that photo plus B's photos registers it in B's frame; it is already registered in A's frame, which gives the A-to-B transform including scale.
2. Door matching: a door in A and a door in B with similar widths are hypothesised to be the same door; B is placed so the two door centres coincide across a wall-thickness gap, in two possible orientations.

Hypotheses are scored by registration quality, door width agreement, no room overlap (hard constraint), and Manhattan alignment. A maximum spanning tree over the best pairwise transforms places all rooms. Adjacencies with low score margins are flagged as uncertain rather than guessed.

## Drift handling

Multi-room LiDAR and video captures accumulate drift. Correction has three parts, each with an on/off switch for the ablation:

1. Loop closure: the protocol ends the capture where it started. The first and last segments are registered (ICP for LiDAR, MapAnything for video), and the error is distributed over the trajectory by pose-graph optimisation.
2. Plane anchoring: every segment is levelled to a shared floor plane and gravity direction, which removes roll, pitch and height drift.
3. Manhattan yaw anchoring: segment yaw is snapped to the global dominant wall directions when within 5 degrees.

The ablation reports the stitched footprint with corrections off and on against ground truth, on the real video capture and on synthetic LiDAR captures with injected drift.

## Damage, rules and scope

Detection: Grounding DINO (tiny) proposes boxes for damage prompts (water stain, mold, crack, hole, peeling paint) and for context objects (door, window, mirror, sink, toilet, bathtub, shower, stove, refrigerator, washing machine). SAM 2.1 turns boxes into masks. Mask pixels are lifted to 3-D with the tier's depth or point map and assigned to the nearest wall, floor or ceiling surface; the extent is the area of the projected mask on that surface, and repeated detections across views are merged. Detector scores are not treated as probabilities until they are calibrated on staged damage.

Rules: `rules/concealed_damage.yaml` holds rules with an id, trigger conditions, a cited basis (EPA mold guide, IICRC S500/S520 paraphrased) and an action. Examples: ceiling stain implies possible moisture above the ceiling; wall stain touching the floor implies wicking into the cavity; stain within 1 m of a wet fixture implies a supply or drain leak; mold area over 10 sq ft needs containment; crack starting at an opening corner suggests movement. Every flag records the rule id and the inputs that fired it.

Scope: each damage region and flag maps to line items keyed to a surface id, with an Xactimate-style category (DRY, PNT, WTR, CLN, HMR, INS), an activity code (`&` remove and replace, `-` remove, `+` replace, `R` detach and reset), a quantity with interval and a unit. Codes are labelled Xactimate-style, not an official price list.

## Uncertainty

Every number in the output is `{value, lo, hi}` at a 90% nominal level. The error model per tier has a per-capture log-scale term shared by every measurement and a per-measurement additive term that grows with thin evidence (fewer views, low observed fraction of a wall, low light, missing EXIF). Interval width is the model's standard deviation times a multiplier `q` fitted per tier by split conformal on ground-truth rooms, with the room as the unit and leave-one-room-out for reported coverage. Where a tier has fewer than 9 independent rooms, `q` stays at its prior and the report says so; it does not claim calibration it cannot show.

## Output contract

`result.json` validates against `schema/scan2scope.schema.json` (JSON Schema 2020-12). Top level: capture metadata and flags, conventions (measurement definitions), property (stitched plan, footprint, adjacency, drift-correction record), rooms (polygon, walls, openings, surfaces, ceiling height, floor area), damage, concealed_damage_flags, scope, timing, provenance (git commit, model revisions, cache mode). `plan.svg` and `plan.png` render the stitched plan with dimensions and intervals; `rooms/` holds a sheet per room. The console prints a table of every wall, opening and ceiling height with ids that match the render, so a laser check at the walk-in is quick.

## Benchmark and gates

Ground truth lives in `bench/data/<property>/ground_truth.yaml` (format in `docs/ground_truth_protocol.md`). The harness matches predicted rooms, walls and openings to ground truth deterministically (room by folder name or overlap, walls by cyclic alignment of the polygon, openings by wall and centre offset under half the true width) and computes every gate in `bench/gates.yaml` per tier, the repeatability table, calibration coverage with Clopper-Pearson intervals, the head-to-head table and timing.

Assumed thresholds where the brief is silent are marked `assumed: true` in `gates.yaml`: LiDAR wall length within max(2 cm, 1%), floor area within 2% (LiDAR), 6% (video), 16% (photo), and video and photo ceiling and opening widths within the tier's wall-length bound. The five gates listed in the brief are used as written.

## Data plan

- Real (iPhone 17): the home captured at the photo and video tiers, one room captured twice at each of those tiers, one furnished room with staged damage of two classes, tape or laser ground truth for every reported dimension, magicplan scans of two rooms chosen before scanning.
- Synthetic LiDAR: generated apartments with exact ground truth, captures rendered in the Stray Scanner format with ARKit-like depth noise, confidence and pose drift, including repeat captures and drift injection.
- Head-to-head deviation: with no LiDAR phone, magicplan runs in its non-LiDAR AR mode on the iPhone 17 and is compared against our video and photo tiers on the same two rooms. The compliance matrix records this as a deviation from "LiDAR tier".

## Reproducibility

`uv` with a lockfile, Python 3.12. `scan2scope fetch-weights` downloads pinned Hugging Face revisions with parallel range requests and checks SHA-256. Model outputs are cached by SHA-256 of the input bytes, model revision and preprocessing parameters; `--replay` reproduces the reported numbers from the cache and the default live path recomputes them. Raw benchmark data goes to GitHub Release assets (no Git LFS), with GPS removed from photos and videos. CI runs unit tests on synthetic data on Ubuntu.

## Decision log

| Date | Decision | Alternatives considered | Reason |
|---|---|---|---|
| 2026-10-03 | Route 2 stock protocol | Own iOS app (Route 1) | Free tools only, no Apple Developer account, 30-hour budget |
| 2026-10-03 | Stray Scanner for LiDAR | Polycam raw export, Record3D, 3D Scanner App | Only free app verified to export per-frame depth, poses and intrinsics with no purchase |
| 2026-10-03 | MapAnything Apache-2.0 weights for photo and video geometry | VGGT, Pi3, MASt3R, DA3, CUT3R | Metric output, accepts intrinsics, runs on Apple MPS (8 views in 8 s measured), permissive licence; the others are non-commercial, gated or CUDA-only |
| 2026-10-03 | Grounding DINO tiny + SAM 2.1 for semantics | SAM 3, Florence-2, local VLM | Apache-2.0, ungated, returns scores; SAM 3 is gated and its install needs CUDA; VLM confidences were 0.95-1.0 on wrong labels in a test |
| 2026-10-03 | Shared layout core for all tiers | Tier-specific layout code | One place to debug and fix; tiers differ only in how the Scene is produced |
| 2026-10-03 | magicplan free Starter for head-to-head | Polycam (free tier exports GLTF only) | Free plan exports per-room numbers and a dimensioned sketch |
| 2026-10-03 | Own schema and gates, marked as assumptions | Wait for Round 1 material | None was provided |
