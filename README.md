# scan2scope

Turns an iPhone capture of a home into a dimensioned floor plan of the whole property, damage regions on walls, floors and ceilings, concealed-damage flags that name the rule that fired, and repair scope line items keyed to those surfaces. Every measurement carries a 90% interval. Three input tiers share one output contract:

| Tier | Input | Phones |
|---|---|---|
| Photo | one folder per room, 2 to 8 photos each | any iPhone 15 or newer |
| Video | one handheld walkthrough clip that ends where it started | any iPhone 15 or newer |
| LiDAR | a Stray Scanner export (depth, poses, intrinsics) | iPhone 15 Pro to 18 Pro Max |

Only an iPhone 17 has been used for real captures; [docs/device_matrix.md](docs/device_matrix.md) lists every model and what was tested on it. How to capture: [docs/capture_protocol.md](docs/capture_protocol.md) (one page). Design and decision log: [docs/design.md](docs/design.md). Technical report: [docs/technical_report.md](docs/technical_report.md). Running a capture at the walk-in: [docs/walkin_runbook.md](docs/walkin_runbook.md).

## Set up a clean machine

Needs about 10 GB of free disk and a network connection for setup. All timings here are from an M4 MacBook (16 GB, macOS 26). On Linux x86_64 the lockfile installs CPU-only torch and CI runs the unit tests there without the model stack; the model stages have only been timed on the M4.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh        # or: pip install uv (after the script, open a new shell)
git clone https://github.com/abhayraihal/scan2scope.git && cd scan2scope
uv sync --all-extras                                   # Python 3.12 environment from uv.lock
uv run scan2scope fetch-weights                        # 5.8 GB of weights at pinned revisions, hash-checked
uv run scan2scope doctor                               # checks weights, device, disk and decoders
```

Measured on 2026-10-03 from a fresh clone with empty dependency and model caches, at about 11 MB/s (installing uv not included): clone 6 s, `uv sync --all-extras` 75 s, `fetch-weights` 579 s, `doctor` 19 s, then about 50 s for the first photo-tier run including model loading (65 s at commit 7d26d19, whose damage detection is slower). That is 12 to 12.5 minutes from clone to a first result (729 s, or 744 s at 7d26d19), and the weight download is 579 s of it.

## Run a capture

One command per capture:

```sh
uv run scan2scope run ~/Downloads/Scan                 # photo tier: a folder of room folders
uv run scan2scope run ~/Downloads/IMG_0042.MOV         # video tier: one clip
uv run scan2scope run ~/Downloads/recording.zip        # LiDAR tier: a Stray Scanner folder or its zip
```

The tier is detected from the files. A folder of photo folders is the photo tier with one room per folder (a single folder of photos is one room), a video file is the video tier, and a folder with `odometry.csv` and `depth/` is the LiDAR tier. Any of them can come as a zip.

Flags: `--tier` overrides the detection, `--out` sets the output folder, `--no-semantics` skips damage detection, `--no-drift` turns drift correction off (for the ablation), and `--cache` takes `live` (the default: compute model outputs and store them), `replay` (use stored outputs only) or `off`.

Run times on the M4 at commit 7d26d19 (cache off, other jobs running): 65 s for the 5 benchmark photos (22 s model loading, 32 s damage detection) and 264 s for the 57 s benchmark clip (142 s geometry, 120 s damage detection). The benchmark run at 226bcff, before the slower damage checks of 8df5f97, took 55 s for the photos and 162 to 179 s per clip. A synthetic LiDAR capture takes about 15 to 22 s without damage detection.

## Outputs

Each run writes to `out/<capture name>/`, or to the `--out` folder:

- `result.json` holds capture metadata and flags; the property (footprint, extents, adjacency, drift-correction record); rooms (polygon, walls, openings, surfaces, ceiling height, floor area); damage regions; concealed-damage flags with the rule id and the inputs that fired it; scope line items keyed to surface ids; stage timings; and provenance (git commit, model revisions, cache mode). Every measurement is `{value, lo, hi, unit}` at a 90% nominal level. The file is validated against [schema/scan2scope.schema.json](schema/scan2scope.schema.json) on every run, and a capture whose geometry fails still gets a valid file with no rooms and a `geometry_failed` flag.
- `plan.svg` and `plan.png` show the stitched property plan with dimensions, and `rooms/<id>.svg` holds one sheet per room.
- The console prints every wall, opening and ceiling height with its interval and the ids used on the plan (`R1-W3`, `R1-O1`), then damage, flags, scope lines and stage timings.

## Results

[docs/benchmark_report.md](docs/benchmark_report.md) has every gate per tier, the repeatability and calibration tables, the drift ablation and timing. The real benchmark is one furnished bedroom captured on an iPhone 17 and measured with a tape: 162 and 142 in wall to wall and 130 in floor to ceiling (4.115 m, 3.607 m, 3.302 m; floor area 14.84 m2). No LiDAR phone was available, so the LiDAR tier is scored on synthetic captures. The real-room scores below are the same in the benchmark runs at 226bcff and 7d26d19; the changes between those commits (damage detection, one damage rule) do not reach the synthetic run.

| Tier | Scored on | Wall length | Ceiling height | Floor area | Values inside their 90% interval |
|---|---|---|---|---|---|
| Photo | real bedroom: one set of 5 photos, scored as the originals and as the WhatsApp copy (2 captures) | 8 of 8 within 8%, worst 7.5% | 0 of 2 within 8%: 59 and 65 cm low | 2 of 2 within 16%, worst 11.5% | 14 of 14, mean half-width 33% of the true value |
| Video | the same bedroom: clips of 57 s (1080p) and 49 s (4K), each scored as the original and as the WhatsApp copy (4 captures) | 1 of 16 within 3%, median error 7.7%, worst 58% | 0 of 4 within 3%: 44 to 103 cm low | 0 of 4 within 6%: 7 to 26% low | 25 of 28, mean half-width 27% |
| LiDAR | synthetic: 4 properties, 12 captures, 20 rooms, exact ground truth | 209 of 252 within max(2 cm, 1%), 10 not found | 58 of 60 within 1.5 cm, 2 rooms not found | 52 of 60 within 2% | 479 of 560 (85.5%), mean half-width 3.1% |

- Thresholds are the gates in [bench/gates.yaml](bench/gates.yaml). The brief states the photo and video wall-length gates and the LiDAR ceiling gate; the other thresholds in the table are ours and are marked `assumed: true` there.
- Across repeat captures, 155 of 232 synthetic LiDAR wall pairs agree within max(1 cm, 0.5%). In the real room, 1 of 8 video wall pairs do.
- Of the 192 synthetic LiDAR openings, 61 were found within 2 cm, 34 were found more than 2 cm off and 97 were missed; with the 10 phantoms counted as failures, the opening gate passes 61 of 202. The real room's openings were not measured and are not scored.
- A public Stray Scanner recording of an open-plan office (232 s) gives a real LiDAR run without ground truth. Drift correction accepted 71 of 122 loop closures and cut the floor-height spread between trajectory segments from 48 cm to 5.5 cm, and the plan came out as one room with 16 walls and 189.6 m2.
- In the fix loop, synthetic LiDAR repeatability (the worst gate) went from 82 of 236 wall pairs (34.7%) to 155 of 232 (66.8%), against a predicted 85% (75 to 95%). The declaration, both runs, the diff and the post-mortem are in [docs/fixloop/](docs/fixloop/).

## Reproduce the reported numbers

```sh
# real bedroom: download the published captures (6 zips, 331 MB, SHA-256 checked), then run and score them
uv run python scripts/benchmark_data.py fetch bench/data/bedroom
uv run python scripts/benchmark_data.py fetch bench/data/bedroom_whatsapp
uv run scan2scope bench bench/data --out runs/final_real

# synthetic LiDAR: regenerate the 4 properties (12 captures, 0.9 GB), then run and score them
uv run scan2scope synth --out bench/synthetic --properties 4 --seed 0
uv run scan2scope bench bench/synthetic --out runs/final_synth --no-semantics

# fix loop: benchmark the tagged before and after commits from clean worktrees on the synthetic set
BENCH_ARGS=--no-semantics CACHE=off scripts/fixloop.sh fixloop-before fixloop-after bench/synthetic

# real LiDAR recording (not redistributed here): download it from Hugging Face at a pinned revision, run it with and without drift correction
curl -L -o 4e41d0a7da.zip https://huggingface.co/datasets/vslamlab/strayscanner/resolve/25370e394f18124a10c02f99c3f2a48680dc5837/4e41d0a7da.zip
uv run scan2scope run 4e41d0a7da.zip --no-semantics --out out/office_on
uv run scan2scope run 4e41d0a7da.zip --no-semantics --no-drift --out out/office_off

# conformal fit of q on the 20 synthetic rooms (technical report section 5): prints q and leave-one-room-out coverage, writes nothing
uv run python -c "import json; from scan2scope.uncertainty.calibrate import fit_q, loro_coverage; m = json.load(open('runs/final_synth/metrics.json')); r = [x for c in m['metrics'] for x in c['records']]; print(fit_q(r, write=False)['tiers']['lidar']['q'], loro_coverage(r)['lidar']['coverage'])"
```

Each `bench` run writes `benchmark_report.md`, `metrics.json` and `gates.json` to its `--out` folder, and runs every multi-room video or LiDAR capture a second time with drift correction off for the ablation. The fix-loop script writes `runs/fixloop/before`, `runs/fixloop/after` and `runs/fixloop/diff.md`; the committed copies are in `docs/fixloop/`. On the M4, generating the synthetic set took 26 minutes, the synthetic benchmark 5.5 minutes and the fix loop 12.5 minutes. The real benchmark took 12.5 minutes at 226bcff, running alongside the synthetic one; damage detection is slower from 8df5f97 on, and at 7d26d19 the six real captures add up to about 16 minutes (an estimate from stage times, not one measured bench run).

Two reproduction checks were run on 2026-10-04. `synth --seed 0` regenerated all 4 benchmarked synthetic properties byte for byte. Live reruns of bedroom/photo_1 and bedroom/video_1 at 7d26d19 with the cache off matched the published wall lengths, ceiling heights and floor areas, and their intervals, to all four decimals stored in `result.json`, and reported the same damage regions; the other four real captures were not rerun for this check. `--cache replay` reuses only the model outputs that a live run stored in `~/.cache/scan2scope/outputs`, so it works on the machine that made them.

Two analyses are not scripted in this repository. The ARKitScenes runs that measured MapAnything's scale error used scripts and data kept outside it; those numbers and the real bedroom's set the photo and video scale priors. The per-stage drift ablation on 5 synthetic captures switches stages through the `drift_options` argument of `geometry.lidar.build_scene`, which the CLI does not expose (it has only `--no-drift`); [docs/benchmark_report.md](docs/benchmark_report.md) describes the script.

## Repository map

| Path | What it holds |
|---|---|
| `src/scan2scope/cli.py`, `pipeline.py` | the `scan2scope` command and the order of stages |
| `src/scan2scope/ingest` | tier detection, photo EXIF and intrinsics, video frame sampling, Stray Scanner parser |
| `src/scan2scope/geometry` | LiDAR back-projection and drift correction, MapAnything wrapper, photo and video reconstruction, video chunk alignment |
| `src/scan2scope/layout` | floor, ceiling, walls, rooms and openings from a gravity-aligned point cloud (shared by all tiers) |
| `src/scan2scope/stitch` | photo-tier placement of rooms into one property plan |
| `src/scan2scope/semantics`, `rules`, `scope` | damage detection, concealed-damage rules, scope line items |
| `src/scan2scope/uncertainty` | per-tier error model (`priors.yaml`) and conformal calibration of intervals |
| `src/scan2scope/output` | JSON writer, schema validation, plan rendering, console summary |
| `src/scan2scope/bench` | benchmark harness: ground truth, matching, metrics, gates, drift ablation, report |
| `src/scan2scope/synth` | synthetic apartments and Stray Scanner captures with exact ground truth |
| `src/scan2scope/cache.py`, `weights.py` | model-output cache (`live`, `replay`, `off`); pinned weight download and `doctor` |
| `schema/scan2scope.schema.json` | output contract (JSON Schema 2020-12) |
| `bench/gates.yaml`, `bench/data/<property>/`, `bench/templates/` | gate thresholds; ground truth and release manifest per real property; ground-truth template |
| `scripts/` | benchmark data packaging and fetch, laptop-screen blurring, fix-loop runner and comparison, compliance matrix |
| `tests/` | unit tests; `uv run pytest -m "not ml"` runs the ones that need no model weights, as CI does |
| `docs/` | design and decision log, capture and ground-truth protocols, walk-in runbook, module contracts, technical report, benchmark report, fix loop, requirements and compliance matrix |

## Models, data and tools

| Component | Version | Licence | Used for |
|---|---|---|---|
| [MapAnything](https://github.com/facebookresearch/map-anything) code, weights [facebook/map-anything-apache](https://huggingface.co/facebook/map-anything-apache) | code 3d10cf7, weights 00f9c24 | Apache-2.0 | metric geometry and camera poses from photos and video frames |
| [DINOv2](https://github.com/facebookresearch/dinov2) hub code | `main` branch, not pinned | Apache-2.0 | encoder code that MapAnything loads through torch.hub |
| [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | a2bb814 | Apache-2.0 | open-vocabulary boxes for damage, fixtures, mirrors, doors and windows; laptop boxes for screen blurring |
| [SAM 2.1 hiera small](https://huggingface.co/facebook/sam2.1-hiera-small) | ee5bba1 | Apache-2.0 | masks for damage regions |
| [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) | 1.4 | free app, MIT source | LiDAR capture on Pro iPhones; the format the LiDAR tier reads |
| Benchmark data `bedroom`, `bedroom_whatsapp` | GitHub release `benchmark-data-v1`: 6 zips, 331 MB | our own captures | the real-room benchmark; laptop screens blurred with `scripts/blur_screens.py` before publishing, the original photos keep their EXIF (no GPS); every real-room number is computed from these files |
| [ARKitScenes](https://github.com/apple/ARKitScenes) raw data, 3 Validation scans (45663154, 47332885, 48018560) | v1 | Apple ARKitScenes licence (non-commercial grant; commercial use capped by user count) | development only: photo and video geometry checked against iPad Pro LiDAR depth, and MapAnything's scale error measured for the photo and video priors; kept local, not redistributed |
| Stray Scanner recording `4e41d0a7da` from [vslamlab/strayscanner](https://huggingface.co/datasets/vslamlab/strayscanner) | dataset revision 25370e3 | none stated | a real LiDAR run without ground truth; kept local, not redistributed |
| magicplan | free Starter plan; app version not recorded | proprietary app | planned head-to-head; its IFC export (`bench/data/bedroom/magicplan/room.ifc`) had no room geometry, so no comparison was made |

No paid tools or services are used, and nothing calls our own infrastructure. Besides installing, only `fetch-weights` (Hugging Face and GitHub), `benchmark_data.py fetch` (GitHub release assets) and the optional recording download use the network; a full photo run completed on 2026-10-04 with the HTTP and HTTPS proxy variables pointed at a closed port, so runs need no network.

## Limitations and deviations from the brief

- The real benchmark is one furnished bedroom. Photo stitching is shown only in unit tests on synthetic rooms, and multi-room adjacency and the drift ablation are scored only on the synthetic LiDAR captures.
- No LiDAR phone was available. The LiDAR tier is scored on 12 synthetic Stray Scanner captures and run on one public real recording that has no tape ground truth.
- One damage class was staged (a paper sheet with a drawn crack). The real room's openings and damage extents were not measured, so neither is scored on real data.
- The ground truth is whole-inch tape readings with one ceiling reading, where [docs/ground_truth_protocol.md](docs/ground_truth_protocol.md) asks for millimetres, two readings each and three ceiling points.
- The benchmark capture departs from the capture protocol: both clips were filmed in portrait, video_2 at 4K and aimed at furniture, and the room has 5 photos where 6 to 8 are recommended.
- There is no head-to-head. The magicplan free-plan IFC export contained no room geometry, and no other app data was captured.
- No Round 1 schema or gate list was provided, so `schema/` and `bench/gates.yaml` are ours; thresholds the brief does not state are marked `assumed: true`.
- The intervals are not calibrated. The interval multiplier q stays at its prior of 1.0 in every tier (`uncertainty/calibration.yaml`): no tier has 9 real rooms, and q is not fitted on synthetic rooms. The real-room intervals contain 39 of the 42 tape values, but they are wide, with a mean half-width of 33% of the true value at the photo tier and 27% at the video tier. The synthetic LiDAR intervals contain 85.5% of values against the nominal 90%.
- Photo and video metric scale comes from MapAnything alone, so a scale error moves every number of a capture together. Ceilings in the real room came out 44 to 103 cm low on all 6 captures. Media sent as ordinary messaging-app attachments lose EXIF and are recompressed; JPEG quality-70 re-saves of the ARKitScenes kitchen photos cut MapAnything's predicted depth by 35%, so the capture protocol asks for captures sent as files.
- Drift correction is on by default, but on the 12 synthetic LiDAR captures it lowered the mean wall error on 4 and raised it on 8, including all 4 strong-drift captures, and the share of walls in tolerance went up on 4 and down on 5 (drift ablation in the benchmark report).

## AI assistance

The code, tests and documents were written with Claude Code (Anthropic). Claude Code agents also ran the error analyses on the real room and on ARKitScenes, wrote the fixes that followed and ran the benchmarks. The owner took the captures and the tape measurements. The decision log in [docs/design.md](docs/design.md) records each design decision, the alternatives considered and the reason.
