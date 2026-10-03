# scan2scope

Turns an iPhone capture of a home into one dimensioned whole-property floor plan, per-surface damage regions, concealed-damage flags with the rule that fired, and repair scope line items keyed to surfaces. Every measurement carries a 90% interval. Three input tiers share one output contract:

| Tier | Input | Phones |
|---|---|---|
| Photo | one folder per room, 2 to 8 photos each | any iPhone 15 or newer |
| Video | one handheld walkthrough clip | any iPhone 15 or newer |
| LiDAR | a Stray Scanner export (depth, poses, intrinsics) | iPhone 15 Pro to 18 Pro Max |

How to capture: [docs/capture_protocol.md](docs/capture_protocol.md) (one page). Design: [docs/design.md](docs/design.md).

## Run it on a fresh capture

Needs macOS on Apple Silicon or Linux x86_64, about 8 GB of free disk, and a network connection for the first setup.

```sh
curl -LsSf https://astral.sh/uv/install.sh | sh        # or: pip install uv
git clone https://github.com/abhayraihal/scan2scope.git && cd scan2scope
uv sync --all-extras                                   # Python 3.12 environment from uv.lock
uv run scan2scope fetch-weights                        # about 5.8 GB, pinned revisions, SHA-256 checked
uv run scan2scope doctor                               # checks weights, device, disk and decoders
uv run scan2scope run ~/Downloads/Scan                 # photo tier: folder of room folders
uv run scan2scope run ~/Downloads/IMG_0042.MOV         # video tier
uv run scan2scope run ~/Downloads/recording.zip        # LiDAR tier: Stray Scanner share-sheet zip
```

The tier is detected from the files (`--tier` overrides it). Each run writes `out/<capture>/result.json` (validated against [schema/scan2scope.schema.json](schema/scan2scope.schema.json)), `plan.svg` and `plan.png` (the stitched plan), `rooms/<id>.svg`, and prints every wall, opening and ceiling height with its interval and the ids used on the plan.

Useful flags: `--no-drift` turns drift correction off (used for the ablation), `--cache replay` reuses stored model outputs instead of running the models, `--no-semantics` skips damage detection.

## Repository map

| Path | What it holds |
|---|---|
| `src/scan2scope/ingest` | tier detection, photo EXIF and intrinsics, video frame sampling, Stray Scanner parser |
| `src/scan2scope/geometry` | LiDAR back-projection and drift correction, MapAnything wrapper, photo and video reconstruction |
| `src/scan2scope/layout` | floor, ceiling, walls, rooms and openings from a gravity-aligned point cloud (shared by all tiers) |
| `src/scan2scope/stitch` | photo-tier placement of rooms into one property plan |
| `src/scan2scope/semantics`, `rules`, `scope` | damage detection, concealed-damage rules, scope line items |
| `src/scan2scope/uncertainty` | per-tier error model and conformal calibration of intervals |
| `src/scan2scope/output` | JSON writer, schema validation, plan rendering, console summary |
| `src/scan2scope/bench`, `bench/` | ground truth, matching, gates, benchmark report |
| `src/scan2scope/synth` | synthetic Stray Scanner captures with exact ground truth |
| `docs/` | design, capture protocol, ground-truth protocol, module contracts, requirements, compliance matrix |

## Models, data and tools

| Component | Version | Licence | Used for |
|---|---|---|---|
| [MapAnything](https://github.com/facebookresearch/map-anything), weights `facebook/map-anything-apache` | code 3d10cf7, weights 00f9c24 | Apache-2.0 | metric geometry and poses from photos and video frames |
| [Grounding DINO tiny](https://huggingface.co/IDEA-Research/grounding-dino-tiny) | a2bb814 | Apache-2.0 | open-vocabulary boxes for damage, fixtures, mirrors, doors, windows |
| [SAM 2.1 hiera small](https://huggingface.co/facebook/sam2.1-hiera-small) | ee5bba1 | Apache-2.0 | masks for damage regions |
| [Stray Scanner](https://apps.apple.com/us/app/stray-scanner/id1557051662) | 1.4 | free app, MIT source | LiDAR capture on Pro iPhones |
| magicplan | free Starter plan | proprietary app | head-to-head comparison only |

No paid tools or services are used, and nothing calls infrastructure of ours: everything runs on the local machine after `fetch-weights`.

AI coding assistants (Claude Code) were used to write code and documents. Design decisions, their alternatives and the reasons are recorded in [docs/design.md](docs/design.md).
