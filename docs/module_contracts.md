# Module contracts

Interfaces between stages. Types are in `src/scan2scope/types.py`; frames and units are documented there (metres, z-up world, OpenCV camera axes, camera-to-world poses). The orchestration is `src/scan2scope/pipeline.py`.

## Order of stages

ingest → geometry → layout → (photo only) stitch → semantics → rules → uncertainty → scope → output.

Layout, stitch and semantics produce `Measurement` objects with `value` and `evidence` only. `uncertainty.annotate` fills `lo`/`hi` and writes `evidence["sigma"]`. Scope runs after it and derives quantity intervals from the intervals of the measurements it uses.

## cache.py

`OutputCache(mode, root)` with `mode` in `live | replay | off`. `compute(key: dict, fn) -> dict[str, np.ndarray]`: live computes on a miss and stores, replay loads or raises `CacheMiss`, off always computes. The key is hashed with SHA-256 over `json.dumps(key, sort_keys=True)`; keys include input file SHA-256s (`file_sha256(path)`), the model revision and preprocessing parameters, never the device. `effective_mode` reports `live | replay | mixed | none`, `device_label()` the torch device.

## weights.py

`fetch_all(verify=True)` downloads every `config.MODELS` entry at its pinned revision into `ModelSpec.local_dir` with parallel HTTP range requests and resume, checks SHA-256 of large files, and installs the DINOv2 hub code into `TORCH_HOME/hub/facebookresearch_dinov2_main`. `doctor() -> int` prints weights, device, disk and decoder checks and returns a process exit code.

## ingest

- `detect.detect_capture(path, tier=None, work_dir) -> (tier, root, CaptureInfo)`. A zip is extracted into `work_dir/input` first. A folder with `odometry.csv` and `depth/` (at most one level down) is `lidar`; a video file, or a folder with exactly one video, is `video`; a folder of image subfolders is `photo` (one room per subfolder); a folder of images only is `photo` with one room. Anything else raises `ValueError` with a message saying what was expected.
- `images.load_image(path) -> (rgb uint8 HxWx3 upright, ExifInfo)`, `images.intrinsics_from_exif(exif, width, height) -> K | None` (35 mm equivalent focal length on the image diagonal), `images.list_room_folders(root) -> list[tuple[str, list[Path]]]` sorted by folder name, skipping hidden files and Live Photo `.MOV` companions.
- `video.probe(path) -> VideoInfo`, `video.sample_frames(path, out_dir, target_fps, max_frames) -> list[FrameRecord]` writing upright frames with rotation applied and blurred frames dropped.
- `stray.load_stray(root) -> StrayCapture`: per-frame timestamps, camera-to-world poses in OpenCV axes and ARKit world, per-frame intrinsics at RGB resolution, depth and confidence readers (`read_depth(i)` in metres, `read_conf(i)`), RGB frame extraction, count checks.

## geometry

- `lidar.build_scene(root, work_dir, *, drift_correction=True) -> Scene`
- `video.build_scene(video_path, work_dir, *, drift_correction=True, cache) -> Scene`
- `photo.build_room_scenes(root, work_dir, *, cache) -> list[Scene]` (one per room folder, `room_hint` = folder name)
- `mapanything_backend.MapAnythingRunner.get().infer(images, intrinsics, key) -> list[ViewPrediction]`
- `drift` holds the loop-closure, pose-graph and plane-anchoring code used by lidar and video. Every scene records `meta["drift"] = {"enabled": bool, ...}` and `meta["quality"]`.

The returned Scene is gravity-aligned (z up). Views carry point maps in world coordinates so later stages can lift image pixels to 3-D.

## layout

`build_plan(scene, *, single_room=False) -> Plan`. Rooms `R1..Rn`, polygons counter-clockwise, walls `<room>-W<k>` in polygon order, openings `<room>-O<k>` with `connects_to` set when another room is on the other side, adjacency from shared openings (`source="shared_frame"`), footprint = sum of room floor areas, extents = bounding box of all rooms. Photo rooms are labelled from the folder name (`"02 kitchen"` → `"kitchen"`).

## stitch

`stitch_rooms(room_scenes, room_plans, work_dir, *, cache) -> (Plan, list[Scene])` places photo-tier rooms in one property frame (doorway-photo registration first, door matching second, no-overlap as a hard constraint), renumbers rooms in folder order, fills adjacency (`doorway_photo` or `door_match`) and `connects_to`, transforms the scenes into the property frame, and records `plan.meta["stitch"]`. Rooms it cannot place are laid out without overlap and flagged `placement_uncertain:<room>`.

## semantics

`analyze(scenes, plan, work_dir, *, cache) -> SemanticsResult(damage, objects)`. Grounding DINO boxes and SAM 2.1 masks, lifted through view point maps onto the nearest room surface, merged across views. `objects` are fixtures, mirrors, doors and windows with room id and plan position, used by the rules.

## rules, uncertainty, scope

- `rules.evaluate(plan, damage, objects) -> list[ConcealedFlag]` from `rules/concealed_damage.yaml`.
- `uncertainty.annotate(plan, damage, *, tier, quality)` fills every interval in place.
- `scope.generate(plan, damage, flags) -> list[LineItem]` from `scope/catalog.yaml`.

## output

`writer.build_result(...) -> dict` (matches `schema/scan2scope.schema.json`), `schema.validate(result)`, `render.render_all(result, out_dir)` (`plan.svg`, `plan.png`, `rooms/<id>.svg`), `console.print_summary(result)`.

## bench and synth

`bench.runner.run_benchmark(data_root, out_dir, *, cache_mode, only, skip_run)` runs every capture listed in each property's `ground_truth.yaml`, matches results to ground truth, evaluates `bench/gates.yaml` per tier, and writes `benchmark_report.md` and `metrics.json`. `synth.generate.generate_benchmark(out, n_properties, seed)` writes synthetic Stray Scanner captures with exact ground truth in the same layout.
