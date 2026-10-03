# Benchmark capture plan

What gets captured for the benchmark, with the iPhone 17 (photo and video tiers). The LiDAR tier is benchmarked on synthetic captures because no LiDAR phone is available; see `docs/design.md`.

## Property

At least three rooms plus the hallway or corridor that connects them. One of the rooms is furnished and has the staged damage described in `docs/ground_truth_protocol.md`. Include a bathroom with a mirror if there is one, and a room with a window.

## Captures, in this order

| Id | Tier | What | Why |
|---|---|---|---|
| photo_1 | photo | every room, per the protocol, one folder per room | photo gates, photo stitch |
| video_1 | video | one walkthrough of every room, ending where it started | video gates, drift ablation |
| photo_2 | photo | every room again, standing in slightly different spots | repeatability (at least the damage room) |
| video_2 | video | the walkthrough again | repeatability, second multi-room sample |
| photo_dim | photo | one room with the main light off | low-light failure mode |
| magicplan | head-to-head | magicplan free plan, two rooms chosen before scanning: the damage room and the hallway | Part 3 |

If time is short, `photo_2` and `video_2` can cover only the damage room. `photo_dim` is optional.

## magicplan (free Starter plan)

Install magicplan from the App Store and create a free account. New project, scan the two rooms with its camera mode (the iPhone 17 has no LiDAR, so it uses its AR mode). Make no manual edits. Export the Statistics CSV and the Sketch PDF with all dimensions shown, in metric units, and note the app version from its settings or About screen.

## Measurements

After the captures, measure everything in `docs/ground_truth_protocol.md` and fill `bench/templates/ground_truth.yaml`. Numbers in any format are fine; they get transcribed into the YAML.

## Transfer

AirDrop everything to the Mac and put it under `~/code/scan2scope-data/home/raw/` with the ids above as folder or file names.
