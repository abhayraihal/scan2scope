# Walk-in runbook

How to run the pipeline on a capture made at the defense. The capture itself follows `docs/capture_protocol.md`.

## Before the session

```sh
uv run scan2scope doctor          # weights present, Apple GPU found, disk free, HEVC and HEIC decoders present
```

Bring a USB-C cable: the demo Mac has AirDrop switched off by policy, so files come over the cable (Image Capture for photos and videos, Finder's Files tab for the Stray Scanner folder).

## Receiving the files

| Tier | What arrives | Command |
|---|---|---|
| Photo | a folder (`Scan`) holding one folder per room | `uv run scan2scope run ~/Downloads/Scan` |
| Video | one `.MOV` | `uv run scan2scope run ~/Downloads/IMG_1234.MOV` |
| LiDAR | a Stray Scanner `.zip` | `uv run scan2scope run ~/Downloads/<recording>.zip` |

If the photos arrive as loose files instead of room folders, make one folder per room in walking order (`01 hallway`, `02 kitchen`, ...) and move each room's photos into it. Image Capture and AirDrop keep the original HEIC files with their EXIF focal length; ordinary messaging-app sends strip it, which the pipeline flags and answers with wider intervals.

## Reading the result

The console prints, per room, every wall with its length and 90% interval, the ceiling height, the floor area and every opening with its width and height. The ids (`R2-W3`, `R2-O1`) match the labels on `out/<capture>/plan.png`, so a laser reading can be compared against the right number directly. Damage regions, concealed-damage flags with their rule ids, and the scope line items follow.

Typical run times on the M4 (16 GB): photo tier about 1 minute per 5 rooms plus up to 3 minutes of room stitching; video tier about 2.5 minutes per minute of walkthrough; LiDAR tier under a minute plus about a minute of damage detection. The first run loads the models (about 20 s).

## If something goes wrong

- `geometry_failed` in the capture flags: the result is still written, with no rooms; rerun with `--tier` set explicitly if the tier was misdetected.
- A video that the phone recorded in HDR is flagged `video_hdr` and still runs.
- `--no-semantics` skips damage detection if time is short; geometry and intervals are unaffected.
