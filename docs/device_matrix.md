# Device matrix

Which tier runs on which iPhone, the capture tool for each tier, and the accuracy each tier delivered in the benchmark. One phone was available, an iPhone 17, and it was used for the photo and video tiers. Every other row is what the capture tools and the pipeline support on paper; none of those phones has been tested. The LiDAR tier has also run on four real recordings made on LiDAR devices we did not have: the public office recording and three recordings provided with the problem statement. None of them comes with measurements.

## Tiers and tools per model

The capture tools are the same on every model (`docs/capture_protocol.md`):

- The photo tier uses the Camera app in Photo mode with the 1x lens, flash and Live Photo off: one folder per room, 2 to 8 photos.
- The video tier uses the Camera app in Video mode with the 1x lens at 1080p and 30 fps, with HDR Video, Enhanced Stabilization and Auto FPS off and Lock Camera on. 4K at 30 fps also runs: video_2 of the benchmark is 4K.
- The LiDAR tier uses Stray Scanner 1.4 (free, no in-app purchases, needs iOS 18.6 or later) at 30 fps; the pipeline reads its recording folder or its share-sheet zip. The three provided recordings were made at 60 fps (45 to 46 fps on average after dropped frames) and ran with the default settings.

The protocol uses the 1x lens on every tier because the 16e, Air and 17e have a single rear camera, with no Ultra Wide. The LiDAR column is from Apple's technical specifications. To check a phone, open Settings > General > About: if Model Name contains "Pro" and iOS Version is 18.6 or later, all three tiers run; otherwise use photo or video.

| Model | LiDAR | Photo tier | Video tier | LiDAR tier | Tested by us | Notes |
|---|---|---|---|---|---|---|
| iPhone 15 | no | Camera app | Camera app | not available | no | |
| iPhone 15 Plus | no | Camera app | Camera app | not available | no | |
| iPhone 15 Pro | yes | Camera app | Camera app | Stray Scanner 1.4 | no | |
| iPhone 15 Pro Max | yes | Camera app | Camera app | Stray Scanner 1.4 | no | |
| iPhone 16 | no | Camera app | Camera app | not available | no | |
| iPhone 16 Plus | no | Camera app | Camera app | not available | no | |
| iPhone 16e | no | Camera app | Camera app | not available | no | single rear camera |
| iPhone 16 Pro | yes | Camera app | Camera app | Stray Scanner 1.4 | no | |
| iPhone 16 Pro Max | yes | Camera app | Camera app | Stray Scanner 1.4 | no | |
| iPhone 17 | no | Camera app | Camera app | not available | photo and video | the benchmark phone; its files were JPEG photos and H.264 SDR video |
| iPhone Air | no | Camera app | Camera app | not available | no | single rear camera |
| iPhone 17e | no | Camera app | Camera app | not available | no | single rear camera |
| iPhone 17 Pro | yes | Camera app | Camera app | Stray Scanner 1.4 | no | |
| iPhone 17 Pro Max | yes | Camera app | Camera app | Stray Scanner 1.4 | no | |
| iPhone 18 Pro | yes | Camera app | Camera app | Stray Scanner 1.4 | no | Stray Scanner 1.4 not confirmed to run on this model |
| iPhone 18 Pro Max | yes | Camera app | Camera app | Stray Scanner 1.4 | no | Stray Scanner 1.4 not confirmed to run on this model |

## Accuracy each tier delivered

Signed errors against the tape (real room) or the exact truth (synthetic). Thresholds are the gates in `bench/gates.yaml`: walls within 8% (photo), 3% (video) and max(2 cm, 1%) (LiDAR); ceilings within 1.5 cm and floor areas within 2% for LiDAR. Coverage is the share of scored values inside their 90% interval, with the mean interval half-width as a share of the true value. Full tables are in `docs/benchmark_report.md`.

| Tier | Source | n | Wall length | Ceiling height | Floor area | Openings | 90% interval coverage |
|---|---|---|---|---|---|---|---|
| Photo | real room, iPhone 17 photos sent as documents (5712x4284, EXIF kept) | 1 capture of 1 room, 5 photos | 4 of 4 within 8%: -6.4% (4.115 m pair), -5.5% (3.607 m pair) | -19.8% | -11.5% | not measured | 7 of 7; half-width 29% |
| Photo | the same photos after WhatsApp (1280x960, no EXIF) | 1 capture of 1 room | 4 of 4 within 8%: -7.5%, -0.3% | -17.8% | -7.8% | not measured | 7 of 7; half-width 37% |
| Video | real room, iPhone 17 clips sent as documents: 57 s at 1080p, 49 s at 4K | 2 clips of 1 room | 0 of 8 within 3%: 1080p clip -6.3% and -4.5% (4 walls); 4K clip +19.1% to -58.1% (8 walls) | -13.3% (1080p), -31.0% (4K) | -10.5%, -7.3% | not measured | 13 of 14; half-width 29% |
| Video | the same clips after WhatsApp (464x832) | 2 clips of 1 room | 1 of 8 within 3%: first clip -8.7% and -6.8% (4 walls); second clip -2.9% to -42.6% (6 walls) | -14.4%, -26.4% | -14.9%, -25.5% | not measured | 12 of 14; half-width 25% |
| LiDAR | synthetic, ordinary drift (lidar_1 and lidar_2 of 4 properties) | 8 captures, 20 rooms | 149 of 168 within max(2 cm, 1%); median error 0.71 cm | 40 of 40 within 1.5 cm; median 0.11 cm | 38 of 40 within 2%; median 0.27% | 43 of 128 within 2 cm; 61 missed, 3 phantoms | 340 of 390 (87.2%); half-width 2.9% |
| LiDAR | synthetic, strong drift (lidar_drift of 4 properties) | 4 captures, 20 rooms | 60 of 84 within max(2 cm, 1%), 10 not found; median 1.04 cm over the 74 found | 18 of 20 within 1.5 cm, 2 rooms not found | 14 of 20 within 2%, 2 not found | 18 of 64 within 2 cm; 36 missed, 7 phantoms | 139 of 170 (81.8%); half-width 3.5% |
| LiDAR | real iPhone LiDAR: public Stray Scanner office recording, 232 s | 1 capture, no tape readings | not scored | not scored | not scored | not scored | not scored |
| LiDAR | real LiDAR: 3 Stray Scanner recordings of one apartment provided with the problem statement, 37 to 215 s | 3 captures, no measurements | not scored | not scored | not scored | not scored | not scored |

How to read these rows:

- The photo and video rows are one physical room. The two rows per tier hold the same captures before and after WhatsApp's recompression. They show what happened in that bedroom; the n is too small to state an accuracy for the tier in general. In that room every wall except one, every ceiling and every floor area came out short of the tape, and the intervals contain the tape because they are wide (about ±1 m on a 4 m photo wall).
- The LiDAR rows are synthetic. No phone produced those captures, and the numbers assume depth noise, poses and drift like the generator's (`docs/benchmark_report.md`, "What the benchmark contains").
- The office recording has no ground truth. Drift correction accepted 71 of 122 loop closures and cut the spread of floor height across trajectory segments from 47.9 cm to 5.5 cm. The plan came out as one room with 16 walls, 17 openings and 189.6 m2. Nothing here says how close that is to the real office.
- The three provided recordings have no measurements either. c7d28f72c6, which keeps the ceiling in view, gave 7 rooms linked through doorways with ceilings of 3.07 to 3.09 m; in the other two the camera never looks above horizontal, the rooms merge into one outline each and the ceiling is flagged as not observed (`docs/testdata_validation.md`).
- Hand-off changes the input. The WhatsApp rows show what WhatsApp's ordinary media send did to these files. The protocol's hand-off (USB cable, AirDrop, or send as a document) keeps the original files, and the original-file rows were sent as documents.

## What is untested

- Every model except the iPhone 17, which is 15 of the 16 rows. They use the same tools and the same pipeline path, and whether their accuracy matches the iPhone 17 rows has not been measured.
- LiDAR-tier accuracy on any phone. No LiDAR phone was available and Stray Scanner was never run by us. The parser follows the documented Stray Scanner 1.4 format and was checked on the synthetic captures, the public office recording and the three provided recordings; none of the real ones has measurements.
- Stray Scanner 1.4 on the iPhone 18 Pro and 18 Pro Max.
- HEIC photos and HEVC or HDR video from a phone. The iPhone 17 files were JPEG and H.264 SDR. HEVC has been decoded on real files only by the LiDAR tier, from the `rgb.mp4` of the three provided recordings; HEIC and 10-bit HDR decoding is covered only by unit tests on generated files.
- Video held sideways, as the protocol asks. Both benchmark clips were filmed in portrait.
- A measured multi-room capture from a phone. Photo stitching, adjacency and a video drift ablation need one; our only real capture is one bedroom, and the provided 7-room LiDAR recording has no measurements.
- Openings and damage on real captures, which were not measured in the benchmark room.
- Low light and wet floors: no capture of either was made.
