# Capture protocol (one page)

Pick one tier. Install time: none for Photo and Video, under 2 minutes for LiDAR. Capture time: about 1 minute per room.

## Before you start (every tier)

1. Turn on every light. Open every interior door fully. Keep people and pets out of view.
2. Check the phone: Settings > General > About. If Model Name contains "Pro" and iOS Version is 18.6 or later, all three tiers work. Otherwise use Photo or Video.

## Photo tier (any iPhone 15 or newer)

Camera app, Photo mode, tap **1x**, flash off, Live Photo off. Hold the phone sideways (landscape) at chest height and keep it level. Do not zoom.

For each room, including hallways:

1. Stand in each corner with your back to it and take one photo towards the opposite corner. The line where the floor meets the walls and the line where the ceiling meets the walls must both be in the picture.
2. For every doorway in the room, stand about 1.5 m from it inside this room and take one photo through the open door, so the door frame and part of the next room are visible.
3. Keep each room between 2 and 8 photos. 6 to 8 is best.

Hand off, by cable (works on Macs where AirDrop is switched off): connect the iPhone to the Mac with a USB-C cable, unlock it and tap Trust. On the Mac open Image Capture, choose a folder named `Scan` as the destination and import the photos. In `Scan`, make one folder per room named with a two-digit number in walking order and a name (`01 hallway`, `02 kitchen`) and move each room's photos into it; they are numbered in the order they were taken. Where AirDrop is allowed, AirDrop works too. Do not send photos through a messaging app as ordinary photos: that removes the lens information. Send them as a document or file instead. On the Mac run `scan2scope run ~/Scan` (or wherever `Scan` is).

## Video tier (any iPhone 15 or newer)

Once, in Settings > Camera > Record Video: choose 1080p HD at 30 fps (4K at 30 fps also works), turn **off** HDR Video, Enhanced Stabilization and Auto FPS, and turn **on** Lock Camera. In the Camera app: Video mode, tap **1x**, hold the phone sideways.

1. Start recording at the entrance, pointing at a corner of the room so two walls and the floor are in view. Remember this view.
2. In each room, walk slowly (about one step per second) along the walls, 1.5 to 2 m away from them, with the camera pointed at the walls so the floor line and the ceiling line stay in view. Turn slowly at corners.
3. At each doorway, stop 1.5 m in front of the door frame for 2 seconds, then walk through slowly with the camera pointing ahead.
4. After the last room, walk back to the start and end on the same corner view you started with. Stop recording. Expect 45 to 60 seconds per room, under 8 minutes in total.

Hand off: import the video with Image Capture over the USB-C cable (or AirDrop, or a messaging app's send-as-document option). On the Mac run `scan2scope run <video file>`.

## LiDAR tier (Pro models only)

Install **Stray Scanner** (free, by Kenneth Blomqvist) from the App Store and allow camera access. Tap the frame-rate button until it reads **30 fps**. Press record, walk the same route as the video tier (steps 1 to 4 above, including ending where you started), and press stop.

Hand off: connect the USB-C cable, then on the Mac open Finder > the iPhone > Files > Stray Scanner and drag the recording folder to the Mac. Where AirDrop is allowed: open the recording in Stray Scanner > Share > AirDrop (it arrives as a zip). On the Mac run `scan2scope run <recording folder or zip>`.

## Avoid (every tier)

- Fast turns and fast walking: blurred frames lose detail.
- Facing a mirror or a window head-on for more than a second: capture them at an angle.
- Wet or freshly mopped floors: dry them first if you can.
- Dark rooms: if the yellow Night mode icon appears, turn on more lights.
- Closing doors, zooming, switching lenses, Portrait, Panorama or Cinematic mode during the capture.

The pipeline accepts any photos or video and always returns a result, but captures that skip these steps get wider intervals and may place rooms with less confidence.
