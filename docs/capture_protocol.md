# Capture protocol (one page)

Pick one tier. Install time: none for Photo and Video, under 2 minutes for LiDAR. Capture time: about 1 minute per room.

## Before you start (every tier)

1. Turn on every light. Open every interior door fully. Keep people and pets out of view.
2. Check the phone: Settings > General > About. If Model Name contains "Pro" and iOS Version is 18.6 or later, all three tiers work. Otherwise use Photo or Video.

## Photo tier (any iPhone 15 or newer)

Camera app, Photo mode, tap **1x**, flash off, Live Photo off. Hold the phone sideways at chest height, level. Do not zoom. For each room, including hallways:

1. Stand in each corner with your back to it and take one photo towards the opposite corner, with the floor line and the ceiling line both in the picture.
2. For every doorway, stand 1.5 m from it inside this room and take one photo through the open door, showing the frame and part of the next room.
3. Keep each room between 2 and 8 photos; 6 to 8 is best.

## Video tier (any iPhone 15 or newer)

Once, in Settings > Camera > Record Video: 1080p HD at 30 fps; HDR Video, Enhanced Stabilization and Auto FPS **off**; Lock Camera **on**. Camera app: Video mode, **1x**, phone sideways.

1. Start at the entrance, pointing at a room corner so two walls and the floor are in view. Remember this view.
2. In each room, walk slowly (one step per second) along the walls, 1.5 to 2 m from them, keeping the floor line and the ceiling line in view. Turn slowly at corners.
3. Stop 2 seconds in front of each doorway, then walk through with the camera pointing ahead.
4. Walk back to the start and end on the same corner view. About 1 minute per room, under 8 minutes in total.

## LiDAR tier (Pro models only)

Install **Stray Scanner** (free, by Kenneth Blomqvist) and allow camera access. Tap the frame-rate button until it reads **30 fps**, press record, walk the video-tier route (steps 1 to 4), press stop.

## Hand off to the Mac

Connect the iPhone with a USB-C cable, unlock it and tap Trust.

- Photos and video: on the Mac, open Image Capture and import them into a folder named `Scan`. For photos, make one folder per room inside `Scan`, named in walking order (`01 hallway`, `02 kitchen`), and move each room's photos into it (they are numbered in the order taken).
- LiDAR: Finder > the iPhone > Files > Stray Scanner, drag the recording folder to the Mac.
- AirDrop also works where the Mac allows it. Never send photos or video through a messaging app as ordinary media: that strips the lens data. Send as a document or file.

Then run `scan2scope run <Scan folder, video file, or recording folder>`.

## Avoid (every tier)

- Fast turns and fast walking.
- Facing a mirror or a window head-on for more than a second: capture them at an angle.
- Wet or freshly mopped floors: dry them first if you can.
- Dark rooms: if the yellow Night mode icon appears, turn on more lights.
- Closing doors, zooming, switching lenses, Portrait, Panorama or Cinematic mode.

Any photos or video still give a result, but captures that skip these steps get wider intervals.
