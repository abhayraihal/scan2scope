# Ground-truth protocol

Every number the pipeline reports is scored against a hand measurement taken with these definitions. The definitions matter more than the instrument: a skirting board is 15 to 30 mm thick and a door leaf about 35 mm, so measuring at the wrong spot shifts a reading by more than the 2 cm gates.

## Tools

A steel tape (5 m or longer) or a laser distance meter. Record which one and its model in `ground_truth.yaml`. A laser meter must have its reference set to the rear edge when its back sits against a wall.

## Rules for every reading

- Take each reading twice. If the two differ by more than 5 mm, take a third and record the median.
- Record metres to the millimetre (for example `3.412`).
- Measure after all captures are done, so the tape and helper are not in any capture.

## What to measure

Walls. Length is face to face between the two neighbouring walls, at 1.0 m height (above skirting boards and most furniture; use 1.5 m if furniture is in the way). List walls in order: start with the wall that contains the door you entered through, then go to the next wall on your right as you stand inside the room facing that first wall, and so on around the room. A rectangular room has 4 walls; an L-shaped room has 6.

Ceiling height. Floor to ceiling at three points: the middle of the room, and 0.5 m from two opposite corners. Record all three.

Doors and open passages. Clear width between the two jamb faces (the inner faces of the frame, not the decorative trim) at 1.0 m height, and height from the floor to the underside of the frame head. Record the wall it sits in, its offset from the left end of that wall as seen from inside the room, which room it leads to, and the wall thickness at the jamb.

Windows. Width and height of the opening inside the frame, sill height from the floor, the wall it sits in and its offset from the left end of that wall.

Damage. For each damaged region: class (`water_stain`, `mold`, `crack`, `hole`, `peeling_paint`), the surface (wall id, `ceiling` or `floor`), width and height of its bounding box, its offset from the left end of the wall and its height of the lowest point above the floor. For a crack, also record its length.

Footprint. The harness computes the property footprint as the sum of room floor areas from the wall lengths. For a room that is not a rectangle, also measure one diagonal so the shape is fixed.

## Staged damage (benchmark room)

Two removable classes, both disclosed as staged in the report:

- Water stain: dab cold strong tea or coffee onto a sheet of white paper in an irregular blotch with a darker rim, let it dry, and tape it flat to the wall or ceiling with clear tape.
- Crack: on a strip of white paper or masking tape, draw a thin jagged dark line 30 to 60 cm long, and tape it to the wall, ideally starting at a corner of a door or window frame.

## File format

Copy `bench/templates/ground_truth.yaml` to `bench/data/<property>/ground_truth.yaml` and fill it in. Room ids must match the photo-tier folder names.
