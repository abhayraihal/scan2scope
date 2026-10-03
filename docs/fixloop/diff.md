# Fix loop: before and after

Before: `2d6f3b47221b96d9820c9b1d6367006a91376a2f`. After: `9661e655b3f4daca6bf5561c39f0eeb0b28084f1` (9661e65 Test rooms across a hallway whose parallel wall faces are 10 and 14 cm apart).

## Worst gate before the fix

lidar repeatability: 82/236 walls pass, strict reading 44/236; worst synth_0/01 hallway/W1 lidar_1 vs lidar_2: |delta| 205.4 cm (allowed 1.0 cm) (threshold: |delta| <= max(1.0 cm, 0.5%) per wall between captures).

After: fail, 155/232 walls pass, strict reading 92/232; worst synth_0/02 kitchen/W4 lidar_1 vs lidar_drift: |delta| 389.1 cm (allowed 2.0 cm) (improved).

## Gates

improved: 7, unchanged: 27

| Tier | Gate | Before | After | Change | Measured before | Measured after | Threshold |
|---|---|---|---|---|---|---|---|
| photo | result_produced | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | every capture returns a scored result with geometry |
| photo | wall_length | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 8.0% of GT on every wall |
| photo | ceiling_height | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 8.0% of GT in every room |
| photo | ceiling_spread | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | max - min across captures <= 1.0 cm per room |
| photo | opening_width | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 8.0% of GT on >= 85.0% of openings (misses and phantoms fail) |
| photo | floor_area | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 16.0% of GT in every room |
| photo | footprint | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 8.0% of GT per multi-room capture |
| photo | stitch | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | adjacency exact, max pairwise overlap <= 0.05 m2, footprint within 8.0% |
| photo | repeatability | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures |
| photo | repeat_structure | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | repeat captures find the same walls and openings in each room |
| photo | calibration | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | 95.0% Clopper-Pearson CI of coverage contains 0.90 |
| photo | confident_garbage | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | no miss larger than 2 x half-width |
| video | result_produced | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | every capture returns a scored result with geometry |
| video | wall_length | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 3.0% of GT on every wall |
| video | ceiling_height | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 3.0% of GT in every room |
| video | ceiling_spread | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | max - min across captures <= 1.0 cm per room |
| video | opening_width | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 3.0% of GT on >= 85.0% of openings (misses and phantoms fail) |
| video | floor_area | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|err\| <= 6.0% of GT in every room |
| video | repeatability | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures |
| video | repeat_structure | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | repeat captures find the same walls and openings in each room |
| video | drift_ablation | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | drift correction on for every multi-room capture, with an off run for the ablation |
| video | calibration | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | 95.0% Clopper-Pearson CI of coverage contains 0.90 |
| video | confident_garbage | n.a. | n.a. | unchanged | no captures in this tier | no captures in this tier | no miss larger than 2 x half-width |
| lidar | result_produced | pass | pass | unchanged | 12/12 pass | 12/12 pass | every capture returns a scored result with geometry |
| lidar | wall_length | fail | fail | improved | worst 102.40x allowed (synth_0/lidar_1/01 hallway/W1), 147/252 pass, 8 missing | worst 98.05x allowed (synth_0/lidar_drift/02 kitchen/W4), 209/252 pass, 10 missing | \|err\| <= max(2.0 cm, 1.0% of GT) on every wall |
| lidar | ceiling_height | fail | fail | unchanged | worst 0.5 cm (synth_3/lidar_drift/05 bedroom), 58/60 pass, 2 missing | worst 0.5 cm (synth_3/lidar_drift/04 kitchen), 58/60 pass, 2 missing | \|err\| <= 1.5 cm in every room |
| lidar | ceiling_spread | pass | pass | unchanged | 20/20 pass; synth_0/01 hallway: spread 0.1 cm over 2 captures; synth_0/02 kitchen: spread 0.2 cm over 3 captures; synth_0/03 living: spread 0.3 cm over 3 captures (+17 more) | 20/20 pass; synth_0/01 hallway: spread 0.1 cm over 2 captures; synth_0/02 kitchen: spread 0.3 cm over 3 captures; synth_0/03 living: spread 0.3 cm over 3 captures (+17 more) | max - min across captures <= 1.0 cm per room |
| lidar | opening_width | fail | fail | improved | 62/209 pass (29.7%); 110 missed, 17 phantom, 20 out of tolerance | 61/202 pass (30.2%); 97 missed, 10 phantom, 34 out of tolerance | \|err\| <= 2.0 cm on >= 85.0% of openings (misses and phantoms fail) |
| lidar | floor_area | fail | fail | improved | worst 65.7% (synth_0/lidar_drift/02 kitchen), 34/60 pass, 2 missing | worst 62.9% (synth_0/lidar_drift/02 kitchen), 52/60 pass, 2 missing | \|err\| <= 2.0% of GT in every room |
| lidar | repeatability | fail | fail | improved | 82/236 walls pass, strict reading 44/236; worst synth_0/01 hallway/W1 lidar_1 vs lidar_2: \|delta\| 205.4 cm (allowed 1.0 cm) | 155/232 walls pass, strict reading 92/232; worst synth_0/02 kitchen/W4 lidar_1 vs lidar_drift: \|delta\| 389.1 cm (allowed 2.0 cm) | \|delta\| <= max(1.0 cm, 0.5%) per wall between captures |
| lidar | repeat_structure | fail | fail | improved | 28/56 pass; synth_0/01 hallway lidar_1 vs lidar_2: walls 8/4, openings 2/2; synth_0/02 kitchen lidar_1 vs lidar_2: walls 8/4, openings 1/2; synth_0/02 kitchen lidar_1 vs lidar_drift: walls 8/22, openings 1/3 (+25 more) | 36/56 pass; synth_0/01 hallway lidar_1 vs lidar_2: walls 4/4, openings 4/2; synth_0/02 kitchen lidar_1 vs lidar_2: walls 8/4, openings 1/1; synth_0/02 kitchen lidar_1 vs lidar_drift: walls 8/20, openings 1/4 (+17 more) | repeat captures find the same walls and openings in each room |
| lidar | drift_ablation | pass | pass | unchanged | 12/12 pass; synth_0/lidar_1: footprint err on 1.2%, off 0.2%; synth_0/lidar_2: footprint err on 1.0%, off 0.8%; synth_0/lidar_drift: footprint err on -0.4%, off -2.4% (+9 more) | 12/12 pass; synth_0/lidar_1: footprint err on -1.1%, off -1.0%; synth_0/lidar_2: footprint err on -0.2%, off -0.1%; synth_0/lidar_drift: footprint err on -2.8%, off -2.4% (+9 more) | drift correction on for every multi-room capture, with an off run for the ablation |
| lidar | calibration | fail | fail | improved | 325/536 covered (60.6%), CI [56.4%, 64.8%], 20 rooms | 410/560 covered (73.2%), CI [69.3%, 76.8%], 20 rooms | 95.0% Clopper-Pearson CI of coverage contains 0.90 |
| lidar | confident_garbage | fail | fail | improved | 152 of 536 values, worst 139.3x half-width (synth_1/lidar_drift/05 kitchen/W3) | 105 of 560 values, worst 132.0x half-width (synth_1/lidar_1/05 kitchen/W3) | no miss larger than 2 x half-width |

## Changed files

```
 docs/fixloop/before/benchmark_report.md |  537 ++++++
 docs/fixloop/before/gates.json          | 2861 +++++++++++++++++++++++++++++++
 docs/fixloop/declaration.md             |   44 +
 scripts/fixloop.sh                      |    7 +-
 src/scan2scope/layout/core.py           |   55 +-
 src/scan2scope/layout/refine.py         |  429 +++++
 tests/layout_fixtures.py                |   35 +
 tests/test_layout_multi_room.py         |   28 +
 tests/test_layout_refine.py             |  134 ++
 9 files changed, 4124 insertions(+), 6 deletions(-)
```

## Source diff (src/)

```diff
diff --git a/src/scan2scope/layout/core.py b/src/scan2scope/layout/core.py
index 03d59e0..57af5d0 100644
--- a/src/scan2scope/layout/core.py
+++ b/src/scan2scope/layout/core.py
@@ -5,13 +5,17 @@ from __future__ import annotations
 import logging
 import re
 import time
+from collections.abc import Callable
 from dataclasses import dataclass
 
 import numpy as np
+import shapely
+from shapely.geometry import Polygon
 
 from scan2scope.layout import cells as C
 from scan2scope.layout import floor_ceiling as FC
 from scan2scope.layout import openings as OP
+from scan2scope.layout import refine as R
 from scan2scope.layout import walls as W
 from scan2scope.types import Adjacency, Measurement, Opening, Plan, Room, Scene, Wall
 
@@ -325,6 +329,27 @@ def _face_evidence(lines: list[W.WallLine], axis: int, coord: float, n_sign: int
     return _Face(int(m.sum()), rms, q.sigma, flags, q.slope, q.t_mid)
 
 
+def _claimable(cx: C.Complex, room_of_cell: np.ndarray, k: int) -> Callable[[Polygon], bool]:
+    """Test for a region of the plan: True when no room other than room k holds any of its cells."""
+
+    def ok(region: Polygon) -> bool:
+        return bool(np.isin(room_of_cell[_cells_in(cx, region)], (-1, k)).all())
+
+    return ok
+
+
+def _cells_in(cx: C.Complex, region: Polygon) -> tuple[np.ndarray, np.ndarray]:
+    """Indices of the cells whose centres lie inside the region."""
+    x0, y0, x1, y1 = region.bounds
+    i0, i1 = max(int(np.searchsorted(cx.xs, x0, "right")) - 1, 0), int(np.searchsorted(cx.xs, x1))
+    j0, j1 = max(int(np.searchsorted(cx.ys, y0, "right")) - 1, 0), int(np.searchsorted(cx.ys, y1))
+    ii, jj = np.meshgrid(np.arange(i0, min(i1, cx.shape[0])), np.arange(j0, min(j1, cx.shape[1])),
+                         indexing="ij")
+    ii, jj = ii.ravel(), jj.ravel()
+    inside = shapely.contains_xy(region, 0.5 * (cx.xs[ii] + cx.xs[ii + 1]), 0.5 * (cx.ys[jj] + cx.ys[jj + 1]))
+    return ii[inside], jj[inside]
+
+
 def _cell_of(cx: C.Complex, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
     nx, ny = cx.shape
     i = np.searchsorted(cx.xs, xy[:, 0]) - 1
@@ -348,12 +373,14 @@ def _assemble(scene: Scene, d: _Data, lines: list[W.WallLine], cx: C.Complex, ro
     else:
         up, down = d.N[:, 2] > FC.HORIZONTAL_NZ, d.N[:, 2] < -FC.HORIZONTAL_NZ
     cam_room = np.where(d.cam_ok, _room_at(cx, room_of_cell, d.cams[:, :2]), -1) if len(d.cams) else []
+    pts = R.WallPoints.build(d.P, d.N, d.wp, d.wf, fc.floor.z - 0.3, fc.ceiling.z + 0.3)
     rooms: list[Room] = []
     polys_m: list[np.ndarray] = []
+    n_filled = n_steps = 0
     for k, g in enumerate(groups):
         rid = f"R{k + 1}"
         rflags: list[str] = []
-        poly_m, hole = C.cells_polygon(cx, [c for r in g for c in regs[r].cells])
+        poly_c, hole = C.cells_polygon(cx, [c for r in g for c in regs[r].cells])
         if hole > 1e-6:
             rflags.append(f"holes_filled:{hole:.2f}")
         fm, cm = (vroom == k) & up, (vroom == k) & down
@@ -375,6 +402,20 @@ def _assemble(scene: Scene, d: _Data, lines: list[W.WallLine], cx: C.Complex, ro
                   "floor_tilt": fl.tilt, "ceiling_tilt": ce.tilt, "observed_fraction": ceil_obs,
                   "ceiling_observed": bool(ce.observed), "noise_sigma": sigma}
 
+        # the room's own faces: furniture notches filled, edges refitted, short steps removed
+        out = R.room_outline(poly_c, pts, fl.z, ce.z, sigma, _claimable(cx, room_of_cell, k))
+        if out is None:
+            poly_m, fits = poly_c, None
+            rflags.append("outline_not_refined")
+        else:
+            poly_m, fits = out.polygon, [e.fit for e in out.edges]
+            for region in out.filled:
+                room_of_cell[_cells_in(cx, region)] = k
+            if out.filled:
+                rflags.append(f"furniture_filled:{len(out.filled)}")
+            n_filled += len(out.filled)
+            n_steps += out.steps
+
         K = len(poly_m)
         edges = []
         for e in range(K):
@@ -383,7 +424,11 @@ def _assemble(scene: Scene, d: _Data, lines: list[W.WallLine], cx: C.Complex, ro
             axis, coord, t0, t1 = (1, p[1], p[0], q[0]) if abs(dx) >= abs(dy) else (0, p[0], p[1], q[1])
             nin = np.array([-dy, dx]) / max(float(np.hypot(dx, dy)), 1e-12)
             n_sign = int(np.sign(nin[axis])) or 1
-            face = _face_evidence(lines, axis, coord, n_sign, min(t0, t1), max(t0, t1))
+            if fits is None:
+                face = _face_evidence(lines, axis, coord, n_sign, min(t0, t1), max(t0, t1))
+            else:
+                fit = fits[e]
+                face = _Face(fit.n, fit.rms, fit.sigma, fit.flags, fit.slope, fit.t_mid)
             frame = OP.WallFrame(axis, float(coord), n_sign, float(t0), 1 if t1 > t0 else -1,
                                  float(abs(t1 - t0)), fl.z, ce.z, face.slope, face.t_mid)
             wa = OP.analyze_wall(frame, d.P, d.N, d.wp, d.wn, d.O, d.E, face.sigma if face.n >= 30 else sigma)
@@ -405,7 +450,8 @@ def _assemble(scene: Scene, d: _Data, lines: list[W.WallLine], cx: C.Complex, ro
             walls.append(Wall(wid, rid, Rw @ p, Rw @ q, length, _m(hgt, "height", **lev_ev), Rw @ nin,
                               float(np.clip(wa.observed_fraction, 0, 1)),
                               {"n_points": face.n, "fit_rms": face.rms, "face_sigma": face.sigma,
-                               "n_plane_points": wa.n_points}, list(face.flags) + wa.flags))
+                               "n_plane_points": wa.n_points, "refined": bool(fits and fits[e].refined)},
+                              list(face.flags) + wa.flags))
             for f in wa.openings:
                 t = frame.t_start + frame.t_dir * 0.5 * (f.u0 + f.u1)
                 c_m = np.array([frame.coord, t]) if frame.axis == 0 else np.array([t, frame.coord])
@@ -439,8 +485,9 @@ def _assemble(scene: Scene, d: _Data, lines: list[W.WallLine], cx: C.Complex, ro
                           {"n_cells": sum(len(regs[r].cells) for r in g), "coverage": regs[g[0]].coverage,
                            "boundary_support": regs[g[0]].support,
                            "n_cameras": int(sum(regs[r].n_cams for r in g)),
-                           "polygon_manhattan": poly_m.tolist()}))
+                           "polygon_manhattan": poly_m.tolist(), "polygon_cells": poly_c.tolist()}))
         polys_m.append(poly_m)
+    meta["outline"] = {"furniture_filled": n_filled, "steps_removed": n_steps}
 
     adjacency = _connect(rooms, cx, room_of_cell)
     allp = np.concatenate(polys_m)
diff --git a/src/scan2scope/layout/refine.py b/src/scan2scope/layout/refine.py
new file mode 100644
index 0000000..94114ca
--- /dev/null
+++ b/src/scan2scope/layout/refine.py
@@ -0,0 +1,429 @@
+"""Room outlines from each room's own wall faces: furniture filter, per-room wall refinement, regularisation.
+
+The cell complex puts every room edge on a wall line fitted once for the whole property. Two parallel faces of
+different rooms a few centimetres apart then share one line placed between them, an outline steps between the
+two faces of a wall, and furniture fronts cut notches into rooms. Per room:
+
+1. Furniture filter: an edge whose face points never reach the band CEILING_BAND below the room's ceiling is a
+   furniture front or side. A run of such edges between two walls is replaced by those walls when that adds
+   floor no other room holds, at most FURNITURE_DEPTH deep. This runs on the cell lines before refinement, and
+   once more on the refined outline for notches whose walls sat on the two faces of one wall body.
+2. Refinement: each edge is refitted from the wall points within BAND of it that face into the room and lie
+   along it: a trimmed mean of their offset, started at the densest offset and iterated, or a slanted line
+   when that explains the points clearly better (the edge then sits where the line crosses its midpoint). An
+   edge with fewer than MIN_POINTS such points keeps its cell-complex position and is flagged
+   wall_not_refined. Corners are where consecutive refined lines meet.
+3. Regularisation: an edge shorter than STEP_MAX between two parallel edges goes. Between edges that run the
+   same way it is a step, and both snap to the face with more support over their joint extent and merge into
+   one edge; between edges that run opposite ways it is the end of a strip that thin (a wall body, a slot),
+   and the strip is cut back. It also runs once on the cell lines, so the walls around a notch line up first.
+
+Coordinates are in the Manhattan frame. An outline is rectilinear and counter-clockwise, held as a cyclic list
+of edges whose axes alternate; vertex k is where edge k - 1 meets edge k.
+"""
+
+from __future__ import annotations
+
+import logging
+from collections.abc import Callable
+from dataclasses import dataclass, field
+
+import numpy as np
+from scipy.ndimage import gaussian_filter1d
+from shapely.geometry import Polygon
+
+from scan2scope.layout import walls as W
+from scan2scope.layout.floor_ceiling import robust_sigma
+
+log = logging.getLogger("scan2scope.layout")
+
+BAND = 0.15  # wall points this close to an edge are candidates for its face
+STEP_MAX = 0.25  # shorter edges between parallel edges are removed
+CEILING_BAND = 0.5  # a face whose points never come this close to the ceiling is furniture
+MIN_POINTS = 30  # fewer face voxels than this: the edge keeps its cell-complex position
+END_MARGIN = 0.03  # voxels this close to either end of an edge are left out (corner voxels mix two faces)
+WIN_MIN, WIN_MAX = 0.015, 0.08  # trimming window of the face fit
+MODE_RES = 0.005
+TOP_QUANTILE = 0.99
+COLLINEAR_TOL = 0.05  # walls this close on one line are one wall when the notch between them is filled
+FURNITURE_DEPTH = 1.0  # deepest notch that is filled
+FURNITURE_AREA = 0.25  # largest filled area as a share of the room
+SLANT_FLAG = float(np.tan(np.radians(1.0)))
+
+
+@dataclass
+class WallPoints:
+    """Wall voxels per face direction (walls.DIRS), sorted by the coordinate across the face."""
+
+    v: list[np.ndarray] = field(default_factory=list)  # across the face
+    t: list[np.ndarray] = field(default_factory=list)  # along the face
+    z: list[np.ndarray] = field(default_factory=list)
+    wa: list[np.ndarray] = field(default_factory=list)  # one vote per voxel: mode, support, heights
+    wf: list[np.ndarray] = field(default_factory=list)  # fit weight
+
+    @classmethod
+    def build(cls, P: np.ndarray, N: np.ndarray, w_area: np.ndarray, w_fit: np.ndarray, z_lo: float,
+              z_hi: float) -> WallPoints:
+        m = np.flatnonzero((np.abs(N[:, 2]) < W.WALL_NZ) & (P[:, 2] > z_lo) & (P[:, 2] < z_hi))
+        codes = W.direction_codes(N[m, :2])
+        out = cls()
+        for k, (axis, _) in enumerate(W.DIRS):
+            idx = m[codes == k]
+            idx = idx[np.argsort(P[idx, axis], kind="stable")]
+            out.v.append(P[idx, axis])
+            out.t.append(P[idx, 1 - axis])
+            out.z.append(P[idx, 2])
+            out.wa.append(w_area[idx])
+            out.wf.append(w_fit[idx])
+        return out
+
+    def select(self, axis: int, sign: int, lo: float, hi: float, t_lo: float, t_hi: float, z_lo: float,
+               z_hi: float) -> tuple[np.ndarray, ...]:
+        """v, t, z, wa, wf of the voxels facing sign along axis with v in [lo, hi] and t, z in range."""
+        k = W.DIRS.index((axis, sign))
+        i0, i1 = np.searchsorted(self.v[k], [lo, hi])
+        t, z = self.t[k][i0:i1], self.z[k][i0:i1]
+        m = (t >= t_lo) & (t <= t_hi) & (z > z_lo) & (z < z_hi)
+        return self.v[k][i0:i1][m], t[m], z[m], self.wa[k][i0:i1][m], self.wf[k][i0:i1][m]
+
+
+@dataclass
+class FaceFit:
+    """The face an edge lies on: coord + slope * (t - t_mid) across the edge, from n inlier voxels."""
+
+    coord: float
+    slope: float = 0.0
+    t_mid: float = 0.0
+    n: int = 0
+    rms: float = 0.0
+    sigma: float = 0.0
+    mass: float = 0.0
+    refined: bool = False
+    flags: tuple[str, ...] = ()
+
+
+@dataclass
+class Edge:
+    axis: int  # 0: the line x = coord, 1: the line y = coord
+    coord: float
+    sign: int  # +1 when the room lies on the positive side of the line
+    fit: FaceFit | None = None
+    kind: str = "unknown"  # furniture filter: wall | furniture | unknown
+
+
+@dataclass
+class Outline:
+    polygon: np.ndarray
+    edges: list[Edge]  # edge k runs from polygon[k] to polygon[k + 1]
+    filled: list[Polygon] = field(default_factory=list)  # floor added behind furniture
+    steps: int = 0  # steps and strips removed
+
+
+def edges_of(poly: np.ndarray) -> list[Edge] | None:
+    """Edges of a counter-clockwise rectilinear polygon without collinear vertices, else None."""
+    K = len(poly)
+    if K < 4 or K % 2:
+        return None
+    out = []
+    for e in range(K):
+        d = np.asarray(poly[(e + 1) % K], float) - np.asarray(poly[e], float)
+        axis = 1 if abs(d[0]) >= abs(d[1]) else 0
+        run = d[1 - axis]
+        if abs(d[axis]) > 1e-9 or abs(run) < 1e-9:
+            return None
+        sign = int(np.sign(run)) if axis == 1 else -int(np.sign(run))  # inward normal (-dy, dx)
+        out.append(Edge(axis, float(poly[e][axis]), sign))
+    if any(out[k].axis == out[k - 1].axis for k in range(K)):
+        return None
+    return out
+
+
+def outline(edges: list[Edge]) -> np.ndarray:
+    P = np.empty((len(edges), 2))
+    for k, e in enumerate(edges):
+        prev = edges[k - 1].coord
+        P[k] = (e.coord, prev) if e.axis == 0 else (prev, e.coord)
+    return P
+
+
+def _span(edges: list[Edge], k: int) -> tuple[float, float]:
+    """Along-line coordinates of the start and the end of edge k."""
+    return edges[k - 1].coord, edges[(k + 1) % len(edges)].coord
+
+
+def _length(edges: list[Edge], k: int) -> float:
+    a, b = _span(edges, k)
+    return abs(b - a)
+
+
+def consistent(edges: list[Edge]) -> bool:
+    """Axes alternate, every edge runs the way its inward side requires, and the outline is simple."""
+    K = len(edges)
+    if K < 4 or K % 2:
+        return False
+    for k, e in enumerate(edges):
+        if edges[k - 1].axis == e.axis:
+            return False
+        a, b = _span(edges, k)
+        if abs(b - a) < 1e-4:
+            return False
+        if (int(np.sign(b - a)) if e.axis == 1 else -int(np.sign(b - a))) != e.sign:
+            return False
+    poly = Polygon(outline(edges))
+    return bool(poly.is_valid and poly.exterior.is_ccw)
+
+
+def _quantile(x: np.ndarray, w: np.ndarray, q: float) -> float:
+    o = np.argsort(x)
+    cw = np.cumsum(w[o])
+    return float(x[o][min(int(np.searchsorted(cw, q * cw[-1])), len(x) - 1)])
+
+
+def _mode(v: np.ndarray, w: np.ndarray, c0: float, sigma: float) -> float:
+    """Densest offset within BAND of c0 (weighted histogram smoothed at the noise level)."""
+    nb = round(2 * BAND / MODE_RES)
+    i = np.clip(((v - (c0 - BAND)) / MODE_RES).astype(np.int64), 0, nb - 1)
+    h = np.bincount(i, weights=w, minlength=nb).astype(float)
+    sm = gaussian_filter1d(h, max(sigma, 0.01) / MODE_RES, mode="constant")
+    return c0 - BAND + (int(np.argmax(sm)) + 0.5) * MODE_RES
+
+
+def _trim(v: np.ndarray, t: np.ndarray, w: np.ndarray, a: float, b: float, tm: float,
+          sigma: float) -> tuple[float, float, np.ndarray]:
+    """Trimmed mean offset of the line a + b (t - tm): window 2.5 robust sigmas, iterated."""
+    s = max(sigma, WIN_MIN / 2.5)
+    for _ in range(6):
+        r = v - a - b * (t - tm)
+        m = np.abs(r) < float(np.clip(2.5 * s, WIN_MIN, WIN_MAX))
+        if m.sum() < 3 or w[m].sum() <= 0:
+            break
+        da = float(np.average(r[m], weights=w[m]))
+        a += da
+        s = robust_sigma(r[m] - da, w[m])
+    r = v - a - b * (t - tm)
+    return a, s, np.abs(r) < float(np.clip(2.5 * s, WIN_MIN, WIN_MAX))
+
+
+@dataclass
+class _Room:
+    """The wall points within one room's height range and the capture's wall noise."""
+
+    pts: WallPoints
+    z0: float
+    z1: float
+    sigma: float
+
+    def band(self, e: Edge, t0: float, t1: float, center: float | None = None,
+             half: float = BAND) -> tuple[np.ndarray, ...]:
+        lo, hi = min(t0, t1), max(t0, t1)
+        m = min(END_MARGIN, 0.25 * (hi - lo))
+        c = e.coord if center is None else center
+        return self.pts.select(e.axis, e.sign, c - half, c + half, lo + m, hi - m, self.z0, self.z1)
+
+    def fit(self, e: Edge, t0: float, t1: float, start: float | None = None) -> FaceFit:
+        """Face of edge e over [t0, t1], from the densest offset within BAND or else from `start`."""
+        tm = 0.5 * (t0 + t1)
+        c0 = e.coord if start is None else start
+        v, t, _, wa, wf = self.band(e, t0, t1, c0)
+        if len(v) < MIN_POINTS:
+            sparse = "wall_face_sparse" if len(v) else "wall_face_missing"
+            return FaceFit(e.coord, t_mid=tm, n=len(v), sigma=self.sigma, flags=(sparse, "wall_not_refined"))
+        c = _mode(v, wa, c0, self.sigma) if start is None else c0
+        a, s, inl = _trim(v, t, wf, c, 0.0, tm, self.sigma)
+        slope = 0.0
+        if inl.sum() >= 20 and (len(v) > 1.15 * inl.sum() or s > 1.5 * self.sigma):
+            # a wall a few degrees off the Manhattan axis spreads over many offsets: try one rotated line
+            a2, b2, tm2 = W._slant_fit(t, v, wf, a, max(2.5 * self.sigma, 0.02))
+            a2 += b2 * (tm - tm2)
+            r2 = v - a2 - b2 * (t - tm)
+            inl2 = np.abs(r2) < max(float(np.clip(2.5 * s, WIN_MIN, WIN_MAX)), 2.5 * self.sigma)
+            s2 = robust_sigma(r2[inl2], wf[inl2]) if inl2.sum() > 10 else s
+            if W.SLANT_MIN < abs(b2) <= W.SLANT_MAX and (inl2.sum() >= 1.15 * inl.sum() or s2 < 0.7 * s):
+                a, s, inl = _trim(v, t, wf, a2, b2, tm, self.sigma)
+                slope = b2
+        n = int(inl.sum())
+        if n < MIN_POINTS:
+            return FaceFit(e.coord, t_mid=tm, n=n, sigma=self.sigma,
+                           flags=("wall_face_sparse", "wall_not_refined"))
+        r = v[inl] - a - slope * (t[inl] - tm)
+        rms = float(np.sqrt(np.average(r ** 2, weights=wf[inl])))
+        flags = (f"wall_slanted:{np.degrees(np.arctan(slope)):.1f}deg",) if abs(slope) > SLANT_FLAG else ()
+        return FaceFit(float(a), float(slope), float(tm), n, rms, float(s), float(wa[inl].sum()), True, flags)
+
+    def support(self, e: Edge, t0: float, t1: float) -> float:
+        """Voxels on the face of e (its fitted line, else its coordinate) over [t0, t1]."""
+        f = e.fit
+        if f is not None and f.refined:
+            a, b, tm, s = f.coord, f.slope, f.t_mid, f.sigma
+        else:
+            a, b, tm, s = e.coord, 0.0, 0.5 * (t0 + t1), self.sigma
+        win = float(np.clip(2.5 * s, WIN_MIN, WIN_MAX))
+        v, t, _, wa, _ = self.band(e, t0, t1, a, win + abs(b) * abs(t1 - t0))
+        return float(wa[np.abs(v - a - b * (t - tm)) < win].sum())
+
+    def kind(self, e: Edge, t0: float, t1: float, ceil_z: float) -> str:
+        """furniture when the face points never reach the band CEILING_BAND below the ceiling."""
+        _, _, z, wa, _ = self.band(e, t0, t1)
+        if len(z) < MIN_POINTS:
+            return "unknown"
+        return "furniture" if _quantile(z, wa, TOP_QUANTILE) < ceil_z - CEILING_BAND else "wall"
+
+
+def _furniture_runs(edges: list[Edge]) -> list[tuple[int, int, list[int]]]:
+    """(wall before, wall after, edges between) for each run between walls that holds a furniture edge."""
+    K = len(edges)
+    walls = [k for k, e in enumerate(edges) if e.kind == "wall"]
+    out = []
+    for i, a in enumerate(walls):
+        b = walls[(i + 1) % len(walls)]
+        run = [(a + j) % K for j in range(1, (b - a) % K or K)]
+        if run and b != a and any(edges[k].kind == "furniture" for k in run):
+            out.append((a, b, run))
+    return out
+
+
+def _bridge(edges: list[Edge], a: int, b: int, run: list[int], room: _Room) -> list[Edge] | None:
+    """The outline without `run`: wall a continues straight into wall b, or meets it at a corner."""
+    ea, eb = edges[a], edges[b]
+    drop = set(run)
+    if ea.axis == eb.axis:
+        if ea.sign != eb.sign or abs(ea.coord - eb.coord) > COLLINEAR_TOL:
+            return None
+        drop.add(b)
+        t0, t1 = _span(edges, a)[0], _span(edges, b)[1]
+        if abs(ea.coord - eb.coord) > 1e-9:
+            # one wall on two nearby lines: keep the line with more face points along the joint extent
+            best = max((ea, eb), key=lambda q: room.support(q, t0, t1))
+            ea = Edge(ea.axis, best.coord, ea.sign, kind="wall")
+    out = [ea if k == a else e for k, e in enumerate(edges) if k not in drop]
+    if len(out) < 4 or not consistent(out):
+        return None
+    # a wall left shorter than a step was the notch's own side, and it reaches the ceiling band: no furniture
+    if any((e is ea or e is eb) and _length(out, k) < STEP_MAX for k, e in enumerate(out)):
+        return None
+    return out
+
+
+def _thin(region: Polygon, width: float) -> bool:
+    return all(min(g.bounds[2] - g.bounds[0], g.bounds[3] - g.bounds[1]) <= width + 1e-9
+               for g in getattr(region, "geoms", [region]) if g.area > 1e-9)
+
+
+def _added(edges: list[Edge], new: list[Edge]) -> Polygon | None:
+    """The floor a fill adds, at most FURNITURE_AREA of the room and FURNITURE_DEPTH deep; it may give up
+    only slivers where the two walls it joins were not exactly on one line."""
+    p0, p1 = Polygon(outline(edges)), Polygon(outline(new))
+    if p1.area <= p0.area + 1e-6 or p1.area - p0.area > FURNITURE_AREA * p0.area:
+        return None
+    if not _thin(p0.difference(p1), COLLINEAR_TOL):
+        return None
+    region = p1.difference(p0)
+    return region if _thin(region, FURNITURE_DEPTH) else None
+
+
+def _fill(edges: list[Edge], room: _Room, ceil_z: float,
+          claimable: Callable[[Polygon], bool]) -> tuple[list[Edge], list[Polygon]]:
+    """Replace runs of furniture edges between two walls by the walls behind them."""
+    for k, e in enumerate(edges):
+        e.kind = room.kind(e, *_span(edges, k), ceil_z)
+    filled: list[Polygon] = []
+    while len(edges) > 4:
+        for a, b, run in _furniture_runs(edges):
+            new = _bridge(edges, a, b, run, room)
+            region = None if new is None else _added(edges, new)
+            if region is not None and claimable(region):
+                log.debug("furniture notch filled: %.2f m2", region.area)
+                filled.append(region)
+                edges = new
+                break
+        else:
+            break
+    return edges, filled
+
+
+def _merge_step(edges: list[Edge], e: int, room: _Room, refit: bool) -> list[Edge]:
+    """Remove the step edge e: its neighbours snap to the better-supported face and become one edge,
+    refitted over the joint extent when `refit`."""
+    K = len(edges)
+    ia, ib = (e - 1) % K, (e + 1) % K
+    a, b = edges[ia], edges[ib]
+    t0, t1 = edges[(e - 2) % K].coord, edges[(e + 2) % K].coord
+    sa, sb = room.support(a, t0, t1), room.support(b, t0, t1)
+    best = a if sa >= sb else b
+    merged = Edge(a.axis, best.coord, a.sign, kind=a.kind)
+    if refit:
+        merged.fit = room.fit(merged, t0, t1, start=best.coord)
+        if merged.fit.refined:
+            merged.coord = merged.fit.coord
+    log.debug("step %.3f m removed: faces %.3f (support %.0f) and %.3f (%.0f) -> %.3f",
+              abs(b.coord - a.coord), a.coord, sa, b.coord, sb, merged.coord)
+    return [merged if k == ia else q for k, q in enumerate(edges) if k not in (e, ib)]
+
+
+def _cut_strip(edges: list[Edge], e: int) -> list[Edge]:
+    """Remove a strip narrower than STEP_MAX whose end is edge e (its sides e - 1 and e + 1 run opposite
+    ways): the end and the shorter side go, and the edge beyond the shorter side extends to the longer
+    side."""
+    K = len(edges)
+    la = abs(edges[e].coord - edges[(e - 2) % K].coord)
+    lb = abs(edges[(e + 2) % K].coord - edges[e].coord)
+    drop = {(e - 1) % K, e} if la <= lb else {e, (e + 1) % K}
+    log.debug("strip %.3f m wide, %.3f m long removed", abs(edges[(e + 1) % K].coord - edges[e - 1].coord),
+              min(la, lb))
+    return [q for k, q in enumerate(edges) if k not in drop]
+
+
+def _regularise(edges: list[Edge], room: _Room, refit: bool) -> tuple[list[Edge], int]:
+    """Remove edges shorter than STEP_MAX between parallel edges, shortest first: a step between edges that
+    run the same way, or the end of a thin strip between edges that run opposite ways."""
+    n = 0
+    while len(edges) > 4:
+        K = len(edges)
+        step, e = min((abs(edges[(k + 1) % K].coord - edges[k - 1].coord), k) for k in range(K))
+        if step >= STEP_MAX:
+            break
+        if edges[e - 1].sign == edges[(e + 1) % K].sign:
+            edges = _merge_step(edges, e, room, refit)
+        else:
+            edges = _cut_strip(edges, e)
+        n += 1
+    return edges, n
+
+
+def _refit_all(edges: list[Edge], room: _Room, keep_face: bool) -> None:
+    """Fit every edge over its current extent, then move each onto its face (all at once)."""
+    fits = [room.fit(e, *_span(edges, k), start=e.coord if keep_face else None) for k, e in enumerate(edges)]
+    for e, f in zip(edges, fits, strict=True):
+        e.fit = f
+        if f.refined:
+            e.coord = f.coord
+
+
+def room_outline(poly: np.ndarray, pts: WallPoints, floor_z: float, ceil_z: float, sigma: float,
+                 claimable: Callable[[Polygon], bool]) -> Outline | None:
+    """The room's outline from its cell polygon. None when the polygon is not rectilinear or the refined
+    outline is not a simple counter-clockwise polygon; the caller then keeps the cell polygon.
+
+    claimable(region) is False for plan regions that hold cells of another room.
+    """
+    edges = edges_of(poly)
+    if edges is None:
+        return None
+    room = _Room(pts, floor_z + 0.05, ceil_z - 0.03, sigma)
+    edges, steps = _regularise(edges, room, refit=False)
+    edges, filled = _fill(edges, room, ceil_z, claimable)
+    _refit_all(edges, room, keep_face=False)
+    edges, n = _regularise(edges, room, refit=True)
+    steps += n
+    # a notch between walls on the two faces of one wall body lines up only once the walls are refitted
+    edges, late = _fill(edges, room, ceil_z, claimable)
+    if late:
+        edges, n = _regularise(edges, room, refit=True)
+        steps += n
+    _refit_all(edges, room, keep_face=True)
+    if not consistent(edges):
+        log.debug("refined outline is not simple; keeping the cell polygon")
+        return None
+    P = outline(edges)
+    start = int(np.lexsort((P[:, 0], P[:, 1]))[0])
+    return Outline(np.roll(P, -start, axis=0), edges[start:] + edges[:start], filled + late, steps)
```
