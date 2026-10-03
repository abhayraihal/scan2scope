"""Hand-built two-room property used by test_uncertainty.py and test_output.py.

R1 (4 x 3 m) and R2 (3 x 3 m) share a 0.1 m wall with a door through it; R1 has a window. Damage: a wall stain
and a ceiling stain in R2, a crack in R1. One concealed-damage flag and two scope items.
"""

from __future__ import annotations

import numpy as np

from scan2scope.types import (
    Adjacency,
    CaptureInfo,
    ConcealedFlag,
    DamageRegion,
    LineItem,
    Measurement,
    Opening,
    Plan,
    Room,
    Wall,
)


def M(v: float, kind: str = "length", unit: str = "m", **evidence) -> Measurement:
    return Measurement(float(v), unit=unit, kind=kind, evidence=dict(evidence))


def poly_room(rid: str, label: str, poly: np.ndarray, height: float = 2.5, observed: tuple[float, ...] | None = None,
              source_hint: str | None = None) -> Room:
    """Room with one wall per edge of a counter-clockwise polygon."""
    poly = np.asarray(poly, float)
    n = len(poly)
    observed = observed or (0.9,) * n
    walls = []
    for k in range(n):
        a, b = poly[k], poly[(k + 1) % n]
        d = b - a
        length = float(np.linalg.norm(d))
        u = d / length
        walls.append(Wall(id=f"{rid}-W{k + 1}", room_id=rid, start=a.copy(), end=b.copy(),
                          length=M(length, n_points=5000, residual=0.004), height=M(height, "height"),
                          normal_in=np.array([-u[1], u[0]]), observed_fraction=observed[k]))
    x, y = poly[:, 0], poly[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(np.roll(x, -1), y)))
    perimeter = sum(w.length.value for w in walls)
    return Room(id=rid, label=label, polygon=poly, walls=walls, openings=[], floor_z=0.0, ceiling_z=height,
                ceiling_height=M(height, "height"), floor_area=M(area, "area", "m2"), perimeter=M(perimeter),
                source_hint=source_hint)


def rect_room(rid: str, label: str, x0: float, y0: float, x1: float, y1: float, height: float = 2.5,
              observed: tuple[float, ...] = (0.9, 0.9, 0.9, 0.9), source_hint: str | None = None) -> Room:
    poly = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)
    return poly_room(rid, label, poly, height, observed, source_hint)


def make_plan() -> Plan:
    r1 = rect_room("R1", "living", 0.0, 0.0, 4.0, 3.0, observed=(0.9, 0.85, 0.2, 0.9), source_hint="01 living")
    r2 = rect_room("R2", "kitchen", 4.1, 0.0, 7.1, 3.0, source_hint="02 kitchen")
    # R1-W2 runs (4,0)->(4,3) and R2-W4 runs (4.1,3)->(4.1,0); the door spans y 1.0..1.8 on both.
    r1.openings = [
        Opening("R1-O1", "R1", "R1-W2", "door", M(1.0, "offset"), M(0.8, "width"), M(2.0, "height"),
                center=np.array([4.0, 1.4]), connects_to="R2", confidence=0.9),
        Opening("R1-O2", "R1", "R1-W3", "window", M(1.5, "offset"), M(1.2, "width"), M(1.1, "height"),
                sill=M(0.9, "height"), center=np.array([1.9, 3.0]), confidence=0.8),
    ]
    r2.openings = [
        Opening("R2-O1", "R2", "R2-W4", "door", M(1.2, "offset"), M(0.8, "width"), M(2.0, "height"),
                center=np.array([4.1, 1.4]), connects_to="R1", confidence=0.9),
    ]
    return Plan(rooms=[r1, r2], adjacency=[Adjacency("R1", "R2", "R1-O1", "R2-O1", 0.9, "shared_frame")],
                footprint_area=M(21.0, "area", "m2"), extent_x=M(7.1), extent_y=M(3.0),
                meta={"drift": {"enabled": True, "loop_closure": True, "plane_anchoring": True,
                                "yaw_anchoring": False, "loop_residual_m": 0.031}})


def make_damage() -> list[DamageRegion]:
    return [
        DamageRegion("D1", "R2", "R2-W2", "water_stain", 0.62, area=M(0.18, "area", "m2"), width=M(0.6),
                     height=M(0.4), u_range=(0.5, 1.1), v_range=(0.0, 0.4), view_ids=["v3", "v4"]),
        DamageRegion("D2", "R2", "R2-CEIL", "water_stain", 0.55, area=M(0.25, "area", "m2"), width=M(0.6),
                     height=M(0.5), u_range=(5.0, 5.6), v_range=(1.0, 1.5), view_ids=["v5"]),
        DamageRegion("D3", "R1", "R1-W1", "crack", 0.41, area=M(0.02, "area", "m2"), width=M(0.4), height=M(0.5),
                     u_range=(1.0, 1.4), v_range=(1.0, 1.5), length=M(0.55), view_ids=["v1"]),
    ]


def make_flags() -> list[ConcealedFlag]:
    return [ConcealedFlag("F1", "ceiling_stain_moisture_above", "Possible moisture above the ceiling",
                          "EPA mold guide (paraphrased)", "R2", ["R2-CEIL"], ["D2"], "medium",
                          "Check the ceiling cavity with a moisture meter", inputs={"stain_area_m2": 0.25})]


def make_scope() -> list[LineItem]:
    return [
        LineItem("L1", "R2", "R2-W2", "DRY", "1/2", "&", "Drywall 1/2 in - remove and replace",
                 Measurement(2.0, 1.6, 2.5, unit="SF", kind="area"), "SF", damage_ids=["D1"]),
        LineItem("L2", "R2", "R2-CEIL", "PNT", "SP", "+", "Seal and paint the ceiling",
                 Measurement(3.0, 2.4, 3.7, unit="SF", kind="area"), "SF", damage_ids=["D2"], flag_ids=["F1"],
                 rule_id="ceiling_stain_moisture_above"),
    ]


def make_info(tier: str = "video") -> CaptureInfo:
    return CaptureInfo("test_capture", tier, "/tmp/test_capture", device={"model": "iPhone 17"},
                       input_stats={"n_frames": np.int64(240), "fps": np.float32(30.0)})


TIMING = {"total_s": 12.5, "stages": {"ingest": 0.5, "geometry": 8.0, "layout": 2.0, "uncertainty": 0.01}}
PROVENANCE = {"pipeline_version": "0.1.0", "git_commit": "abc123", "cache_mode": "live", "device": "cpu",
              "models": [{"name": "facebook/map-anything-apache", "revision": "00f9c24", "license": "Apache-2.0"}]}
