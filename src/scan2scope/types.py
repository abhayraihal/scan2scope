"""Core data types shared by every stage.

Frames and units: metres; the world frame is right-handed with z up (gravity-aligned) after the
geometry backend runs. Plan coordinates are the world (x, y). Cameras use the OpenCV convention
(x right, y down, z forward) and poses are camera-to-world.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

TIERS = ("photo", "video", "lidar")


@dataclass
class Measurement:
    """A measured quantity. lo/hi are filled by the uncertainty stage (90% nominal interval)."""

    value: float
    lo: float | None = None
    hi: float | None = None
    unit: str = "m"
    kind: str = "length"  # length | height | width | offset | area
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        lo = self.value if self.lo is None else self.lo
        hi = self.value if self.hi is None else self.hi
        return {"value": round(float(self.value), 4), "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                "unit": self.unit}


@dataclass
class CameraView:
    """One image with its pose and, when available, a per-pixel world point map.

    pointmap is (h, w, 3) in world coordinates and may be lower resolution than the image; pixel (i, j)
    of the point map covers image pixel ((j + 0.5) * width / w - 0.5, (i + 0.5) * height / h - 0.5).
    """

    id: str
    image_path: Path | None
    width: int
    height: int
    K: np.ndarray  # (3, 3) intrinsics in image pixels
    T_wc: np.ndarray  # (4, 4) camera-to-world
    pointmap: np.ndarray | None = None
    valid: np.ndarray | None = None  # (h, w) bool
    conf: np.ndarray | None = None  # (h, w) float, larger is better
    timestamp: float | None = None
    room_hint: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def center(self) -> np.ndarray:
        return self.T_wc[:3, 3]


@dataclass
class Scene:
    """Gravity-aligned geometry of one capture (or one room for the photo tier)."""

    tier: str
    views: list[CameraView]
    points: np.ndarray  # (N, 3)
    normals: np.ndarray  # (N, 3) unit, oriented towards the observing camera
    weights: np.ndarray  # (N,) confidence in [0, 1]
    view_index: np.ndarray  # (N,) index into views of the view that produced the point
    scale_log_sigma: float = 0.0  # 1-sigma uncertainty of the global metric scale, in log units
    room_hint: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Wall:
    id: str  # "<room>-W<k>"
    room_id: str
    start: np.ndarray  # (2,) plan coordinates, room polygon is counter-clockwise
    end: np.ndarray
    length: Measurement
    height: Measurement
    normal_in: np.ndarray  # (2,) unit normal pointing into the room
    observed_fraction: float = 0.0  # share of the wall face covered by observations
    evidence: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class Opening:
    id: str  # "<room>-O<k>"
    room_id: str
    wall_id: str
    type: str  # door | window | opening
    offset: Measurement  # wall start to the near edge of the opening, along the wall
    width: Measurement
    height: Measurement
    sill: Measurement | None = None  # windows only
    center: np.ndarray | None = None  # (2,) plan coordinates
    connects_to: str | None = None  # room id on the other side
    confidence: float = 0.0
    evidence: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


@dataclass
class Room:
    id: str  # "R1", "R2", ...
    label: str
    polygon: np.ndarray  # (K, 2) counter-clockwise, plan coordinates
    walls: list[Wall]
    openings: list[Opening]
    floor_z: float
    ceiling_z: float
    ceiling_height: Measurement
    floor_area: Measurement
    perimeter: Measurement
    view_ids: list[str] = field(default_factory=list)
    source_hint: str | None = None  # photo folder name when known
    flags: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class Adjacency:
    room_a: str
    room_b: str
    opening_a: str | None
    opening_b: str | None
    confidence: float
    source: str  # "shared_frame" | "doorway_photo" | "door_match"


@dataclass
class Plan:
    rooms: list[Room]
    adjacency: list[Adjacency]
    footprint_area: Measurement
    extent_x: Measurement
    extent_y: Measurement
    flags: list[str] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)  # drift record, stitch record


@dataclass
class DamageRegion:
    id: str  # "D1", ...
    room_id: str
    surface_id: str  # "<room>-W<k>", "<room>-FLOOR" or "<room>-CEIL"
    cls: str  # water_stain | mold | crack | hole | peeling_paint
    score: float  # detector score after cross-view merge, not a calibrated probability
    area: Measurement
    width: Measurement  # along the surface's u axis
    height: Measurement  # along the surface's v axis
    u_range: tuple[float, float]  # on the surface: walls u from wall start, v from floor; floor/ceiling plan x/y
    v_range: tuple[float, float]
    length: Measurement | None = None  # cracks
    view_ids: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConcealedFlag:
    id: str  # "F1", ...
    rule_id: str
    title: str
    basis: str
    room_id: str
    surface_ids: list[str]
    damage_ids: list[str]
    severity: str  # low | medium | high
    recommendation: str
    inputs: dict[str, Any] = field(default_factory=dict)


@dataclass
class LineItem:
    id: str  # "L1", ...
    room_id: str
    surface_id: str
    category: str  # Xactimate-style category code, e.g. DRY, PNT, WTR
    selector: str
    activity: str  # "&" remove and replace, "-" remove, "+" replace or apply, "R" detach and reset, "I" install
    description: str
    quantity: Measurement
    unit: str  # SF, LF, EA, m2, m
    damage_ids: list[str] = field(default_factory=list)
    flag_ids: list[str] = field(default_factory=list)
    rule_id: str | None = None


@dataclass
class CaptureInfo:
    id: str
    tier: str
    path: str
    device: dict[str, Any] = field(default_factory=dict)
    input_stats: dict[str, Any] = field(default_factory=dict)
    flags: list[str] = field(default_factory=list)


def surface_ids(room: Room) -> list[str]:
    return [w.id for w in room.walls] + [f"{room.id}-FLOOR", f"{room.id}-CEIL"]
