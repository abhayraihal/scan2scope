from types import SimpleNamespace

import numpy as np

from scan2scope.pipeline import _reject_mirror_openings
from scan2scope.types import Adjacency, Measurement, Opening, Plan, Room


def _opening(oid, center, connects_to=None):
    return Opening(id=oid, room_id="R1", wall_id="R1-W1", type="window", offset=Measurement(0.5),
                   width=Measurement(0.9), height=Measurement(1.2), center=np.array(center, float),
                   connects_to=connects_to)


def _plan(openings):
    room = Room(id="R1", label="bathroom", polygon=np.array([[0, 0], [3, 0], [3, 2], [0, 2]], float), walls=[],
                openings=openings, floor_z=0.0, ceiling_z=2.5, ceiling_height=Measurement(2.5),
                floor_area=Measurement(6.0), perimeter=Measurement(10.0))
    adj = [Adjacency("R1", "R2", "R1-O2", None, 0.9, "shared_frame")]
    return Plan(rooms=[room], adjacency=adj, footprint_area=Measurement(6.0), extent_x=Measurement(3.0),
                extent_y=Measurement(2.0))


def test_mirror_on_a_window_removes_it_but_keeps_doors_between_rooms():
    plan = _plan([_opening("R1-O1", [1.0, 0.0]), _opening("R1-O2", [2.5, 0.0], connects_to="R2")])
    mirror = SimpleNamespace(cls="mirror", xy=(1.1, 0.05))
    flags = _reject_mirror_openings(plan, [mirror, SimpleNamespace(cls="sink", xy=(2.5, 0.0))])
    assert flags == ["opening_rejected_mirror:R1-O1"]
    assert [o.id for o in plan.rooms[0].openings] == ["R1-O2"]
    assert len(plan.adjacency) == 1


def test_far_mirror_keeps_the_window():
    plan = _plan([_opening("R1-O1", [1.0, 0.0])])
    assert _reject_mirror_openings(plan, [SimpleNamespace(cls="mirror", xy=(2.9, 1.9))]) == []
    assert len(plan.rooms[0].openings) == 1
