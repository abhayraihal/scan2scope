"""Room outlines from the room's own faces (layout.refine) on synthetic wall voxels and cell polygons."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from scan2scope.layout import refine as R

CEIL = 2.5


def face(axis: int, coord: float, sign: int, t0: float, t1: float, z0: float = 0.0, z1: float = CEIL,
         noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """2 cm voxel centres on the face axis = coord between t0 and t1, normals sign along the axis."""
    t, z = np.meshgrid(np.arange(t0 + 0.01, t1, 0.02), np.arange(z0 + 0.01, z1, 0.02))
    P = np.zeros((t.size, 3))
    P[:, axis] = coord + np.random.default_rng(seed).normal(0.0, noise, t.size)
    P[:, 1 - axis], P[:, 2] = t.ravel(), z.ravel()
    N = np.zeros_like(P)
    N[:, axis] = sign
    return P, N


def box_room(x0: float, y0: float, x1: float, y1: float, noise: float = 0.0) -> list:
    """The four faces of a rectangular room, normals into the room."""
    return [face(0, x0, 1, y0, y1, noise=noise, seed=1), face(0, x1, -1, y0, y1, noise=noise, seed=2),
            face(1, y0, 1, x0, x1, noise=noise, seed=3), face(1, y1, -1, x0, x1, noise=noise, seed=4)]


def points(faces: list) -> R.WallPoints:
    P = np.concatenate([p for p, _ in faces])
    N = np.concatenate([n for _, n in faces])
    w = np.ones(len(P))
    return R.WallPoints.build(P, N, w, w, -0.3, CEIL + 0.3)


def run(poly, faces, claimable=lambda region: True) -> R.Outline:
    out = R.room_outline(np.asarray(poly, float), points(faces), 0.0, CEIL, 0.01, claimable)
    assert out is not None
    assert Polygon(out.polygon).exterior.is_ccw
    return out


def same_outline(out: R.Outline, truth) -> bool:
    return Polygon(out.polygon).symmetric_difference(Polygon(truth)).area < 1e-3


RECT = [(0, 0), (4, 0), (4, 3), (0, 3)]


def test_edges_move_onto_the_room_faces_within_15cm():
    faces = box_room(0, 0, 4, 3, noise=0.01)
    # a parallel face of another room 14 cm away, beyond this room's extent, must not pull the west wall
    faces.append(face(0, 0.14, 1, 3.2, 9.5, noise=0.01, seed=5))
    out = run([(-0.07, 0.05), (4.06, 0.05), (4.06, 2.92), (-0.07, 2.92)], faces)
    assert len(out.edges) == 4
    np.testing.assert_allclose(np.sort(out.polygon, axis=0), np.sort(np.array(RECT, float), axis=0),
                               atol=0.002)
    assert all(e.fit.refined and e.fit.n > 1000 for e in out.edges)


def test_step_onto_the_far_face_of_a_wall_is_removed():
    # a hallway whose north edge runs on the far face of its wall (y = 1.62) for half its length
    faces = [*box_room(0, 0, 6, 1.5), face(1, 1.62, 1, 0, 6, seed=6)]  # the far face looks into the next room
    poly = [(0, 0), (6, 0), (6, 1.62), (3.4, 1.62), (3.4, 1.5), (0, 1.5)]
    out = run(poly, faces)
    assert len(out.edges) == 4 and out.steps >= 1
    assert same_outline(out, [(0, 0), (6, 0), (6, 1.5), (0, 1.5)])


def test_l_shape_and_alcove_steps_are_kept():
    L = [(0, 0), (5, 0), (5, 2.5), (2.5, 2.5), (2.5, 5), (0, 5)]
    faces = [face(1, 0, 1, 0, 5), face(0, 5, -1, 0, 2.5), face(1, 2.5, -1, 2.5, 5), face(0, 2.5, -1, 2.5, 5),
             face(1, 5, -1, 0, 2.5), face(0, 0, 1, 0, 5)]
    out = run(L, faces)
    assert len(out.edges) == 6 and same_outline(out, L)
    # a 0.3 m deep alcove is deeper than a step and stays
    alcove = [(0, 0), (1.5, 0), (1.5, -0.3), (2.5, -0.3), (2.5, 0), (4, 0), (4, 3), (0, 3)]
    faces = [face(1, 0, 1, 0, 1.5), face(0, 1.5, 1, -0.3, 0), face(1, -0.3, 1, 1.5, 2.5),
             face(0, 2.5, -1, -0.3, 0), face(1, 0, 1, 2.5, 4), face(0, 4, -1, 0, 3), face(1, 3, -1, 0, 4),
             face(0, 0, 1, 0, 3)]
    out = run(alcove, faces)
    assert len(out.edges) == 8 and same_outline(out, alcove)


def test_wall_body_strip_attached_to_a_room_is_cut():
    # the strip between y = 3.0 and 3.13 (a wall body) runs on past the room to x = 6.3
    faces = [*box_room(0, 0, 4, 3), face(1, 3.13, 1, 0, 7, seed=7), face(1, 3.0, -1, 4.1, 7, seed=8)]
    poly = [(0, 0), (4, 0), (4, 3.0), (6.3, 3.0), (6.3, 3.13), (0, 3.13)]
    out = run(poly, faces)
    assert len(out.edges) == 4
    assert same_outline(out, RECT)


def counter(height: float) -> list:
    """A box 1.5 m wide and 0.6 m deep against the y = 0 wall, its front and sides facing the room."""
    return [face(1, 0.6, 1, 1.0, 2.5, z1=height), face(0, 1.0, -1, 0.0, 0.6, z1=height),
            face(0, 2.5, 1, 0.0, 0.6, z1=height)]


NOTCH = [(0, 0), (1, 0), (1, 0.6), (2.5, 0.6), (2.5, 0), (4, 0), (4, 3), (0, 3)]


def test_furniture_notch_is_filled_up_to_the_wall_behind():
    # the wall behind the counter is seen above it
    faces = box_room(0, 0, 4, 3) + counter(0.9)
    out = run(NOTCH, faces)
    assert len(out.edges) == 4 and same_outline(out, RECT)
    assert len(out.filled) == 1 and out.filled[0].area == pytest.approx(0.9, abs=1e-6)


def test_full_height_box_and_claimed_floor_keep_the_notch():
    # a unit whose front reaches into the band 0.5 m below the ceiling is a wall, by the furniture rule
    out = run(NOTCH, box_room(0, 0, 4, 3) + counter(2.2))
    assert len(out.edges) == 8 and not out.filled and same_outline(out, NOTCH)
    # floor that another room holds is never added
    out = run(NOTCH, box_room(0, 0, 4, 3) + counter(0.9), claimable=lambda region: False)
    assert len(out.edges) == 8 and not out.filled


def test_wall_without_points_keeps_its_cell_position_and_is_flagged():
    faces = box_room(0, 0, 4, 3)[:3]  # no points on the north wall
    out = run([(0, 0), (4, 0), (4, 3.04), (0, 3.04)], faces)
    north = next(e for e in out.edges if e.axis == 1 and e.sign == -1)
    assert north.coord == pytest.approx(3.04)
    assert not north.fit.refined and "wall_not_refined" in north.fit.flags
    assert "wall_face_missing" in north.fit.flags


def test_non_rectilinear_polygon_is_left_alone():
    tri = np.array([(0, 0), (4, 0), (0, 3)], float)
    assert R.room_outline(tri, points(box_room(0, 0, 4, 3)), 0.0, CEIL, 0.01, lambda region: True) is None
