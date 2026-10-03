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


def run(poly, faces) -> R.Outline:
    out = R.room_outline(np.asarray(poly, float), points(faces), 0.0, CEIL, 0.01)
    assert out is not None
    assert Polygon(out.polygon).exterior.is_ccw
    return out


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


def test_wall_without_points_keeps_its_cell_position_and_is_flagged():
    faces = box_room(0, 0, 4, 3)[:3]  # no points on the north wall
    out = run([(0, 0), (4, 0), (4, 3.04), (0, 3.04)], faces)
    north = next(e for e in out.edges if e.axis == 1 and e.sign == -1)
    assert north.coord == pytest.approx(3.04)
    assert not north.fit.refined and "wall_not_refined" in north.fit.flags
    assert "wall_face_missing" in north.fit.flags


def test_non_rectilinear_polygon_is_left_alone():
    tri = np.array([(0, 0), (4, 0), (0, 3)], float)
    assert R.room_outline(tri, points(box_room(0, 0, 4, 3)), 0.0, CEIL, 0.01) is None
