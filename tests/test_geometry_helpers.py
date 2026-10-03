import json
from pathlib import Path

import numpy as np

from scan2scope.geometry import gravity, pointmaps, se3

ROOT = Path(__file__).resolve().parents[1]


def test_umeyama_recovers_similarity():
    rng = np.random.default_rng(0)
    T = se3.make_T(se3.rotvec_to_R(np.array([0.1, -0.2, 0.3])), np.array([1.0, 2.0, -0.5]), 1.3)
    P = rng.normal(size=(200, 3))
    est = se3.umeyama(P, se3.apply(T, P))
    assert np.allclose(est, T, atol=1e-9)
    assert np.allclose(se3.invert(est) @ T, np.eye(4), atol=1e-9)


def test_rotation_between_maps_vectors():
    for a, b in [((0, 0, 1), (0, 1, 0)), ((1, 2, 3), (-1, 0, 2)), ((0, 0, 1), (0, 0, -1))]:
        a, b = np.array(a, float), np.array(b, float)
        R = se3.rotation_between(a, b)
        assert np.allclose(R @ (a / np.linalg.norm(a)), b / np.linalg.norm(b), atol=1e-9)
        assert np.isclose(np.linalg.det(R), 1.0)


def test_backprojection_and_normals_of_a_plane():
    K = np.array([[100.0, 0, 32], [0, 100.0, 24], [0, 0, 1]])
    depth = np.full((48, 64), 2.0)
    pm = pointmaps.backproject_depth(depth, K, np.eye(4))
    assert np.allclose(pm[..., 2], 2.0)
    n, ok = pointmaps.normals_from_pointmap(pm, np.ones(depth.shape, bool), cam_center=np.zeros(3))
    assert ok.sum() > 0.8 * ok.size
    assert np.allclose(n[ok], [0, 0, -1], atol=1e-6)


def test_voxel_downsample_averages_attributes():
    pts = np.array([[0.01, 0.01, 0.01], [0.02, 0.02, 0.02], [1.0, 1.0, 1.0]])
    w = np.array([1.0, 3.0, 5.0])
    p2, w2 = pointmaps.voxel_downsample(pts, 0.1, w)
    assert len(p2) == 2
    assert sorted(w2.tolist()) == [2.0, 5.0]


def test_manhattan_angle_and_up_estimate():
    th = np.radians(17.0)
    dirs = np.array([[np.cos(th + k * np.pi / 2), np.sin(th + k * np.pi / 2)] for k in range(4)] * 25)
    assert np.isclose(gravity.manhattan_angle(dirs), th, atol=1e-6)
    rng = np.random.default_rng(1)
    up_true = np.array([0.05, -0.02, 1.0]) / np.linalg.norm([0.05, -0.02, 1.0])
    normals = np.vstack([np.tile(up_true, (300, 1)), np.tile(-up_true, (200, 1)), rng.normal(size=(300, 3)) * [1, 1, 0.05]])
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)
    up = gravity.estimate_up(normals, np.ones(len(normals)), np.array([0.0, 0.0, 1.0]))
    assert np.degrees(np.arccos(np.clip(up @ up_true, -1, 1))) < 1.0


def test_schema_is_valid_json_schema():
    import jsonschema

    schema = json.loads((ROOT / "schema/scan2scope.schema.json").read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
