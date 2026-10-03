import math

import numpy as np
import pytest

from scan2scope.geometry import drift as dr
from scan2scope.geometry.se3 import apply, make_T, rot_z, rotvec_to_R


def _plane(origin, u, v, n, su, sv, step=0.04):
    a, b = np.meshgrid(np.arange(0, su, step), np.arange(0, sv, step))
    p = np.asarray(origin) + a.reshape(-1, 1) * np.asarray(u) + b.reshape(-1, 1) * np.asarray(v)
    return p, np.tile(np.asarray(n, float), (len(p), 1))


def _cloud(planes, rng):
    p = np.concatenate([x[0] for x in planes])
    n = np.concatenate([x[1] for x in planes])
    return p + rng.normal(0, 0.001, p.shape), n


def _corner(rng):
    return _cloud([_plane([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2, 2),
                   _plane([0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], 2, 2),
                   _plane([0, 0, 0], [1, 0, 0], [0, 0, 1], [0, 1, 0], 2, 2)], rng)


def test_disabled_returns_poses_unchanged():
    poses = np.tile(np.eye(4), (5, 1, 1))
    poses[:, 0, 3] = np.arange(5)
    out, rec = dr.correct_lidar_poses([dr.KeyFrame(i, float(i)) for i in range(5)], poses, lambda k: None,
                                      enabled=False)
    np.testing.assert_array_equal(out, poses)
    assert out is not poses
    assert rec == {"enabled": False}


def test_icp_recovers_a_rigid_motion_on_a_corner():
    rng = np.random.default_rng(1)
    tgt, tgt_n = _corner(rng)
    T_true = make_T(rotvec_to_R(np.radians([1.0, -0.5, 2.0])), np.array([0.03, -0.02, 0.01]))
    Ti = np.linalg.inv(T_true)
    src, src_n = apply(Ti, tgt), tgt_n @ Ti[:3, :3].T
    res = dr.icp_point_to_plane(src, src_n, tgt, tgt_n)
    E = res.T @ Ti
    assert np.degrees(np.linalg.norm(dr.Rotation.from_matrix(E[:3, :3]).as_rotvec())) < 0.05
    assert np.linalg.norm(apply(E, src.mean(0)) - src.mean(0)) < 0.002
    assert res.dof == 6
    assert res.rmse < 0.003 and res.overlap > 0.9


def test_icp_on_wall_and_floor_leaves_sliding_free():
    rng = np.random.default_rng(2)
    tgt, tgt_n = _cloud([_plane([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 2, 2),
                         _plane([0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], 2, 2)], rng)
    src = tgt + np.array([0.0, 0.05, 0.02])  # 2 cm above the floor and slid 5 cm along the wall (y)
    res = dr.icp_point_to_plane(src, tgt_n, tgt, tgt_n)
    move = apply(res.T, src).mean(0) - src.mean(0)
    assert move[2] == pytest.approx(-0.02, abs=0.002)
    assert abs(move[1]) < 0.005  # no information along the wall, so no step along it
    assert res.dof == 5
    ev, U = np.linalg.eigh(res.info)
    assert ev[0] < 1e-9 and abs(U[4, 0]) > 0.99  # the free direction is translation along y
    assert dr._accept(res, src.mean(0)) is None  # still a useful loop edge


def test_icp_floor_only_is_rejected():
    rng = np.random.default_rng(3)
    tgt, tgt_n = _cloud([_plane([0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], 3, 3)], rng)
    res = dr.icp_point_to_plane(tgt + [0.1, 0.0, 0.01], tgt_n, tgt, tgt_n)
    assert res.dof == 3
    assert dr._accept(res, tgt.mean(0)) == "degenerate"


def _circle_poses(n):
    T = np.tile(np.eye(4), (n, 1, 1))
    for k in range(n):
        th = 2 * np.pi * k / n
        T[k, :3, :3] = rot_z(th + np.pi / 2)
        T[k, :3, 3] = [2 * np.cos(th), 2 * np.sin(th), 1.4]
    return T


def test_pose_graph_closes_a_drifted_loop():
    n = 16
    true = _circle_poses(n)
    true = np.concatenate([true, true[:1]])  # back at the start
    raw = true.copy()
    psi = np.radians(0.25) * np.arange(n + 1)  # VIO-like: yaw error grows, each step applied with it
    for k in range(1, n + 1):
        raw[k, :3, 3] = raw[k - 1, :3, 3] + rot_z(psi[k - 1]) @ (true[k, :3, 3] - true[k - 1, :3, 3])
        raw[k, :3, :3] = rot_z(psi[k]) @ true[k, :3, :3]
    edges = [dr.Edge(s, s + 1, np.linalg.inv(raw[s]) @ raw[s + 1], 0.02, math.radians(0.3)) for s in range(n)]
    D = raw @ np.linalg.inv(true)
    sqrt_info = np.diag([1 / 0.001] * 3 + [1 / 0.005] * 3)  # an accurate registration: 0.06 deg, 5 mm
    loop = dr.LoopEdge(0, n, D[0] @ np.linalg.inv(D[n]), raw[0, :3, 3].copy(), sqrt_info, raw[0], raw[n])
    params = dr.solve_pose_graph(raw, edges, loops=[loop])
    X = dr._corrections(params) @ raw
    err_raw = np.linalg.norm(raw[:, :3, 3] - true[:, :3, 3], axis=1).max()
    err = np.linalg.norm(X[:, :3, 3] - true[:, :3, 3], axis=1).max()
    assert err_raw > 0.1
    assert err < 0.1 * err_raw
    assert np.degrees(abs(dr._yaw_of(X[n, :3, :3] @ true[n, :3, :3].T))) < 0.2


def test_priors_level_floor_and_snap_yaw():
    n = 6
    anchors = _circle_poses(n)
    edges = [dr.Edge(s, s + 1, np.linalg.inv(anchors[s]) @ anchors[s + 1], 0.5, math.radians(20)) for s in range(n - 1)]
    z_priors = {s: (1.4 + 0.03, 0.01) for s in range(n)}
    yaw_priors = {s: (math.radians(1.0), math.radians(0.5)) for s in range(n)}
    params = dr.solve_pose_graph(anchors, edges, z_priors=z_priors, yaw_priors=yaw_priors)
    X = dr._corrections(params) @ anchors
    np.testing.assert_allclose(X[:, 2, 3], 1.43, atol=0.002)
    np.testing.assert_allclose(np.degrees(params[:, 0]), 1.0, atol=0.05)


def test_interpolate_corrections():
    C = np.stack([np.eye(4), make_T(rot_z(np.radians(2.0)), np.array([0.2, 0.0, 0.0]))])
    out = dr.interpolate_corrections(np.array([0.0, 10.0]), C, np.array([-1.0, 0.0, 5.0, 10.0, 12.0]))
    np.testing.assert_allclose(out[0], C[0], atol=1e-12)
    np.testing.assert_allclose(out[3], C[1], atol=1e-12)
    np.testing.assert_allclose(out[4], C[1], atol=1e-12)
    assert np.degrees(dr._yaw_of(out[2, :3, :3])) == pytest.approx(1.0)
    assert out[2, 0, 3] == pytest.approx(0.1)


def test_voxel_accumulator_weights_and_tags():
    acc = dr.VoxelAccumulator(0.1)
    acc.add(np.array([[0.01, 0.01, 0.01], [0.03, 0.01, 0.01]]), np.array([[0, 0, 1.0], [0, 0, 1.0]]),
            np.array([1.0, 0.25]), tag=np.array([4, 7]))
    acc.add(np.array([[0.05, 0.05, 0.05], [0.55, 0.0, 0.0]]), np.array([[0, 0, 1.0], [1.0, 0, 0]]),
            np.array([2.0, 1.0]), tag=9)
    p, n, w, tag = acc.result()
    order = np.argsort(p[:, 0])
    p, n, w, tag = p[order], n[order], w[order], tag[order]
    assert len(p) == 2
    np.testing.assert_allclose(p[0], (1.0 * np.array([0.01, 0.01, 0.01]) + 0.25 * np.array([0.03, 0.01, 0.01])
                                      + 2.0 * np.array([0.05, 0.05, 0.05])) / 3.25)
    assert w[0] == pytest.approx(3.25 / 3) and tag[0] == 9 and tag[1] == 9
