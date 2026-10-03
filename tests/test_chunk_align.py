import numpy as np

from scan2scope.geometry import chunk_align as ca
from scan2scope.geometry import se3


def _random_sim3(rng, rot=0.3, trans=1.0, log_s=0.1):
    return se3.make_T(se3.rotvec_to_R(rng.normal(size=3) * rot), rng.normal(size=3) * trans,
                      float(np.exp(rng.normal() * log_s)))


def _trajectory(n=10, radius=4.0):
    """Chunk poses on a closed circle, each turned to face along the path."""
    nodes = []
    for k in range(n):
        a = 2 * np.pi * k / n
        nodes.append(se3.make_T(se3.rot_z(a + np.pi / 2), np.array([radius * np.cos(a), radius * np.sin(a), 0.0])))
    return nodes


def test_so3_exp_log_roundtrip():
    rng = np.random.default_rng(0)
    v = np.vstack([rng.normal(size=(50, 3)), [[0.0, 0.0, 0.0], [1e-9, 0.0, 0.0], [0.0, np.pi - 1e-4, 0.0]]])
    R = ca._so3_exp(v)
    assert np.allclose(R @ np.transpose(R, (0, 2, 1)), np.eye(3), atol=1e-9)
    assert np.allclose(R, np.stack([se3.rotvec_to_R(x) for x in v]), atol=1e-8)
    v2 = ca._so3_log(R)
    assert np.allclose(ca._so3_exp(v2), R, atol=1e-8)


def test_robust_sim3_recovers_transform_with_outliers():
    rng = np.random.default_rng(1)
    T = _random_sim3(rng)
    src = rng.uniform(-2, 2, size=(3000, 3))
    dst = se3.apply(T, src) + rng.normal(scale=0.005, size=src.shape)
    bad = rng.random(len(src)) < 0.35
    dst[bad] += rng.normal(scale=1.0, size=(bad.sum(), 3))
    fit = ca.robust_sim3(src, dst, thresh=0.03)
    assert fit.ok
    assert fit.inlier_frac > 0.55
    assert fit.residual_m < 0.01
    err = ca.sim3_error(T, fit.T)
    assert err["trans"] < 0.005 and err["rot_deg"] < 0.1 and abs(err["log_scale"]) < 0.002


def test_robust_sim3_degenerate_inputs_fail_cleanly():
    fit = ca.robust_sim3(np.zeros((3, 3)), np.zeros((3, 3)))
    assert not fit.ok and np.allclose(fit.T, np.eye(4))
    pts = np.full((100, 3), np.nan)
    assert not ca.robust_sim3(pts, pts).ok


def test_chain_composes_relative_transforms():
    rng = np.random.default_rng(2)
    rel = [_random_sim3(rng) for _ in range(4)]
    out = ca.chain(rel)
    assert len(out) == 5 and np.allclose(out[0], np.eye(4))
    assert np.allclose(out[4], rel[0] @ rel[1] @ rel[2] @ rel[3])


def test_pose_graph_is_exact_on_consistent_edges():
    rng = np.random.default_rng(3)
    truth = [np.eye(4)] + [_random_sim3(rng, trans=2.0) for _ in range(5)]
    edges = [ca.Edge(k, k + 1, se3.invert(truth[k]) @ truth[k + 1]) for k in range(5)]
    edges.append(ca.Edge(0, 5, se3.invert(truth[0]) @ truth[5], kind="loop"))
    init = [T @ _random_sim3(rng, rot=0.02, trans=0.05, log_s=0.01) for T in truth]
    init[0] = truth[0]
    res = ca.optimize_pose_graph(init, edges)
    assert res.success
    for A, B in zip(truth, res.nodes):
        err = ca.sim3_error(A, B)
        assert err["trans"] < 1e-6 and err["rot_deg"] < 1e-5 and abs(err["log_scale"]) < 1e-7
    assert res.cost_after < 1e-12


def test_pose_graph_distributes_loop_error():
    rng = np.random.default_rng(4)
    truth = _trajectory(12)
    truth = [se3.invert(truth[0]) @ T for T in truth]
    # every sequential edge carries the same small yaw and scale bias, so the chain drifts steadily
    bias = se3.make_T(se3.rot_z(np.radians(1.5)), np.array([0.03, 0.0, 0.01]), 1.01)
    seq = [se3.invert(truth[k]) @ truth[k + 1] @ bias for k in range(11)]
    chained = ca.chain(seq)
    loop = ca.Edge(0, 11, se3.invert(truth[0]) @ truth[11], kind="loop")
    edges = [ca.Edge(k, k + 1, seq[k]) for k in range(11)] + [loop]
    res = ca.optimize_pose_graph(chained, edges)

    def pos_err(nodes):
        return np.array([np.linalg.norm(nodes[k][:3, 3] - truth[k][:3, 3]) for k in range(12)])

    before, after = pos_err(chained), pos_err(res.nodes)
    assert before[-1] > 0.5
    assert after.max() < 0.35 * before.max()
    loop_res = [e for e in res.edge_residuals if e["kind"] == "loop"][0]
    assert loop_res["trans"] < 0.2 * ca.sim3_error(loop.T_ij, chained[11])["trans"]
    seq_rot = [e["rot_deg"] for e in res.edge_residuals if e["kind"] == "seq"]
    assert max(seq_rot) - min(seq_rot) < 0.5  # the error is spread, not dumped on one edge


def test_pose_graph_priors_level_and_snap_chunks():
    truth = [se3.make_T(np.eye(3), np.array([2.0 * k, 0.0, 0.0])) for k in range(4)]
    tilt = se3.make_T(se3.rotvec_to_R(np.radians([1.5, -1.0, 3.0])), np.array([0.0, 0.0, 0.08]))
    drifted = [truth[0]] + [T @ tilt for T in truth[1:]]
    edges = [ca.Edge(k, k + 1, se3.invert(drifted[k]) @ drifted[k + 1]) for k in range(3)]
    floor_pt = np.array([0.5, 0.3, -1.4])
    priors = []
    for k in range(1, 4):
        priors.append(ca.Prior(k, R_world=np.eye(3)))
        z_true = se3.apply(truth[k], floor_pt)[2]
        priors.append(ca.Prior(k, point=floor_pt, z_world=float(z_true)))
    res = ca.optimize_pose_graph(drifted, edges, priors)
    for k in range(1, 4):
        R = se3.decompose_sim3(res.nodes[k])[1]
        assert ca.rotation_angle_deg(R) < 0.2
        assert abs(se3.apply(res.nodes[k], floor_pt)[2] - se3.apply(truth[k], floor_pt)[2]) < 0.01


def test_pose_graph_without_free_nodes_returns_input():
    nodes = [np.eye(4), se3.make_T(np.eye(3), np.ones(3))]
    res = ca.optimize_pose_graph(nodes, [], fixed=(0, 1))
    assert all(np.allclose(a, b) for a, b in zip(nodes, res.nodes))
