import math

from scan2scope.geometry.scale import door_height_cue, fuse_scale


def test_fuse_scale_inverse_variance():
    v, s = fuse_scale([(0.10, 0.05), (0.00, 0.10)])
    w1, w2 = 1 / 0.05 ** 2, 1 / 0.10 ** 2
    assert math.isclose(v, (0.10 * w1) / (w1 + w2))
    assert math.isclose(s, 1 / math.sqrt(w1 + w2))
    assert s < 0.05


def test_fuse_scale_single_cue_is_identity():
    assert fuse_scale([(0.2, 0.08)]) == (0.2, 0.08)


def test_fuse_scale_ignores_bad_cues_and_handles_empty():
    assert fuse_scale([]) == (0.0, math.inf)
    assert fuse_scale([(math.nan, 0.1), (0.1, math.inf), (0.3, -1.0)]) == (0.0, math.inf)
    v, s = fuse_scale([(math.nan, 0.1), (0.05, 0.1)])
    assert math.isclose(v, 0.05) and math.isclose(s, 0.1)


def test_fuse_scale_exact_cue_wins():
    assert fuse_scale([(0.3, 0.0), (0.0, 0.01), (0.1, 0.0)]) == (0.2, 0.0)


def test_door_height_cue():
    v, s = door_height_cue(2.05)
    assert abs(v) < 1e-12 and math.isclose(s, 0.06 / 2.05)
    v, _ = door_height_cue(1.90)
    assert math.isclose(math.exp(v) * 1.90, 2.05)
    _, s2 = door_height_cue(1.90, meas_sigma_m=0.03)
    assert s2 > s
    assert door_height_cue(0.0) == (0.0, math.inf)
    assert door_height_cue(float("nan")) == (0.0, math.inf)


def test_door_cue_pulls_a_prior_towards_it():
    prior = (0.0, 0.08)
    v, s = fuse_scale([prior, door_height_cue(1.85)])
    assert 0.0 < v < math.log(2.05 / 1.85)
    assert s < 0.08
