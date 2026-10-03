import numpy as np

from scan2scope.layout.floor_ceiling import estimate


def _plane(rng, x, y, z, n, nz):
    xy = np.column_stack([rng.uniform(*x, n), rng.uniform(*y, n)])
    return xy, np.full(n, z) + rng.normal(0, 0.005, n), np.full(n, nz)


def _scene(with_ceiling: bool, seed: int = 0):
    rng = np.random.default_rng(seed)
    parts = [_plane(rng, (0, 4), (0, 3), 0.0, 4000, 1.0),  # floor
             _plane(rng, (1.0, 1.8), (2.95, 3.1), 2.1, 400, -1.0)]  # door head: underside of a lintel
    if with_ceiling:
        parts.append(_plane(rng, (0, 4), (0, 3), 2.7, 3000, -1.0))
    xy = np.vstack([p[0] for p in parts])
    z = np.concatenate([p[1] for p in parts])
    nz = np.concatenate([p[2] for p in parts])
    w = np.ones(len(z))
    return z, nz, w, w, xy, np.full(10, 1.4)


def test_door_head_is_not_taken_as_the_ceiling():
    fc = estimate(*_scene(with_ceiling=False))
    assert not fc.ceiling.observed
    assert "ceiling_not_observed" in fc.flags
    assert any(f.startswith("ceiling_candidates_rejected") for f in fc.flags)


def test_room_wide_ceiling_is_accepted():
    fc = estimate(*_scene(with_ceiling=True))
    assert fc.ceiling.observed
    assert abs(fc.ceiling.z - 2.7) < 0.01
