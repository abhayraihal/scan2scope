"""MapAnything runner on real images. Needs the weights; run under the GPU lock:
lockf -k /tmp/scan2scope-gpu.lock python -m pytest tests/test_mapanything_ml.py -q"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scan2scope.config import MODELS

pytestmark = pytest.mark.ml
torch = pytest.importorskip("torch")
pytest.importorskip("mapanything")

KITCHEN = Path("/tmp/georesearch/testimgs/kitchen")
if not (MODELS["mapanything"].local_dir / "model.safetensors").exists():
    pytest.skip("MapAnything weights not downloaded", allow_module_level=True)
if not (KITCHEN / "05.png").exists():
    pytest.skip("kitchen test images not available", allow_module_level=True)

from scan2scope.geometry import mapanything_backend as mb  # noqa: E402


class NpzCache:
    """Stand-in for scan2scope.cache.OutputCache: npz files keyed by SHA-256 of the sorted JSON key."""

    def __init__(self, root: Path):
        self.root, self.misses, self.hits = root, 0, 0

    def compute(self, key, fn):
        path = self.root / (hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest() + ".npz")
        if path.exists():
            self.hits += 1
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        self.misses += 1
        out = fn()
        np.savez(path, **out)
        return out


def _kitchen(n=6, start=0):
    return [np.asarray(Image.open(KITCHEN / f"{i:02d}.png").convert("RGB")) for i in range(start, start + n)]


def test_preprocessing_matches_mapanything():
    from mapanything.utils.cropping import crop_resize_if_necessary
    from mapanything.utils.image import find_closest_aspect_ratio

    for aspect in np.linspace(0.4, 3.2, 57):
        assert mb.target_size([(int(1000 * aspect), 1000)]) == find_closest_aspect_ratio(int(1000 * aspect) / 1000, 518)
    rng = np.random.default_rng(0)
    for w, h in [(779, 520), (640, 480), (300, 520), (1920, 1080), (97, 61)]:
        img = rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)
        tgt = mb.target_size([(w, h)])
        ours = np.asarray(mb.preprocess_image(img, tgt))
        theirs = np.asarray(crop_resize_if_necessary(Image.fromarray(img), resolution=tgt)[0])
        assert np.array_equal(ours, theirs), (w, h)


def test_runner_on_kitchen_frames_with_cache(tmp_path):
    cache = NpzCache(tmp_path)
    runner = mb.MapAnythingRunner.get()
    imgs = _kitchen(6)
    preds = runner.infer(imgs, None, key={"test": "kitchen6"}, cache=cache)
    assert cache.misses == 1 and len(preds) == 6
    for p in preds:
        assert p.model_size == (518, 336) and p.pts3d.shape == (336, 518, 3) and p.pts3d.dtype == np.float32
        assert p.conf.shape == (336, 518) and p.mask.shape == (336, 518) and p.mask.dtype == bool
        assert p.mask.mean() > 0.5
        assert np.isfinite(p.pts3d).all() and np.isfinite(p.conf).all() and (p.conf[p.mask] >= 1.0).all()
        assert np.isfinite(p.T_wc).all() and np.allclose(p.T_wc[:3, :3] @ p.T_wc[:3, :3].T, np.eye(3), atol=1e-3)
        assert 0.1 < p.metric_scale < 100 and p.resized_size == (518, 345) and p.crop == (0, 4)
        assert 200 < p.K[0, 0] < 1000 and abs(p.K[0, 2] - 258.5) < 20
        # the point map is consistent with the pose and intrinsics: reproject the valid pixels
        pc = (p.pts3d[p.mask] - p.T_wc[:3, 3]) @ p.T_wc[:3, :3]
        _, u = np.nonzero(p.mask)
        uu = p.K[0, 0] * pc[:, 0] / pc[:, 2] + p.K[0, 2]
        assert np.median(np.abs(uu - u)) < 1.0
    assert np.allclose(preds[0].T_wc, np.eye(4), atol=0.05)
    again = runner.infer(imgs, None, key={"test": "kitchen6"}, cache=cache)
    assert cache.hits == 1 and cache.misses == 1
    for a, b in zip(preds, again):
        assert np.array_equal(a.pts3d, b.pts3d) and np.array_equal(a.mask, b.mask) and a.metric_scale == b.metric_scale


def test_runner_follows_given_intrinsics():
    imgs = _kitchen(4, start=10)
    W, H = imgs[0].shape[1], imgs[0].shape[0]
    K = np.array([[640.0, 0.0, (W - 1) / 2], [0.0, 640.0, (H - 1) / 2], [0.0, 0.0, 1.0]])
    preds = mb.MapAnythingRunner.get().infer(imgs, [K] * 4, key={"test": "intrinsics"})
    for p in preds:
        assert p.intrinsics_given
        assert np.allclose(p.K_image(), K, rtol=0.02, atol=2.0)
