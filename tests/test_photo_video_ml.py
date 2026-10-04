"""Photo and video scenes from real kitchen frames through MapAnything. Needs the weights and the kitchen
frames (KITCHEN below), which are not in the repository; skips without them. On a shared machine, run it under
a lock file: lockf -k /tmp/scan2scope-gpu.lock python -m pytest tests/test_photo_video_ml.py -q"""

import hashlib
import json
import math
import shutil
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scan2scope.config import MODELS

pytestmark = pytest.mark.ml
pytest.importorskip("torch")
pytest.importorskip("mapanything")

KITCHEN = Path("/tmp/georesearch/testimgs/kitchen")
if not (MODELS["mapanything"].local_dir / "model.safetensors").exists():
    pytest.skip("MapAnything weights not downloaded", allow_module_level=True)
if not (KITCHEN / "24.png").exists():
    pytest.skip("kitchen test images not available", allow_module_level=True)

from scan2scope.geometry import photo, video


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


def _check_scene(s):
    n = len(s.points)
    assert n > 1000 and s.points.dtype == np.float32
    assert s.normals.shape == (n, 3) and s.weights.shape == (n,) and s.view_index.shape == (n,)
    assert np.isfinite(s.points).all() and np.isfinite(s.normals).all()
    assert np.allclose(np.linalg.norm(s.normals, axis=1), 1.0, atol=1e-3)
    assert s.weights.min() >= 0.0 and s.weights.max() <= 1.0
    assert s.view_index.min() >= 0 and s.view_index.max() < len(s.views)
    for v in s.views:
        assert v.image_path.exists() and v.K.shape == (3, 3) and np.isfinite(v.T_wc).all()
        assert v.pointmap.ndim == 3 and v.valid.shape == v.pointmap.shape[:2] == v.conf.shape
        assert abs(v.pointmap.shape[1] / v.pointmap.shape[0] - v.width / v.height) < 0.02
    json.dumps(s.meta["quality"])
    json.dumps(s.meta["drift"])


def test_photo_rooms_from_kitchen_frames(tmp_path):
    root = tmp_path / "Scan"
    room = root / "01 kitchen"
    room.mkdir(parents=True)
    import pillow_heif

    pillow_heif.register_heif_opener()
    for k, i in enumerate(range(0, 24, 4)):
        im = Image.open(KITCHEN / f"{i:02d}.png").convert("RGB")
        exif = Image.Exif()
        exif.get_ifd(0x8769)[0xA405] = 30
        im.save(room / f"IMG_{k:04d}.HEIC", exif=exif, quality=95)
    (root / "02 single").mkdir()
    shutil.copy(KITCHEN / "05.png", root / "02 single" / "only.png")

    cache = NpzCache(tmp_path)
    scenes = photo.build_room_scenes(root, tmp_path / "work", cache=cache)
    assert [s.room_hint for s in scenes] == ["01 kitchen", "02 single"]
    kitchen, single = scenes
    for s in scenes:
        _check_scene(s)
    assert len(kitchen.views) == 6 and kitchen.meta["quality"]["exif_focal"]
    assert all(v.meta["intrinsics_given"] for v in kitchen.views)
    # the model follows the EXIF focal length, mapped back to the photo resolution
    for v in kitchen.views:
        K_exif = np.array(v.meta["K_exif"])
        assert np.allclose(v.K, K_exif, rtol=0.03, atol=3.0)
    assert "thin" in single.meta["flags"] and not single.meta["quality"]["exif_focal"]
    # cameras look down at the table, so their down axes point below the horizon after gravity alignment
    assert kitchen.meta["gravity"]["refined"]
    assert all(v.T_wc[2, 1] < -0.3 for v in kitchen.views)
    up = kitchen.normals[:, 2] > 0.9
    assert up.mean() > 0.1  # table top and floor face up
    assert np.median(kitchen.normals[up, 2]) > 0.98
    assert cache.misses == 2
    again = photo.build_room_scenes(root, tmp_path / "work", cache=cache)
    assert cache.hits == 2 and np.allclose(again[0].points, kitchen.points)


def _encode_kitchen_video(path: Path, fps: float = 2.0):
    import av

    imgs = [np.asarray(Image.open(KITCHEN / f"{i:02d}.png").convert("RGB"))[:, :778] for i in range(25)]
    with av.open(str(path), "w") as c:
        st = c.add_stream("libx264", rate=int(fps))
        st.width, st.height, st.pix_fmt = imgs[0].shape[1], imgs[0].shape[0], "yuv420p"
        st.options = {"crf": "14"}
        for img in imgs:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(img), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)


def test_video_scene_from_kitchen_clip(tmp_path):
    clip = tmp_path / "kitchen.mp4"
    _encode_kitchen_video(clip)
    cache = NpzCache(tmp_path)
    # the clip is 2 fps; sample all of it
    on = video.build_scene(clip, tmp_path / "work_on", drift_correction=True, cache=cache, chunk_size=8, overlap=3,
                           loop_frames=4, target_fps=2.0)
    q, d = on.meta["quality"], on.meta["drift"]
    assert q["n_frames"] >= 20 and abs(q["duration_s"] - 12.5) < 0.6
    assert q["n_chunks"] == len(video.plan_chunks(q["n_frames"], 8, 3))
    _check_scene(on)
    assert len(on.views) == math.ceil(q["n_frames"] / video.KEYFRAME_EVERY)
    assert all(c["align_method"] in ("reference", "points") for c in d["chunks"]), d["chunks"]
    rel = [c["relative_scale"] for c in d["chunks"][1:]]
    assert all(0.7 < r < 1.4 for r in rel), rel
    assert d["enabled"] and d["loop_closure"]["attempted"]
    assert on.scale_log_sigma >= video.SCALE_SIGMA_BASE
    assert on.meta["frames"]["T_wc"].shape == (q["n_frames"], 4, 4)

    misses = cache.misses
    off = video.build_scene(clip, tmp_path / "work_off", drift_correction=False, cache=cache, chunk_size=8,
                            overlap=3, loop_frames=4, target_fps=2.0)
    assert cache.misses == misses  # every chunk came from the cache
    assert not off.meta["drift"]["enabled"] and not off.meta["drift"]["loop_closure"]["attempted"]
    _check_scene(off)
