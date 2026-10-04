"""Smoke test with the real Grounding DINO and SAM 2.1 weights; skips without them. LOCAL_IMAGES are not in
the repository, and a synthetic wall image stands in when they are missing. On a shared machine, run it under
a lock file:

lockf -k /tmp/scan2scope-gpu.lock python -m pytest tests/test_semantics_ml.py -q -m ml
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from scan2scope.config import MODELS, setup_env

pytestmark = pytest.mark.ml

LOCAL_IMAGES = [Path("/tmp/scan2scope-scratch/semantics/wikimedia/w01.jpg"),
                Path("/tmp/georesearch/testimgs/kitchen/00.png")]


def _weights_present() -> bool:
    return all((MODELS[k].local_dir / "model.safetensors").exists() for k in ("grounding_dino", "sam2"))


@pytest.fixture(scope="module")
def models():
    if not _weights_present():
        pytest.skip("Grounding DINO / SAM 2.1 weights not downloaded")
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    setup_env()
    from scan2scope.semantics.detector import GroundingDinoDetector
    from scan2scope.semantics.segmenter import Sam2Segmenter

    return GroundingDinoDetector(), Sam2Segmenter()


def _test_image() -> np.ndarray:
    from PIL import Image

    for p in LOCAL_IMAGES:  # local-only files, never committed
        if p.exists():
            return np.asarray(Image.open(p).convert("RGB"))
    img = np.full((480, 640, 3), 235, np.uint8)  # pale wall with a brown blotch
    yy, xx = np.mgrid[:480, :640]
    blob = ((xx - 330) / 90.0) ** 2 + ((yy - 200) / 60.0) ** 2 < 1
    img[blob] = (150, 110, 60)
    return img


def test_detector_and_segmenter_run_on_a_test_image(models):
    from scan2scope.semantics.detector import DAMAGE_PROMPTS, OBJECT_PROMPTS, decode, working_size

    det, seg = models
    rgb = _test_image()
    import cv2

    w, h = working_size(rgb.shape[1], rgb.shape[0], det.cfg.long_side)
    img = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    dets = []
    for prompt in (DAMAGE_PROMPTS, OBJECT_PROMPTS):
        raw = det.predict(img, prompt)
        assert raw["boxes"].shape[1] == 4 and raw["phrase_scores"].shape[1] == len(prompt.phrases)
        assert (raw["boxes"] >= 0).all() and (raw["boxes"] <= 1).all()
        out = decode(raw, prompt, w, h, det.cfg)
        for d in out:
            assert d.cls in prompt.classes and prompt.box_threshold < d.score <= 1
            assert 0 <= d.box[0] < d.box[2] <= w and 0 <= d.box[1] < d.box[3] <= h
        dets += out
    boxes = np.stack([d.box for d in dets]) if dets else np.array([[w * 0.3, h * 0.3, w * 0.7, h * 0.7]])
    raw = seg.predict(img, boxes)
    assert raw["masks"].shape[0] == len(boxes) and tuple(raw["shape"]) == (h, w)
    assert ((raw["iou"] >= 0) & (raw["iou"] <= 1.05)).all()
    masks = seg.masks(raw, boxes)
    assert len(masks) == len(boxes)
    for m, iou, fallback in masks:
        assert m.shape == (h, w) and m.dtype == bool and m.any()


def test_analyze_with_real_models_on_a_synthetic_wall(models, tmp_path):
    from PIL import Image
    from semantics_fixtures import box_view, make_plan, rect_room

    from scan2scope.semantics import analyze
    from scan2scope.types import Scene

    path = tmp_path / "view.png"
    Image.fromarray(_test_image()).resize((640, 480)).save(path)
    view = box_view("v0", (2.0, 1.0, 1.2), (2.0, 3.0, 1.2), (0.0, 0.0, 0.0), (4.0, 3.0, 2.5), image_path=path)
    scene = Scene(tier="photo", views=[view], points=np.zeros((0, 3)), normals=np.zeros((0, 3)),
                  weights=np.zeros(0), view_index=np.zeros(0, int))
    det, seg = models
    res = analyze([scene], make_plan(rect_room()), tmp_path / "work", detector=det, segmenter=seg)
    assert (tmp_path / "work" / "semantics" / "detections.json").exists()
    for d in res.damage:
        assert d.surface_id.startswith("R1-") and d.area.value > 0 and 0 <= d.score <= 1
