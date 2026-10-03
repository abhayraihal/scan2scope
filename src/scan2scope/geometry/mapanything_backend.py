"""MapAnything wrapper: load-once runner, array preprocessing that matches mapanything.utils.image.load_images,
and cached inference.

Every image of one call is resized to cover the call's model size (518 fixed mapping, chosen from the mean aspect
ratio) and centre-cropped. Pixel centres are at integer coordinates: input pixel u maps to the model pixel
u_m = sx * (u + 0.5) - 0.5 - ox, with sx = resized width / input width and ox the crop offset.
"""

from __future__ import annotations

import contextlib
import gc
import hashlib
import io
import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Protocol

import numpy as np
from PIL import Image

from scan2scope.config import MODELS, setup_env, torch_device

log = logging.getLogger("scan2scope.geometry")

MAX_VIEWS = 32
RESOLUTION_SET = 518
NORM_TYPE = "dinov2"
# mapanything.utils.image.RESOLUTION_MAPPINGS[518]: aspect ratio -> (width, height)
ASPECT_SIZES = {
    1.000: (518, 518), 1.321: (518, 392), 1.542: (518, 336), 1.762: (518, 294), 2.056: (518, 252),
    3.083: (518, 168), 0.757: (392, 518), 0.649: (336, 518), 0.567: (294, 518), 0.486: (252, 518),
}
INFER_ARGS = {"memory_efficient_inference": True, "minibatch_size": 1, "use_amp": True, "amp_dtype": "bf16",
              "apply_mask": True, "mask_edges": True, "apply_confidence_mask": False}
# MapAnything confidence is 1 + exp(x); indoor surfaces measured 10 to 60, unreliable pixels sit near 1.
CONF_REF = 30.0
HEAVY_CROP = 0.2  # share of a resized image cut away to fit the call's aspect ratio
# MapAnything accepts intrinsics on some views and not others; False drops them all unless every view has them.
ALLOW_PARTIAL_INTRINSICS = True
# The capture protocol's 1x main lens (iPhone 15 or newer, 24 or 26 mm equivalent) has a focal of 0.69 to 0.75 times
# the long side of a 4:3 photo, and more in 16:9 video (stabilisation crops). On recompressed images MapAnything
# predicts far shorter focals and too small a scene: ARKitScenes kitchen photos re-saved at JPEG quality 70 went
# from 0.80 to 0.43 (true 0.84) with metric depth 35% short, at any resolution; the WhatsApp home captures came out
# at 0.53 to 0.55 against 0.75 to 0.85 from vanishing points and two-view self-calibration.
PROTOCOL_MIN_FOCAL_LONG = 0.6
FOCAL_MISMATCH_LOG = 0.1  # a predicted focal this far (log) from the EXIF one: the model did not take it


def confidence_weight(conf: np.ndarray) -> np.ndarray:
    """Absolute map of MapAnything confidence to [0, 1]: log(conf) / log(CONF_REF), clipped."""
    c = np.maximum(np.nan_to_num(np.asarray(conf, np.float64), nan=1.0), 1.0)
    return np.clip(np.log(c) / np.log(CONF_REF), 0.0, 1.0).astype(np.float32)

def _rank_in_mask(conf: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Percentile rank (0..1) of each masked pixel's confidence within the view; 0 outside the mask."""
    out = np.zeros(conf.shape, np.float32)
    vals = conf[mask]
    if vals.size < 2:
        out[mask] = 1.0
        return out
    order = np.argsort(vals, kind="stable")
    ranks = np.empty(vals.size, np.float32)
    ranks[order] = np.arange(vals.size, dtype=np.float32) / (vals.size - 1)
    out[mask] = ranks
    return out



try:
    from scan2scope.cache import CacheMiss
except ImportError:  # cache.py is written in parallel; keep the name available for except clauses
    class CacheMiss(Exception):
        pass


class OutputCacheLike(Protocol):
    def compute(self, key: dict, fn: Callable[[], dict[str, np.ndarray]]) -> dict[str, np.ndarray]: ...


def target_size(sizes: Sequence[tuple[int, int]]) -> tuple[int, int]:
    """Model input (width, height) for images of the given (width, height), as in load_images."""
    aspect = sum(w / h for w, h in sizes) / len(sizes)
    key = min(sorted(ASPECT_SIZES), key=lambda a: abs(a - aspect))
    return ASPECT_SIZES[key]


def resize_geometry(width: int, height: int, target: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    """(resized size, crop offset) of mapanything's crop_resize_if_necessary without intrinsics."""
    size = np.array([width, height], float)
    scale_final = max(np.array(target, float) / size) + 1e-8
    w2, h2 = np.floor(size * scale_final).astype(int)
    return (int(w2), int(h2)), (int((w2 - target[0]) // 2), int((h2 - target[1]) // 2))


def preprocess_image(img: np.ndarray, target: tuple[int, int]) -> Image.Image:
    """Resize (LANCZOS down, BICUBIC up) and centre-crop exactly like crop_resize_if_necessary."""
    pil = Image.fromarray(img)
    (w2, h2), (ox, oy) = resize_geometry(pil.size[0], pil.size[1], target)
    scale_final = max(np.array(target, float) / np.array(pil.size, float)) + 1e-8
    resample = Image.Resampling.LANCZOS if scale_final < 1 else Image.Resampling.BICUBIC
    pil = pil.resize((w2, h2), resample=resample)
    return pil.crop((ox, oy, ox + target[0], oy + target[1]))


def intrinsics_to_model(K: np.ndarray, image_size: tuple[int, int], target: tuple[int, int]) -> np.ndarray:
    """Intrinsics at the input image resolution -> intrinsics of the cropped model image."""
    (w2, h2), (ox, oy) = resize_geometry(image_size[0], image_size[1], target)
    sx, sy = w2 / image_size[0], h2 / image_size[1]
    K = np.asarray(K, float)
    out = np.eye(3)
    out[0, 0], out[0, 1], out[1, 1] = K[0, 0] * sx, K[0, 1] * sx, K[1, 1] * sy
    out[0, 2] = sx * (K[0, 2] + 0.5) - 0.5 - ox
    out[1, 2] = sy * (K[1, 2] + 0.5) - 0.5 - oy
    return out


def array_sha256(a: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(f"{a.shape}|{a.dtype}".encode())
    h.update(np.ascontiguousarray(a).data)
    return h.hexdigest()


def as_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """Coerce grey, RGBA or float images to (H, W, 3) uint8."""
    a = np.asarray(img)
    if a.ndim == 2:
        a = np.repeat(a[..., None], 3, axis=2)
    if a.ndim == 3 and a.shape[2] == 4:
        a = a[..., :3]
    if a.ndim != 3 or a.shape[2] != 3:
        raise ValueError(f"expected an RGB image, got shape {a.shape}")
    if a.dtype != np.uint8:
        f = a.astype(np.float64)
        if np.nanmax(f) <= 1.0:
            f = f * 255.0
        a = np.clip(np.nan_to_num(f), 0, 255).round().astype(np.uint8)
    return np.ascontiguousarray(a)


def _valid_K(K: Any) -> np.ndarray | None:
    if K is None:
        return None
    K = np.asarray(K, float)
    if K.shape != (3, 3) or not np.isfinite(K).all() or K[0, 0] <= 0 or K[1, 1] <= 0:
        return None
    return K


@dataclass
class ViewPrediction:
    """MapAnything output for one view. Arrays are at model resolution; world is the call's frame (view 0)."""

    pts3d: np.ndarray  # (h, w, 3) float32, zero where mask is False
    conf: np.ndarray  # (h, w) float32, >= 1, larger is better
    mask: np.ndarray  # (h, w) bool
    T_wc: np.ndarray  # (4, 4) camera-to-world, OpenCV axes
    K: np.ndarray  # (3, 3) at model resolution
    metric_scale: float  # MapAnything's metric scaling factor for the call
    image_size: tuple[int, int]  # (width, height) of the image passed in
    resized_size: tuple[int, int]  # (width, height) after resizing, before the centre crop
    crop: tuple[int, int]  # (ox, oy) crop offset in resized pixels
    intrinsics_given: bool = False
    pose_ok: bool = True  # False when MapAnything returned a non-finite pose; T_wc is then a placeholder
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def model_size(self) -> tuple[int, int]:
        return int(self.pts3d.shape[1]), int(self.pts3d.shape[0])

    @cached_property
    def weight(self) -> np.ndarray:
        """Confidence in [0, 1], zero outside the mask: half absolute, half the pixel's rank within its view.

        The rank term keeps each view's most confident pixels usable when the model is unsure about a whole
        clip (median confidence near its floor of 1 on some real walkthroughs).
        """
        return np.where(self.mask, 0.5 * confidence_weight(self.conf) + 0.5 * _rank_in_mask(self.conf, self.mask),
                        0.0).astype(np.float32)

    @property
    def scale(self) -> tuple[float, float]:
        """(sx, sy): resized pixels per input-image pixel; with crop this is the image-to-model mapping."""
        return self.scale_to()

    def scale_to(self, width: int | None = None, height: int | None = None) -> tuple[float, float]:
        """Resized-frame pixels per pixel of an image with the same content at (width, height)."""
        W, H = (width or self.image_size[0]), (height or self.image_size[1])
        return self.resized_size[0] / W, self.resized_size[1] / H

    def model_to_image(self, u, v, width: int | None = None, height: int | None = None):
        sx, sy = self.scale_to(width, height)
        return (np.asarray(u) + self.crop[0] + 0.5) / sx - 0.5, (np.asarray(v) + self.crop[1] + 0.5) / sy - 0.5

    def image_to_model(self, u, v, width: int | None = None, height: int | None = None):
        sx, sy = self.scale_to(width, height)
        return sx * (np.asarray(u) + 0.5) - 0.5 - self.crop[0], sy * (np.asarray(v) + 0.5) - 0.5 - self.crop[1]

    def K_image(self, width: int | None = None, height: int | None = None) -> np.ndarray:
        """Model intrinsics expressed for an image of the same content at (width, height)."""
        sx, sy = self.scale_to(width, height)
        K = np.eye(3)
        K[0, 0], K[0, 1], K[1, 1] = self.K[0, 0] / sx, self.K[0, 1] / sx, self.K[1, 1] / sy
        K[0, 2] = (self.K[0, 2] + self.crop[0] + 0.5) / sx - 0.5
        K[1, 2] = (self.K[1, 2] + self.crop[1] + 0.5) / sy - 0.5
        return K

    def uncrop(self, arr: np.ndarray, fill: float | bool = 0) -> np.ndarray:
        """Pad a model-resolution map to the resized frame so that it spans the whole image. Pixel (i, j) then
        covers image pixel ((j + 0.5) * width / w - 0.5, (i + 0.5) * height / h - 0.5), the CameraView rule."""
        (w2, h2), (ox, oy) = self.resized_size, self.crop
        h, w = arr.shape[:2]
        out = np.full((h2, w2) + arr.shape[2:], fill, dtype=arr.dtype)
        out[oy:oy + h, ox:ox + w] = arr
        return out


def focal_check(preds: Sequence[ViewPrediction], ref_focal_long: float | None = None) -> tuple[dict, list[str]]:
    """Compare MapAnything's predicted focal (median over views, times the image's long side) with the EXIF focal
    when known, else with the shortest focal the protocol's camera can have. The log distance beyond those bounds
    is returned as an extra 1-sigma log-scale uncertainty: the captures where it was off had metric errors of 7 to
    41%, and passing intrinsics does not fix it (MapAnything keeps its own, wider field of view)."""
    fl = [p.K_image()[0, 0] / max(p.image_size) for p in preds if p.pose_ok and np.isfinite(p.K).all()]
    rec: dict[str, Any] = {"ma_focal_long": None, "ref_focal_long": ref_focal_long, "sigma_log": 0.0}
    if not fl:
        return rec, []
    f = float(np.median(fl))
    rec["ma_focal_long"] = f
    flags = []
    if ref_focal_long is not None and ref_focal_long > 0:
        d = abs(math.log(f / ref_focal_long))
        rec["ref_kind"] = "exif"
        if d > FOCAL_MISMATCH_LOG:
            rec["sigma_log"] = d
            flags.append(f"mapanything_focal_mismatch:{f:.2f}/{ref_focal_long:.2f}")
    else:
        rec["ref_kind"] = "protocol_min"
        rec["ref_focal_long"] = PROTOCOL_MIN_FOCAL_LONG
        if f < PROTOCOL_MIN_FOCAL_LONG:
            rec["sigma_log"] = math.log(PROTOCOL_MIN_FOCAL_LONG / f)
            flags.append(f"mapanything_focal_implausible:{f:.2f}")
    return rec, flags


class MapAnythingRunner:
    """Process-wide MapAnything instance. The model loads on the first cache miss, so replay needs no weights."""

    _instance: MapAnythingRunner | None = None

    def __init__(self) -> None:
        self.spec = MODELS["mapanything"]
        self.device: str | None = None
        self._model = None

    @classmethod
    def get(cls) -> MapAnythingRunner:
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load(self):
        if self._model is None:
            setup_env()
            from mapanything.models import MapAnything

            self.device = torch_device()
            t0 = time.perf_counter()
            chatter = io.StringIO()  # torch.hub and uniception print progress lines while loading
            with contextlib.redirect_stdout(chatter), contextlib.redirect_stderr(chatter):
                model = MapAnything.from_pretrained(str(self.spec.local_dir), local_files_only=True)
            log.debug("MapAnything load output: %s", chatter.getvalue().strip())
            self._model = model.to(self.device).eval()
            log.info("MapAnything loaded on %s in %.1fs", self.device, time.perf_counter() - t0)
        return self._model

    def release(self) -> None:
        """Drop the model and free accelerator memory."""
        if self._model is not None:
            self._model = None
            _free_memory()

    def infer(self, images: Sequence[np.ndarray], intrinsics: Sequence[np.ndarray | None] | None = None,
              key: dict | None = None, cache: OutputCacheLike | None = None) -> list[ViewPrediction]:
        """Run MapAnything on one set of views (at most MAX_VIEWS).

        intrinsics: per view a 3x3 matrix at that image's resolution, or None. Views with intrinsics get
        calibrated rays and the others are left to the model; the API accepts the mix (see Example 6 in the
        MapAnything README and _encode_and_fuse_ray_dirs, which zeroes ray features of views without them).
        """
        n = len(images)
        if n == 0:
            raise ValueError("MapAnything needs at least one image")
        if n > MAX_VIEWS:
            raise ValueError(f"at most {MAX_VIEWS} views per MapAnything call on this machine, got {n}")
        imgs = [as_rgb_uint8(im) for im in images]
        Ks = list(intrinsics) if intrinsics is not None else [None] * n
        if len(Ks) != n:
            raise ValueError(f"{n} images but {len(Ks)} intrinsics")
        Ks = [_valid_K(K) for K in Ks]
        flags = []
        if not ALLOW_PARTIAL_INTRINSICS and any(K is None for K in Ks) and any(K is not None for K in Ks):
            flags.append("intrinsics_dropped_partial")
            Ks = [None] * n
        sizes = [(im.shape[1], im.shape[0]) for im in imgs]
        target = target_size(sizes)
        geoms = [resize_geometry(w, h, target) for w, h in sizes]
        Km = [None if K is None else intrinsics_to_model(K, sz, target) for K, sz in zip(Ks, sizes)]
        n_given = sum(K is not None for K in Km)
        if 0 < n_given < n:
            flags.append(f"intrinsics_partial:{n_given}/{n}")
        cropped = [1.0 - target[0] * target[1] / (w2 * h2) for (w2, h2), _ in geoms]
        n_heavy = sum(c > HEAVY_CROP for c in cropped)
        if n_heavy:
            flags.append(f"aspect_crop:{n_heavy}/{n}")

        def run() -> dict[str, np.ndarray]:
            return self._run(imgs, Km, target)

        if cache is None:
            arrays = run()
        else:
            full_key = {
                "model": self.spec.repo, "revision": self.spec.revision, "resolution": RESOLUTION_SET,
                "target_size": list(target), "norm": NORM_TYPE, "args": INFER_ARGS,
                "images": [array_sha256(im) for im in imgs],
                "intrinsics": [None if K is None else np.round(K, 3).tolist() for K in Km],
                "caller": key or {},
            }
            arrays = cache.compute(full_key, run)
        bad = [i for i in range(n) if not (np.isfinite(arrays["T_wc"][i]).all() and np.isfinite(arrays["K"][i]).all())]
        if bad:  # fp16 overflow can leave a view without a usable pose; keep it, empty
            flags.append("invalid_view:" + ",".join(map(str, bad)))
            log.warning("MapAnything returned a non-finite pose or intrinsics for views %s", bad)
        ms = np.asarray(arrays["metric_scale"], float)
        preds = []
        for i in range(n):
            (w2, h2), crop = geoms[i]
            pts = np.asarray(arrays["pts3d"][i], np.float32)
            mask = np.asarray(arrays["mask"][i], bool) & np.isfinite(pts).all(-1) & (i not in bad)
            T_wc, K = np.asarray(arrays["T_wc"][i], float), np.asarray(arrays["K"][i], float)
            if i in bad:
                T_wc = np.eye(4)
                K = np.array([[0.8 * target[0], 0.0, (target[0] - 1) / 2], [0.0, 0.8 * target[0], (target[1] - 1) / 2],
                              [0.0, 0.0, 1.0]])
            preds.append(ViewPrediction(
                pts3d=np.where(mask[..., None], pts, 0.0).astype(np.float32),
                conf=np.nan_to_num(np.asarray(arrays["conf"][i], np.float32), nan=1.0, posinf=1.0, neginf=1.0),
                mask=mask, T_wc=T_wc, K=K,
                metric_scale=float(ms[i]) if np.isfinite(ms[i]) else 1.0, image_size=sizes[i],
                resized_size=(w2, h2), crop=crop, intrinsics_given=Km[i] is not None, pose_ok=i not in bad,
                meta={"flags": list(flags), "crop_fraction": round(cropped[i], 4)},
            ))
        return preds

    def _run(self, imgs: list[np.ndarray], Km: list[np.ndarray | None], target: tuple[int, int]) -> dict:
        import torch
        import torchvision.transforms as tvf
        from uniception.models.encoders.image_normalizations import IMAGE_NORMALIZATION_DICT

        model = self.load()
        norm = IMAGE_NORMALIZATION_DICT[NORM_TYPE]
        to_input = tvf.Compose([tvf.ToTensor(), tvf.Normalize(mean=norm.mean, std=norm.std)])
        views = []
        for i, img in enumerate(imgs):
            pil = preprocess_image(img, target)
            view = {"img": to_input(pil)[None], "data_norm_type": [NORM_TYPE],
                    "true_shape": np.int32([pil.size[::-1]]), "idx": i, "instance": str(i)}
            if Km[i] is not None:
                view["intrinsics"] = torch.from_numpy(Km[i].astype(np.float32))[None]
            views.append(view)
        t0 = time.perf_counter()
        try:
            preds, oom = None, False
            try:
                preds = model.infer(views, **INFER_ARGS)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom = True
            if oom:  # retry outside the except block: the live traceback would pin the failed call's tensors
                log.warning("MapAnything ran out of memory on %d views, retrying once", len(imgs))
                _free_memory()
                preds = model.infer(views, **INFER_ARGS)
            out = {
                "pts3d": np.stack([p["pts3d"][0].float().cpu().numpy() for p in preds]).astype(np.float32),
                "conf": np.stack([p["conf"][0].float().cpu().numpy() for p in preds]).astype(np.float32),
                "mask": np.stack([p["mask"][0, ..., 0].cpu().numpy().astype(bool) for p in preds]),
                "T_wc": np.stack([p["camera_poses"][0].float().cpu().numpy() for p in preds]).astype(np.float64),
                "K": np.stack([p["intrinsics"][0].float().cpu().numpy() for p in preds]).astype(np.float64),
                "metric_scale": np.array([float(p["metric_scaling_factor"].reshape(-1)[0]) for p in preds]),
            }
            del preds
        finally:
            del views
            _free_memory()
        log.info("MapAnything: %d views at %dx%d in %.1fs", len(imgs), target[0], target[1],
                 time.perf_counter() - t0)
        return out


def _free_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        elif torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
