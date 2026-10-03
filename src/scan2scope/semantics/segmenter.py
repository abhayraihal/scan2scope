"""SAM 2.1 (small) masks from box prompts.

For each box the three multimask outputs are scored by the model's predicted IoU and the best one is kept.
Masks are cached bit-packed at the working image size; `postprocess` clips them to the prompt box and falls
back to the box itself when SAM returns almost nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from scan2scope.config import MODELS, torch_device

log = logging.getLogger("scan2scope.semantics")


@dataclass
class SegmenterConfig:
    batch: int = 16  # boxes per mask-decoder call
    clip_margin: float = 0.05  # masks are clipped to the box grown by this share of its size
    min_mask_frac: float = 0.02  # masks covering less of their box fall back to the box


def pack_masks(masks: np.ndarray) -> np.ndarray:
    return np.packbits(masks.reshape(len(masks), -1).astype(bool), axis=1)


def unpack_masks(packed: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = int(shape[0]), int(shape[1])
    if len(packed) == 0:
        return np.zeros((0, h, w), bool)
    return np.unpackbits(packed, axis=1, count=h * w).reshape(len(packed), h, w).astype(bool)


def postprocess(mask: np.ndarray, box: np.ndarray, cfg: SegmenterConfig) -> tuple[np.ndarray, bool]:
    """Clip a mask to its (slightly grown) prompt box; returns (mask, used_box_fallback)."""
    h, w = mask.shape
    x0, y0, x1, y1 = (float(v) for v in box)
    mx, my = cfg.clip_margin * (x1 - x0), cfg.clip_margin * (y1 - y0)
    c0, r0 = max(0, int(np.floor(x0 - mx))), max(0, int(np.floor(y0 - my)))
    c1, r1 = min(w, int(np.ceil(x1 + mx))), min(h, int(np.ceil(y1 + my)))
    out = np.zeros_like(mask, dtype=bool)
    out[r0:r1, c0:c1] = mask[r0:r1, c0:c1]
    box_area = max(1.0, (x1 - x0) * (y1 - y0))
    if out.sum() >= cfg.min_mask_frac * box_area:
        return out, False
    out[:] = False
    out[max(0, int(round(y0))):min(h, int(round(y1))), max(0, int(round(x0))):min(w, int(round(x1)))] = True
    return out, True


class Sam2Segmenter:
    """Wraps Sam2Model and Sam2Processor on MPS (or CUDA) with a CPU fallback."""

    spec = MODELS["sam2"]

    def __init__(self, config: SegmenterConfig | None = None, device: str | None = None) -> None:
        self.cfg = config or SegmenterConfig()
        self.device = device
        self.model = None
        self.processor = None

    def cache_key(self, image_sha256: str, size: tuple[int, int], boxes: np.ndarray) -> dict:
        return {"stage": "semantics.sam2", "image_sha256": image_sha256, "size": list(size),
                "boxes": np.round(np.asarray(boxes, float), 1).tolist(), "model": self.spec.repo,
                "revision": self.spec.revision, "multimask": True, "select": "max_pred_iou"}

    def _load(self) -> None:
        if self.model is not None:
            return
        d = self.spec.local_dir
        if not (d / "config.json").exists() or not (d / "model.safetensors").exists():
            raise RuntimeError(f"SAM 2.1 weights not found in {d}; run `scan2scope fetch-weights`")
        try:
            from transformers import Sam2Model, Sam2Processor
        except ImportError as exc:
            raise RuntimeError("semantics needs torch and transformers (install the ml extra)") from exc
        self.device = self.device or torch_device()
        self.processor = Sam2Processor.from_pretrained(d, local_files_only=True)
        model = Sam2Model.from_pretrained(d, local_files_only=True).eval()
        try:
            self.model = model.to(self.device)
        except (RuntimeError, TypeError) as exc:
            log.warning("SAM 2.1 cannot use %s (%s); using cpu", self.device, exc)
            self.device = "cpu"
            self.model = model
        log.info("SAM 2.1 loaded on %s", self.device)

    def _run(self, image: np.ndarray, boxes: np.ndarray) -> dict[str, np.ndarray]:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        h, w = image.shape[:2]
        dev = self.device
        with torch.inference_mode():
            pix = self.processor(images=Image.fromarray(image), return_tensors="pt")
            emb = self.model.get_image_embeddings(pix["pixel_values"].to(dev))
            masks, ious, choice = [], [], []
            for s in range(0, len(boxes), self.cfg.batch):
                chunk = np.asarray(boxes[s:s + self.cfg.batch], float)
                prompt = self.processor(original_sizes=[[h, w]], input_boxes=[chunk.tolist()], return_tensors="pt")
                out = self.model(image_embeddings=emb, input_boxes=prompt["input_boxes"].to(dev),
                                 multimask_output=True)
                iou = out.iou_scores[0].float()  # (n, 3)
                best = iou.argmax(-1)
                n = len(chunk)
                logits = out.pred_masks[0][torch.arange(n, device=best.device), best]  # (n, 256, 256)
                up = F.interpolate(logits[:, None].float(), size=(h, w), mode="bilinear", align_corners=False)[:, 0]
                masks.append(pack_masks((up > 0).cpu().numpy()))
                ious.append(iou.gather(1, best[:, None])[:, 0].cpu().numpy())
                choice.append(best.cpu().numpy())
        return {"masks": np.concatenate(masks), "shape": np.array([h, w], np.int64),
                "iou": np.concatenate(ious).astype(np.float32), "choice": np.concatenate(choice).astype(np.int8)}

    def predict(self, image: np.ndarray, boxes: np.ndarray) -> dict[str, np.ndarray]:
        """Best-of-three masks for xyxy pixel boxes on an RGB uint8 image (cacheable arrays)."""
        self._load()
        try:
            return self._run(image, boxes)
        except (RuntimeError, NotImplementedError) as exc:
            if self.device == "cpu":
                raise
            log.warning("SAM 2.1 failed on %s (%s); retrying on cpu", self.device, exc)
            self.device = "cpu"
            self.model = self.model.to("cpu")
            return self._run(image, boxes)

    def masks(self, raw: dict[str, np.ndarray], boxes: np.ndarray) -> list[tuple[np.ndarray, float, bool]]:
        """(mask, predicted IoU, used_box_fallback) per box from cached raw outputs."""
        m = unpack_masks(np.asarray(raw["masks"]), tuple(np.asarray(raw["shape"]).tolist()))
        iou = np.asarray(raw["iou"], float)
        out = []
        for k, box in enumerate(np.asarray(boxes, float)):
            mk, fallback = postprocess(m[k], box, self.cfg)
            out.append((mk, float(iou[k]), fallback))
        return out
