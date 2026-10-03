"""Grounding DINO (tiny) boxes for damage and context-object prompts.

The model is loaded lazily from the pinned local snapshot, so cached outputs replay without weights. Raw outputs
(every query with a phrase score above a low floor) are what gets cached; thresholds and NMS run afterwards in
`decode`, so retuning them does not invalidate the cache.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from scan2scope.config import MODELS, torch_device

log = logging.getLogger("scan2scope.semantics")


@dataclass(frozen=True)
class PromptSet:
    """Phrases in Grounding DINO text format (lowercase, each ending with a period) and their classes."""

    kind: str  # "damage" | "object"
    phrases: tuple[str, ...]
    classes: tuple[str, ...]
    box_threshold: float
    text_threshold: float = 0.25

    @property
    def text(self) -> str:
        return " ".join(f"{p}." for p in self.phrases)

    def phrase_char_spans(self) -> list[tuple[int, int]]:
        spans, pos = [], 0
        for p in self.phrases:
            spans.append((pos, pos + len(p)))
            pos += len(p) + 2  # the period and the space
        return spans


DAMAGE_PROMPTS = PromptSet(
    "damage",
    ("water stain", "mold", "crack", "hole", "peeling paint"),
    ("water_stain", "mold", "crack", "hole", "peeling_paint"),
    box_threshold=0.35,
)
OBJECT_PROMPTS = PromptSet(
    "object",
    ("door", "window", "mirror", "sink", "toilet", "bathtub", "shower", "stove", "refrigerator", "washing machine"),
    ("door", "window", "mirror", "sink", "toilet", "bathtub", "shower", "stove", "refrigerator", "washing_machine"),
    box_threshold=0.30,
)


@dataclass
class DetectorConfig:
    long_side: int = 1024  # images are resized so their long side is about this many pixels
    damage: PromptSet = DAMAGE_PROMPTS
    objects: PromptSet = OBJECT_PROMPTS
    nms_iou: float = 0.5
    nms_containment: float = 0.85  # a box this much inside a higher-scoring box of the same class is dropped
    max_box_frac: float = 0.9  # near-whole-image boxes are a known false-positive mode
    min_box_px: float = 4.0
    keep_floor: float = 0.15  # raw queries below this phrase score are not cached


@dataclass
class Detection:
    cls: str
    kind: str  # "damage" | "object"
    box: np.ndarray  # (4,) x0, y0, x1, y1 in pixels of the image the detector saw
    score: float
    phrase_scores: np.ndarray = field(default_factory=lambda: np.zeros(0))


def working_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    s = long_side / max(width, height)
    return max(1, round(width * s)), max(1, round(height * s))


def cxcywh_to_xyxy(b: np.ndarray) -> np.ndarray:
    cx, cy, w, h = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def box_iou_matrix(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """IoU and intersection over the area of each box in `a`, for xyxy boxes."""
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / np.maximum(union, 1e-12), inter / np.maximum(area_a[:, None], 1e-12)


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float, containment: float) -> np.ndarray:
    """Indices kept by greedy NMS that also drops boxes mostly contained in a better one."""
    order = np.argsort(-scores, kind="stable")
    keep: list[int] = []
    for i in order:
        if keep:
            ious, inside = box_iou_matrix(boxes[i:i + 1], boxes[keep])
            if (ious[0] > iou).any() or (inside[0] > containment).any():
                continue
        keep.append(int(i))
    return np.array(keep, dtype=int)


def decode(raw: dict[str, np.ndarray], prompt: PromptSet, width: int, height: int,
           cfg: DetectorConfig) -> list[Detection]:
    """Thresholds, size filters and per-class NMS on cached raw outputs. Boxes come back in pixels."""
    ps = np.asarray(raw.get("phrase_scores", np.zeros((0, len(prompt.phrases)))), float)
    boxes = np.asarray(raw.get("boxes", np.zeros((0, 4))), float)
    if len(ps) == 0 or ps.ndim != 2 or ps.shape[1] != len(prompt.phrases):
        return []
    cls_idx = ps.argmax(1)
    score = ps.max(1)
    keep = (score > prompt.box_threshold) & (ps[np.arange(len(ps)), cls_idx] > prompt.text_threshold)
    px = np.clip(boxes, 0.0, 1.0) * np.array([width, height, width, height], float)
    bw, bh = px[:, 2] - px[:, 0], px[:, 3] - px[:, 1]
    keep &= (bw >= cfg.min_box_px) & (bh >= cfg.min_box_px)
    keep &= bw * bh <= cfg.max_box_frac * width * height
    out: list[Detection] = []
    for c in np.unique(cls_idx[keep]):
        idx = np.flatnonzero(keep & (cls_idx == c))
        for k in idx[nms(px[idx], score[idx], cfg.nms_iou, cfg.nms_containment)]:
            out.append(Detection(prompt.classes[c], prompt.kind, px[k].copy(), float(score[k]), ps[k].copy()))
    out.sort(key=lambda d: -d.score)
    return out


class GroundingDinoDetector:
    """Wraps GroundingDinoForObjectDetection on MPS (or CUDA) with a CPU fallback."""

    spec = MODELS["grounding_dino"]

    def __init__(self, config: DetectorConfig | None = None, device: str | None = None) -> None:
        self.cfg = config or DetectorConfig()
        self.device = device
        self.model = None
        self.processor = None

    def cache_key(self, image_sha256: str, size: tuple[int, int], prompt: PromptSet) -> dict:
        return {"stage": "semantics.grounding_dino", "image_sha256": image_sha256, "size": list(size),
                "prompt": prompt.text, "model": self.spec.repo, "revision": self.spec.revision,
                "long_side": self.cfg.long_side, "keep_floor": self.cfg.keep_floor}

    def _load(self) -> None:
        if self.model is not None:
            return
        d = self.spec.local_dir
        if not (d / "config.json").exists() or not (d / "model.safetensors").exists():
            raise RuntimeError(f"Grounding DINO weights not found in {d}; run `scan2scope fetch-weights`")
        try:
            import torch
            from transformers import AutoProcessor, GroundingDinoForObjectDetection
        except ImportError as exc:
            raise RuntimeError("semantics needs torch and transformers (install the ml extra)") from exc
        self.device = self.device or torch_device()
        self.processor = AutoProcessor.from_pretrained(d, local_files_only=True)
        model = GroundingDinoForObjectDetection.from_pretrained(d, local_files_only=True).eval()
        try:
            self.model = model.to(self.device)
        except (RuntimeError, TypeError) as exc:
            log.warning("Grounding DINO cannot use %s (%s); using cpu", self.device, exc)
            self.device = "cpu"
            self.model = model
        log.info("Grounding DINO loaded on %s", self.device)

    def _phrase_token_spans(self, prompt: PromptSet, input_ids: list[int]) -> list[list[int]]:
        tok = self.processor.tokenizer
        enc = tok(prompt.text, return_offsets_mapping=True, add_special_tokens=True)
        spans: list[list[int]] = [[] for _ in prompt.phrases]
        if list(enc["input_ids"]) == list(input_ids):
            for t, (a, b) in enumerate(enc["offset_mapping"]):
                for p, (s, e) in enumerate(prompt.phrase_char_spans()):
                    if b > a and a >= s and b <= e:
                        spans[p].append(t)
        else:  # fall back to splitting on the period token
            dot = tok.convert_tokens_to_ids(".")
            p = 0
            for t, i in enumerate(input_ids):
                if i in (tok.cls_token_id, tok.sep_token_id, tok.pad_token_id):
                    continue
                if i == dot:
                    p += 1
                elif p < len(spans):
                    spans[p].append(t)
        if any(not s for s in spans):
            raise RuntimeError(f"could not map prompt phrases to tokens: {prompt.text!r}")
        return spans

    def _forward(self, inputs: dict):
        import torch

        with torch.inference_mode():
            try:
                return self.model(**{k: v.to(self.device) for k, v in inputs.items()})
            except (RuntimeError, NotImplementedError) as exc:
                if self.device == "cpu":
                    raise
                log.warning("Grounding DINO failed on %s (%s); retrying on cpu", self.device, exc)
                self.device = "cpu"
                self.model = self.model.to("cpu")
                return self.model(**{k: v.to("cpu") for k, v in inputs.items()})

    def predict(self, image: np.ndarray, prompt: PromptSet) -> dict[str, np.ndarray]:
        """Raw outputs for an RGB uint8 image already resized to the working size."""
        from PIL import Image

        self._load()
        inputs = self.processor(images=Image.fromarray(image), text=prompt.text, return_tensors="pt",
                                do_resize=False)
        spans = self._phrase_token_spans(prompt, inputs["input_ids"][0].tolist())
        out = self._forward(dict(inputs))
        probs = out.logits.sigmoid()[0].float().cpu().numpy()
        boxes = out.pred_boxes[0].float().cpu().numpy()
        ps = np.stack([probs[:, s].max(1) for s in spans], 1)
        keep = ps.max(1) >= self.cfg.keep_floor
        return {"boxes": np.clip(cxcywh_to_xyxy(boxes[keep]), 0.0, 1.0).astype(np.float32),
                "phrase_scores": ps[keep].astype(np.float32)}
