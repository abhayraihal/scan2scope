"""Blur laptops (their screens show text) in photos and videos before they are published as benchmark data.

    python scripts/blur_screens.py SRC_DIR DST_DIR

Mirrors SRC_DIR into DST_DIR: images are re-saved as JPEG (quality 95) with screens blurred, videos are
re-encoded (H.264, same size and frame rate) with screens blurred on every frame. Boxes come from Grounding
DINO; in videos they are detected every few frames and held in between; frames are written upright. Photo EXIF
(focal length, Apple MakerNote) is kept. Benchmark numbers are computed from the blurred copies, so the
published inputs regenerate them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from PIL import Image, ImageFilter

PROMPT = "laptop ."
MAX_BOX_FRACTION = 0.08  # a dark window can score as a screen; real laptop boxes are small
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
VIDEO_EXT = {".mp4", ".mov", ".m4v"}


class ScreenFinder:
    def __init__(self, threshold: float = 0.3):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        from scan2scope.config import MODELS, setup_env, torch_device

        setup_env()
        path = MODELS["grounding_dino"].local_dir
        self.device = torch_device()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(path, local_files_only=True).to(self.device)
        self.model.eval()
        self.torch = torch
        self.threshold = threshold

    def boxes(self, img: Image.Image) -> list[tuple[int, int, int, int]]:
        inputs = self.processor(images=img, text=PROMPT, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model(**inputs)
        res = self.processor.post_process_grounded_object_detection(
            out, inputs.input_ids, threshold=self.threshold, text_threshold=0.25, target_sizes=[img.size[::-1]])[0]
        w, h = img.size
        boxes = [tuple(round(v) for v in b) for b in res["boxes"].cpu().numpy().tolist()]
        return [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) <= MAX_BOX_FRACTION * w * h]


def blur_boxes(img: Image.Image, boxes: list[tuple[int, int, int, int]], pad: float = 0.08) -> Image.Image:
    out = img.copy()
    w, h = img.size
    for x0, y0, x1, y1 in boxes:
        px, py = int((x1 - x0) * pad) + 4, int((y1 - y0) * pad) + 4
        box = (max(0, x0 - px), max(0, y0 - py), min(w, x1 + px), min(h, y1 + py))
        if box[2] <= box[0] or box[3] <= box[1]:
            continue
        region = out.crop(box).filter(ImageFilter.GaussianBlur(radius=max(8, (box[2] - box[0]) // 6)))
        out.paste(region, box)
    return out


def blur_image(finder: ScreenFinder, src: Path, dst: Path) -> int:
    from scan2scope.ingest.images import load_image

    rgb, _ = load_image(src)
    img = Image.fromarray(rgb)
    boxes = finder.boxes(img)
    raw = Image.open(src)
    exif = raw.getexif()
    if exif.get(0x0112, 1) != 1:  # pixels are saved upright, so the orientation tag must say so
        exif[0x0112] = 1
        exif_bytes = exif.tobytes()
    else:
        exif_bytes = raw.info.get("exif", exif.tobytes())  # keeps the Apple MakerNote byte for byte
    blur_boxes(img, boxes).save(dst.with_suffix(".jpg"), quality=95, exif=exif_bytes)
    return len(boxes)


def blur_video(finder: ScreenFinder, src: Path, dst: Path, every: int = 3) -> int:
    import av

    from scan2scope.ingest.video import probe, rotate_upright

    rotation = probe(src).rotation_deg  # frames are written upright, without a rotation tag
    with av.open(str(src)) as inp:
        stream = inp.streams.video[0]
        rate = stream.average_rate or 30
        with av.open(str(dst.with_suffix(".mp4")), "w") as out:
            ov = out.add_stream("libx264", rate=rate)
            w, h = stream.codec_context.width, stream.codec_context.height
            ov.width, ov.height = (h, w) if rotation in (90, 270) else (w, h)
            ov.pix_fmt = "yuv420p"
            ov.options = {"crf": "17", "preset": "medium"}
            boxes, n_boxes = [], 0
            for k, frame in enumerate(inp.decode(stream)):
                img = Image.fromarray(rotate_upright(frame.to_ndarray(format="rgb24"), rotation))
                if k % every == 0:
                    boxes = finder.boxes(img)
                    n_boxes += len(boxes)
                new = av.VideoFrame.from_image(blur_boxes(img, boxes))
                for packet in ov.encode(new):
                    out.mux(packet)
            for packet in ov.encode():
                out.mux(packet)
    return n_boxes


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src", type=Path)
    p.add_argument("dst", type=Path)
    a = p.parse_args()
    if a.dst.exists() and any(a.dst.iterdir()):
        sys.exit(f"{a.dst} is not empty")
    finder = ScreenFinder()
    for f in sorted(a.src.rglob("*")):
        if f.is_dir() or f.name.startswith("."):
            continue
        rel = f.relative_to(a.src)
        out = a.dst / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        ext = f.suffix.lower()
        if ext in IMAGE_EXT:
            print(f"{rel}: {blur_image(finder, f, out)} screens blurred")
        elif ext in VIDEO_EXT:
            print(f"{rel}: {blur_video(finder, f, out)} screen detections blurred")
        else:
            shutil.copy2(f, out)


if __name__ == "__main__":
    main()
