"""Paths, pinned model revisions and device selection."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

CACHE_DIR = Path(os.environ.get("SCAN2SCOPE_CACHE", Path.home() / ".cache" / "scan2scope"))
TORCH_HOME = CACHE_DIR / "torch_home"
OUTPUT_CACHE_DIR = CACHE_DIR / "outputs"


@dataclass(frozen=True)
class ModelSpec:
    name: str
    repo: str
    revision: str
    license: str
    files: tuple[str, ...]

    @property
    def local_dir(self) -> Path:
        return CACHE_DIR / "hf" / self.repo.split("/")[-1]


MODELS: dict[str, ModelSpec] = {
    "mapanything": ModelSpec(
        "mapanything", "facebook/map-anything-apache", "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a", "Apache-2.0",
        ("config.json", "model.safetensors", "README.md"),
    ),
    "grounding_dino": ModelSpec(
        "grounding_dino", "IDEA-Research/grounding-dino-tiny", "a2bb814dd30d776dcf7e30523b00659f4f141c71",
        "Apache-2.0",
        ("config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
         "special_tokens_map.json", "vocab.txt", "added_tokens.json"),
    ),
    "sam2": ModelSpec(
        "sam2", "facebook/sam2.1-hiera-small", "ee5bba1d82bb8749febdf90f45e84b687142ba03", "Apache-2.0",
        ("config.json", "model.safetensors", "preprocessor_config.json", "processor_config.json"),
    ),
}

DINOV2_HUB_ZIP = "https://github.com/facebookresearch/dinov2/archive/refs/heads/main.zip"


def torch_device() -> str:
    forced = os.environ.get("SCAN2SCOPE_DEVICE")
    if forced:
        return forced
    import torch

    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def setup_env() -> None:
    """Point torch.hub at our cache and keep Hugging Face offline when weights are present."""
    os.environ.setdefault("TORCH_HOME", str(TORCH_HOME))
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if all((spec.local_dir / "config.json").exists() for spec in MODELS.values()):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
