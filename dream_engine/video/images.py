"""Scene images for dream videos (the visual association cortex made literal).

``SDTurboBackend`` runs Stability AI's SD-Turbo locally (CPU is fine: 1 step), so the
dream content never leaves the machine. Images are cached by prompt + settings, so
re-rendering a video does not regenerate them. Another backend (an image API) only
needs to implement ``render``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from PIL import Image


class ImageBackend(Protocol):
    name: str

    def render(self, prompt: str, seed: int) -> Image.Image: ...


class SDTurboBackend:
    def __init__(self, cfg: dict[str, Any]):
        v = cfg["video"]
        self.model_id, self.size, self.steps = v["model"], tuple(v["image_size"]), v["steps"]
        self.variant = v.get("weights_variant")
        self.name = f"{self.model_id}@{self.size[0]}x{self.size[1]}/{self.steps}"
        self._pipe = None

    def _load(self):
        if self._pipe is None:
            import torch
            from diffusers import AutoPipelineForText2Image

            # fp16 weights halve the download; they are upcast to float32 for CPU inference
            self._pipe = AutoPipelineForText2Image.from_pretrained(self.model_id, variant=self.variant,
                                                                   torch_dtype=torch.float32)
            self._pipe.set_progress_bar_config(disable=True)
        return self._pipe

    def render(self, prompt: str, seed: int) -> Image.Image:
        import torch

        pipe = self._load()
        w, h = self.size
        return pipe(prompt, width=w, height=h, num_inference_steps=self.steps, guidance_scale=0.0,
                    generator=torch.Generator().manual_seed(seed)).images[0]


class CachedImages:
    """Wraps a backend with a disk cache keyed by (backend, prompt, seed)."""

    def __init__(self, backend: ImageBackend, cache_dir: Path):
        self.backend, self.dir = backend, cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.generated = 0

    def get(self, prompt: str, seed: int) -> Path:
        key = hashlib.sha1(json.dumps([self.backend.name, prompt, seed]).encode()).hexdigest()[:16]
        path = self.dir / f"{key}.png"
        if not path.exists():
            self.backend.render(prompt, seed).save(path)
            (self.dir / f"{key}.txt").write_text(prompt, encoding="utf-8")  # which prompt made this image
            self.generated += 1
        return path


def make_backend(cfg: dict[str, Any]) -> ImageBackend:
    if cfg["video"]["backend"] == "sd-turbo":
        return SDTurboBackend(cfg)
    raise ValueError(f"Unknown video image backend: {cfg['video']['backend']}")
