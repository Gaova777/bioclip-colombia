from collections.abc import Sequence
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

MODELS = {
    "bioclip": ("hf-hub:imageomics/bioclip", None),
    "clip": ("ViT-B-16", "openai"),
}


class OpenClipEncoder:
    def __init__(self, name: str, batch_size: int = 32, device: str | None = None):
        model_name, pretrained = MODELS[name]
        self.name = name
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.model.to(self.device).eval()
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.logit_scale = self.model.logit_scale.exp().item()

    @torch.no_grad()
    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            batch = paths[start : start + self.batch_size]
            images = torch.stack([self.preprocess(Image.open(p).convert("RGB")) for p in batch])
            chunks.append(self._normalize(self.model.encode_image(images.to(self.device))))
            print(f"  {self.name}: {start + len(batch)}/{len(paths)} images", end="\r")
        print()
        return np.concatenate(chunks)

    @torch.no_grad()
    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        return self._normalize(self.model.encode_text(self.tokenizer(list(texts)).to(self.device)))

    @staticmethod
    def _normalize(x: torch.Tensor) -> np.ndarray:
        return (x / x.norm(dim=-1, keepdim=True)).float().cpu().numpy()
