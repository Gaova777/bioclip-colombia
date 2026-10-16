from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class ImageTextEncoder(Protocol):
    """Anything that puts images and texts in the same space: BioCLIP, CLIP, an ONNX export..."""

    name: str
    logit_scale: float

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        """One L2-normalized row per image."""
        ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """One L2-normalized row per text."""
        ...
