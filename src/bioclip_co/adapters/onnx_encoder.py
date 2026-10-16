from collections.abc import Sequence
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

# CLIP's preprocessing (BioCLIP keeps it): bicubic resize of the short side, center crop, normalize
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
TOKENIZERS = {"bioclip": "hf-hub:imageomics/bioclip", "clip": "ViT-B-16"}


def preprocess(image) -> np.ndarray:
    img = Image.open(image).convert("RGB")
    scale = SIZE / min(img.size)
    img = img.resize((round(img.width * scale), round(img.height * scale)), Image.BICUBIC)
    left, top = (img.width - SIZE) // 2, (img.height - SIZE) // 2
    img = img.crop((left, top, left + SIZE, top + SIZE))
    pixels = (np.asarray(img, dtype=np.float32) / 255 - MEAN) / STD
    return pixels.transpose(2, 0, 1)  # HWC -> CHW


class OnnxEncoder:
    """The same ImageTextEncoder port, running the exported ONNX files on onnxruntime (CPU)."""

    def __init__(self, name: str, model_dir: str | Path = "models", int8: bool = False, batch_size: int = 32):
        model_dir = Path(model_dir)
        image_file = f"{name}_image_int8.onnx" if int8 else f"{name}_image.onnx"  # the text encoder is never INT8
        self.name = name
        self.batch_size = batch_size
        self.model_dir = model_dir
        self.image = ort.InferenceSession(model_dir / image_file, providers=["CPUExecutionProvider"])
        self.logit_scale = float((model_dir / f"{name}_logit_scale.txt").read_text())

    def encode_images(self, paths: Sequence[Path]) -> np.ndarray:
        chunks = []
        for start in range(0, len(paths), self.batch_size):
            batch = np.stack([preprocess(p) for p in paths[start : start + self.batch_size]])
            chunks.append(self.image.run(None, {"pixels": batch})[0])
        return np.concatenate(chunks)

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        import open_clip  # only the tokenizer; it runs once at startup

        # loaded here, not in __init__: a server with precomputed text embeddings never needs it
        text = ort.InferenceSession(self.model_dir / f"{self.name}_text.onnx", providers=["CPUExecutionProvider"])
        tokens = open_clip.get_tokenizer(TOKENIZERS[self.name])(list(texts)).numpy()
        return np.concatenate([
            text.run(None, {"tokens": tokens[i : i + self.batch_size]})[0]
            for i in range(0, len(tokens), self.batch_size)
        ])
