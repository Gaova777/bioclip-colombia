"""Export a model's image and text encoders to ONNX, plus an INT8 version of the image encoder,
and the candidates' text embeddings (so a CPU server like the Lambda needs neither torch nor the text encoder).

usage: uv run --group onnx scripts/export_onnx.py bioclip
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from onnxruntime.quantization import QuantType, quantize_dynamic

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.zero_shot import prompts_for


class ImageTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixels):
        return self.model.encode_image(pixels, normalize=True)


class TextTower(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tokens):
        return self.model.encode_text(tokens, normalize=True)


def export(module, example, path, input_name):
    torch.onnx.export(
        module, (example,), path, input_names=[input_name], output_names=["embedding"],
        dynamic_axes={input_name: {0: "batch"}, "embedding": {0: "batch"}}, opset_version=17, dynamo=False,
    )


def main(args):
    encoder = OpenClipEncoder(args.model, device="cpu")
    # the fused attention kernel has no ONNX equivalent; the plain one computes the same thing
    torch.backends.mha.set_fastpath_enabled(False)
    args.out.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        export(ImageTower(encoder.model), torch.randn(1, 3, 224, 224), args.out / f"{args.model}_image.onnx", "pixels")
        export(TextTower(encoder.model), encoder.tokenizer(["a photo of a bird"]), args.out / f"{args.model}_text.onnx", "tokens")
    # INT8 weights for the matrix multiplications, activations stay float (dynamic quantization).
    # Only the image encoder: quantizing the text encoder drops species accuracy from 78% to 38%,
    # and the text side runs once at startup anyway, so it has nothing to gain.
    image = args.out / f"{args.model}_image.onnx"
    quantize_dynamic(image, image.with_name(image.stem + "_int8.onnx"), weight_type=QuantType.QInt8)
    (args.out / f"{args.model}_logit_scale.txt").write_text(f"{encoder.logit_scale}\n")
    texts = encoder.encode_texts(prompts_for(read_candidates(args.candidates), style=args.model))
    np.save(args.out / f"{args.model}_text_emb.npy", texts)
    for f in sorted(args.out.glob(f"{args.model}_*")):
        print(f"{f.name:32} {f.stat().st_size / 1e6:7.1f} MB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["clip", "bioclip"])
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--out", type=Path, default=Path("models"))
    main(parser.parse_args())
