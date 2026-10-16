"""PyTorch vs. ONNX vs. ONNX INT8 for BioCLIP: file size, latency for one photo, and accuracy on the test set.

usage: uv run --group onnx scripts/benchmark.py   (after scripts/export_onnx.py bioclip)
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_testset  # noqa: E402

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.onnx_encoder import OnnxEncoder
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.metrics import rank_accuracy

LATENCY_RUNS = 50


def variants(model_dir):
    if torch.cuda.is_available():
        yield "pytorch, GPU", lambda: OpenClipEncoder("bioclip", device="cuda"), None
    yield "pytorch, CPU", lambda: OpenClipEncoder("bioclip", device="cpu"), None
    yield "onnx, CPU", lambda: OnnxEncoder("bioclip", model_dir), model_dir / "bioclip_image.onnx"
    yield "onnx int8, CPU", lambda: OnnxEncoder("bioclip", model_dir, int8=True), model_dir / "bioclip_image_int8.onnx"


def main(args):
    taxa = read_candidates(args.candidates)
    paths, truths = load_testset(args.testset, taxa)
    torch_size = sum(p.numel() * p.element_size() for p in OpenClipEncoder("bioclip", device="cpu").model.visual.parameters())

    rows = []
    for label, make, onnx_file in variants(args.models):
        encoder = make()
        encoder.encode_images(paths[:3])  # warm-up
        times = []
        for p in paths[:LATENCY_RUNS]:
            start = time.perf_counter()
            encoder.encode_images([p])
            times.append((time.perf_counter() - start) * 1000)
        txt = encoder.encode_texts(prompts_for(taxa, style="bioclip"))
        probs = species_probabilities(encoder.encode_images(paths), txt, encoder.logit_scale).tolist()
        rows.append({
            "variant": label,
            "image_encoder_mb": round((onnx_file.stat().st_size if onnx_file else torch_size) / 1e6, 1),
            "latency_p50_ms": round(float(np.percentile(times, 50)), 1),
            "latency_p95_ms": round(float(np.percentile(times, 95)), 1),
            "species_accuracy": round(rank_accuracy(taxa, probs, truths, "species"), 4),
            "genus_accuracy": round(rank_accuracy(taxa, probs, truths, "genus"), 4),
        })
        print("  ".join(f"{k}={v}" for k, v in rows[-1].items()), flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "benchmark.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {args.out}/benchmark.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--out", type=Path, default=Path("results/serving"))
    main(parser.parse_args())
