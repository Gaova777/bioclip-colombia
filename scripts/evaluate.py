"""Zero-shot evaluation of CLIP and BioCLIP on the iNaturalist test set.

usage: uv run scripts/evaluate.py --candidates data/candidates/aves_colombia.csv
"""
import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.metrics import mistake_depth, rank_accuracy, risk_coverage
from bioclip_co.domain.taxonomy import RANKS

THRESHOLDS = [1e-9] + [round(t, 2) for t in np.arange(0.1, 1.0, 0.05)] + [0.99]
COLORS = {"clip": "#8a8a8a", "bioclip": "#2a7d4f"}


def load_testset(testset_dir, taxa):
    by_name = {t.scientific_name: t for t in taxa}
    with open(testset_dir / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    paths = [testset_dir / r["path"] for r in rows]
    truths = [by_name[f"{r['genus']} {r['species']}"] for r in rows]
    return paths, truths


def image_embeddings(encoder, paths, cache_dir):
    # encoding ~1000 images on a laptop CPU takes minutes; thresholds and metrics take seconds
    cache = cache_dir / f"{encoder.name}_images.npz"
    keys = np.array([p.as_posix() for p in paths])
    if cache.exists():
        saved = np.load(cache)
        if np.array_equal(saved["keys"], keys):
            return saved["emb"]
    emb = encoder.encode_images(paths)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.savez(cache, keys=keys, emb=emb)
    return emb


def plot(curves, out):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for model, points in curves.items():
        ax.plot([p.species_coverage for p in points], [p.species_error for p in points],
                marker="o", ms=3, color=COLORS[model], label=model)
    ax.set_xlabel("coverage: share of images answered down to species")
    ax.set_ylabel("error among species answers")
    ax.set_title("Risk vs. coverage, 200 Colombian birds")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)


def main(args):
    taxa = read_candidates(args.candidates)
    paths, truths = load_testset(args.testset, taxa)
    print(f"{len(paths)} images, {len(taxa)} candidate species")
    args.out.mkdir(parents=True, exist_ok=True)

    curves, rank_rows, curve_rows = {}, [], []
    for model in ["clip", "bioclip"]:
        encoder = OpenClipEncoder(model)
        img = image_embeddings(encoder, paths, args.cache)
        txt = encoder.encode_texts(prompts_for(taxa, style=model))
        probs = species_probabilities(img, txt, encoder.logit_scale).tolist()

        # every candidate is a bird, so kingdom, phylum and class are always right
        accuracy = {rank: rank_accuracy(taxa, probs, truths, rank) for rank in RANKS[3:]}
        depths = mistake_depth(taxa, probs, truths)
        rank_rows.append({"model": model, **accuracy, "mean_mistake_depth": float(np.mean(depths))})

        curves[model] = risk_coverage(taxa, probs, truths, THRESHOLDS)
        curve_rows += [{"model": model, **vars(p)} for p in curves[model]]

        print(f"\n== {model} ==")
        print("  top-1 accuracy: " + "  ".join(f"{r} {a:.1%}" for r, a in accuracy.items()))
        print(f"  wrong species still share {np.mean(depths):.2f} of 7 ranks with the truth on average")

    for name, rows in [("accuracy_by_rank.csv", rank_rows), ("risk_coverage.csv", curve_rows)]:
        with open(args.out / name, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    plot(curves, args.out / "risk_coverage.png")
    print(f"\nresults in {args.out}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/aves_colombia"))
    main(parser.parse_args())
