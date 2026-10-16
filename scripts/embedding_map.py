"""Where do the test photos land in each model's embedding space? UMAP, colored by order.

The plot is for looking. The number is neighbor agreement, in the full 512-d space: for each photo,
the share of its 10 nearest photos (cosine) that belong to the same order, family, genus or species.

usage: uv run --group analysis scripts/embedding_map.py   (after scripts/evaluate.py has cached the embeddings)
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import umap

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.domain.taxonomy import RANKS

TOP_ORDERS = 8
NEIGHBORS = 10


def neighbor_agreement(emb, labels):
    sims = emb @ emb.T
    np.fill_diagonal(sims, -np.inf)  # a photo is not its own neighbor
    nearest = np.argsort(-sims, axis=1)[:, :NEIGHBORS]
    labels = np.array(labels)
    return float((labels[nearest] == labels[:, None]).mean())


def main(args):
    taxa = {t.scientific_name: t for t in read_candidates(args.candidates)}
    with open(args.testset / "manifest.csv", encoding="utf-8") as f:
        truths = [taxa[f"{r['genus']} {r['species']}"] for r in csv.DictReader(f)]

    orders = [t.at("order") for t in truths]
    shown = [o for o, _ in Counter(orders).most_common(TOP_ORDERS)]
    colors = dict(zip(shown, plt.get_cmap("tab10").colors))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    rows = []
    for ax, model in zip(axes, ["clip", "bioclip"]):
        emb = np.load(args.cache / f"{args.testset.name}_{model}_images.npz")["emb"]
        # labels are full paths, so two genera with the same name in different families don't match
        rows.append({"model": model} | {
            rank: neighbor_agreement(emb, [" ".join(t.prefix(rank)) for t in truths]) for rank in RANKS[3:]
        })
        xy = umap.UMAP(metric="cosine", random_state=0).fit_transform(emb)

        other = [o not in colors for o in orders]
        ax.scatter(xy[other, 0], xy[other, 1], s=6, c="#cccccc", label="other orders")
        for order, color in colors.items():
            mask = [o == order for o in orders]
            ax.scatter(xy[mask, 0], xy[mask, 1], s=8, color=color, label=order)
        ax.set_title(f"{model}: {rows[-1]['order']:.0%} of nearest neighbors share the order")
        ax.set_xticks([])
        ax.set_yticks([])
        print(f"{model}: neighbors in the same " + ", ".join(f"{r} {rows[-1][r]:.1%}" for r in RANKS[3:]))

    axes[1].legend(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, markerscale=2)
    fig.suptitle("1000 test photos in each model's image space, colored by order")
    fig.tight_layout()
    fig.savefig(args.out / "umap_by_order.png", dpi=150)

    with open(args.out / "embedding_clusters.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {args.out}/umap_by_order.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--out", type=Path, default=Path("results/aves_colombia"))
    main(parser.parse_args())
