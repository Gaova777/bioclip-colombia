"""Fit a temperature per model and pick the threshold, on the validation set (never on the test set).

The threshold is the one that answers the most photos down to species while keeping
the share of wrong answers at or below --max-error.

usage: uv run scripts/calibrate.py --max-error 0.05
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import confidence_and_correct, image_embeddings, load_testset  # noqa: E402

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.calibration import fit_logit_scale
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.metrics import calibration_error, pick_threshold, risk_coverage

THRESHOLDS = [round(t, 2) for t in np.arange(0.3, 1.0, 0.01)]


def main(args):
    taxa = read_candidates(args.candidates)
    paths, truths = load_testset(args.valset, taxa)
    truth_idx = np.array([taxa.index(t) for t in truths])
    print(f"{len(paths)} validation images")

    result = {"max_error": args.max_error, "validation_images": len(paths)}
    for model in ["clip", "bioclip"]:
        encoder = OpenClipEncoder(model)
        img = image_embeddings(encoder, paths, args.cache, args.valset.name)
        txt = encoder.encode_texts(prompts_for(taxa, style=model))

        scale = fit_logit_scale(img, txt, truth_idx)
        before = species_probabilities(img, txt, encoder.logit_scale).tolist()
        after = species_probabilities(img, txt, scale).tolist()
        points = risk_coverage(taxa, after, truths, THRESHOLDS)
        try:
            point = pick_threshold(points, args.max_error)
        except ValueError:
            point = None

        result[model] = {
            "logit_scale": scale,
            "threshold": point.threshold if point else None,
            "validation": {
                "logit_scale_original": encoder.logit_scale,
                "ece_original": calibration_error(*confidence_and_correct(taxa, before, truths)),
                "ece_calibrated": calibration_error(*confidence_and_correct(taxa, after, truths)),
                **({k: v for k, v in vars(point).items() if k != "threshold"} if point else {}),
            },
        }
        v = result[model]["validation"]
        print(f"\n== {model} ==")
        print(f"  logit scale {encoder.logit_scale:.1f} -> {scale:.1f}, "
              f"calibration error {v['ece_original']:.3f} -> {v['ece_calibrated']:.3f}")
        if point:
            print(f"  threshold {point.threshold:.0%}: {point.species_coverage:.1%} reach species, "
                  f"{point.any_error:.1%} of answers wrong")
        else:
            print(f"  no threshold keeps the error at or below {args.max_error:.0%}")

    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--valset", type=Path, default=Path("data/valset"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--max-error", type=float, default=0.05)
    parser.add_argument("--out", type=Path, default=Path("data/calibration.json"))
    main(parser.parse_args())
