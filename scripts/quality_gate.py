"""Model quality gate: evaluate on the test set, log the run to MLflow, fail if the model got worse.

The limits live in quality_gate.json. With --register, a model that passes becomes a new version
of "bioclip-colombia" in the MLflow model registry.

usage:
    uv run --group mlops scripts/quality_gate.py --register
    uv run --group mlops scripts/quality_gate.py --model clip     # a worse model, to see the gate fail
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import confidence_and_correct, load_testset  # noqa: E402

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.metrics import calibration_error, rank_accuracy, risk_coverage


def check(metrics: dict, limits: dict) -> list[str]:
    failures = []
    for name, limit in limits.items():
        value = metrics[name]
        if "min" in limit and value < limit["min"]:
            failures.append(f"{name} = {value:.3f}, below {limit['min']}")
        if "max" in limit and value > limit["max"]:
            failures.append(f"{name} = {value:.3f}, above {limit['max']}")
    return failures


def main(args):
    taxa = read_candidates(args.candidates)
    paths, truths = load_testset(args.testset, taxa)
    calibration = json.loads(args.calibration.read_text())[args.model]
    limits = json.loads(args.limits.read_text())

    encoder = OpenClipEncoder(args.model)
    txt = encoder.encode_texts(prompts_for(taxa, style=args.model))
    probs = species_probabilities(encoder.encode_images(paths), txt, calibration["logit_scale"]).tolist()
    threshold = calibration["threshold"] or 0.99
    [point] = risk_coverage(taxa, probs, truths, [threshold])
    metrics = {
        "species_accuracy": rank_accuracy(taxa, probs, truths, "species"),
        "genus_accuracy": rank_accuracy(taxa, probs, truths, "genus"),
        "calibration_error": calibration_error(*confidence_and_correct(taxa, probs, truths)),
        "species_coverage_at_threshold": point.species_coverage,
        "any_error_at_threshold": point.any_error,
    }
    failures = check(metrics, limits)

    mlflow.set_tracking_uri(args.tracking_uri)
    mlflow.set_experiment("bioclip-colombia")
    with mlflow.start_run(run_name=f"{args.model} quality gate") as run:
        mlflow.log_params({
            "model": args.model, "logit_scale": calibration["logit_scale"], "threshold": threshold,
            "test_images": len(paths), "candidates": len(taxa),
        })
        mlflow.log_metrics(metrics)
        mlflow.set_tag("gate", "failed" if failures else "passed")
        mlflow.log_dict(limits, "quality_gate.json")
        if not failures and args.register and args.model == "bioclip":
            from bioclip_co.entrypoints.mlflow_model import BioclipClassifier

            mlflow.pyfunc.log_model(
                name="model",
                python_model=BioclipClassifier(),
                artifacts={"candidates": str(args.candidates), "calibration": str(args.calibration)},
                code_paths=["src/bioclip_co"],
                registered_model_name="bioclip-colombia",
            )

    for name, value in metrics.items():
        print(f"  {name:32} {value:.3f}")
    print(f"MLflow run {run.info.run_id}")
    if failures:
        print("QUALITY GATE FAILED:\n  " + "\n  ".join(failures))
        sys.exit(1)
    print("quality gate passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["bioclip", "clip"], default="bioclip")
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    parser.add_argument("--limits", type=Path, default=Path("quality_gate.json"))
    parser.add_argument("--tracking-uri", default="sqlite:///mlflow.db")
    parser.add_argument("--register", action="store_true")
    main(parser.parse_args())
