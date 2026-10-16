"""AWS Lambda entrypoint: the same /classify answer, BioCLIP only, on ONNX Runtime with no torch.

The text embeddings of the candidates are precomputed (scripts/export_onnx.py), so a cold start only
loads the image encoder. API Gateway (HTTP API) sends the photo base64-encoded in the body.
"""
import base64
import io
import json
import os
from pathlib import Path

import numpy as np

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.onnx_encoder import OnnxEncoder
from bioclip_co.application.zero_shot import species_probabilities
from bioclip_co.domain.taxonomy import RANKS, marginalize

ROOT = Path(os.environ.get("LAMBDA_TASK_ROOT", "."))

# module level: loaded once per container, reused by every request it serves
TAXA = read_candidates(ROOT / "data/candidates/aves_colombia.csv")
CALIBRATION = json.loads((ROOT / "data/calibration.json").read_text())["bioclip"]
ENCODER = OnnxEncoder("bioclip", ROOT / "models", int8=os.environ.get("INT8") == "1")
TEXT_EMB = np.load(ROOT / "models/bioclip_text_emb.npy")


def handler(event, context):
    body = event.get("body") or ""
    image = base64.b64decode(body) if event.get("isBase64Encoded") else body.encode("latin-1")
    try:
        probs = species_probabilities(ENCODER.encode_images([io.BytesIO(image)]), TEXT_EMB, CALIBRATION["logit_scale"])
    except OSError:
        return {"statusCode": 400, "body": json.dumps({"detail": "the body is not an image"})}
    nodes = [
        {"rank": rank, "path": list(prefix), "p": round(p, 6)}
        for rank in RANKS
        for prefix, p in marginalize(TAXA, probs[0].tolist(), rank).items()
    ]
    return {
        "statusCode": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"bioclip": nodes, "threshold": CALIBRATION["threshold"]}),
    }
