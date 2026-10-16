"""HTTP API for the web demo: a photo goes in, the probability of every node of the tree comes out, per model.

The threshold is not applied here. The browser walks down the tree itself, so moving the slider
never calls the server.

usage: uv run uvicorn bioclip_co.entrypoints.api:app
"""
import io
import json
import os
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.taxonomy import RANKS, Taxon, marginalize
from bioclip_co.ports.encoder import ImageTextEncoder

CANDIDATES = os.environ.get("CANDIDATES", "data/candidates/aves_colombia.csv")
CALIBRATION = Path(os.environ.get("CALIBRATION", "data/calibration.json"))  # made by scripts/calibrate.py
DEFAULT_THRESHOLD = 0.8
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")


class Classifier:
    def __init__(self, encoder: ImageTextEncoder, taxa: Sequence[Taxon], logit_scale: float | None = None):
        self.encoder = encoder
        self.taxa = taxa
        self.logit_scale = logit_scale or encoder.logit_scale
        # the candidates never change, so their texts are encoded once at startup
        self.text_emb = encoder.encode_texts(prompts_for(taxa, style=encoder.name))

    def tree(self, image: io.BytesIO) -> list[dict]:
        img = self.encoder.encode_images([image])
        probs = species_probabilities(img, self.text_emb, self.logit_scale)[0].tolist()
        return [
            {"rank": rank, "path": list(prefix), "p": round(p, 6)}
            for rank in RANKS
            for prefix, p in marginalize(self.taxa, probs, rank).items()
        ]


def load_models() -> dict:
    from bioclip_co.adapters.openclip_encoder import OpenClipEncoder  # imports torch, keep it out of the tests

    taxa = read_candidates(CANDIDATES)
    calibration = json.loads(CALIBRATION.read_text()) if CALIBRATION.exists() else {}
    models = {
        name: Classifier(OpenClipEncoder(name), taxa, calibration.get(name, {}).get("logit_scale"))
        for name in ["clip", "bioclip"]
    }
    threshold = calibration.get("bioclip", {}).get("threshold") or DEFAULT_THRESHOLD
    return {"taxa": taxa, "models": models, "threshold": threshold}


def create_app(load: Callable[[], dict] = load_models) -> FastAPI:
    state = {}

    @asynccontextmanager
    async def lifespan(app):
        state.update(load())
        yield

    app = FastAPI(title="bioclip-colombia", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=ALLOWED_ORIGINS, allow_methods=["GET", "POST"])

    @app.get("/health")
    def health():
        return {"status": "ok", "models": list(state["models"]), "threshold": state["threshold"]}

    @app.get("/candidates")
    def candidates():
        return [{"lineage": list(t.lineage), "common_name": t.common_name} for t in state["taxa"]]

    @app.post("/classify")
    def classify(file: UploadFile):
        data = file.file.read()
        try:
            Image.open(io.BytesIO(data)).verify()
        except OSError:  # UnidentifiedImageError and truncated files
            raise HTTPException(400, "the file is not an image")
        return {name: model.tree(io.BytesIO(data)) for name, model in state["models"].items()}

    return app


app = create_app()
