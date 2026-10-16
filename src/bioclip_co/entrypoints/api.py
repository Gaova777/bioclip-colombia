"""HTTP API for the web demo: a photo goes in, the probability of every node of the tree comes out, per model.

The threshold is not applied here. The browser walks down the tree itself, so moving the slider
never calls the server.

usage: uv run uvicorn bioclip_co.entrypoints.api:app
"""
import io
import json
import os
import urllib.request
import uuid
from collections.abc import Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langgraph.types import Command
from PIL import Image
from pydantic import BaseModel

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.taxonomy import RANKS, Taxon, marginalize
from bioclip_co.ports.encoder import ImageTextEncoder

CANDIDATES = os.environ.get("CANDIDATES", "data/candidates/aves_colombia.csv")
CALIBRATION = Path(os.environ.get("CALIBRATION", "data/calibration.json"))  # made by scripts/calibrate.py
DEFAULT_THRESHOLD = 0.8
ALLOWED_ORIGINS = os.environ.get("ALLOWED_ORIGINS", "http://localhost:5173").split(",")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3:4b-instruct")


class Classifier:
    def __init__(self, encoder: ImageTextEncoder, taxa: Sequence[Taxon], logit_scale: float | None = None):
        self.encoder = encoder
        self.taxa = taxa
        self.logit_scale = logit_scale or encoder.logit_scale
        # the candidates never change, so their texts are encoded once at startup
        self.text_emb = encoder.encode_texts(prompts_for(taxa, style=encoder.name))

    def species_probs(self, image: io.BytesIO) -> list[float]:
        img = self.encoder.encode_images([image])
        return species_probabilities(img, self.text_emb, self.logit_scale)[0].tolist()

    def tree(self, image: io.BytesIO) -> list[dict]:
        probs = self.species_probs(image)
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
    return {"taxa": taxa, "models": models, "threshold": threshold, "guide": load_field_guide(taxa, threshold)}


def load_field_guide(taxa, threshold):
    """The agent needs a local Ollama server; without one the API still classifies, it just can't ask questions."""
    try:
        urllib.request.urlopen(f"{OLLAMA_URL}/api/version", timeout=2)
    except OSError:
        print(f"no Ollama at {OLLAMA_URL}, /agent is off")
        return None
    from bioclip_co.adapters.gbif import Gbif
    from bioclip_co.adapters.ollama_writer import OllamaWriter
    from bioclip_co.adapters.wikipedia import Wikipedia
    from bioclip_co.application.field_guide import build_field_guide

    return build_field_guide(taxa, threshold, Gbif(), Wikipedia(), OllamaWriter(OLLAMA_MODEL, OLLAMA_URL))


class Answer(BaseModel):
    thread_id: str
    answer: str


def agent_reply(result: dict, thread_id: str) -> dict:
    reply = {"thread_id": thread_id, "trace": result.get("trace", [])}
    if "__interrupt__" in result:
        pause = result["__interrupt__"][0].value
        return reply | {"status": "question", "question": pause["question"], "candidates": pause["candidates"]}
    return reply | {"status": "done", "card": result["card"]}


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
        return {
            "status": "ok", "models": list(state["models"]), "threshold": state["threshold"],
            "agent": state.get("guide") is not None,
        }

    @app.get("/candidates")
    def candidates():
        return [{"lineage": list(t.lineage), "common_name": t.common_name} for t in state["taxa"]]

    def read_image(file: UploadFile) -> bytes:
        data = file.file.read()
        try:
            Image.open(io.BytesIO(data)).verify()
        except OSError:  # UnidentifiedImageError and truncated files
            raise HTTPException(400, "the file is not an image")
        return data

    @app.post("/classify")
    def classify(file: UploadFile):
        data = read_image(file)
        return {name: model.tree(io.BytesIO(data)) for name, model in state["models"].items()}

    @app.post("/agent/start")
    def agent_start(file: UploadFile, region: str | None = Form(None)):
        guide = require_guide()
        data = read_image(file)
        thread_id = uuid.uuid4().hex
        probs = state["models"]["bioclip"].species_probs(io.BytesIO(data))
        result = guide.invoke({"probs": probs, "region": region or None}, {"configurable": {"thread_id": thread_id}})
        return agent_reply(result, thread_id)

    @app.post("/agent/answer")
    def agent_answer(body: Answer):
        guide = require_guide()
        config = {"configurable": {"thread_id": body.thread_id}}
        if not guide.get_state(config).next:
            raise HTTPException(404, "no question is waiting for that thread")
        return agent_reply(guide.invoke(Command(resume=body.answer), config), body.thread_id)

    def require_guide():
        if state.get("guide") is None:
            raise HTTPException(503, "the field guide needs a local Ollama server")
        return state["guide"]

    return app


app = create_app()
