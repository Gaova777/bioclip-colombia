import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from bioclip_co.application.field_guide import build_field_guide
from bioclip_co.domain.taxonomy import Taxon
from bioclip_co.entrypoints.api import Classifier, create_app

TAXA = [
    Taxon(("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae", "Amazilia", "tzacatl"), "rufous-tailed hummingbird"),
    Taxon(("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae", "Amazilia", "amazilia"), "amazilia hummingbird"),
    Taxon(("Animalia", "Chordata", "Aves", "Passeriformes", "Thraupidae", "Sicalis", "flaveola"), "saffron finch"),
]


class FakeEncoder:
    """Text i is the unit vector e_i; every image looks mostly like the first candidate."""

    name = "bioclip"
    logit_scale = 10.0

    def encode_texts(self, texts):
        return np.eye(len(texts))

    def encode_images(self, images):
        img = np.array([[0.8, 0.5, 0.33]])
        return img / np.linalg.norm(img)


class FakeWriter:
    def question(self, candidates, notes, already_asked):
        return "Is the tail rufous?"

    def fits(self, candidate, note, question, answer):
        return candidate["name"] == "Amazilia tzacatl" if answer == "yes" else None

    def card(self, name, common, notes):
        return f"about {name}"


class NoRecords:
    def records_in(self, name, region):
        return 0

    def red_list(self, name):
        return None


class Notes:
    def summary(self, name):
        return {"title": name, "extract": "notes", "url": "https://example.org"}


def load(threshold=0.7):
    guide = build_field_guide(TAXA, threshold, NoRecords(), Notes(), FakeWriter())
    return {"taxa": TAXA, "models": {"bioclip": Classifier(FakeEncoder(), TAXA)}, "threshold": threshold, "guide": guide}


@pytest.fixture
def client():
    with TestClient(create_app(load=load)) as c:
        yield c


def png():
    buf = io.BytesIO()
    Image.new("RGB", (8, 8)).save(buf, format="PNG")
    return buf.getvalue()


def test_classify_returns_every_node_of_the_tree(client):
    nodes = client.post("/classify", files={"file": ("bird.png", png())}).json()["bioclip"]
    by_rank = {}
    for n in nodes:
        by_rank.setdefault(n["rank"], []).append(n)

    assert len(by_rank["species"]) == 3
    assert sum(n["p"] for n in by_rank["species"]) == pytest.approx(1, abs=1e-4)
    assert by_rank["species"][0]["path"][-2:] == ["Amazilia", "tzacatl"]
    # the genus is the sum of its two species
    amazilia = next(n for n in by_rank["genus"] if n["path"][-1] == "Amazilia")
    assert amazilia["p"] == pytest.approx(by_rank["species"][0]["p"] + by_rank["species"][1]["p"], abs=1e-4)
    assert by_rank["class"] == [{"rank": "class", "path": ["Animalia", "Chordata", "Aves"], "p": 1.0}]


def test_classify_rejects_files_that_are_not_images(client):
    resp = client.post("/classify", files={"file": ("notes.txt", b"not a photo")})
    assert resp.status_code == 400


def test_candidates_and_health(client):
    assert len(client.get("/candidates").json()) == 3
    assert client.get("/health").json() == {"status": "ok", "models": ["bioclip"], "threshold": 0.7, "agent": True}


def test_the_agent_asks_then_answers():
    # the fake image is 94% the first candidate: at a 0.99 threshold it stops at the genus (99.1%)
    with TestClient(create_app(load=lambda: load(threshold=0.99))) as client:
        start = client.post("/agent/start", files={"file": ("bird.png", png())}).json()
        assert start["status"] == "question"
        assert start["question"] == "Is the tail rufous?"

        done = client.post("/agent/answer", json={"thread_id": start["thread_id"], "answer": "yes"}).json()
        assert done["status"] == "done"
        assert done["card"]["name"] == "Amazilia tzacatl"


def test_answering_a_thread_that_isnt_waiting(client):
    assert client.post("/agent/answer", json={"thread_id": "nope", "answer": "yes"}).status_code == 404
