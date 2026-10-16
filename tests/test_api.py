import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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


@pytest.fixture
def client():
    app = create_app(load=lambda: {"taxa": TAXA, "models": {"bioclip": Classifier(FakeEncoder(), TAXA)}, "threshold": 0.7})
    with TestClient(app) as c:
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
    assert client.get("/health").json() == {"status": "ok", "models": ["bioclip"], "threshold": 0.7}
