import base64
import json
import urllib.request
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from bioclip_co.domain.field_marks import COLOR, VOCABULARY

Color = Literal[tuple(COLOR)]


class SeenMarks(BaseModel):
    """What a vision-language model sees in one photo; null for parts it can't see clearly."""

    crown: Color | None = Field(description="main colour of the top of the head")
    throat: Color | None = Field(description="main colour of the throat")
    underparts: Color | None = Field(description="main colour of the breast and belly")
    back: Color | None = Field(description="main colour of the back")
    tail: Color | None = Field(description="main colour of the tail")
    bill_color: Literal[tuple(VOCABULARY["bill_color"])] | None
    bill_shape: Literal[tuple(VOCABULARY["bill_shape"])] | None
    tail_shape: Literal[tuple(VOCABULARY["tail_shape"])] | None


PROMPT = (
    "Look at the bird in this photo and fill in its field marks: the main colour you can see of each body part, "
    "and the shape of the bill and tail. Use null for any part you can't see clearly. Pick from the allowed values."
)


class OllamaVision:
    """Field marks seen in a photo, by a local vision-language model served by Ollama."""

    def __init__(self, model: str = "qwen3-vl:8b-instruct", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url

    def marks(self, photo: Path) -> dict:
        body = {
            "model": self.model,
            "messages": [{"role": "user", "content": PROMPT, "images": [base64.b64encode(photo.read_bytes()).decode()]}],
            "format": SeenMarks.model_json_schema(),
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096},
        }
        req = urllib.request.Request(
            f"{self.base_url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp:
            return SeenMarks.model_validate_json(json.load(resp)["message"]["content"]).model_dump()
