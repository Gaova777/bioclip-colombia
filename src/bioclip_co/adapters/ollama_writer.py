from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from bioclip_co.domain.field_marks import COLOR, VOCABULARY

Color = Literal[tuple(COLOR)]


class FieldMarks(BaseModel):
    """The adult male, one value per part from a closed vocabulary; null when the text doesn't say."""

    crown: Color | None = Field(description="main colour of the top of the head")
    throat: Color | None = Field(description="main colour of the throat")
    underparts: Color | None = Field(description="main colour of the breast and belly")
    back: Color | None = Field(description="main colour of the back")
    tail: Color | None = Field(description="main colour of the tail")
    bill_color: Literal[tuple(VOCABULARY["bill_color"])] | None
    bill_shape: Literal[tuple(VOCABULARY["bill_shape"])] | None
    tail_shape: Literal[tuple(VOCABULARY["tail_shape"])] | None
    sexes_differ: bool = Field(description="true if the text says females look different from males")


class OllamaWriter:
    """FieldGuideWriter on a local model served by Ollama, plus the field marks extraction (scripts/field_marks.py)."""

    def __init__(self, model: str = "qwen3:4b-instruct", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0)

    def field_marks(self, name: str, common: str, notes: str) -> dict:
        prompt = (
            f"Description of {name} ({common}):\n\n{notes}\n\n"
            "Fill in the field marks of the adult male from this description only. For each part pick the single "
            "closest value from the allowed list. Use null when the description doesn't say; never guess "
            "from what you know about other birds. Iridescent or glittering colours count as their hue "
            "(glittering green is green)."
        )
        return self.llm.with_structured_output(FieldMarks).invoke(prompt).model_dump()

    def card(self, name: str, common: str, notes: str) -> str:
        prompt = (
            f"Write a field guide entry for {name}{f' ({common})' if common else ''} in at most 70 words, "
            "plain English, no headings, no lists. Mention how to recognize it and where it lives. "
            f"Use only these notes and don't add facts that aren't in them:\n\n{notes}"
        )
        return self.llm.invoke(prompt).content.strip()
