from typing import Literal

from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

from bioclip_co.ports.field_guide import Candidate


class Question(BaseModel):
    question: str = Field(description="one short yes/no question about a single visible feature")


class Fit(BaseModel):
    verdict: Literal["agrees", "contradicts", "not mentioned"]


def describe(candidates: list[Candidate], notes: dict[str, str]) -> str:
    return "\n\n".join(f"{c['name']} ({c['common']}):\n{notes.get(c['name']) or 'no notes'}" for c in candidates)


class OllamaWriter:
    """FieldGuideWriter on a local model served by Ollama."""

    def __init__(self, model: str = "qwen3:4b-instruct", base_url: str = "http://localhost:11434"):
        self.llm = ChatOllama(model=model, base_url=base_url, temperature=0)

    def question(self, candidates: list[Candidate], notes: dict[str, str], already_asked: list[str]) -> str:
        asked = "".join(f"\n- {q}" for q in already_asked)
        prompt = (
            "You help a birdwatcher in Colombia. Their photo is one of these species:\n\n"
            f"{describe(candidates, notes)}\n\n"
            "Write ONE short yes/no question about a single visible feature (one colour, the bill, the tail, one marking) "
            "that best tells these species apart. Only one feature: a 'no' to a question about two features is ambiguous. "
            "Base it only on the notes. Don't name the species in the question."
            + (f"\nDon't repeat these, they were already asked:{asked}" if asked else "")
        )
        return self.llm.with_structured_output(Question).invoke(prompt).question

    def fits(self, candidate: Candidate, note: str, question: str, answer: str) -> bool | None:
        prompt = (
            f"Notes about {candidate['name']} ({candidate['common']}):\n{note or 'no notes'}\n\n"
            f"A birdwatcher looking at a bird was asked: {question}\nThey answered: {answer}\n\n"
            f"Does that answer agree with the notes about {candidate['name']}, contradict them, "
            "or do the notes not mention it?"
        )
        verdict = self.llm.with_structured_output(Fit).invoke(prompt).verdict
        return {"agrees": True, "contradicts": False}.get(verdict)

    def card(self, name: str, common: str, notes: str) -> str:
        prompt = (
            f"Write a field guide entry for {name}{f' ({common})' if common else ''} in at most 70 words, "
            "plain English, no headings, no lists. Mention how to recognize it and where it lives. "
            f"Use only these notes and don't add facts that aren't in them:\n\n{notes}"
        )
        return self.llm.invoke(prompt).content.strip()
