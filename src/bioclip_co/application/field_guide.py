"""The field guide agent: what to do when the classifier stops before the species.

    classify ──(reached species)──────────────────────────────▶ card
       │
       ▼
    narrow by region ──(one candidate left, or one is sure)───▶ card
       │
       ▼
    ask ──(no useful question left)───────────────────────────▶ card
     │ ▲
     ▼ └──(still unsure, < 3 questions)──┐
    wait for the human (interrupt) ──▶ update ──(sure, or 3 questions)──▶ card

The question is the one with the most expected information about the candidates (domain/field_marks.py),
and the answer updates the classifier's probabilities with Bayes' rule. The language model only writes
the final card from cited text; it never chooses a species.
"""
import operator
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from bioclip_co.domain import field_marks
from bioclip_co.domain.abstention import descend
from bioclip_co.domain.taxonomy import RANKS, Taxon
from bioclip_co.ports.field_guide import Candidate, Encyclopedia, FieldGuideWriter, Occurrences

POOL = 8  # candidates the agent keeps track of; the rest of the node barely matters
MAX_QUESTIONS = 3
THREATENED = {"VU", "EN", "CR"}


class Question(TypedDict):
    feature: str
    value: str
    text: str


class GuideState(TypedDict, total=False):
    probs: list[float]  # species probabilities from the classifier
    region: str | None  # Colombian department, as GBIF spells it
    path: tuple[str, ...]  # the deepest node the agent trusts
    candidates: list[Candidate]
    question: Question | None
    asked: list[tuple[str, str]]
    species: str | None
    card: dict
    trace: Annotated[list[str], operator.add]  # what happened, step by step, for the page


def top_two(candidates: list[Candidate]) -> str:
    return ", ".join(f"{c['name']} {c['p']:.0%}" for c in candidates[:2])


def build_field_guide(
    taxa: Sequence[Taxon],
    threshold: float,
    occurrences: Occurrences,
    encyclopedia: Encyclopedia,
    writer: FieldGuideWriter,
    marks: Mapping[str, Mapping],
):
    by_name = {t.scientific_name: t for t in taxa}

    def settle(candidates: list[Candidate]) -> dict:
        """Where the candidates' probabilities let us stand: the same walk down the tree as the classifier."""
        pred = descend([by_name[c["name"]] for c in candidates], [c["p"] for c in candidates], threshold)
        return {"path": pred.path, "species": pred.name if pred.reached_species else None}

    def classify(state: GuideState) -> GuideState:
        pred = descend(taxa, state["probs"], threshold)
        inside = [(t, p) for t, p in zip(taxa, state["probs"]) if t.lineage[: len(pred.path)] == pred.path]
        inside = sorted(inside, key=lambda tp: -tp[1])[:POOL]
        total = sum(p for _, p in inside)
        candidates = [{"name": t.scientific_name, "common": t.common_name, "p": p / total} for t, p in inside]
        return {
            "path": pred.path,
            "species": pred.name if pred.reached_species else None,
            "candidates": candidates,
            "asked": [],
            "trace": [f"BioCLIP: {pred.rank} {pred.name}, {pred.confidence:.0%} sure"],
        }

    def narrow_by_region(state: GuideState) -> GuideState:
        region, candidates = state["region"], state["candidates"]
        with ThreadPoolExecutor(max_workers=len(candidates)) as pool:
            counts = list(pool.map(lambda c: occurrences.records_in(c["name"], region), candidates))
        kept = [c for c, n in zip(candidates, counts) if n > 0]
        if not kept:  # nothing recorded there: don't throw away what the photo says
            return {"trace": [f"GBIF: none of the {len(candidates)} candidates has records in {region}, keeping them all"]}
        total = sum(c["p"] for c in kept)
        kept = [{**c, "p": c["p"] / total} for c in kept]
        return {"candidates": kept, **settle(kept), "trace": [f"GBIF: {len(kept)} of {len(candidates)} candidates recorded in {region}"]}

    def ask(state: GuideState) -> GuideState:
        candidates = state["candidates"]
        best = field_marks.best_question([c["name"] for c in candidates], [c["p"] for c in candidates], marks, state["asked"])
        if best is None:
            return {"question": None, "trace": ["Field guide: no question would tell these candidates apart"]}
        feature, value, gain = best
        text = field_marks.question_text(feature, value)
        return {
            "question": {"feature": feature, "value": value, "text": text},
            "asked": state["asked"] + [(feature, value)],
            "trace": [f"Field guide asks: {text} (expected information: {gain:.2f} bits)"],
        }

    def wait_for_answer(state: GuideState) -> GuideState:
        # the graph pauses here until the person answers (POST /agent/answer resumes it)
        answer = interrupt({"question": state["question"]["text"], "candidates": state["candidates"]})
        names = [c["name"] for c in state["candidates"]]
        q = state["question"]
        probs = field_marks.update(names, [c["p"] for c in state["candidates"]], marks, q["feature"], q["value"], answer)
        candidates = sorted(({**c, "p": p} for c, p in zip(state["candidates"], probs)), key=lambda c: -c["p"])
        return {"candidates": candidates, **settle(candidates), "trace": [f"You: {answer}", f"Now: {top_two(candidates)}"]}

    def card(state: GuideState) -> GuideState:
        species = state.get("species")
        name = species or state["path"][-1]
        common = next((c["common"] for c in state["candidates"] if c["name"] == species), "")
        try:
            note = encyclopedia.summary(name)
        except OSError:  # no internet, or the API is throttling: the card says so instead of failing
            note = None
        status = occurrences.red_list(species) if species else None
        sources = [note["url"]] if note else []
        if species:
            sources.append(f"https://www.gbif.org/species/search?q={species.replace(' ', '%20')}")
        return {"card": {
            "name": name,
            "rank": "species" if species else RANKS[len(state["path"]) - 1],
            "common": common,
            "text": writer.card(name, common, note["extract"]) if note else "No notes found for this taxon.",
            "red_list": status,
            # a threatened species' exact location helps poachers, so the page shouldn't show it
            "hide_location": status in THREATENED,
            "sources": sources,
        }}

    def after_classify(state: GuideState) -> str:
        if state.get("species"):
            return "card"
        return "narrow_by_region" if state.get("region") else "ask"

    def after_answer(state: GuideState) -> str:
        return "card" if state.get("species") or len(state["asked"]) >= MAX_QUESTIONS else "ask"

    graph = StateGraph(GuideState)
    for node in [classify, narrow_by_region, ask, wait_for_answer, card]:
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify, ["card", "narrow_by_region", "ask"])
    graph.add_conditional_edges("narrow_by_region", lambda s: "card" if s.get("species") else "ask", ["card", "ask"])
    graph.add_conditional_edges("ask", lambda s: "wait_for_answer" if s["question"] else "card", ["wait_for_answer", "card"])
    graph.add_conditional_edges("wait_for_answer", after_answer, ["card", "ask"])
    graph.add_edge("card", END)
    # the checkpointer keeps each conversation's state between the question and the answer
    return graph.compile(checkpointer=InMemorySaver())
