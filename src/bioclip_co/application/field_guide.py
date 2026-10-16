"""The field guide agent: what to do when the classifier stops before the species.

    classify ──(reached species)──────────────────────────────▶ card
       │
       ▼
    narrow by region ──(one candidate left, or one is sure)───▶ card
       │
       ▼
    ask ──▶ wait for the human (interrupt) ──▶ decide ──(settled, or 2 questions asked)──▶ card
     ▲                                            │
     └──────────────(still unsure)────────────────┘

The classifier proposes the candidates and their probabilities; the language model writes the question,
judges for each candidate whether the answer fits its notes, and writes the card. It never names a species of its own.
"""
import operator
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from bioclip_co.domain.abstention import descend
from bioclip_co.domain.taxonomy import RANKS, Taxon
from bioclip_co.ports.field_guide import Candidate, Encyclopedia, FieldGuideWriter, Occurrences

MAX_CANDIDATES = 3  # more than three is too much to tell apart with one question
MAX_QUESTIONS = 2
REGION_CHECKS = 8  # only the most likely candidates are looked up in GBIF; the rest barely matter
THREATENED = {"VU", "EN", "CR"}


class GuideState(TypedDict, total=False):
    probs: list[float]  # species probabilities from the classifier
    region: str | None  # Colombian department, as GBIF spells it
    path: tuple[str, ...]  # the deepest node the classifier trusts
    candidates: list[Candidate]
    notes: dict[str, str]
    question: str
    answer: str
    asked: list[str]
    species: str | None
    card: dict
    trace: Annotated[list[str], operator.add]  # what happened, step by step, for the page


def common_path(taxa: Sequence[Taxon]) -> tuple[str, ...]:
    """The deepest node all these taxa share."""
    path = []
    for names in zip(*(t.lineage for t in taxa)):
        if len(set(names)) > 1:
            break
        path.append(names[0])
    return tuple(path)


def build_field_guide(
    taxa: Sequence[Taxon],
    threshold: float,
    occurrences: Occurrences,
    encyclopedia: Encyclopedia,
    writer: FieldGuideWriter,
):
    by_name = {t.scientific_name: t for t in taxa}

    def classify(state: GuideState) -> GuideState:
        pred = descend(taxa, state["probs"], threshold)
        inside = [(t, p) for t, p in zip(taxa, state["probs"]) if t.lineage[: len(pred.path)] == pred.path]
        total = sum(p for _, p in inside)
        candidates = sorted(
            ({"name": t.scientific_name, "common": t.common_name, "p": p / total} for t, p in inside),
            key=lambda c: -c["p"],
        )
        update = {"path": pred.path, "candidates": candidates, "trace": [f"BioCLIP: {pred.rank} {pred.name}, {pred.confidence:.0%} sure"]}
        if pred.reached_species:
            update["species"] = pred.name
        return update

    def narrow_by_region(state: GuideState) -> GuideState:
        region, candidates = state["region"], state["candidates"][:REGION_CHECKS]
        with ThreadPoolExecutor(max_workers=REGION_CHECKS) as pool:
            counts = list(pool.map(lambda c: occurrences.records_in(c["name"], region), candidates))
        kept = [c for c, n in zip(candidates, counts) if n > 0]
        if not kept:  # nothing recorded there: don't throw away what the photo says
            return {"trace": [f"GBIF: none of the {len(candidates)} candidates has records in {region}, keeping them all"]}
        total = sum(c["p"] for c in kept)
        kept = [{**c, "p": c["p"] / total} for c in kept]
        update = {"candidates": kept, "trace": [f"GBIF: {len(kept)} of {len(candidates)} candidates recorded in {region}"]}
        if len(kept) == 1 or kept[0]["p"] >= threshold:
            update["species"] = kept[0]["name"]
        return update

    def ask(state: GuideState) -> GuideState:
        top = state["candidates"][:MAX_CANDIDATES]
        with ThreadPoolExecutor(max_workers=MAX_CANDIDATES) as pool:
            found = list(pool.map(lambda c: encyclopedia.summary(c["name"]), top))
        notes = state.get("notes", {}) | {c["name"]: note["extract"] if note else "" for c, note in zip(top, found)}
        asked = state.get("asked", [])
        question = writer.question(top, notes, asked)
        return {
            "candidates": top, "notes": notes, "question": question, "asked": asked + [question],
            "trace": [f"Field guide asks: {question}"],
        }

    def wait_for_answer(state: GuideState) -> GuideState:
        # the graph pauses here until the person answers (POST /agent/answer resumes it)
        answer = interrupt({"question": state["question"], "candidates": state["candidates"]})
        return {"answer": answer, "trace": [f"You: {answer}"]}

    def decide(state: GuideState) -> GuideState:
        candidates = state["candidates"]
        verdicts = [writer.fits(c, state["notes"].get(c["name"], ""), state["question"], state["answer"]) for c in candidates]
        kept = [c for c, v in zip(candidates, verdicts) if v is not False] or candidates  # all ruled out: keep them
        ruled_out = [c["name"] for c, v in zip(candidates, verdicts) if v is False]
        total = sum(c["p"] for c in kept)
        kept = [{**c, "p": c["p"] / total} for c in kept]
        # the classifier's probabilities still decide among the candidates that are left
        species = kept[0]["name"] if len(kept) == 1 or kept[0]["p"] >= threshold else None
        path = by_name[species].lineage if species else common_path([by_name[c["name"]] for c in kept])
        said = f"rules out {', '.join(ruled_out)}" if ruled_out else "the answer doesn't rule anything out"
        return {"candidates": kept, "species": species, "path": path, "trace": [f"Field guide: {said}"]}

    def card(state: GuideState) -> GuideState:
        species = state.get("species")
        name = species or state["path"][-1]
        common = next((c["common"] for c in state["candidates"] if c["name"] == species), "")
        note = encyclopedia.summary(name)
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

    graph = StateGraph(GuideState)
    for node in [classify, narrow_by_region, ask, wait_for_answer, decide, card]:
        graph.add_node(node.__name__, node)
    graph.add_edge(START, "classify")
    graph.add_conditional_edges("classify", after_classify, ["card", "narrow_by_region", "ask"])
    graph.add_conditional_edges("narrow_by_region", lambda s: "card" if s.get("species") else "ask", ["card", "ask"])
    graph.add_edge("ask", "wait_for_answer")
    graph.add_edge("wait_for_answer", "decide")
    graph.add_conditional_edges(
        "decide",
        lambda s: "card" if s.get("species") or len(s["asked"]) >= MAX_QUESTIONS else "ask",
        ["card", "ask"],
    )
    graph.add_edge("card", END)
    # the checkpointer keeps each conversation's state between the question and the answer
    return graph.compile(checkpointer=InMemorySaver())
