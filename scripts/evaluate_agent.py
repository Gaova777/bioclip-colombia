"""How much does the field guide help? Simulated users on the test photos where BioCLIP stops before the species.

The simulated user answers from the true species' field marks ("can't tell" when they say nothing),
and sometimes gets it wrong on purpose (--noise). It runs the real LangGraph agent, without the
region step, Wikipedia or the LLM (they don't change which species it ends on).
This is the best case for the questions: the user agrees with the field guide. Real people will do worse.

usage: uv run scripts/evaluate_agent.py   (after scripts/evaluate.py has cached the embeddings)
"""
import argparse
import csv
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import image_embeddings, load_testset  # noqa: E402

from langgraph.types import Command

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
from bioclip_co.application.field_guide import build_field_guide
from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.abstention import descend


class Offline:
    def records_in(self, name, region):
        return 0

    def red_list(self, name):
        return None

    def summary(self, name):
        return None

    def card(self, name, common, notes):
        return ""


def answer(truth_marks, question, rng, noise):
    mark = (truth_marks or {}).get(question["feature"])
    if mark is None:
        return "can't tell"
    said = "yes" if mark == question["value"] else "no"
    if rng.random() < noise:
        said = "no" if said == "yes" else "yes"
    return said


def run(guide, probs, truth_marks, rng, noise, thread):
    config = {"configurable": {"thread_id": thread}}
    result = guide.invoke({"probs": probs, "region": None}, config)
    questions = 0
    while "__interrupt__" in result:
        state = guide.get_state(config).values
        questions += 1
        result = guide.invoke(Command(resume=answer(truth_marks, state["question"], rng, noise)), config)
    return guide.get_state(config).values, questions


def main(args):
    taxa = read_candidates(args.candidates)
    paths, truths = load_testset(args.testset, taxa)
    cal = json.loads(args.calibration.read_text())["bioclip"]
    marks = {k: v for k, v in json.loads(args.field_marks.read_text()).items() if v}

    encoder = OpenClipEncoder("bioclip")
    img = image_embeddings(encoder, paths, args.cache, args.testset.name)
    txt = encoder.encode_texts(prompts_for(taxa, style="bioclip"))
    all_probs = species_probabilities(img, txt, cal["logit_scale"]).tolist()

    preds = [descend(taxa, p, cal["threshold"]) for p in all_probs]
    unsure = [(p, t) for p, t, pred in zip(all_probs, truths, preds) if not pred.reached_species]
    print(f"{len(unsure)} of {len(truths)} photos stop before the species at {cal['threshold']:.0%}")
    # BioCLIP on its own, over all the photos: the baseline the agent has to beat
    sure_right = sum(pred.name == t.scientific_name for pred, t in zip(preds, truths) if pred.reached_species)
    sure_wrong = sum(pred.name != t.scientific_name for pred, t in zip(preds, truths) if pred.reached_species)
    unsure_wrong = sum(t.lineage[: len(pred.path)] != pred.path for pred, t in zip(preds, truths) if not pred.reached_species)
    total = len(truths)
    print(f"BioCLIP alone: right species {sure_right / total:.1%}, wrong answers {(sure_wrong + unsure_wrong) / total:.1%}")

    offline = Offline()
    guide = build_field_guide(taxa, cal["threshold"], offline, offline, offline, marks)
    rows = []
    for noise in args.noise:
        rng = random.Random(0)
        counts = {"right species": 0, "wrong species": 0, "right branch, higher rank": 0, "wrong branch": 0}
        asked, in_pool, has_marks = 0, 0, 0
        for i, (probs, truth) in enumerate(unsure):
            state, questions = run(guide, probs, marks.get(truth.scientific_name), rng, noise, f"{noise}-{i}")
            asked += questions
            in_pool += truth.scientific_name in [c["name"] for c in state["candidates"]]
            has_marks += truth.scientific_name in marks
            if state.get("species"):
                counts["right species" if state["species"] == truth.scientific_name else "wrong species"] += 1
            else:
                on_branch = truth.lineage[: len(state["path"])] == tuple(state["path"])
                counts["right branch, higher rank" if on_branch else "wrong branch"] += 1
        n = len(unsure)
        system_right = (sure_right + counts["right species"]) / total
        system_wrong = (sure_wrong + counts["wrong species"] + counts["wrong branch"]) / total
        rows.append({
            "user_error_rate": noise, **{k: v / n for k, v in counts.items()}, "questions_per_photo": asked / n,
            "all_photos_right_species": system_right, "all_photos_wrong_answers": system_wrong,
        })
        print(f"\nsimulated user wrong {noise:.0%} of the time:")
        for k, v in counts.items():
            print(f"  {k:28} {v:4d}  ({v / n:.1%})")
        print(f"  questions per photo          {asked / n:.2f}")
        print(f"  over all {total} photos: right species {system_right:.1%}, wrong answers {system_wrong:.1%}")
    print(f"\nthe true species is among the agent's candidates in {in_pool / n:.1%} of these photos, "
          f"and has field marks in {has_marks / n:.1%}")

    args.out.mkdir(parents=True, exist_ok=True)
    with open(args.out / "agent_eval.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    parser.add_argument("--field-marks", type=Path, default=Path("data/field_marks.json"))
    parser.add_argument("--noise", type=float, nargs="+", default=[0.0, 0.1, 0.2])
    parser.add_argument("--out", type=Path, default=Path("results/agent"))
    main(parser.parse_args())
