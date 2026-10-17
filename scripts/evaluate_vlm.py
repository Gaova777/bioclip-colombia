"""Why not just ask a multimodal LLM? A local vision-language model on the same test photos.

closed: the photo plus the 200 candidates; it must pick one and say how sure it is (0-100).
        Same task as BioCLIP, so accuracy per rank and abstention on its own confidence are comparable.
open:   just "which species is this?"; counts how often it names something that isn't a candidate.

Answers are saved as they come (results/vlm/answers_*.jsonl), so a run can be stopped and resumed.

usage: uv run scripts/evaluate_vlm.py --one-per-species                (needs Ollama running)
       uv run scripts/evaluate_vlm.py --one-per-species --mode open
"""
import argparse
import base64
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from evaluate import load_testset  # noqa: E402

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.domain.metrics import rank_accuracy, risk_coverage, shared_ranks
from bioclip_co.domain.taxonomy import RANKS

THRESHOLDS = [0, 50, 60, 70, 80, 90, 95, 99]


def chat(url, model, prompt, image, schema):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(image.read_bytes()).decode()]}],
        "format": schema,
        "stream": False,
        # the 200-species list plus the image need ~4k tokens
        "options": {"temperature": 0, "num_ctx": 6144},
        "think": False,
    }
    req = urllib.request.Request(f"{url}/api/chat", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(json.load(resp)["message"]["content"])


def bioclip_on_the_same_photos(args, taxa, paths, truths):
    """BioCLIP (calibrated) on exactly these photos, from the embeddings scripts/evaluate.py cached."""
    import numpy as np

    from bioclip_co.adapters.openclip_encoder import OpenClipEncoder
    from bioclip_co.application.zero_shot import prompts_for, species_probabilities

    cache = np.load(args.cache / f"{args.testset.name}_bioclip_images.npz")
    index = {k: i for i, k in enumerate(cache["keys"])}
    img = cache["emb"][[index[p.as_posix()] for p in paths]]
    cal = json.loads(args.calibration.read_text())["bioclip"]
    # on the CPU: the vision-language model is still holding the GPU
    txt = OpenClipEncoder("bioclip", device="cpu").encode_texts(prompts_for(taxa, style="bioclip"))
    probs = species_probabilities(img, txt, cal["logit_scale"]).tolist()
    [point] = risk_coverage(taxa, probs, truths, [cal["threshold"]])
    return [{"metric": f"bioclip_{rank}_accuracy", "value": rank_accuracy(taxa, probs, truths, rank)} for rank in RANKS[3:]] + [
        {"metric": "bioclip_species_coverage_at_threshold", "value": point.species_coverage},
        {"metric": "bioclip_any_error_at_threshold", "value": point.any_error},
    ]


def main(args):
    taxa = read_candidates(args.candidates)
    paths, truths = load_testset(args.testset, taxa)
    if args.one_per_species:  # the first photo of each species: 200 photos instead of 1000, same species mix
        seen = set()
        keep = [i for i, t in enumerate(truths) if not (t.scientific_name in seen or seen.add(t.scientific_name))]
        paths, truths = [paths[i] for i in keep], [truths[i] for i in keep]
    paths, truths = paths[: args.limit], truths[: args.limit]
    by_name = {t.scientific_name: t for t in taxa}
    names = list(by_name)
    listing = "\n".join(f"- {t.scientific_name} ({t.common_name})" for t in taxa)

    if args.mode == "closed":
        prompt = (
            "This photo was taken in Colombia. It shows one of these bird species:\n" + listing +
            "\n\nAnswer with the scientific name of the species, exactly as written in the list, "
            "and how confident you are from 0 to 100."
        )
        schema = {"type": "object", "properties": {"species": {"type": "string", "enum": names},
                  "confidence": {"type": "integer"}}, "required": ["species", "confidence"]}
    else:
        prompt = "This photo was taken in Colombia. Which bird species is it? Answer with its scientific name."
        schema = {"type": "object", "properties": {"species": {"type": "string"}}, "required": ["species"]}

    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / f"answers_{args.mode}_{args.model.replace(':', '_')}.jsonl"
    done = {json.loads(line)["path"]: json.loads(line) for line in log.open()} if log.exists() else {}
    start, new = time.time(), 0
    with log.open("a") as f:
        for i, (path, truth) in enumerate(zip(paths, truths), 1):
            if str(path) in done:
                continue
            reply = chat(args.url, args.model, prompt, path, schema)
            done[str(path)] = row = {"path": str(path), "truth": truth.scientific_name, **reply}
            f.write(json.dumps(row) + "\n")
            f.flush()
            new += 1
            if new % 50 == 0:
                print(f"  {i}/{len(paths)}  {(time.time() - start) / new:.1f} s per photo", flush=True)

    answers = [done[str(p)] for p in paths]
    n = len(answers)
    if args.mode == "closed":
        rows = []
        for rank in RANKS[3:]:
            depth = RANKS.index(rank) + 1
            right = sum(shared_ranks(by_name[a["species"]], by_name[a["truth"]]) >= depth for a in answers)
            rows.append({"metric": f"{rank}_accuracy", "value": right / n})
        for t in THRESHOLDS:
            kept = [a for a in answers if a["confidence"] >= t]
            wrong = sum(a["species"] != a["truth"] for a in kept)
            rows.append({"metric": f"answers_with_confidence_>={t}", "value": len(kept) / n})
            rows.append({"metric": f"error_among_them_>={t}", "value": wrong / len(kept) if kept else 0.0})
        rows += bioclip_on_the_same_photos(args, taxa, paths, truths)
        with open(args.out / f"closed_{args.model.replace(':', '_')}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["metric", "value"])
            writer.writeheader()
            writer.writerows(rows)
        for r in rows:
            print(f"  {r['metric']:32} {r['value']:.3f}")
    else:
        said = [a["species"].strip() for a in answers]
        right = sum(s.lower() == a["truth"].lower() for s, a in zip(said, answers))
        off_list = sum(s.lower() not in {x.lower() for x in names} for s in said)
        summary = {"photos": n, "right_species": right / n, "not_a_candidate": off_list / n}
        (args.out / f"open_{args.model.replace(':', '_')}.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--mode", choices=["closed", "open"], default="closed")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--one-per-species", action="store_true")
    parser.add_argument("--url", default="http://localhost:11434")
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--testset", type=Path, default=Path("data/testset"))
    parser.add_argument("--cache", type=Path, default=Path("data/cache"))
    parser.add_argument("--calibration", type=Path, default=Path("data/calibration.json"))
    parser.add_argument("--out", type=Path, default=Path("results/vlm"))
    main(parser.parse_args())
