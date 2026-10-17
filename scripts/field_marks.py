"""Field marks for every candidate species, seen in photos: a local vision-language model looks at the
validation photos of each species (never the test ones) and fills in a form with a closed vocabulary.
A mark is kept only if most photos agree. The result is committed so it can be reviewed and fixed by hand.

I first extracted the marks from Wikipedia descriptions with a text model. Many descriptions don't give
the colours (or there's no article), and the model filled the gaps by guessing, so photos replaced the text.

usage: uv run scripts/field_marks.py   (needs Ollama running)
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from bioclip_co.adapters.ollama_vision import OllamaVision
from bioclip_co.domain.field_marks import consensus


def main(args):
    with open(args.valset / "manifest.csv", encoding="utf-8") as f:
        photos = defaultdict(list)
        for row in csv.DictReader(f):
            photos[f"{row['genus']} {row['species']}"].append(args.valset / row["path"])

    # every answer is kept, so an interrupted run picks up where it stopped
    seen = {}
    if args.seen.exists():
        seen = {r["photo"]: r["marks"] for r in map(json.loads, args.seen.open())}
    vision = OllamaVision(args.model)
    args.seen.parent.mkdir(parents=True, exist_ok=True)
    result = {}
    with args.seen.open("a") as log:
        for i, (species, paths) in enumerate(sorted(photos.items()), 1):
            for path in paths:
                if str(path) not in seen:
                    seen[str(path)] = vision.marks(path)
                    log.write(json.dumps({"photo": str(path), "marks": seen[str(path)]}) + "\n")
                    log.flush()
            votes = [seen[str(p)] for p in paths]
            result[species] = consensus(votes, min_votes=args.min_votes) | {"photos": len(votes)}
            known = sum(v is not None for k, v in result[species].items() if k != "photos")
            print(f"[{i}/{len(photos)}] {species}: {known}/8 marks agreed on by {args.min_votes}+ of {len(votes)} photos", flush=True)
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valset", type=Path, default=Path("data/valset"))
    parser.add_argument("--model", default="qwen3-vl:8b-instruct")
    parser.add_argument("--min-votes", type=int, default=3)
    parser.add_argument("--seen", type=Path, default=Path("data/cache/field_marks_seen.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("data/field_marks.json"))
    main(parser.parse_args())
