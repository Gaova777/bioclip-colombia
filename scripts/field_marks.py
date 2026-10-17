"""Field marks for every candidate species: a local LLM reads the Wikipedia description and fills in a form
with a closed vocabulary (crown colour, bill shape...). The result is committed so it can be reviewed and fixed by hand.

usage: uv run scripts/field_marks.py   (needs Ollama running)
"""
import argparse
import json
import time
from pathlib import Path

from bioclip_co.adapters.candidates_csv import read_candidates
from bioclip_co.adapters.ollama_writer import OllamaWriter
from bioclip_co.adapters.wikipedia import Wikipedia


def main(args):
    taxa = read_candidates(args.candidates)
    done = json.loads(args.out.read_text()) if args.out.exists() else {}
    wiki, llm = Wikipedia(), OllamaWriter(args.model)
    for i, t in enumerate(taxa, 1):
        name = t.scientific_name
        if done.get(name) is not None:
            continue
        time.sleep(1)  # be gentle with the Wikipedia API
        note = wiki.summary(name, max_chars=3000)
        if note is None:
            done[name] = None
            print(f"[{i}/{len(taxa)}] {name}: no Wikipedia article")
        else:
            done[name] = llm.field_marks(name, t.common_name, note["extract"]) | {"source": note["url"]}
            known = sum(v is not None for k, v in done[name].items() if k not in ("sexes_differ", "source"))
            print(f"[{i}/{len(taxa)}] {name}: {known}/8 field marks", flush=True)
        args.out.write_text(json.dumps(done, indent=1, ensure_ascii=False) + "\n")  # resumable


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", type=Path, default=Path("data/candidates/aves_colombia.csv"))
    parser.add_argument("--model", default="qwen3:4b-instruct")
    parser.add_argument("--out", type=Path, default=Path("data/field_marks.json"))
    main(parser.parse_args())
