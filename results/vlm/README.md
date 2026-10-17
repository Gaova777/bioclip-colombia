# Why not just ask a multimodal LLM?

The question every reviewer asks. So I asked one: **Qwen3-VL 8B instruct**, a local vision-language model through Ollama,
on the same test photos as BioCLIP. One photo per species (the first of each), 200 photos, because with the
200-name list in the prompt the 8B model runs at ~15 s per photo on this GPU.

```
uv run scripts/evaluate_vlm.py --one-per-species              # the photo + the 200 candidates, pick one + confidence
uv run scripts/evaluate_vlm.py --one-per-species --mode open  # just "which species is this?"
```

## Same task as BioCLIP: pick one of the 200 candidates

| on the same 200 photos | order | family | genus | species |
|---|---:|---:|---:|---:|
| Qwen3-VL 8B (~8,000 M parameters) | 90.0% | 75.0% | 36.0% | 30.5% |
| BioCLIP (~150 M parameters) | **94.0%** | **89.5%** | **83.5%** | **80.5%** |

([`closed_qwen3-vl_8b-instruct.csv`](closed_qwen3-vl_8b-instruct.csv), answers in
[`answers_closed_qwen3-vl_8b-instruct.jsonl`](answers_closed_qwen3-vl_8b-instruct.jsonl))

**Its confidence means nothing.** Asked for a confidence from 0 to 100, it said exactly 95 for 184 of the 200 photos,
and it's wrong on 67% of the answers it gives with 95 or more. So it can't abstain: there's no threshold that
separates its good answers from its bad ones. BioCLIP, calibrated and at its validation threshold, gets 1.5%
wrong answers on these same photos and still names the species for 68.5% of them.

## Without the list: "which species is this?"

| | |
|---|---:|
| right species | 18.0% |
| right genus | 44.5% |
| names a species that isn't one of the 200 candidates | **71.0%** |

([`open_qwen3-vl_8b-instruct.json`](open_qwen3-vl_8b-instruct.json))

Most of those are real birds, just not these: the eared dove becomes a mourning dove (*Zenaida macroura*), the
great thrush a rufous-bellied thrush, the southern lapwing a bogotá rail. A confident, plausible, wrong name is
exactly what this project is trying to avoid.

## What I take from it

- **A specialist 50 times smaller is more than twice as accurate**, and, more importantly, knows when it's guessing.
  General multimodal models know a lot about birds in general and little about telling 200 of them apart.
- **This is why the agent is built the way it is.** The language model never picks the species; it doesn't even pick
  the questions. The vision-language model did one job it's good at, offline: looking at photos and describing colours
  (the field marks). That's a much easier task than naming the species, and a majority vote over 5 photos cleans it up.
- **Caveats.** One open model, one size, one prompt, 200 photos. A bigger or newer model (or a cloud one) will do better;
  the point is the pattern, not the exact numbers. Its answers are in the `.jsonl` files if you want to look.
