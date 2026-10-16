# bioclip-colombia

**A species classifier that knows when to say "I'm not sure", tested on the birds of Colombia.**

![The web demo: BioCLIP names the species, CLIP only commits to "a bird"](docs/demo.png)

Show a general-purpose vision model a photo of a hummingbird and it will often say "a bird", or name the wrong
species with a lot of confidence. [BioCLIP](https://imageomics.github.io/bioclip/) (CVPR 2024) fixes a big part of that.
It's a CLIP model trained on 10 million images of living things, where the caption of every image is its full taxonomy:
*Animalia Chordata Aves Apodiformes Trochilidae Amazilia tzacatl*. Related species share almost all of that text, so the
model ends up learning the shape of the tree of life, and it can recognize species it never saw.

But like any classifier, BioCLIP always names a species, even when it's guessing.
This project takes the paper and adds what a biologist would do:
**if you can't tell the species, say the genus or the family, and say how sure you are.**

It's project 1 of *Paper → Product*, a series where I take a recent AI paper, reproduce part of it,
add something of my own and build it the way it would work in the real world.

## How it works

1. **Zero-shot classification.** Every candidate species becomes a sentence
   (`a photo of Animalia Chordata Aves ... Amazilia tzacatl with common name rufous-tailed hummingbird`).
   The photo and all the sentences go through BioCLIP, and a softmax over their similarities gives one probability per species.
2. **Add it up the tree.** A genus is as likely as the sum of its species, a family as the sum of its genera,
   and so on up to the kingdom.
3. **Walk down with a threshold.** Start at the kingdom and go down one rank at a time, always taking the most likely child.
   As soon as the best option drops below the threshold, stop and answer with the last rank you were sure about.

```
$ uv run first_look.py data/samples/saucerottia_saucerottei_anori.jpg --candidates data/candidates/aves_colombia.csv

== clip ==
     32.5%  andean emerald (Uranomitra franciae)
     29.9%  rufous-tailed jacamar (Galbula ruficauda)
  answer at 80%: class Aves (100.0%)  <- not sure about the species

== bioclip ==
     98.8%  steely-vented hummingbird (Saucerottia saucerottei)
  answer at 80%: species Saucerottia saucerottei (98.8%)
```

CLIP's second guess there is a jacamar, a bird from a different order, so the only thing it can honestly say is "a bird".

## What I found

I built a test set of **1,000 photos of the 200 most observed birds in Colombia**, 5 per species, all from
iNaturalist and all taken in 2024 or later, so BioCLIP can't have seen them during training.
No training and no tuning: both models are used zero-shot.

**How often the top guess is right, at each rank:**

| | order | family | genus | species |
|---|---:|---:|---:|---:|
| CLIP | 78.0% | 59.5% | 32.9% | 29.1% |
| BioCLIP | **93.1%** | **87.1%** | **80.2%** | **77.8%** |

**What the threshold buys you.** Each point on the plot is a threshold. Moving right means answering more photos down
to species; moving up means more of those answers are wrong.

<img src="results/aves_colombia/risk_coverage.png" alt="Risk vs. coverage for CLIP and BioCLIP" width="560">

At an 80% threshold, BioCLIP names the species for 72% of the photos and gets 7.5% of those wrong
(down from 22.5% wrong when it's forced to always answer). The other photos stop at genus, family or order instead of guessing.
CLIP, to get to the same error rate, has to give up on 96% of the photos.

A few things I didn't expect:

- **BioCLIP is overconfident.** 29 of its wrong answers come with more than 90% probability. A threshold helps a lot,
  but it can't fix a model that's sure and wrong; calibration is next on the list.
- **Its mistakes aren't milder, there are just fewer of them.** A wrong BioCLIP species shares, on average, 4.2 of 7 ranks
  with the right one. CLIP: 4.0. About a third of BioCLIP's mistakes already pick the wrong order.
- **Some confusions make total sense** (two black-and-red woodpeckers, two kingfishers of the same genus), and some
  are just color: a yellow warbler called a saffron finch, four times out of five.

The full notes, with the caveats, are in [results/aves_colombia](results/aves_colombia/README.md).

## The demo

A small web app to try it with your own photos: upload a bird, both models walk down the tree side by side,
and a slider moves the threshold. The branches stay solid while a model is sure and turn dotted where it stops.

- **Backend:** FastAPI ([`entrypoints/api.py`](src/bioclip_co/entrypoints/api.py)). `POST /classify` returns the probability
  of every node of the tree for both models, once per photo.
- **Frontend:** React + Vite + TypeScript ([`web/`](web/)). The threshold is applied in the browser
  ([`tree.ts`](web/src/tree.ts) is the same walk as [`abstention.py`](src/bioclip_co/domain/abstention.py)),
  so moving the slider never calls the server.
- **Docker:** CPU-only image with the weights inside, so it starts without internet.

```bash
uv run uvicorn bioclip_co.entrypoints.api:app      # API on :8000
cd web && npm install && npm run dev               # page on :5173
```

## Run it yourself

```bash
git clone https://github.com/Gaova777/bioclip-colombia.git
cd bioclip-colombia
uv sync
uv run pytest -q
uv run first_look.py data/samples/amazilia_tzacatl_yarumal.jpg --candidates data/candidates/aves_colombia.csv
```

The first run downloads the BioCLIP and CLIP weights (about 600 MB each). It works on a CPU and uses the GPU if there is one.

To rebuild the test set and evaluate both models (the test set takes ~15 minutes because the iNaturalist API is rate limited;
the evaluation takes seconds on a GPU, a few minutes on a laptop):

```bash
uv run scripts/build_testset.py data/candidates/aves_colombia.csv -k 5
uv run scripts/evaluate.py
```

## How the code is organized

The code follows a clean-architecture split, so the interesting logic can be read and tested without loading a model:

```
src/bioclip_co/
├── domain/          plain Python, no ML libraries
│   ├── taxonomy.py      Taxon, adding probabilities up the tree
│   ├── abstention.py    walking down the tree with a threshold
│   └── metrics.py       accuracy per rank, mistake depth, risk-coverage
├── ports/           what the app needs from a model (ImageTextEncoder)
├── adapters/        BioCLIP/CLIP through OpenCLIP, the candidates CSV reader
├── application/     prompts and zero-shot probabilities
└── entrypoints/     the FastAPI app
scripts/             fetch candidates, build the test set, evaluate
web/                 the React demo
first_look.py        try it on one photo
tests/
```

Swapping BioCLIP for an ONNX or OpenVINO version later means writing one more adapter; nothing else changes.
The API tests already do something like that: they plug in a fake encoder and never load a model.

## Things that didn't work (yet)

- My first test set had zero photos of *Tyrannus tyrannus*. Searching iNaturalist by name returned other *Tyrannus*
  species, and my filter threw them all away. Photos are now looked up by taxon id.
- The threshold isn't tuned. The numbers above describe the whole curve; picking one value has to happen on a separate
  validation split, not on the test set.
- Five photos per species is enough for the totals, not for per-species conclusions.

## Roadmap

- [x] Zero-shot CLIP vs. BioCLIP on Colombian birds
- [x] Marginalization and hierarchical abstention, with tests
- [x] Metrics: accuracy per rank, mistake depth, risk-coverage
- [x] Full evaluation on a 1,000-photo iNaturalist test set
- [x] Web demo: FastAPI + React, Docker image for the API
- [ ] Tune the threshold on a validation split, and calibrate BioCLIP
- [ ] Look inside the embedding space (UMAP by order and family)
- [ ] ONNX / OpenVINO: latency vs. accuracy
- [ ] A "field guide" agent (LangGraph + a local LLM) that narrows candidates by region and asks follow-up questions
- [ ] CI/CD with a model quality gate, and experiment tracking with MLflow
- [ ] Public demo online

## Data and credits

- Candidate species and test photos come from [iNaturalist](https://www.inaturalist.org/), Colombia only.
- Every image has its author and license, in [data/samples/CREDITS.md](data/samples/CREDITS.md) and in the test set
  [manifest](data/testset/manifest.csv). The test images themselves are not redistributed in this repo.
- Models: [BioCLIP](https://huggingface.co/imageomics/bioclip) and OpenAI CLIP ViT-B/16, through
  [OpenCLIP](https://github.com/mlfoundations/open_clip).

## Reference

Stevens, S., Wu, J., Thompson, M. J., et al. *BioCLIP: A Vision Foundation Model for the Tree of Life.* CVPR 2024.
[arXiv:2311.18803](https://arxiv.org/abs/2311.18803)
