# Zero-shot on 200 Colombian birds: CLIP vs. BioCLIP, with abstention

## Setup

- **Candidates:** the 200 most observed bird species in Colombia on iNaturalist
  ([`aves_colombia.csv`](../../data/candidates/aves_colombia.csv)), 21 orders, 48 families, 153 genera.
- **Test set:** 5 research-grade iNaturalist photos per species, all taken in Colombia, all observed in 2024 or later,
  1000 images total ([`manifest.csv`](../../data/testset/manifest.csv), with author and license for every photo).
  BioCLIP was trained on iNat21 and a 2023 EOL crawl, so it can't have seen these photos.
- **Models:** CLIP ViT-B/16 (OpenAI weights, via open_clip) with common-name prompts (`a photo of a <common name>`) and
  `imageomics/bioclip` with taxonomic + common-name prompts, the best text type for each model in the paper.
- **Abstention:** species probabilities are summed up the tree (genus, family, order...). The model walks down
  from the root and stops at the first rank where the best child has probability below a threshold τ.
- No training, no tuning. Everything is zero-shot.

```
uv run scripts/build_testset.py data/candidates/aves_colombia.csv -k 5
uv run scripts/evaluate.py
```

## Top-1 accuracy per rank

Kingdom, phylum and class are left out: every candidate is a bird, so they are always right.

| model   | order | family | genus | species |
|---------|------:|-------:|------:|--------:|
| CLIP    | 78.0% |  59.5% | 32.9% |   29.1% |
| BioCLIP | 93.1% |  87.1% | 80.2% |   77.8% |

([`accuracy_by_rank.csv`](accuracy_by_rank.csv))

## Risk vs. coverage

Each point is one threshold τ. Coverage is the share of images the model answers down to species;
the y axis is how many of those species answers are wrong.

![risk vs coverage](risk_coverage.png)

| τ    | CLIP coverage | CLIP error | BioCLIP coverage | BioCLIP error |
|------|--------------:|-----------:|-----------------:|--------------:|
| none |          100% |      72.0% |             100% |         22.5% |
| 0.5  |         19.3% |      38.9% |            89.2% |         15.5% |
| 0.8  |          4.2% |       7.1% |            72.1% |          7.5% |
| 0.9  |          1.7% |       0.0% |            64.9% |          4.5% |
| 0.99 |          0.1% |       0.0% |            44.0% |          1.1% |

([`risk_coverage.csv`](risk_coverage.csv))

At τ = 0.8, BioCLIP answers 721 of the 1000 photos down to species. The rest stop at genus (40),
family (79), order (70) or just "a bird" (90). CLIP reaches species on 42 photos and says only "a bird" on 492.

## What I take from it

- **Abstention is only useful if the model knows something.** With BioCLIP, the threshold trades coverage
  for error smoothly: going from τ = none to 0.9 cuts the species error from 22.5% to 4.5% and still answers
  65% of the photos at species level. With CLIP the same trade is useless: to get under 10% error it has to
  give up on 96% of the photos.
- **BioCLIP is overconfident.** 29 of its 222 wrong species answers come with more than 90% probability,
  and even at τ = 0.99 it still makes mistakes. CLIP has the opposite problem: none of its mistakes are
  above 90%, but it is rarely above 90% about anything.
- **Mistakes didn't get milder, there are just fewer of them.** I expected BioCLIP's wrong answers to stay
  closer to the truth in the tree. They barely do: a wrong species shares on average 4.21 of 7 ranks with the
  right one for BioCLIP and 4.04 for CLIP. About a third of BioCLIP's mistakes (71 of 222) already pick
  the wrong order.
- **Some mistakes make sense, some don't.**
  - Look-alikes in the same genus or family: *Campephilus melanoleucos* → *Dryocopus lineatus* (4×),
    *Psarocolius angustifrons* → *P. decumanus* (3×), *Chloroceryle amazona* → *C. americana* (3×).
  - Same colour, different family: *Setophaga aestiva* (yellow warbler) → *Sicalis flaveola* (saffron finch), 4×.
    The saffron finch is the most observed species in the list; I don't know yet if that matters.
  - CLIP's worst ones cross orders: a buff-tailed coronet (hummingbird) called a rufous-tailed jacamar 5 out of 5 times.

## What didn't work / caveats

- The first version of the test set had no photos of *Tyrannus tyrannus*: searching iNaturalist by name
  returned other *Tyrannus* species and the script dropped them. Photos are now looked up by taxon id.
- iNaturalist returns the newest observations first, so rebuilding the test set later gives slightly
  different photos (between the two builds, 989 of 993 photos were the same).
- 5 photos per species is small. A single photo moves a species' accuracy by 20 points, so per-species
  numbers aren't meaningful yet; the totals over 1000 images are.
- τ is not chosen yet. The table above describes the whole curve; picking τ has to happen on a separate
  validation split, not on this test set.
- Only the first photo of each observation is used, and I haven't checked them by hand.

## Next

- Split validation / test and pick τ on validation.
- Calibration: check if a temperature on BioCLIP's logits fixes the overconfidence before thresholding.
- Look at the 29 confident mistakes one by one.
