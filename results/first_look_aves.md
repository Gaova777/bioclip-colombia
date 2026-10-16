# First look: 3 hummingbird photos vs. 200 Colombian birds

Candidates: the 200 most observed bird species in Colombia on iNaturalist (21 orders, 48 families, 153 genera).
Threshold for the answer: 80%. Command:

```
uv run first_look.py data/samples/<photo>.jpg --candidates data/candidates/aves_colombia.csv
```

| Photo | CLIP top-1 species | CLIP answer at 80% | BioCLIP top-1 species | BioCLIP answer at 80% |
|---|---|---|---|---|
| *Amazilia tzacatl* | ✅ 43.6% | family Trochilidae (92.2%) | ✅ 100% | **species** *Amazilia tzacatl* (100%) |
| *Saucerottia saucerottei* | ❌ Andean emerald 32.5% | class Aves (100%) | ✅ 98.8% | **species** *Saucerottia saucerottei* (98.8%) |
| *Uranomitra franciae* | ❌ lesser violetear 57.0% | family Trochilidae (85.1%) | ✅ 69.0% | family Trochilidae (99.7%) |

Notes:
- With the threshold, neither model gives a wrong answer. CLIP pays for it by saying very little: for the steely-vented
  hummingbird its second guess is a jacamar (a different order), so it can only commit to "a bird".
- BioCLIP got more confident with 200 candidates than with 30 hummingbirds (98.8% vs 72.4% for the steely-vented).
  Not obvious why; something to look at when we evaluate on more images.
- Three photos are anecdotes, not an evaluation. Next step is a proper test set.
