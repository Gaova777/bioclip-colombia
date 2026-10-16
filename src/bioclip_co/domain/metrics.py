from collections.abc import Sequence
from dataclasses import dataclass

from bioclip_co.domain.abstention import Prediction, descend
from bioclip_co.domain.taxonomy import Taxon, marginalize


def is_correct(pred: Prediction, truth: Taxon) -> bool:
    """A prediction is right if its whole path agrees with the truth, however short it is."""
    return truth.lineage[: len(pred.path)] == pred.path


def shared_ranks(a: Taxon, b: Taxon) -> int:
    """How many ranks two taxa share from the root (7 = same species, 0 = different kingdoms)."""
    n = 0
    for x, y in zip(a.lineage, b.lineage):
        if x != y:
            break
        n += 1
    return n


def rank_accuracy(taxa: Sequence[Taxon], probs: Sequence[Sequence[float]], truths: Sequence[Taxon], rank: str) -> float:
    """Top-1 accuracy if we only had to name the `rank` (genus, family...)."""
    hits = 0
    for row, truth in zip(probs, truths):
        best = next(iter(marginalize(taxa, row, rank)))
        hits += best == truth.prefix(rank)
    return hits / len(truths)


def mistake_depth(taxa: Sequence[Taxon], probs: Sequence[Sequence[float]], truths: Sequence[Taxon]) -> list[int]:
    """For each wrong top-1 species, how many ranks it still got right. Higher = milder mistake."""
    depths = []
    for row, truth in zip(probs, truths):
        guess = taxa[max(range(len(row)), key=row.__getitem__)]
        if guess != truth:
            depths.append(shared_ranks(guess, truth))
    return depths


@dataclass(frozen=True)
class OperatingPoint:
    threshold: float
    species_coverage: float  # share of images answered down to species
    species_error: float  # error rate among those species answers
    any_error: float  # share of images whose answer (at any rank) is wrong
    mean_depth: float  # average number of ranks in the answer, 7 = species


def risk_coverage(
    taxa: Sequence[Taxon],
    probs: Sequence[Sequence[float]],
    truths: Sequence[Taxon],
    thresholds: Sequence[float],
) -> list[OperatingPoint]:
    points = []
    for threshold in thresholds:
        preds = [descend(taxa, row, threshold) for row in probs]
        at_species = [(p, t) for p, t in zip(preds, truths) if p.reached_species]
        species_wrong = sum(not is_correct(p, t) for p, t in at_species)
        points.append(OperatingPoint(
            threshold=threshold,
            species_coverage=len(at_species) / len(preds),
            species_error=species_wrong / len(at_species) if at_species else 0.0,
            any_error=sum(not is_correct(p, t) for p, t in zip(preds, truths)) / len(preds),
            mean_depth=sum(len(p.path) for p in preds) / len(preds),
        ))
    return points
