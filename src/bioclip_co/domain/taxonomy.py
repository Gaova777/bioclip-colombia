from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

RANKS = ("kingdom", "phylum", "class", "order", "family", "genus", "species")


@dataclass(frozen=True)
class Taxon:
    """One leaf of the tree of life, e.g. Animalia > ... > Amazilia > tzacatl."""

    lineage: tuple[str, ...]
    common_name: str = ""

    def __post_init__(self):
        if len(self.lineage) != len(RANKS):
            raise ValueError(f"expected {len(RANKS)} ranks, got {len(self.lineage)}: {self.lineage}")

    def at(self, rank: str) -> str:
        return self.lineage[RANKS.index(rank)]

    def prefix(self, rank: str) -> tuple[str, ...]:
        # The whole path down to `rank`, not just the name at that rank:
        # the same genus name can exist in two different families (homonyms).
        return self.lineage[: RANKS.index(rank) + 1]

    @property
    def scientific_name(self) -> str:
        return f"{self.at('genus')} {self.at('species')}"

    @property
    def taxonomic_name(self) -> str:
        """The flattened string BioCLIP was trained on (paper, section 3.2)."""
        return " ".join(self.lineage)


def marginalize(taxa: Sequence[Taxon], probs: Sequence[float], rank: str) -> dict[tuple[str, ...], float]:
    """Turn species probabilities into probabilities at a higher rank.

    P(genus) is the sum of its species, P(family) the sum of its genera, and so on.
    The result is sorted from most to least likely.
    """
    if len(taxa) != len(probs):
        raise ValueError(f"{len(taxa)} taxa but {len(probs)} probabilities")

    totals: dict[tuple[str, ...], float] = defaultdict(float)
    for taxon, p in zip(taxa, probs):
        totals[taxon.prefix(rank)] += p
    return dict(sorted(totals.items(), key=lambda item: -item[1]))
