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
