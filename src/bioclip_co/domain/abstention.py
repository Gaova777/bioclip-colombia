from collections.abc import Sequence
from dataclasses import dataclass

from bioclip_co.domain.taxonomy import RANKS, Taxon, marginalize


@dataclass(frozen=True)
class Prediction:
    path: tuple[str, ...]  # from the kingdom down to the deepest rank we trust
    confidence: float

    @property
    def rank(self) -> str | None:
        return RANKS[len(self.path) - 1] if self.path else None

    @property
    def name(self) -> str | None:
        if self.rank == "species":
            return f"{self.path[-2]} {self.path[-1]}"
        return self.path[-1] if self.path else None

    @property
    def reached_species(self) -> bool:
        return self.rank == "species"


def descend(taxa: Sequence[Taxon], probs: Sequence[float], threshold: float) -> Prediction:
    """Walk down the tree and stop at the first rank where we are less sure than `threshold`.

    At each rank we only look at the children of the node we already chose,
    so the answer is always a single path from the root.
    """
    if not 0 < threshold <= 1:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    node: tuple[str, ...] = ()
    confidence = 1.0
    for rank in RANKS:
        children = {
            prefix: p
            for prefix, p in marginalize(taxa, probs, rank).items()
            if prefix[:-1] == node
        }
        best, p = next(iter(children.items()))
        if p < threshold:
            break
        node, confidence = best, p
    return Prediction(node, confidence)
