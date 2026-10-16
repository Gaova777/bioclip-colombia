import csv
from pathlib import Path

from bioclip_co.domain.taxonomy import RANKS, Taxon


def read_candidates(path: str | Path) -> list[Taxon]:
    """Read a CSV like the ones made by scripts/fetch_candidates.py."""
    with open(path, encoding="utf-8") as f:
        return [
            Taxon(tuple(row[r] for r in RANKS), row.get("common_name", "").lower())
            for row in csv.DictReader(f)
        ]
