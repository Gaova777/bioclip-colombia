"""Download the most observed species of a family in Colombia from iNaturalist.

usage: uv run scripts/fetch_candidates.py Trochilidae 30
"""
import csv
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.inaturalist.org/v1"
COLOMBIA = 7196
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]


def get(endpoint, retries=3, **params):
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-colombia"})
    for attempt in range(retries):
        time.sleep(1)  # iNaturalist asks for ~1 request per second
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except (urllib.error.URLError, ConnectionError):
            if attempt == retries - 1:
                raise


def family_lineage(taxon_id):
    taxon = get(f"taxa/{taxon_id}")["results"][0]
    ranks = {a["rank"]: a["name"] for a in taxon["ancestors"]}
    return [ranks[r] for r in RANKS[:-2]] + [taxon["name"]]


def main(family, n):
    family_id = get("taxa", q=family, rank="family")["results"][0]["id"]
    above_genus = family_lineage(family_id)
    counts = get(
        "observations/species_counts",
        taxon_id=family_id, place_id=COLOMBIA, quality_grade="research",
        per_page=n, locale="en",
    )["results"]

    out = Path(f"data/candidates/{family.lower()}_colombia.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RANKS + ["species", "common_name", "observations"])
        for row in counts:
            t = row["taxon"]
            genus, epithet = t["name"].split()[:2]
            writer.writerow(above_genus + [genus, epithet, t.get("preferred_common_name", ""), row["count"]])
    print(f"{len(counts)} species -> {out}")


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 30)
