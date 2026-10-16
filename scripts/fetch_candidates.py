"""Download the most observed species of one or more taxa in Colombia from iNaturalist.

usage:
    uv run scripts/fetch_candidates.py Trochilidae -n 30
    uv run scripts/fetch_candidates.py Aves Trochilidae -n 150 -o data/candidates/aves_colombia.csv
"""
import argparse
import csv
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.inaturalist.org/v1"
COLOMBIA = 7196
RANKS = ["kingdom", "phylum", "class", "order", "family", "genus"]


def get(endpoint, retries=4, **params):
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-colombia"})
    for attempt in range(retries):
        time.sleep(1 + 2**attempt - 1)  # iNaturalist asks for ~1 request per second
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.load(resp)
        except OSError:  # covers URLError, dropped connections and timeouts
            if attempt == retries - 1:
                raise


def find_taxon_id(name):
    for taxon in get("taxa", q=name)["results"]:
        if taxon["name"] == name:
            return taxon["id"]
    raise SystemExit(f"taxon not found on iNaturalist: {name}")


def top_species(taxon_id, n):
    return get(
        "observations/species_counts",
        taxon_id=taxon_id, place_id=COLOMBIA, quality_grade="research",
        per_page=n, locale="en",
    )["results"]


def ancestor_names(ids):
    """Map taxon id -> (rank, name), asking the API for 30 ids at a time."""
    ids = sorted(ids)
    names = {}
    for i in range(0, len(ids), 30):
        batch = ",".join(map(str, ids[i : i + 30]))
        for t in get(f"taxa/{batch}")["results"]:
            names[t["id"]] = (t["rank"], t["name"])
    return names


def main(taxa, n, out):
    rows = {}
    for name in taxa:
        for row in top_species(find_taxon_id(name), n):
            rows.setdefault(row["taxon"]["id"], row)

    names = ancestor_names({a for r in rows.values() for a in r["taxon"]["ancestor_ids"]})

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(RANKS + ["species", "common_name", "observations"])
        written = 0
        for row in sorted(rows.values(), key=lambda r: -r["count"]):
            t = row["taxon"]
            lineage = dict(names[a] for a in t["ancestor_ids"] if a in names)
            if missing := [r for r in RANKS if r not in lineage]:
                print(f"skipping {t['name']}: no {', '.join(missing)} in its lineage")
                continue
            epithet = t["name"].split()[1]
            writer.writerow([lineage[r] for r in RANKS] + [epithet, t.get("preferred_common_name", ""), row["count"]])
            written += 1
    print(f"{written} species -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("taxa", nargs="+", help="taxon names, e.g. Aves or Trochilidae")
    parser.add_argument("-n", type=int, default=30, help="species per taxon")
    parser.add_argument("-o", "--out", type=Path)
    args = parser.parse_args()
    out = args.out or Path(f"data/candidates/{args.taxa[0].lower()}_colombia.csv")
    main(args.taxa, args.n, out)
