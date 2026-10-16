"""Build a test set: a few iNaturalist photos per candidate species, all taken in Colombia.

Only observations from 2024 on are used. BioCLIP's training data (iNat21 and the 2023 EOL crawl)
is older, so these photos can't have been seen during training.
Images stay out of git; the manifest with credits is committed.

usage:
    uv run scripts/build_testset.py data/candidates/aves_colombia.csv -k 5
    uv run scripts/build_testset.py data/candidates/aves_colombia.csv -k 5 -o data/valset --exclude data/testset/manifest.csv
"""
import argparse
import csv
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from fetch_candidates import COLOMBIA, find_taxon_id, get  # noqa: E402

LICENSES = "cc0,cc-by,cc-by-nc,cc-by-sa,cc-by-nc-sa"
SINCE = "2024-01-01"
FIELDS = ["path", "genus", "species", "observation", "observed_on", "license", "attribution"]


def observations(scientific_name, k):
    # taxon_name can resolve to the genus: "Tyrannus tyrannus" gave only other Tyrannus species.
    # The id is exact, and it still includes subspecies.
    taxon_id = find_taxon_id(scientific_name, rank="species")
    return get(
        "observations",
        taxon_id=taxon_id, place_id=COLOMBIA, quality_grade="research",
        photo_license=LICENSES, d1=SINCE, per_page=k,
    )["results"]


def download(url, dest):
    if dest.exists():
        return
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-colombia"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())


def main(candidates, k, out_dir, exclude):
    with open(candidates, encoding="utf-8") as f:
        species = list(csv.DictReader(f))
    # a validation set must not share photos with the test set
    taken = set()
    if exclude:
        with open(exclude, encoding="utf-8") as f:
            taken = {int(row["observation"]) for row in csv.DictReader(f)}

    # 1) metadata from the API, one request per species (rate limited)
    rows, jobs = [], []
    for i, sp in enumerate(species, 1):
        name = f"{sp['genus']} {sp['species']}"
        obs = [o for o in observations(name, k + k * bool(taken)) if o["id"] not in taken][:k]
        for o in obs:
            photo = o["photos"][0]
            dest = out_dir / "images" / f"{sp['genus']}_{sp['species']}_{o['id']}.jpg"
            jobs.append((photo["url"].replace("square", "medium"), dest))
            rows.append({
                "path": dest.relative_to(out_dir).as_posix(),
                "genus": sp["genus"],
                "species": sp["species"],
                "observation": o["id"],
                "observed_on": o["observed_on"],
                "license": photo["license_code"],
                "attribution": photo["attribution"],
            })
        print(f"[{i}/{len(species)}] {name}: {len(obs)} photos", flush=True)

    # 2) the images themselves come from a static bucket, so they can go in parallel
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda job: download(*job), jobs))

    with (out_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} images -> {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("candidates", type=Path)
    parser.add_argument("-k", type=int, default=5, help="photos per species")
    parser.add_argument("-o", "--out", type=Path, default=Path("data/testset"))
    parser.add_argument("--exclude", type=Path, help="manifest whose photos must not be reused")
    args = parser.parse_args()
    main(args.candidates, args.k, args.out, args.exclude)
