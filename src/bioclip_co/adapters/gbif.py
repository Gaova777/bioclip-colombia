import json
import urllib.parse
import urllib.request
from functools import lru_cache

API = "https://api.gbif.org/v1"


def get(endpoint: str, **params) -> dict:
    url = f"{API}/{endpoint}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"User-Agent": "bioclip-colombia"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return json.load(resp)


class Gbif:
    """Occurrence counts per Colombian department and the IUCN category, from the GBIF API."""

    @lru_cache(maxsize=1024)
    def species_key(self, scientific_name: str) -> int | None:
        match = get("species/match", name=scientific_name, kingdom="Animalia")
        return match.get("usageKey") if match.get("matchType") != "NONE" else None

    @lru_cache(maxsize=4096)
    def records_in(self, scientific_name: str, region: str) -> int:
        key = self.species_key(scientific_name)
        if key is None:
            return 0
        return get("occurrence/search", taxonKey=key, country="CO", stateProvince=region, limit=0)["count"]

    @lru_cache(maxsize=1024)
    def red_list(self, scientific_name: str) -> str | None:
        key = self.species_key(scientific_name)
        if key is None:
            return None
        try:
            code = get(f"species/{key}/iucnRedListCategory").get("code")
        except OSError:
            return None
        return None if code in (None, "NE", "DD") else code
