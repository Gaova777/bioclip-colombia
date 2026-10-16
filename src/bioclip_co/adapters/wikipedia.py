import json
import re
import urllib.parse
import urllib.request
from functools import lru_cache

from bioclip_co.ports.field_guide import Note

API = "https://en.wikipedia.org/w/api.php"
MAX_CHARS = 1500  # enough for a plumage description, short enough for a small model's prompt


class Wikipedia:
    """The lead and the "Description" section of the English Wikipedia article.

    The lead alone often says where a bird lives but not what it looks like, and telling
    two species apart needs the plumage. Redirects are followed, so a scientific name lands
    on the common-name article.
    """

    @lru_cache(maxsize=1024)
    def summary(self, name: str) -> Note | None:
        params = {"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": name, "format": "json"}
        req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": "bioclip-colombia"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                page = next(iter(json.load(resp)["query"]["pages"].values()))
        except (OSError, KeyError, StopIteration):
            return None
        text = page.get("extract")
        if not text:
            return None
        # split on the "== Section ==" headings; the first chunk is the lead
        sections = re.split(r"\n==+ (.+?) ==+\n", text)
        lead, rest = sections[0], dict(zip(sections[1::2], sections[2::2]))
        description = rest.get("Description", "")
        extract = f"{lead}\n\n{description}".strip()[:MAX_CHARS]
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}"
        return {"title": page["title"], "extract": extract, "url": url}
