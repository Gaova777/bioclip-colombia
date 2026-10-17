import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from functools import lru_cache

from bioclip_co.ports.field_guide import Note

API = "https://en.wikipedia.org/w/api.php"
# Wikimedia throttles generic user agents hard; their policy asks for a name and a way to reach the author
USER_AGENT = "bioclip-colombia/0.1 (https://github.com/Gaova777/bioclip-colombia)"
MAX_CHARS = 1500  # enough for a plumage description, short enough for a small model's prompt


def fetch(params: dict, retries: int = 5) -> dict:
    req = urllib.request.Request(f"{API}?{urllib.parse.urlencode(params)}", headers={"User-Agent": USER_AGENT})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as e:
            if e.code != 429 or attempt == retries - 1:
                raise
            time.sleep(int(e.headers.get("Retry-After") or 2 ** (attempt + 1)))


class Wikipedia:
    """The lead and the "Description" section of the English Wikipedia article.

    The lead alone often says where a bird lives but not what it looks like, and telling
    two species apart needs the plumage. Redirects are followed, so a scientific name lands
    on the common-name article.
    """

    @lru_cache(maxsize=1024)
    def summary(self, name: str, max_chars: int = MAX_CHARS) -> Note | None:
        """None only when there is no article; network errors are raised, not hidden."""
        data = fetch({"action": "query", "prop": "extracts", "explaintext": 1, "redirects": 1, "titles": name, "format": "json"})
        page = next(iter(data["query"]["pages"].values()))
        if "missing" in page or not page.get("extract"):
            return None
        # split on the "== Section ==" headings; the first chunk is the lead
        sections = re.split(r"\n==+ (.+?) ==+\n", page["extract"])
        lead, rest = sections[0], dict(zip(sections[1::2], sections[2::2]))
        description = rest.get("Description", "")
        extract = f"{lead}\n\n{description}".strip()[:max_chars]
        url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page['title'].replace(' ', '_'))}"
        return {"title": page["title"], "extract": extract, "url": url}
