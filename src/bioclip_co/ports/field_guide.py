from typing import Protocol, TypedDict


class Note(TypedDict):
    title: str
    extract: str
    url: str


class Encyclopedia(Protocol):
    """Short, citable text about a taxon (Wikipedia, a field guide, a local database...)."""

    def summary(self, name: str) -> Note | None: ...


class Occurrences(Protocol):
    """Where a species has been recorded, and how threatened it is."""

    def records_in(self, scientific_name: str, region: str) -> int: ...

    def red_list(self, scientific_name: str) -> str | None:
        """IUCN category code (LC, NT, VU, EN, CR...) or None if unknown."""
        ...


class Candidate(TypedDict):
    name: str  # scientific name
    common: str
    p: float  # probability within the node the classifier trusts


class FieldGuideWriter(Protocol):
    """The language model. It never chooses a species: it only writes about the one it's given."""

    def card(self, name: str, common: str, notes: str) -> str:
        """A short field-guide card, using only the notes."""
        ...
