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
    """The language model. It never names a species on its own: it only works with the candidates it's given."""

    def question(self, candidates: list[Candidate], notes: dict[str, str], already_asked: list[str]) -> str:
        """One new yes/no question about something visible that tells the candidates apart."""
        ...

    def fits(self, candidate: Candidate, note: str, question: str, answer: str) -> bool | None:
        """Does the answer agree with this one candidate's notes? None if the notes don't say.

        One candidate at a time on purpose: a small model asked to pick among three
        sometimes picked the one the answer had just ruled out.
        """
        ...

    def card(self, name: str, common: str, notes: str) -> str:
        """A short field-guide card, using only the notes."""
        ...
