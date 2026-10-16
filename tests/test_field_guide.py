from langgraph.types import Command

from bioclip_co.application.field_guide import build_field_guide
from bioclip_co.domain.taxonomy import Taxon

HUMMINGBIRDS = ("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae")
TAXA = [
    Taxon(HUMMINGBIRDS + ("Colibri", "coruscans"), "sparkling violetear"),
    Taxon(HUMMINGBIRDS + ("Colibri", "cyanotus"), "lesser violetear"),
    Taxon(HUMMINGBIRDS + ("Colibri", "delphinae"), "brown violetear"),
]


class FakeOccurrences:
    def __init__(self, records):
        self.records = records

    def records_in(self, name, region):
        return self.records.get(name, 0)

    def red_list(self, name):
        return "EN" if name == "Colibri delphinae" else "LC"


class FakeEncyclopedia:
    def summary(self, name):
        return {"title": name, "extract": f"notes about {name}", "url": f"https://example.org/{name}"}


class FakeWriter:
    def __init__(self):
        self.calls = []

    def question(self, candidates, notes, already_asked):
        self.calls.append("question")
        return "Is the tail long?" if already_asked else "Is the belly blue?"

    def fits(self, candidate, note, question, answer):
        self.calls.append(f"fits {candidate['name']}")
        if answer == "yes":  # "yes, the belly is blue": only the sparkling violetear has one
            return candidate["name"] == "Colibri coruscans"
        if answer == "no":
            return False if candidate["name"] == "Colibri coruscans" else None
        return None

    def card(self, name, common, notes):
        return f"card for {name} from {notes}"


def run(probs, region=None, records=None, writer=None):
    writer = writer or FakeWriter()
    guide = build_field_guide(TAXA, 0.8, FakeOccurrences(records or {}), FakeEncyclopedia(), writer)
    config = {"configurable": {"thread_id": "test"}}
    return guide, config, guide.invoke({"probs": probs, "region": region}, config), writer


def test_a_sure_classifier_goes_straight_to_the_card():
    _, _, result, writer = run([0.9, 0.05, 0.05])
    assert result["card"]["name"] == "Colibri coruscans"
    assert result["card"]["text"] == "card for Colibri coruscans from notes about Colibri coruscans"
    assert writer.calls == []  # no question needed


def test_the_region_can_settle_it_without_asking():
    # unsure between two violetears, but only one has been recorded in that department
    _, _, result, writer = run([0.5, 0.45, 0.05], region="Amazonas", records={"Colibri cyanotus": 12})
    assert result["card"]["name"] == "Colibri cyanotus"
    assert writer.calls == []
    assert "GBIF: 1 of 3 candidates recorded in Amazonas" in result["trace"]


def test_it_asks_and_waits_for_the_answer():
    guide, config, result, writer = run([0.5, 0.45, 0.05])
    assert result["__interrupt__"][0].value["question"] == "Is the belly blue?"
    assert "card" not in result

    result = guide.invoke(Command(resume="yes"), config)
    assert result["card"]["name"] == "Colibri coruscans"
    # the question isn't written twice when resuming, and each candidate is judged once
    assert writer.calls == ["question"] + [f"fits {t.scientific_name}" for t in TAXA]


def test_ruling_one_out_lets_the_classifier_decide_among_the_rest():
    guide, config, _, _ = run([0.5, 0.45, 0.05])
    result = guide.invoke(Command(resume="no"), config)
    # coruscans is out; cyanotus has 0.45 / 0.5 = 90% of what's left
    assert result["card"]["name"] == "Colibri cyanotus"


def test_after_two_unhelpful_answers_the_card_stays_at_the_genus():
    guide, config, _, _ = run([0.5, 0.45, 0.05])
    result = guide.invoke(Command(resume="not sure"), config)
    assert result["__interrupt__"][0].value["question"] == "Is the tail long?"  # a second, different question
    result = guide.invoke(Command(resume="not sure"), config)
    assert result["card"]["name"] == "Colibri"
    assert result["card"]["rank"] == "genus"


def test_threatened_species_hide_their_location():
    _, _, result, _ = run([0.05, 0.05, 0.9])
    assert result["card"]["red_list"] == "EN"
    assert result["card"]["hide_location"]
