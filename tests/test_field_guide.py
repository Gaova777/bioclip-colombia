from langgraph.types import Command

from bioclip_co.application.field_guide import build_field_guide
from bioclip_co.domain.taxonomy import Taxon

HUMMINGBIRDS = ("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae")
TAXA = [
    Taxon(HUMMINGBIRDS + ("Colibri", "coruscans"), "sparkling violetear"),
    Taxon(HUMMINGBIRDS + ("Colibri", "cyanotus"), "lesser violetear"),
    Taxon(HUMMINGBIRDS + ("Colibri", "delphinae"), "brown violetear"),
]
MARKS = {
    "Colibri coruscans": {"underparts": "blue", "back": "green"},
    "Colibri cyanotus": {"underparts": "green", "back": "green"},
    "Colibri delphinae": {"underparts": "brown", "back": "brown"},
}


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
    def card(self, name, common, notes):
        return f"card for {name} from {notes}"


def run(probs, region=None, records=None, marks=MARKS):
    guide = build_field_guide(TAXA, 0.8, FakeOccurrences(records or {}), FakeEncyclopedia(), FakeWriter(), marks)
    config = {"configurable": {"thread_id": "test"}}
    return guide, config, guide.invoke({"probs": probs, "region": region}, config)


def test_a_sure_classifier_goes_straight_to_the_card():
    _, _, result = run([0.9, 0.05, 0.05])
    assert result["card"]["name"] == "Colibri coruscans"
    assert result["card"]["text"] == "card for Colibri coruscans from notes about Colibri coruscans"


def test_the_region_can_settle_it_without_asking():
    # unsure between two violetears, but only one has been recorded in that department
    _, _, result = run([0.5, 0.45, 0.05], region="Amazonas", records={"Colibri cyanotus": 12})
    assert result["card"]["name"] == "Colibri cyanotus"
    assert "GBIF: 1 of 3 candidates recorded in Amazonas" in result["trace"]


def test_it_asks_about_what_separates_the_likely_candidates():
    _, _, result = run([0.5, 0.45, 0.05])
    # the back is green for both likely ones, so it asks about the belly
    assert result["__interrupt__"][0].value["question"].startswith("Are the breast and belly mostly")


def test_the_answer_updates_the_probabilities_and_settles_it():
    guide, config, result = run([0.5, 0.45, 0.05])
    question = result["__interrupt__"][0].value["question"]
    answer = "yes" if "blue" in question else "no"  # the bird has a blue belly: it's the sparkling violetear
    result = guide.invoke(Command(resume=answer), config)
    assert result["card"]["name"] == "Colibri coruscans"


def test_cant_tell_three_times_stays_at_the_genus():
    guide, config, result = run([0.5, 0.45, 0.05])
    for _ in range(3):
        if "card" in result:
            break
        result = guide.invoke(Command(resume="can't tell"), config)
    assert result["card"]["name"] == "Colibri"
    assert result["card"]["rank"] == "genus"


def test_without_field_marks_it_doesnt_ask_and_stays_at_the_genus():
    _, _, result = run([0.5, 0.45, 0.05], marks={})
    assert "Field guide: no question would tell these candidates apart" in result["trace"]
    assert result["card"]["rank"] == "genus"


def test_threatened_species_hide_their_location():
    _, _, result = run([0.05, 0.05, 0.9])
    assert result["card"]["red_list"] == "EN"
    assert result["card"]["hide_location"]
