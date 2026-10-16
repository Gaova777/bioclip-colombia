import pytest

from bioclip_co.domain.abstention import descend
from bioclip_co.domain.taxonomy import Taxon

BIRDS = ("Animalia", "Chordata", "Aves")
HUMMINGBIRDS = BIRDS + ("Apodiformes", "Trochilidae")

# same numbers BioCLIP gave for the steely-vented hummingbird photo
TAXA = [
    Taxon(HUMMINGBIRDS + ("Saucerottia", "saucerottei")),
    Taxon(HUMMINGBIRDS + ("Saucerottia", "cyanifrons")),
    Taxon(HUMMINGBIRDS + ("Eriocnemis", "vestita")),
    Taxon(BIRDS + ("Passeriformes", "Thraupidae", "Sicalis", "flaveola")),
]
PROBS = [0.724, 0.173, 0.094, 0.009]


def test_confident_enough_goes_all_the_way_to_species():
    pred = descend(TAXA, PROBS, threshold=0.7)
    assert pred.reached_species
    assert pred.name == "Saucerottia saucerottei"
    assert pred.confidence == pytest.approx(0.724)


def test_stops_at_genus_when_the_species_is_not_clear():
    pred = descend(TAXA, PROBS, threshold=0.8)
    assert pred.rank == "genus"
    assert pred.name == "Saucerottia"
    assert pred.confidence == pytest.approx(0.897)


def test_a_stricter_threshold_stops_higher_in_the_tree():
    pred = descend(TAXA, PROBS, threshold=0.95)
    assert pred.rank == "family"
    assert pred.name == "Trochilidae"


def test_path_always_starts_at_the_kingdom():
    pred = descend(TAXA, PROBS, threshold=0.8)
    assert pred.path == HUMMINGBIRDS + ("Saucerottia",)


def test_descent_only_follows_children_of_the_chosen_node():
    # Family A wins (0.6) but splits evenly into two genera (0.3 each).
    # Genus B1 (0.4) is the best genus overall, yet it is not inside family A.
    order = BIRDS + ("Passeriformes",)
    taxa = [
        Taxon(order + ("A", "A1", "x")),
        Taxon(order + ("A", "A2", "y")),
        Taxon(order + ("B", "B1", "z")),
    ]
    pred = descend(taxa, [0.3, 0.3, 0.4], threshold=0.3)
    assert pred.path[4] == "A"
    assert pred.path[5] != "B1"


def test_when_nothing_passes_the_threshold_the_answer_is_empty():
    plant = Taxon(("Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", "Moraceae", "Morus", "alba"))
    bird = Taxon(("Animalia", "Chordata", "Aves", "Suliformes", "Sulidae", "Morus", "bassanus"))
    pred = descend([plant, bird], [0.5, 0.5], threshold=0.9)
    assert pred.rank is None
    assert pred.name is None


@pytest.mark.parametrize("threshold", [0, -0.1, 1.5])
def test_threshold_must_be_a_probability(threshold):
    with pytest.raises(ValueError):
        descend(TAXA, PROBS, threshold)
