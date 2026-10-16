import pytest

from bioclip_co.domain.taxonomy import RANKS, Taxon, marginalize

HUMMINGBIRDS = ("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae")

# The BioCLIP output for the steely-vented hummingbird photo in data/samples:
# it was unsure about the species, but both top answers share the genus.
TAXA = [
    Taxon(HUMMINGBIRDS + ("Saucerottia", "saucerottei")),
    Taxon(HUMMINGBIRDS + ("Saucerottia", "cyanifrons")),
    Taxon(HUMMINGBIRDS + ("Eriocnemis", "vestita")),
    Taxon(("Animalia", "Chordata", "Aves", "Passeriformes", "Thraupidae", "Sicalis", "flaveola")),
]
PROBS = [0.724, 0.173, 0.094, 0.009]


def test_genus_probability_is_the_sum_of_its_species():
    by_genus = marginalize(TAXA, PROBS, "genus")
    assert by_genus[HUMMINGBIRDS + ("Saucerottia",)] == pytest.approx(0.897)


def test_results_are_sorted_most_likely_first():
    by_genus = marginalize(TAXA, PROBS, "genus")
    assert list(by_genus.values()) == sorted(by_genus.values(), reverse=True)


def test_higher_ranks_get_more_confident():
    family = marginalize(TAXA, PROBS, "family")[HUMMINGBIRDS]
    genus = marginalize(TAXA, PROBS, "genus")[HUMMINGBIRDS + ("Saucerottia",)]
    species = max(PROBS)
    assert species < genus < family


@pytest.mark.parametrize("rank", RANKS)
def test_probabilities_still_add_up_to_one_at_every_rank(rank):
    assert sum(marginalize(TAXA, PROBS, rank).values()) == pytest.approx(1.0)


def test_homonyms_in_different_families_are_kept_apart():
    # Morus is both the mulberry tree and the gannet
    mulberry = Taxon(("Plantae", "Tracheophyta", "Magnoliopsida", "Rosales", "Moraceae", "Morus", "alba"))
    gannet = Taxon(("Animalia", "Chordata", "Aves", "Suliformes", "Sulidae", "Morus", "bassanus"))
    by_genus = marginalize([mulberry, gannet], [0.5, 0.5], "genus")
    assert len(by_genus) == 2


def test_lengths_must_match():
    with pytest.raises(ValueError):
        marginalize(TAXA, [1.0], "genus")
