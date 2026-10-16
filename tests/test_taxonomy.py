import pytest

from bioclip_co.domain.taxonomy import Taxon

RUFOUS_TAILED = Taxon(
    ("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae", "Amazilia", "tzacatl"),
    "rufous-tailed hummingbird",
)


def test_at_returns_the_name_at_each_rank():
    assert RUFOUS_TAILED.at("kingdom") == "Animalia"
    assert RUFOUS_TAILED.at("family") == "Trochilidae"
    assert RUFOUS_TAILED.at("species") == "tzacatl"


def test_prefix_keeps_the_path_from_the_root():
    assert RUFOUS_TAILED.prefix("order") == ("Animalia", "Chordata", "Aves", "Apodiformes")


def test_names_used_in_prompts():
    assert RUFOUS_TAILED.scientific_name == "Amazilia tzacatl"
    assert RUFOUS_TAILED.taxonomic_name == "Animalia Chordata Aves Apodiformes Trochilidae Amazilia tzacatl"


def test_lineage_must_have_seven_ranks():
    with pytest.raises(ValueError):
        Taxon(("Animalia", "Chordata", "Aves"))
