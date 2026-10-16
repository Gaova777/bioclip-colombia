from bioclip_co.adapters.candidates_csv import read_candidates


def test_reads_rows_into_taxa(tmp_path):
    csv_file = tmp_path / "birds.csv"
    csv_file.write_text(
        "kingdom,phylum,class,order,family,genus,species,common_name,observations\n"
        "Animalia,Chordata,Aves,Apodiformes,Trochilidae,Amazilia,tzacatl,Rufous-tailed Hummingbird,2853\n",
        encoding="utf-8",
    )
    [taxon] = read_candidates(csv_file)
    assert taxon.scientific_name == "Amazilia tzacatl"
    assert taxon.common_name == "rufous-tailed hummingbird"


def test_the_real_candidates_file_loads():
    taxa = read_candidates("data/candidates/trochilidae_colombia.csv")
    assert len(taxa) == 30
    assert {t.at("family") for t in taxa} == {"Trochilidae"}
