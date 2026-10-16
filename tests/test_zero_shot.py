import numpy as np
import pytest

from bioclip_co.application.zero_shot import prompts_for, species_probabilities
from bioclip_co.domain.taxonomy import Taxon

TAXON = Taxon(("Animalia", "Chordata", "Aves", "Apodiformes", "Trochilidae", "Amazilia", "tzacatl"), "rufous-tailed hummingbird")


def test_each_model_gets_the_names_it_knows_best():
    assert prompts_for([TAXON], "clip") == ["a photo of a rufous-tailed hummingbird"]
    assert prompts_for([TAXON], "bioclip") == [
        "a photo of Animalia Chordata Aves Apodiformes Trochilidae Amazilia tzacatl "
        "with common name rufous-tailed hummingbird"
    ]


def test_probabilities_are_one_row_per_image_and_sum_to_one():
    images = np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]])
    texts = np.array([[1.0, 0.0], [0.0, 1.0]])
    probs = species_probabilities(images, texts, logit_scale=100)
    assert probs.shape == (3, 2)
    assert probs.sum(axis=1) == pytest.approx([1, 1, 1])
    assert probs[0].argmax() == 0 and probs[1].argmax() == 1


def test_temperature_controls_how_sharp_the_answer_is():
    images = np.array([[0.6, 0.8]])
    texts = np.array([[1.0, 0.0], [0.0, 1.0]])
    soft = species_probabilities(images, texts, logit_scale=1)[0].max()
    sharp = species_probabilities(images, texts, logit_scale=100)[0].max()
    assert soft < sharp


def test_unknown_prompt_style():
    with pytest.raises(ValueError):
        prompts_for([TAXON], "gpt")
