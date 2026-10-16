import pytest

from bioclip_co.domain.abstention import Prediction
from bioclip_co.domain.metrics import (
    OperatingPoint,
    calibration_error,
    is_correct,
    mistake_depth,
    pick_threshold,
    rank_accuracy,
    risk_coverage,
    shared_ranks,
)
from bioclip_co.domain.taxonomy import Taxon

BIRDS = ("Animalia", "Chordata", "Aves")
HUMMINGBIRDS = BIRDS + ("Apodiformes", "Trochilidae")

STEELY = Taxon(HUMMINGBIRDS + ("Saucerottia", "saucerottei"))
INDIGO = Taxon(HUMMINGBIRDS + ("Saucerottia", "cyanifrons"))
EMERALD = Taxon(HUMMINGBIRDS + ("Uranomitra", "franciae"))
FINCH = Taxon(BIRDS + ("Passeriformes", "Thraupidae", "Sicalis", "flaveola"))
TAXA = [STEELY, INDIGO, EMERALD, FINCH]


def test_a_short_answer_on_the_right_branch_is_correct():
    assert is_correct(Prediction(HUMMINGBIRDS + ("Saucerottia",), 0.9), INDIGO)


def test_a_short_answer_on_the_wrong_branch_is_wrong():
    assert not is_correct(Prediction(HUMMINGBIRDS + ("Uranomitra",), 0.9), INDIGO)


def test_shared_ranks():
    assert shared_ranks(STEELY, STEELY) == 7
    assert shared_ranks(STEELY, INDIGO) == 6
    assert shared_ranks(STEELY, FINCH) == 3


def test_rank_accuracy_forgives_errors_inside_the_right_genus():
    # one image of each Saucerottia, but the model always says "steely-vented"
    probs = [[0.6, 0.3, 0.1, 0.0], [0.6, 0.3, 0.1, 0.0]]
    truths = [STEELY, INDIGO]
    assert rank_accuracy(TAXA, probs, truths, "species") == 0.5
    assert rank_accuracy(TAXA, probs, truths, "genus") == 1.0


def test_mistake_depth_only_counts_wrong_answers():
    probs = [[0.9, 0.1, 0, 0], [0.9, 0.1, 0, 0], [0, 0, 0, 1.0]]
    truths = [STEELY, INDIGO, EMERALD]
    # first is right; second confuses two Saucerottia; third calls a hummingbird a finch
    assert mistake_depth(TAXA, probs, truths) == [6, 3]


def test_a_higher_threshold_trades_coverage_for_fewer_errors():
    probs = [
        [0.95, 0.05, 0, 0],  # sure and right
        [0.55, 0.45, 0, 0],  # unsure between two Saucerottia, truth is the second
    ]
    truths = [STEELY, INDIGO]
    loose, strict = risk_coverage(TAXA, probs, truths, thresholds=[0.5, 0.9])

    assert loose.species_coverage == 1.0
    assert loose.species_error == 0.5
    assert strict.species_coverage == 0.5
    assert strict.species_error == 0.0
    assert strict.any_error == 0.0
    assert strict.mean_depth == pytest.approx((7 + 6) / 2)


def test_calibration_error_is_zero_when_confidence_matches_accuracy():
    # four answers at 75% confidence, three of them right
    assert calibration_error([0.75] * 4, [True, True, True, False]) == pytest.approx(0)


def test_calibration_error_catches_overconfidence():
    # always 99% sure, right only half the time
    assert calibration_error([0.99] * 4, [True, False, True, False]) == pytest.approx(0.49)


def test_pick_threshold_takes_the_most_coverage_within_the_error_budget():
    points = [
        OperatingPoint(0.5, species_coverage=0.9, species_error=0.2, any_error=0.18, mean_depth=6.8),
        OperatingPoint(0.8, species_coverage=0.7, species_error=0.06, any_error=0.05, mean_depth=6.2),
        OperatingPoint(0.9, species_coverage=0.6, species_error=0.03, any_error=0.03, mean_depth=6.0),
    ]
    assert pick_threshold(points, max_error=0.05).threshold == 0.8
    with pytest.raises(ValueError):
        pick_threshold(points, max_error=0.01)
