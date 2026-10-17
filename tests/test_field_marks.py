import pytest

from bioclip_co.domain.field_marks import best_question, question_text, update

# two violet-crowned hummingbirds and one with a white nape band
MARKS = {
    "Uranomitra franciae": {"crown": "violet", "underparts": "white", "tail": "green"},
    "Thalurania colombica": {"crown": "violet", "underparts": "blue", "tail": "blue", "tail_shape": "forked"},
    "Florisuga mellivora": {"crown": "blue", "underparts": "white", "tail": "white"},
}
NAMES = list(MARKS)


def test_a_feature_every_candidate_shares_is_never_asked():
    names = ["Uranomitra franciae", "Thalurania colombica"]
    feature, value, gain = best_question(names, [0.6, 0.4], MARKS)
    assert feature != "crown"  # both are violet: asking would tell nothing
    assert gain > 0


def test_the_best_question_splits_the_likely_candidates():
    feature, value, _ = best_question(NAMES, [0.45, 0.45, 0.1], MARKS)
    # underparts white/blue or tail green/blue separate the two likely ones; the crown doesn't
    assert (feature, value) in {("underparts", "white"), ("underparts", "blue"), ("tail", "green"), ("tail", "blue"),
                                ("tail_shape", "forked")}


def test_no_question_when_nothing_is_known():
    assert best_question(["a", "b"], [0.5, 0.5], {}) is None


def test_questions_already_asked_are_skipped():
    names = ["Uranomitra franciae", "Thalurania colombica"]
    first = best_question(names, [0.5, 0.5], MARKS)
    second = best_question(names, [0.5, 0.5], MARKS, asked=[first[:2]])
    assert second[:2] != first[:2]


def test_a_yes_moves_probability_to_the_candidates_that_match():
    after = update(NAMES, [0.4, 0.4, 0.2], MARKS, "underparts", "white", "yes")
    assert after[0] > 0.4 and after[2] > 0.2  # both white-bellied ones go up
    assert after[1] < 0.1
    assert sum(after) == pytest.approx(1)


def test_cant_tell_changes_nothing():
    assert update(NAMES, [0.4, 0.4, 0.2], MARKS, "tail", "green", "can't tell") == [0.4, 0.4, 0.2]


def test_a_no_weighs_less_when_the_female_may_look_different():
    marks = {"a": {"crown": "violet", "sexes_differ": True}, "b": {"crown": "violet"}, "c": {"crown": "green"}}
    after = update(["a", "b", "c"], [1 / 3] * 3, marks, "crown", "violet", "no")
    assert after[0] > after[1]  # a "no" can't rule out a species whose female may lack the violet crown


def test_question_text():
    assert question_text("underparts", "white") == "Are the breast and belly mostly white?"
