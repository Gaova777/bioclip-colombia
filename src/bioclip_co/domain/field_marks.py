"""Choosing the question that best splits the candidates, and updating their probabilities with the answer.

Each species has field marks from a closed vocabulary (data/field_marks.json). A question is
"is <feature> <value>?". The best one is the one with the largest expected information gain,
weighted by the classifier's probabilities, so a feature every candidate shares is never asked.
Answers update the probabilities with Bayes' rule instead of throwing candidates away.
"""
import math
from collections.abc import Mapping, Sequence

COLOR = ["black", "white", "grey", "brown", "rufous", "buff", "yellow", "orange", "red", "pink", "green", "blue", "violet", "olive"]
VOCABULARY = {
    "crown": COLOR,
    "throat": COLOR,
    "underparts": COLOR,
    "back": COLOR,
    "tail": COLOR,
    "bill_color": ["black", "grey", "yellow", "orange", "red", "pink", "two-toned"],
    "bill_shape": ["short and conical", "straight", "long and straight", "curved", "hooked", "flat and broad"],
    "tail_shape": ["forked", "square", "rounded", "long and pointed"],
}
PHRASES = {
    "crown": "Is the top of the head mostly {}?",
    "throat": "Is the throat mostly {}?",
    "underparts": "Are the breast and belly mostly {}?",
    "back": "Is the back mostly {}?",
    "tail": "Is the tail mostly {}?",
    "bill_color": "Is the bill mostly {}?",
    "bill_shape": "Is the bill {}?",
    "tail_shape": "Is the tail {}?",
}

# P(answer "yes" | the candidate's field mark says...)
MATCH = 0.9  # ...the same value
MISMATCH = 0.1  # ...another value
UNKNOWN = 0.5  # ...nothing (the description doesn't say)
# the marks describe the adult male; when the female looks different the photo may be of her,
# so both a yes and a no say less about that species
MATCH_IF_SEXES_DIFFER = 0.65
MISMATCH_IF_SEXES_DIFFER = 0.35
MIN_GAIN = 0.05  # bits; below this a question isn't worth the person's time


def question_text(feature: str, value: str) -> str:
    return PHRASES[feature].format(value)


def p_yes(marks: Mapping | None, feature: str, value: str) -> float:
    mark = (marks or {}).get(feature)
    if mark is None:
        return UNKNOWN
    if marks.get("sexes_differ"):
        return MATCH_IF_SEXES_DIFFER if mark == value else MISMATCH_IF_SEXES_DIFFER
    return MATCH if mark == value else MISMATCH


def entropy(probs: Sequence[float]) -> float:
    return -sum(p * math.log2(p) for p in probs if p > 0)


def normalize(probs: Sequence[float]) -> list[float]:
    total = sum(probs)
    return [p / total for p in probs]


def posterior(prior: Sequence[float], likelihood: Sequence[float]) -> list[float]:
    return normalize([p * l for p, l in zip(prior, likelihood)])


def best_question(
    names: Sequence[str], prior: Sequence[float], marks: Mapping[str, Mapping], asked: Sequence[tuple[str, str]] = ()
) -> tuple[str, str, float] | None:
    """(feature, value, expected information gain in bits) of the most useful yes/no question, or None."""
    best = None
    for feature in VOCABULARY:
        # only values that some candidate actually has can split them
        for value in {m.get(feature) for n in names if (m := marks.get(n))} - {None}:
            if (feature, value) in asked:
                continue
            yes = [p_yes(marks.get(n), feature, value) for n in names]
            p_answer_yes = sum(p * l for p, l in zip(prior, yes))
            expected = (
                p_answer_yes * entropy(posterior(prior, yes))
                + (1 - p_answer_yes) * entropy(posterior(prior, [1 - l for l in yes]))
            )
            gain = entropy(prior) - expected
            if best is None or gain > best[2]:
                best = (feature, value, gain)
    return best if best and best[2] >= MIN_GAIN else None


def update(
    names: Sequence[str], prior: Sequence[float], marks: Mapping[str, Mapping], feature: str, value: str, answer: str
) -> list[float]:
    """Bayes' rule with the answer ("yes", "no", or anything else for "can't tell")."""
    yes = [p_yes(marks.get(n), feature, value) for n in names]
    if answer == "yes":
        return posterior(prior, yes)
    if answer == "no":
        return posterior(prior, [1 - l for l in yes])
    return list(prior)
