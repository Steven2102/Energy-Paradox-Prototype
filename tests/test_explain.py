"""The justification (src/explain.py) on all three fixtures, from recorded replies.

No test here calls a language model: tests/conftest.py points src/llm.py at
tests/recordings/ with live calls off, so each test replays the reply recorded
for its exact input, and fails if that input has changed since. After changing
the prompt or the fact sheet, delete tests/recordings/ and record afresh:

    LLM_RECORD=1 python -m pytest tests/test_explain.py

The same checks against fresh replies, which calls the provider:

    python -m pytest -m live
"""

import re

import pytest

from src import llm
from src.explain import (LABELS, SYSTEM_PROMPT, UntraceableFigure, answer, explain,
                         justification_request, question_request, untraceable)
from src.revisit import describe_value
from tests.shared import NAMES, RECORDINGS, recommendation

# Figures no recommendation contains, asked of every household.
NOT_IN_THE_RECOMMENDATION = [
    "What is my monthly electricity bill?",
    "What would a 15 kWh battery cost me?",
]

# Telling the household its answer changes...
CHANGE_OF_ANSWER = re.compile(
    r"\b(answer|recommendation|advice)\s+(would\s+|will\s+|could\s+)?"
    r"(changes?|switch(es)?|flips?|becomes?)\b"
    r"|\b(changes?|switch(es)?|flips?|turns?)\s+(the|your)\s+(answer|recommendation|advice)\b",
    re.IGNORECASE)
# ...unless the sentence has already said it doesn't.
NEGATION = re.compile(r"\b(not|no|never|neither|nor|nothing|none)\b|n't\b", re.IGNORECASE)
# Prose for a homeowner: no lists, bold or headings.
NOT_PROSE = re.compile(r"^\s*([-*•]|\d+[.)]|#)\s|\*\*", re.MULTILINE)

STATED_PRIORITY = {"independence": "independence", "backup_power": "backup power"}


def check_justification(rec, prose):
    """What the project lead asked a justification to carry, each read from the object."""
    text = straight_quotes(prose).lower()
    assert not untraceable(prose, justification_request(rec))
    assert not NOT_PROSE.search(prose), NOT_PROSE.search(prose)
    # The recommendation, by its label.
    assert f'"{LABELS[rec.action].lower()}"' in text
    # Which limit it failed, and by how much.
    for test in rec.rule_tests:
        if test.passed is False:
            assert f"{abs(test.margin_years):.1f} years" in prose
    # The near-term cost against the long-term position.
    assert f"{abs(rec.near_term.position_after_year_1):,.0f}" in prose
    assert f"year {rec.long_term.crossover_year}" in text
    # What isn't priced: what the household said matters, and any charge on the bill.
    for item in rec.not_priced:
        if item.stated_priority:
            assert STATED_PRIORITY[item.stated_priority] in text
    for component in rec.unmodelled:
        assert f"{component} charge" in text and "partial" in text
    # The revisit thresholds.
    for revisit in rec.revisit_if:
        if revisit.threshold is not None:
            assert describe_value(revisit, revisit.threshold).lstrip("+-") in prose
    # How much weight it bears.
    if rec.solar is not None:
        assert "indicative" in text
    if rec.time_of_day_source == "form":
        assert "less certain" in text
    check_label_only_thresholds(rec, prose)


def check_label_only_thresholds(rec, prose):
    """A threshold that moves the label but not the advice is never called a change of
    answer; and where no threshold changes the advice, nothing is."""
    sentences = re.split(r"(?<=[.!?])\s+", straight_quotes(prose))
    for revisit in rec.revisit_if:
        if revisit.advice_changes is False:
            figure = describe_value(revisit, revisit.threshold)
            mentions = [sentence for sentence in sentences if figure in sentence]
            assert mentions, f"{figure} is not mentioned"
            for sentence in mentions:
                assert not claims_a_change_of_answer(sentence), sentence
    if not any(revisit.advice_changes for revisit in rec.revisit_if):
        for sentence in sentences:
            assert not claims_a_change_of_answer(sentence), sentence


def claims_a_change_of_answer(sentence):
    """A negation earlier in the sentence ("would not change the answer", "neither
    changes the answer") says the opposite; a quoted label ("Battery not yet") is not one."""
    match = CHANGE_OF_ANSWER.search(sentence)
    if not match:
        return False
    return not NEGATION.search(re.sub(r'"[^"]*"', "", sentence[:match.start()]))


def straight_quotes(prose):
    return prose.replace("“", '"').replace("”", '"')


def check_declined(reply):
    assert re.search(r"(?:\bnot|n't) available|\bunavailable\b", reply, re.IGNORECASE), reply


# ------------------------------------------------------------------ replayed

@pytest.mark.parametrize("name", NAMES)
def test_the_justification_carries_what_the_project_lead_asked_for(name):
    rec = recommendation(name)
    check_justification(rec, explain(rec))


@pytest.mark.parametrize("question", NOT_IN_THE_RECOMMENDATION)
@pytest.mark.parametrize("name", NAMES)
def test_a_figure_the_recommendation_does_not_contain_is_declined_not_invented(name, question):
    # Hard rule 1 as a test. answer() refuses a reply with any figure its input
    # lacks, so an invented bill or battery price fails before the check below.
    check_declined(answer(recommendation(name), question))


def test_a_reply_with_a_figure_not_in_its_input_is_refused(monkeypatch):
    monkeypatch.setattr(llm, "complete", lambda system, user: "A 15 kWh battery costs $14,960.")
    with pytest.raises(UntraceableFigure, match="14,960"):
        answer(recommendation("reference_household"), "What would a 15 kWh battery cost me?")


def test_every_recording_is_one_a_test_replays():
    needed = {llm.cache_key(SYSTEM_PROMPT, justification_request(recommendation(name)))
              for name in NAMES}
    needed |= {llm.cache_key(SYSTEM_PROMPT, question_request(recommendation(name), question))
               for name in NAMES for question in NOT_IN_THE_RECOMMENDATION}
    assert {path.stem for path in RECORDINGS.glob("*.json")} == needed


# ---------------------------------------------------------------------- live

@pytest.fixture
def fresh(monkeypatch, tmp_path):
    """Replies straight from the provider, not the recordings."""
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "LIVE", True)


@pytest.mark.live
@pytest.mark.parametrize("name", NAMES)
def test_a_fresh_justification_passes_the_same_checks(name, fresh):
    rec = recommendation(name)
    check_justification(rec, explain(rec))


@pytest.mark.live
@pytest.mark.parametrize("question", NOT_IN_THE_RECOMMENDATION)
@pytest.mark.parametrize("name", NAMES)
def test_a_fresh_reply_declines_a_figure_it_was_not_given(name, question, fresh):
    check_declined(answer(recommendation(name), question))
