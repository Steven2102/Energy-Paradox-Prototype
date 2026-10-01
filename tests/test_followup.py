"""The follow-up chat (src/followup.py): the three intents on all three fixtures,
from recorded replies, and the closed set it fails closed around.

The classifier's and the answers' replies are replayed from
tests/recordings/followup/; LLM_RECORD=1 records what is missing (as in
tests/test_explain.py). A counterfactual re-runs recommend_for. Its sweeps are
stubbed here, for time: the answer is written from the before-and-after
figures, which do not include them.
"""

import re
from functools import cache

import pytest

from src import llm, recommend
from src.explain import SYSTEM_PROMPT, IncompleteAnswer, UntraceableFigure
from src.followup import CLASSIFIER, DECLINED, Proposal, accepted, ask, changed_inputs
from src.revisit import Revisit
from tests.shared import CONFIG, FIXTURES, INSTALLED, NAMES, RECORDINGS, recommendation

FOLLOWUPS = RECORDINGS / "followup"

EXPLAIN = "Why is the payback so long?"
RESIZE = "What about a 13 kWh battery?"
RETAILER = "Which retailer is cheapest?"
TARIFF = "What if I switched to a flat rate?"
PRICES = "What if prices go up 10% a year?"
ASKED = [(name, question) for name in NAMES for question in (EXPLAIN, RESIZE, RETAILER)] + [
    ("reference_household", TARIFF), ("reference_household", PRICES)]


def without_sweeps(evaluation):
    return (Revisit("battery cost", "$", "action", None, None, evaluation.action, None, None,
                    None, "stubbed: a follow-up answers from the before-and-after figures"),)


@cache
def asked(name, question):
    """The chat's reply for a fixture, from the recordings: each asked once."""
    before = recommendation(name)  # with its sweeps: shared with the other test modules
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(llm, "CACHE_DIR", FOLLOWUPS)
        patch.setattr(recommend, "revisit_if", without_sweeps)
        return ask(question, before, FIXTURES[name], CONFIG, INSTALLED)


# ------------------------------------------------------------ the three intents

@pytest.mark.parametrize("name", NAMES)
def test_explain_answers_from_the_recommendation_as_it_stands_and_briefly(name):
    reply = asked(name, EXPLAIN)
    assert (reply.intent, reply.changed, reply.after) == ("explain", None, None)
    assert len(reply.text.split()) <= 120  # one question answered, not the justification again


@pytest.mark.parametrize("name", NAMES)
def test_explain_never_runs_the_engine(name, monkeypatch):
    before = recommendation(name)

    def no_engine(*args, **kwargs):
        raise AssertionError("an explain question re-ran the engine")

    monkeypatch.setattr("src.engine.simulate_household", no_engine)
    monkeypatch.setattr(llm, "CACHE_DIR", FOLLOWUPS)
    assert ask(EXPLAIN, before, FIXTURES[name], CONFIG, INSTALLED).intent == "explain"


@pytest.mark.parametrize("name", NAMES)
def test_a_counterfactual_re_runs_and_answers_with_the_payback_before_and_after(name):
    before, reply = recommendation(name), asked(name, RESIZE)
    assert (reply.intent, reply.changed) == ("counterfactual", "a 13 kWh battery instead of 10 kWh")
    assert reply.after.evaluated_kwh == 13
    old, new = (f"{rec.basis['payback_years']:.1f}" for rec in (before, reply.after))
    assert f"Battery payback: {old} years before; {new} years after." in reply.sheet
    assert old in reply.text and new in reply.text
    assert len(reply.text.split()) <= 120


@pytest.mark.parametrize("name", NAMES)
def test_out_of_scope_is_declined_with_the_reason_and_nothing_written(name):
    reply = asked(name, RETAILER)
    assert (reply.intent, reply.text, reply.sheet) == (
        "out_of_scope", DECLINED["retailer_or_brand"], None)


@pytest.mark.parametrize("question, topic", [(TARIFF, "tariff_type"), (PRICES, "price_growth")])
def test_the_two_unsupported_counterfactuals_are_declined_with_their_reasons(question, topic):
    assert (asked("reference_household", question).intent,
            asked("reference_household", question).text) == ("out_of_scope", DECLINED[topic])


def test_every_follow_up_recording_is_one_a_test_replays():
    needed = set()
    for name, question in ASKED:
        needed.add(llm.cache_key(CLASSIFIER, question))
        reply = asked(name, question)
        if reply.sheet is not None:
            needed.add(llm.cache_key(SYSTEM_PROMPT, reply.sheet))
    assert {path.stem for path in FOLLOWUPS.glob("*.json")} == needed


# --------------------------------------------------- the closed set, failing closed

@pytest.mark.parametrize("reply, proposal", [
    ('{"intent": "explain"}', Proposal("explain")),
    ('```json\n{"intent": "explain"}\n```', Proposal("explain")),
    ('{"intent": "counterfactual", "parameter": "battery_size", "value": 13}',
     Proposal("counterfactual", "battery_size", 13)),
    ('{"intent": "counterfactual", "parameter": "add_ev", "value": null}',
     Proposal("counterfactual", "add_ev", None)),
    ('{"intent": "out_of_scope", "topic": "price_growth"}',
     Proposal("out_of_scope", topic="price_growth")),
    ('{"intent": "counterfactual", "parameter": "tariff_type", "value": null}',
     Proposal("out_of_scope", topic="tariff_type")),
])
def test_a_proposal_inside_the_closed_set_is_accepted(reply, proposal):
    assert accepted(reply) == proposal


@pytest.mark.parametrize("reply", [
    "It is an explain question.",                                                # does not parse
    "[]",                                                                        # not an object
    '{"intent": "improvise"}',
    '{"intent": "counterfactual", "parameter": "add_heat_pump", "value": 1}',   # outside the set
    '{"intent": "counterfactual", "parameter": "battery_size", "value": "big"}',
    '{"intent": "counterfactual", "parameter": "battery_size", "value": true}',
    '{"intent": "out_of_scope", "topic": "weather"}',
])
def test_anything_else_is_out_of_scope(reply):
    assert accepted(reply) == Proposal("out_of_scope", topic="other")


@pytest.mark.parametrize("name, answers, parameter, value, reason", [
    ("household_b", {}, "add_solar", None, "already have solar"),
    ("household_b", {"ev": "have"}, "add_ev", None, "already have an EV"),
    ("reference_household", {}, "battery_size", None, "Which size?"),
    ("reference_household", {}, "battery_size", 25, "outside"),
    ("reference_household", {}, "add_ev", 9000, "outside"),
    ("reference_household", {}, "add_solar", 60, "outside"),
])
def test_a_counterfactual_python_cannot_run_is_declined_with_the_reason(
        name, answers, parameter, value, reason):
    profile = FIXTURES[name].with_answers(**answers)
    decline = changed_inputs(Proposal("counterfactual", parameter, value), profile, CONFIG)
    assert isinstance(decline, str) and reason in decline


def test_a_what_if_answer_with_an_invented_figure_or_without_both_paybacks_is_refused(
        monkeypatch):
    before = recommendation("reference_household")
    monkeypatch.setattr(recommend, "revisit_if", without_sweeps)

    def stand_in(answer):
        def complete(system, user, **context):
            if system == CLASSIFIER:
                return '{"intent": "counterfactual", "parameter": "battery_size", "value": 13}'
            after = re.search(r"Battery payback: .* before; (.*) after\.", user).group(1)
            return answer.format(after=after)
        return complete

    monkeypatch.setattr(llm, "complete", stand_in("It would be 31.4 years."))
    with pytest.raises(UntraceableFigure):
        ask(RESIZE, before, FIXTURES["reference_household"], CONFIG, INSTALLED)
    monkeypatch.setattr(llm, "complete", stand_in("It would be {after}."))
    with pytest.raises(IncompleteAnswer):
        ask(RESIZE, before, FIXTURES["reference_household"], CONFIG, INSTALLED)
