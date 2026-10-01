"""The form's schema (QUESTIONS in src/profile.py): every question moves a
computed output, answers the form could not give are refused, and the two
guards -- an existing battery, existing solar."""

from dataclasses import replace
from functools import cache

import numpy as np
import pytest

from src.engine import BATTERY_NOW, CANNOT_ASSESS, SOLAR_FIRST, evaluate, not_assessable, tariff_for
from src.generator import household_year
from src.profile import QUESTION, QUESTIONS, check_answers
from src.recommend import (NotAssessed, _comfortable_spend, _not_priced, _unplaced_use,
                           recommend_for)
from src.revisit import _adding_ev, _planned_ev
from tests.shared import CONFIG, FIXTURES, INSTALLED, NAMES, recommendation, run


def evaluation(name="reference_household", **answers):
    """The engine on a fixture with some answers changed: one run, no sweeps."""
    profile = FIXTURES[name].with_answers(**answers)
    return evaluate(profile, tariff_for(profile, CONFIG.tariffs[profile.tariff_ref], CONFIG),
                    CONFIG, INSTALLED)


def load(name="household_b", **answers):
    """The generated year's half-hourly use."""
    profile = FIXTURES[name].with_answers(**answers)
    return household_year(profile, CONFIG.tariffs[profile.tariff_ref], CONFIG.assumptions).load


# ------------------------------------------------ every question moves something

def moves_solar():
    assert recommendation("reference_household").action == SOLAR_FIRST
    assert evaluation(has_solar=True, solar_kw=6.6).action != SOLAR_FIRST


def moves_solar_kw():
    assert (evaluation(has_solar=True, solar_kw=3.0).year.solar.sum()
            < evaluation(has_solar=True, solar_kw=8.0).year.solar.sum())


def moves_battery():
    assert not_assessable(FIXTURES["reference_household"].with_answers(has_battery=True))


def moves_battery_kwh():
    profile = FIXTURES["reference_household"].with_answers(has_battery=True, battery_kwh=13.5)
    assert "13.5 kWh battery" in not_assessable(profile)


def moves_home_during_the_day():
    at_home = load(home_during_the_day=True, work_from_home_days=None)  # the follow-up is not asked
    assert not np.array_equal(at_home, load(home_during_the_day=False))


def moves_work_from_home_days():
    assert not np.array_equal(load(work_from_home_days=0), load(work_from_home_days=3))


def moves_aircon():
    assert not np.array_equal(load(aircon="split"), load(aircon="ducted"))


def moves_pool_pump():
    names = [item.name for item in _unplaced_use(FIXTURES["household_b"].with_answers(
        pool_pump="main_circuit"))]
    assert names == ["pool pump running hours"]


def moves_hot_water():
    names = [item.name for item in _unplaced_use(FIXTURES["household_b"].with_answers(
        hot_water="electric_main_circuit"))]
    assert names == ["hot water heating hours"]


def moves_ev():
    assert _adding_ev(evaluation(ev="have")).not_applicable
    assert "a planned EV" in [item.item for item in _not_priced(evaluation(ev="planning"))]


def moves_ev_planned_year():
    assert _adding_ev(evaluation(ev="planning", ev_planned_year=2028)).planned.year == 2028


def moves_years_expected_in_home():
    stay = evaluation(years_expected_in_home=40).rule_tests[1]
    assert (stay.limit_years, stay.passed) == (40, True)


def moves_motivation():
    drivers = recommendation("household_b").drivers
    assert [driver.priority for driver in drivers] == ["lower_bill", "independence"]


def moves_max_upfront_aud():
    assert _comfortable_spend(cheaper_battery(max_upfront_aud=3000)) is not None


MOVES = {
    "has_solar": moves_solar,
    "solar_kw": moves_solar_kw,
    "has_battery": moves_battery,
    "battery_kwh": moves_battery_kwh,
    "home_during_the_day": moves_home_during_the_day,
    "work_from_home_days": moves_work_from_home_days,
    "aircon": moves_aircon,
    "pool_pump": moves_pool_pump,
    "hot_water": moves_hot_water,
    "ev": moves_ev,
    "ev_planned_year": moves_ev_planned_year,
    "years_expected_in_home": moves_years_expected_in_home,
    "motivation": moves_motivation,
    "max_upfront_aud": moves_max_upfront_aud,
}


@pytest.mark.parametrize("question", QUESTIONS, ids=lambda question: question.key)
def test_every_question_on_the_form_moves_a_computed_output(question):
    # A field that changes nothing computed does not belong on the form. A new
    # question needs a check here showing what it moves, or it should go.
    assert question.moves.strip()
    assert question.key in MOVES, f"show what {question.key} moves, or remove it from the form"
    MOVES[question.key]()


def test_household_size_is_not_asked():
    # The bill fixes the magnitude, so a person count moves nothing computed.
    assert "household_size" not in QUESTION


# -------------------------------------------------------- answers the form can give

@pytest.mark.parametrize("name", NAMES)
def test_each_fixture_answers_only_what_the_form_asks(name):
    check_answers(FIXTURES[name].form)


@pytest.mark.parametrize("answers, problem", [
    ({"has_solar": False, "household_size": 4}, "not a question on the form"),
    ({"has_solar": False, "aircon": "window_unit"}, "expected one of"),
    ({"has_solar": False, "work_from_home_days": 3}, "only asked when home_during_the_day"),
    ({"has_solar": True, "solar_kw": 0.1}, "outside"),
    ({"has_solar": False, "motivation": ["lower_bill", "lower_bill"]}, "distinct"),
    ({}, "has_solar must be answered"),
])
def test_an_answer_the_form_could_not_give_is_refused(answers, problem):
    with pytest.raises(ValueError, match=problem):
        check_answers(answers)


# ------------------------------------------------------------------ the guards

def test_an_existing_battery_gives_cannot_assess_with_the_reason_and_no_payback(monkeypatch):
    profile = FIXTURES["reference_household"].with_answers(has_battery=True, battery_kwh=13.5)

    def no_simulation(*args, **kwargs):
        raise AssertionError("a purchase payback was computed for a household with a battery")

    monkeypatch.setattr("src.engine.simulate_household", no_simulation)
    result = recommend_for(profile, CONFIG, INSTALLED)
    assert isinstance(result, NotAssessed)
    assert result.action == CANNOT_ASSESS
    assert "already has a 13.5 kWh battery" in result.reason
    with pytest.raises(ValueError, match="already has"):
        run(profile)


def test_existing_solar_takes_solar_first_off_the_table():
    assert recommendation("reference_household").action == SOLAR_FIRST
    with_solar = evaluation(has_solar=True, solar_kw=6.6)
    assert with_solar.solar is None  # no indicative comparison: the panels are already there
    assert with_solar.action != SOLAR_FIRST
    assert with_solar.tariff.feed_in_tariff_aud_per_kwh == (
        CONFIG.assumptions.new_solar_feed_in_aud_per_kwh)  # declared: the tariff states none


# ------------------------------------------------------------ the planned EV

@cache
def planned_ev_sweep():
    """The reference household planning an EV, and its adding-an-EV sweep: run once."""
    planning = evaluation(ev="planning")
    return planning, _adding_ev(planning)


@pytest.mark.parametrize("year, stay, inside", [
    (2028, 10, True),     # 2028, and the stay runs to 2036
    (2040, 10, False),
    (2028, None, None),   # no stay given: cannot be said
    (None, 10, None),     # no year given
])
def test_a_planned_ev_is_dated_against_the_stated_stay(year, stay, inside):
    planning, revisit = planned_ev_sweep()
    answered = planning.profile.with_answers(ev_planned_year=year, years_expected_in_home=stay)
    plan = _planned_ev(replace(planning, profile=answered), revisit)
    assert (plan.year, plan.inside_stay) == (year, inside)


def test_a_planned_ev_reads_its_answer_off_the_sweep_and_a_re_run_agrees():
    planning, revisit = planned_ev_sweep()
    typical = run(replace(planning.profile, added_ev_kwh_per_year=revisit.planned.at))
    assert typical.action == revisit.planned.action_then


# ---------------------------------------------------------- comfortable spend

def cheaper_battery(**answers):
    """household_b's engine run with a battery 40% cheaper, where it reaches battery_now."""
    cost = CONFIG.batteries.cost_model
    cheaper = replace(CONFIG, batteries=replace(CONFIG.batteries, cost_model=replace(
        cost, fixed_aud=cost.fixed_aud * 0.6, variable_aud_per_kwh=cost.variable_aud_per_kwh * 0.6)))
    profile = FIXTURES["household_b"].with_answers(**answers)
    return evaluate(profile, CONFIG.tariffs[profile.tariff_ref], cheaper, INSTALLED)


def test_a_recommended_battery_over_the_comfortable_spend_is_declared_not_weighed():
    over = cheaper_battery(max_upfront_aud=3000)  # the net outlay is $4,250
    assert over.action == BATTERY_NOW
    assert _comfortable_spend(over).over_by == pytest.approx(over.payback.net_cost - 3000)
    within = cheaper_battery(max_upfront_aud=20000)
    assert _comfortable_spend(within) is None and within.action == BATTERY_NOW
    # Only a recommended battery is compared: at today's price there is none to declare.
    assert _comfortable_spend(evaluation("household_b", max_upfront_aud=5000)) is None
