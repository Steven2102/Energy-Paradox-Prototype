"""The recommendation: required tests 5, 6, 7 and 8 from CLAUDE.md on all three
fixtures, plus the decision rule, missing form answers, solar_first and the
running position."""

from dataclasses import replace
from datetime import date
from functools import cache
from types import SimpleNamespace

import pytest

from src.config import load_config
from src.engine import (BATTERY_NOT_YET, BATTERY_NOW, CANNOT_ASSESS, NO_ACTION, SOLAR_FIRST,
                        decide, evaluate, rule_test)
from src.profile import load_fixtures
from src.recommend import IncompleteRecommendation, check_complete, recommend
from src.revisit import TOLERANCE, sweep

CONFIG = load_config()
FIXTURES = {profile.name: profile for profile in load_fixtures()}
NAMES = ["reference_household", "household_b", "household_c"]
INSTALLED = date(2026, 9, 29)


def run(profile, tariff=None, config=CONFIG):
    return evaluate(profile, tariff or CONFIG.tariffs[profile.tariff_ref], config, INSTALLED)


@cache
def recommendation(name):
    return recommend(run(FIXTURES[name]))


# ----------------------------------------------------------- required test 5

def test_5_paybacks_span_at_least_three_times():
    paybacks = [recommendation(name).basis["payback_years"] for name in NAMES]
    assert max(paybacks) / min(paybacks) >= 3


def test_5_each_fixture_is_held_back_by_something_different():
    assert {name: recommendation(name).limited_by.label for name in NAMES} == {
        "reference_household": "demand in the peak window",
        "household_b": "battery capacity and available surplus",
        "household_c": "an unpriced charge",
    }


def test_5_at_least_one_fixture_reaches_battery_now_inside_its_sweep_range():
    assert any(revisit.to_action == BATTERY_NOW
               for name in NAMES for revisit in recommendation(name).revisit_if)


# ----------------------------------------------------------- required test 6

@pytest.mark.parametrize("name", NAMES)
def test_6_only_household_c_declares_an_unmodelled_charge(name):
    rec = recommendation(name)
    if name == "household_c":
        assert list(rec.unmodelled) == ["demand"]
        assert rec.action == CANNOT_ASSESS
        assert "demand charge" in [item.item for item in rec.not_priced]
    else:
        assert rec.unmodelled == {}


def test_6_the_declaration_stops_once_demand_charges_are_priced(monkeypatch):
    monkeypatch.setattr("src.tariff.UNMODELLED", ("block", "seasonal", "feed_in_windows"))
    evaluation = run(FIXTURES["household_c"])
    assert evaluation.unmodelled == {}
    assert evaluation.battery_action != CANNOT_ASSESS


# ----------------------------------------------------------- required test 7

@pytest.mark.parametrize("name", NAMES)
def test_7_no_recommendation_ships_incomplete(name):
    rec = recommendation(name)
    assert rec.revisit_if and rec.assumptions and rec.not_priced
    assert set(rec.basis) == {"kwh_shifted", "r_out", "r_in", "efficiency", "annual_saving",
                              "demand_saving", "battery_cost", "rebate", "payback_years"}
    assert all(value is not None for value in rec.basis.values())


@pytest.mark.parametrize("section", ["revisit_if", "assumptions", "not_priced"])
def test_7_a_recommendation_with_an_empty_section_is_refused(section):
    with pytest.raises(IncompleteRecommendation, match=section):
        check_complete(replace(recommendation("household_b"), **{section: ()}))


def test_7_a_recommendation_missing_a_basis_figure_is_refused():
    rec = recommendation("household_b")
    with pytest.raises(IncompleteRecommendation, match="basis.rebate"):
        check_complete(replace(rec, basis={**rec.basis, "rebate": None}))


# ----------------------------------------------------------- required test 8

def rerun(name, revisit, value):
    """Re-run the engine with one parameter moved -- applied here, independently of
    src/revisit.py, so a sweep that moves the wrong input fails too."""
    profile, config = FIXTURES[name], CONFIG
    tariff = CONFIG.tariffs[profile.tariff_ref]
    if revisit.parameter == "battery cost":
        cost = config.batteries.cost_model
        scale = value / (cost.fixed_aud + cost.variable_aud_per_kwh * config.batteries.default_size_kwh)
        scaled = replace(cost, fixed_aud=cost.fixed_aud * scale,
                         variable_aud_per_kwh=cost.variable_aud_per_kwh * scale)
        config = replace(config, batteries=replace(config.batteries, cost_model=scaled))
    elif revisit.parameter == "feed-in tariff":
        tariff = replace(tariff, feed_in_tariff_aud_per_kwh=value)
    elif revisit.parameter == "rate spread":
        dearest = max(tariff.energy_windows, key=lambda window: window.rate_aud_per_kwh)
        tariff = replace(tariff, energy_windows=tuple(
            replace(window, rate_aud_per_kwh=window.rate_aud_per_kwh + value)
            if window is dearest else window for window in tariff.energy_windows))
    elif revisit.parameter == "adding solar":
        profile = replace(profile, has_solar=True, solar_kw=value, annual_solar_export_kwh=None)
        if tariff.feed_in_tariff_aud_per_kwh is None:
            tariff = replace(tariff, feed_in_tariff_aud_per_kwh=(
                CONFIG.assumptions.new_solar_feed_in_aud_per_kwh))
    else:
        raise AssertionError(f"no independent re-run for {revisit.parameter!r}")
    return getattr(run(profile, tariff, config), revisit.tracks)


@pytest.mark.parametrize("name", NAMES)
def test_8_every_reported_threshold_and_every_no_change_is_real(name):
    for revisit in recommendation(name).revisit_if:
        if revisit.not_applicable:
            continue
        if revisit.to_action is None:
            for end in revisit.searched:
                assert rerun(name, revisit, end) == revisit.from_action, revisit
        else:
            assert rerun(name, revisit, revisit.threshold) == revisit.to_action, revisit
            assert rerun(name, revisit, revisit.before) == revisit.from_action, revisit


STEP = TOLERANCE["rate spread"]
LOW, HIGH = CONFIG.assumptions.revisit_ranges.rate_spread_change_aud_per_kwh


def sweep_rate_spread(changed):
    """The rate-spread sweep over its configured range, against a stand-in engine
    whose action changes wherever changed(value) is true."""
    def at(value):
        return SimpleNamespace(action="changed" if changed(value) else "unchanged")
    return sweep(SimpleNamespace(action="unchanged"), "rate spread", "$/kWh", "action",
                 0.0, (LOW, HIGH), at), at


@pytest.mark.parametrize("edge, changed", [
    ("upper", lambda value: value > HIGH - STEP / 2),
    ("lower", lambda value: value < LOW + STEP / 2),
])
def test_8_a_change_within_a_step_of_the_edge_is_no_change_within_the_range_searched(edge, changed):
    # Where the search stopped is not something it found. The range is reported
    # only as far as the answer was verified unchanged, so it holds at both ends.
    revisit, at = sweep_rate_spread(changed)
    low, high = revisit.searched
    assert revisit.to_action is None and revisit.threshold is None
    assert at(low).action == at(high).action == "unchanged"
    assert (high < HIGH) if edge == "upper" else (low > LOW)
    assert high - low >= (HIGH - LOW) - 2 * STEP


def test_8_a_change_further_from_the_edge_is_still_a_threshold():
    flip = HIGH - 2.5 * STEP
    revisit, at = sweep_rate_spread(lambda value: value > flip)
    assert revisit.to_action == "changed"
    assert revisit.before <= flip < revisit.threshold
    assert at(revisit.before).action == "unchanged" and at(revisit.threshold).action == "changed"


# ---------------------------------------------------------- the decision rule

@pytest.mark.parametrize("payback_years, warranty, stay, action", [
    (8.0, 10, 12, BATTERY_NOW),
    (11.0, 10, 12, BATTERY_NOT_YET),     # outside the warranty
    (11.0, 12, 10, BATTERY_NOT_YET),     # outside the stay
    (37.2, 10, None, BATTERY_NOT_YET),   # a failed limit decides it, whatever is missing
    (8.0, 10, None, CANNOT_ASSESS),      # nothing failed, but the answer turns on the stay
])
def test_the_decision_rule(payback_years, warranty, stay, action):
    tests = (rule_test("warranty", payback_years, warranty),
             rule_test("expected stay", payback_years, stay))
    assert decide(SimpleNamespace(annual_saving=100.0), tests, {}) == action


def test_an_unpriced_charge_or_a_battery_that_saves_nothing_decides_first():
    inside = (rule_test("warranty", 5.0, 10), rule_test("expected stay", 5.0, 12))
    assert decide(SimpleNamespace(annual_saving=100.0), inside, {"demand": "..."}) == CANNOT_ASSESS
    assert decide(SimpleNamespace(annual_saving=0.0), inside, {}) == NO_ACTION


def test_the_answer_names_each_failed_limit_and_by_how_much():
    warranty, stay = recommendation("household_b").rule_tests
    assert (warranty.limit, warranty.passed, warranty.limit_years) == ("warranty", False, 10)
    assert (stay.limit, stay.passed, stay.limit_years) == ("expected stay", False, 12)
    assert warranty.margin_years == pytest.approx(stay.margin_years + 2)
    assert 0 < stay.margin_years < 1


# ------------------------------------------------------- missing form answers

def test_missing_answers_are_declared_with_their_effect_never_defaulted_silently():
    rec = recommendation("reference_household")
    missing = {item.name: item for item in rec.assumptions if item.value is None}
    assert set(missing) == {"years_expected_in_home", "occupancy_pattern", "has_aircon / aircon_use"}
    assert missing["years_expected_in_home"].effect.startswith("would not change the answer")
    assert "fails the 10-year warranty on its own" in missing["years_expected_in_home"].effect
    for name in ("occupancy_pattern", "has_aircon / aircon_use"):
        assert missing[name].effect.startswith("would not change the answer (re-run with")
    assert [driver.priority for driver in rec.drivers] == [None]
    assert "No priorities were given" in rec.drivers[0].effect


def test_a_missing_stay_that_would_decide_the_answer_is_said_to():
    profile = FIXTURES["household_b"]
    unknown_stay = replace(profile, form={**profile.form, "years_expected_in_home": None})
    cost = CONFIG.batteries.cost_model
    cheaper = replace(CONFIG, batteries=replace(CONFIG.batteries, cost_model=replace(
        cost, fixed_aud=cost.fixed_aud * 0.6, variable_aud_per_kwh=cost.variable_aud_per_kwh * 0.6)))
    rec = recommend(run(unknown_stay, config=cheaper))
    stay = next(item for item in rec.assumptions if item.name == "years_expected_in_home")
    assert rec.action == CANNOT_ASSESS
    assert stay.effect.startswith("decides the answer")


# ------------------------------------------ solar_first, position, not_priced

def test_solar_first_outranks_the_reference_households_battery():
    rec = recommendation("reference_household")
    assert rec.battery_action == BATTERY_NOT_YET
    assert rec.action == SOLAR_FIRST
    assert rec.solar.payback_years <= (
        CONFIG.assumptions.indicative_solar.payback_ratio * rec.basis["payback_years"])
    assert recommendation("household_b").solar is None  # it already has solar


@pytest.mark.parametrize("name", NAMES)
def test_near_and_long_term_are_a_running_position(name):
    rec = recommendation(name)
    saving, outlay = rec.basis["annual_saving"], rec.near_term.net_outlay
    assert rec.near_term.position_after_year_1 == pytest.approx(saving - outlay)
    for year, position in rec.long_term.positions:
        assert position == pytest.approx(year * saving - outlay)
    crossover = rec.long_term.crossover_year
    assert (crossover - 1) * saving < outlay <= crossover * saving


@pytest.mark.parametrize("name, flagged", [
    ("household_b", {"independence"}),
    ("household_c", {"backup_power"}),
    ("reference_household", set()),
])
def test_not_priced_flags_what_the_household_said_matters(name, flagged):
    assert {item.stated_priority for item in recommendation(name).not_priced} - {None} == flagged
