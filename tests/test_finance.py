"""The headline equation, checked term by term against hand-worked numbers."""

import math
import re

import pytest

from src.config import load_config
from src.finance import battery_cost_aud, payback, show_equation
from tests.shared import NAMES, recommendation


def evaluate(**changes):
    terms = {
        "kwh_shifted": 1000.0,
        "r_out": 0.40,
        "r_in": 0.20,
        "efficiency": 0.8,
        "demand_saving": 50.0,
        "battery_cost": 10_000.0,
        "rebate": 2_000.0,
    }
    terms.update(changes)
    return payback(**terms)


def test_every_term_matches_the_hand_calculation():
    result = evaluate()
    assert result.r_in_after_losses == pytest.approx(0.25)  # 0.20 / 0.8
    assert result.saving_per_kwh == pytest.approx(0.15)     # 0.40 − 0.25
    assert result.energy_saving == pytest.approx(150.0)     # 1000 × 0.15
    assert result.annual_saving == pytest.approx(200.0)     # 150 + 50
    assert result.net_cost == pytest.approx(8_000.0)        # 10,000 − 2,000
    assert result.payback_years == pytest.approx(40.0)      # 8,000 / 200


def test_round_trip_losses_can_erase_a_positive_spread():
    # r_out is above r_in, but storing enough to deliver 1 kWh costs 0.20 / 0.8 = 0.25.
    result = evaluate(r_out=0.22, demand_saving=0.0)
    assert result.annual_saving < 0
    assert result.payback_years == math.inf


def test_a_battery_that_shifts_nothing_never_pays_back():
    assert evaluate(kwh_shifted=0.0, demand_saving=0.0).payback_years == math.inf


def test_demand_saving_is_added_not_folded_into_a_rate():
    # No energy margin at all: the demand term alone pays the battery back.
    result = evaluate(r_out=0.25, demand_saving=400.0)
    assert result.energy_saving == pytest.approx(0.0)
    assert result.payback_years == pytest.approx(20.0)      # 8,000 / 400


def test_cost_fit_reproduces_the_sanity_checks_in_batteries_yaml():
    cost = load_config().batteries.cost_model
    assert battery_cost_aud(10.0, cost.fixed_aud, cost.variable_aud_per_kwh) == pytest.approx(11_390)
    assert battery_cost_aud(13.5, cost.fixed_aud, cost.variable_aud_per_kwh) == pytest.approx(13_889)


@pytest.mark.parametrize("efficiency", [0.0, -0.1, 1.1])
def test_impossible_efficiency_is_refused(efficiency):
    with pytest.raises(ValueError, match="efficiency"):
        evaluate(efficiency=efficiency)


def test_rebate_above_the_battery_cost_is_refused():
    with pytest.raises(ValueError, match="rebate"):
        evaluate(rebate=12_000.0)


# ------------------------------------------------- the equation as the page shows it

SO = r" {14}= "
SUBSTITUTED = re.compile(SO + r"([\d,]+) × \(([\d.]+)c − ([\d.]+)c / ([\d.]+)\) \+ \$([\d,]+)")
SIMPLIFIED = re.compile(SO + r"([\d,]+) × ([\d.]+)c")
SAVING = re.compile(SO + r"\$([\d,]+\.\d\d)")
PAYBACK_SUBSTITUTED = re.compile(SO + r"\(\$([\d,]+) − \$([\d,]+)\) / \$([\d,]+\.\d\d)")
PAYBACK = re.compile(SO + r"([\d.]+) years")


def shown(pattern, line):
    match = pattern.fullmatch(line)
    assert match, line
    return [float(group.replace(",", "")) for group in match.groups()]


@pytest.mark.parametrize("name", NAMES)
def test_the_equation_shows_each_basis_figure_and_its_arithmetic_holds(name):
    basis = recommendation(name).basis
    lines = show_equation(basis).splitlines()
    assert lines[0] == "annual_saving = kwh_shifted × (r_out − r_in / efficiency) + demand_saving"
    assert lines[5] == "payback_years = (battery_cost − rebate) / annual_saving"
    kwh, r_out, r_in, efficiency, demand = shown(SUBSTITUTED, lines[1])
    kwh_again, margin = shown(SIMPLIFIED, lines[2])
    (saving,) = shown(SAVING, lines[3])
    cost, rebate, saving_again = shown(PAYBACK_SUBSTITUTED, lines[6])
    (payback_years,) = shown(PAYBACK, lines[7])

    # Each figure is its basis value, rounded as displayed.
    assert kwh == kwh_again == round(basis["kwh_shifted"])
    assert (r_out, r_in) == (round(basis["r_out"] * 100, 2), round(basis["r_in"] * 100, 2))
    assert efficiency == round(basis["efficiency"], 2)
    assert demand == round(basis["demand_saving"])
    assert margin == round((basis["r_out"] - basis["r_in"] / basis["efficiency"]) * 100, 2)
    assert saving == saving_again == round(basis["annual_saving"], 2)
    assert (cost, rebate) == (round(basis["battery_cost"]), round(basis["rebate"]))
    assert payback_years == round(basis["payback_years"], 1)

    # The results shown are the equation's results...
    assert basis["annual_saving"] == pytest.approx(
        basis["kwh_shifted"] * (basis["r_out"] - basis["r_in"] / basis["efficiency"])
        + basis["demand_saving"])
    assert basis["payback_years"] == pytest.approx(
        (basis["battery_cost"] - basis["rebate"]) / basis["annual_saving"])
    # ...and the arithmetic on the figures shown reaches them, to within the rounding
    # of those figures: half a unit in the last place of each.
    assert r_out - r_in / efficiency == pytest.approx(
        margin, abs=0.005 + 0.005 + 0.005 / efficiency + r_in / efficiency ** 2 * 0.005)
    assert kwh * margin / 100 + demand == pytest.approx(
        saving, abs=0.005 + 0.5 * margin / 100 + kwh * 0.005 / 100 + 0.5)
    net = cost - rebate
    assert net / saving == pytest.approx(
        payback_years, abs=0.05 + 1.0 / saving + net / saving ** 2 * 0.005)


def test_a_missing_term_fails_loudly_rather_than_dropping_its_line():
    basis = dict(recommendation("household_b").basis)
    del basis["rebate"]
    with pytest.raises(KeyError, match="rebate"):
        show_equation(basis)


def test_a_battery_that_moves_nothing_shows_why_and_never_pays_back():
    lines = show_equation(vars(evaluate(kwh_shifted=0.0, r_out=None, r_in=None,
                                        demand_saving=0.0))).splitlines()
    assert lines[1].endswith("= 0 × (no rates: the battery moved nothing) + $0")
    assert lines[2].endswith("= $0.00") and lines[-1].endswith("= never")
