"""The headline equation, checked term by term against hand-worked numbers."""

import math

import pytest

from src.config import load_config
from src.finance import battery_cost_aud, payback


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
