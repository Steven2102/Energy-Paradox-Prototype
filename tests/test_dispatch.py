"""The dispatch simulation: required tests 1, 2, 3 and 9 from CLAUDE.md, each
run against all three fixtures, plus the checks that tie dispatch to the
headline equation."""

from dataclasses import replace
from datetime import date

import numpy as np
import pytest

from src import finance
from src.config import load_config
from src.dispatch import simulate
from src.engine import BATTERY_NOW, evaluate
from src.generator import DAYS, household_year
from src.profile import load_fixtures
from src.tariff import window_by_slot

CONFIG = load_config()
FIXTURES = {profile.name: profile for profile in load_fixtures()}
NAMES = ["reference_household", "household_b", "household_c"]

BATTERY = {
    "capacity_kwh": CONFIG.batteries.default_size_kwh,
    "power_kw": CONFIG.batteries.power_kw,
    "efficiency": CONFIG.batteries.round_trip_efficiency,
    "marginal_throughput_cost_aud_per_kwh": CONFIG.batteries.marginal_throughput_cost_aud_per_kwh,
}
HORIZON = CONFIG.assumptions.dispatch_horizon_intervals
TOL = 1e-9  # kWh

PINNED = date(2026, 9, 26)
COST = finance.battery_cost_aud(
    BATTERY["capacity_kwh"],
    CONFIG.batteries.cost_model.fixed_aud,
    CONFIG.batteries.cost_model.variable_aud_per_kwh,
)
REBATE = finance.rebate(
    BATTERY["capacity_kwh"],
    deeming_factor=CONFIG.incentives.deeming_period_on(PINNED).factor,
    stc_price_aud=CONFIG.incentives.stc_price_aud,
    taper=CONFIG.incentives.capacity_taper,
).rebate_aud


def year_of(profile, tariff=None):
    tariff = tariff or CONFIG.tariffs[profile.tariff_ref]
    return household_year(profile, tariff, CONFIG.assumptions)


def run(year, **battery):
    return simulate(year, horizon_intervals=HORIZON, **{**BATTERY, **battery})


def payback_years(profile, tariff):
    result = run(year_of(profile, tariff))
    return finance.payback(
        kwh_shifted=result.kwh_shifted,
        r_out=result.r_out,
        r_in=result.r_in,
        efficiency=BATTERY["efficiency"],
        demand_saving=0.0,
        battery_cost=COST,
        rebate=REBATE,
    ).payback_years


def dearest_window(tariff):
    return max(tariff.energy_windows, key=lambda window: window.rate_aud_per_kwh)


# ----------------------------------------------------------- required test 1

@pytest.mark.parametrize("name", NAMES)
def test_1_energy_balance_closes_every_interval(name):
    year = year_of(FIXTURES[name])
    result = run(year)
    charged = result.charge_grid + result.charge_solar
    losses = charged * (1 - BATTERY["efficiency"])
    change = np.diff(result.stored, prepend=0.0)
    step = BATTERY["power_kw"] / 2

    assert result.kwh_shifted > 0  # otherwise every check below passes vacuously
    # The battery: what goes in comes out, stays in, or is lost.
    assert np.allclose(charged, result.discharge + change + losses, rtol=0, atol=TOL)
    # The site: grid and solar in = consumption and export out + stored + lost.
    assert np.allclose(result.imports + year.solar,
                       year.load + result.exports + change + losses, rtol=0, atol=TOL)
    # Within the battery's limits.
    assert result.stored.min() >= -TOL
    assert result.stored.max() <= BATTERY["capacity_kwh"] + TOL
    assert charged.max() <= step + TOL
    assert result.discharge.max() <= step + TOL
    assert (result.imports >= -TOL).all() and (result.exports >= -TOL).all()
    assert not ((charged > TOL) & (result.discharge > TOL)).any()


# ----------------------------------------------------------- required test 2

@pytest.mark.parametrize("name", NAMES)
def test_2_a_zero_capacity_battery_reproduces_the_no_battery_bill(name):
    year = year_of(FIXTURES[name])
    result = run(year, capacity_kwh=0.0)
    imported = np.maximum(year.load - year.solar, 0.0)
    exported = np.maximum(year.solar - year.load, 0.0)
    no_battery_bill = (imported * year.import_rate).sum() - (exported * year.feed_in).sum()

    assert result.kwh_shifted == 0
    assert result.r_out is None and result.r_in is None
    assert (result.imports == imported).all() and (result.exports == exported).all()
    assert result.bill_with == no_battery_bill


# ----------------------------------------------------------- required test 3

@pytest.mark.parametrize("name", NAMES)
def test_3_a_lossless_infinite_battery_leaves_no_import_cheaper_energy_could_have_met(name):
    # The ideal battery also costs nothing to cycle, so any cheaper energy is
    # worth moving. "Earlier" means within the dispatch horizon: the battery
    # plans no further ahead.
    year = year_of(FIXTURES[name])
    result = run(year, capacity_kwh=np.inf, power_kw=np.inf, efficiency=1.0,
                 marginal_throughput_cost_aud_per_kwh=0.0)
    still_imported = result.imports - result.charge_grid

    assert result.kwh_shifted > 0
    for t in np.flatnonzero(still_imported > TOL):
        earlier = slice(max(0, t - HORIZON), t)
        rate = year.import_rate[t]
        assert not (year.import_rate[earlier] < rate).any(), f"cheaper grid energy before {t}"
        if year.feed_in < rate:
            assert not (result.exports[earlier] > TOL).any(), f"surplus exported before {t}"


# ----------------------------------------------------------- required test 9

@pytest.mark.parametrize("name", NAMES)
def test_9_more_use_in_the_dearest_window_shifts_more_into_it(name):
    profile = FIXTURES[name]
    tariff = CONFIG.tariffs[profile.tariff_ref]
    dearest = dearest_window(tariff).name
    more = replace(profile, billing_periods=tuple(
        replace(period, kwh_by_window={**period.kwh_by_window,
                                       dearest: period.kwh_by_window[dearest] * 1.10})
        for period in profile.billing_periods
    ))
    in_dearest = np.tile([window.name == dearest for window in window_by_slot(tariff)], DAYS)

    before = run(year_of(profile, tariff)).discharge[in_dearest].sum()
    after = run(year_of(more, tariff)).discharge[in_dearest].sum()
    assert after > before


@pytest.mark.parametrize("name", NAMES)
def test_9_a_wider_rate_spread_shortens_payback(name):
    profile = FIXTURES[name]
    tariff = CONFIG.tariffs[profile.tariff_ref]
    dearest = dearest_window(tariff)
    wider = replace(tariff, energy_windows=tuple(
        replace(window, rate_aud_per_kwh=window.rate_aud_per_kwh + 0.05)
        if window is dearest else window
        for window in tariff.energy_windows
    ))
    assert payback_years(profile, wider) < payback_years(profile, tariff)


@pytest.mark.parametrize("name", NAMES)
def test_9_a_longer_expected_stay_is_more_favourable(name):
    # The stay enters through the decision rule: a longer one can only remove a
    # failed limit, never add one, and never turn battery_now into anything else.
    profile = FIXTURES[name]

    def with_stay(years):
        stayed = replace(profile, form={**profile.form, "years_expected_in_home": years})
        return evaluate(stayed, CONFIG.tariffs[profile.tariff_ref], CONFIG, PINNED)

    def failed(evaluation):
        return {test.limit for test in evaluation.rule_tests if test.passed is False}

    short, long = with_stay(5), with_stay(30)
    assert failed(long) <= failed(short)
    assert long.rule_tests[1].margin_years < short.rule_tests[1].margin_years
    assert short.battery_action != BATTERY_NOW or long.battery_action == BATTERY_NOW


@pytest.mark.parametrize("name", NAMES)
def test_9_a_lower_feed_in_tariff_favours_a_battery_only_where_there_is_solar(name):
    profile = FIXTURES[name]
    tariff = CONFIG.tariffs[profile.tariff_ref]
    higher = payback_years(profile, replace(tariff, feed_in_tariff_aud_per_kwh=0.05))
    lower = payback_years(profile, replace(tariff, feed_in_tariff_aud_per_kwh=0.02))
    if profile.has_solar:
        assert lower < higher
    else:
        assert lower == higher


# ------------------------------------------ dispatch and the headline equation

@pytest.mark.parametrize("name", NAMES)
def test_the_headline_equation_reproduces_the_simulated_saving(name):
    result = run(year_of(FIXTURES[name]))
    equation = result.kwh_shifted * (result.r_out - result.r_in / BATTERY["efficiency"])
    assert equation == pytest.approx(result.bill_without - result.bill_with, rel=1e-9)


@pytest.mark.parametrize("name", NAMES)
def test_the_battery_only_discharges_where_the_margin_after_losses_beats_wear(name):
    year = year_of(FIXTURES[name])
    result = run(year)
    cheapest = np.where(year.solar > year.load,
                        np.minimum(year.import_rate, year.feed_in), year.import_rate)
    for t in np.flatnonzero(result.discharge > TOL):
        earlier = slice(max(0, t - HORIZON), t)
        margin = year.import_rate[t] - cheapest[earlier].min() / BATTERY["efficiency"]
        assert margin > BATTERY["marginal_throughput_cost_aud_per_kwh"]


def test_dispatch_declines_a_margin_below_the_wear_cost_and_accepts_one_above_it():
    # household_c charges at 0.14 and loses 10% on the round trip, so a stored
    # kWh costs 0.1556. Discharged at peak (0.22) it earns 6.44 c; in the
    # shoulder (0.17), 1.44 c.
    profile = FIXTURES["household_c"]
    tariff = CONFIG.tariffs[profile.tariff_ref]
    year = year_of(profile, tariff)
    windows = window_by_slot(tariff)
    shoulder = np.tile([window.name == "shoulder" for window in windows], DAYS)
    peak = np.tile([window.name == "peak" for window in windows], DAYS)

    at_2c = run(year, marginal_throughput_cost_aud_per_kwh=0.02)
    assert at_2c.discharge[shoulder].sum() == 0  # 1.44 c declined
    assert at_2c.discharge[peak].sum() > 0       # 6.44 c accepted

    at_1c = run(year, marginal_throughput_cost_aud_per_kwh=0.01)
    assert at_1c.discharge[shoulder].sum() > 0   # the same 1.44 c clears a lower bar


@pytest.mark.parametrize("name, source", [
    ("reference_household", "grid"),  # no solar: cheap midday import is the only cheap energy
    ("household_b", "solar"),         # flat tariff: grid energy is never cheaper later
    ("household_c", "grid"),
])
def test_each_household_charges_from_the_energy_that_is_cheapest_for_it(name, source):
    result = run(year_of(FIXTURES[name]))
    charged = {"grid": result.charge_grid.sum(), "solar": result.charge_solar.sum()}
    other = "solar" if source == "grid" else "grid"
    assert charged[source] > 0 and charged[other] == 0


def test_a_flat_tariff_without_solar_shifts_nothing():
    # No special case: with every half-hour priced the same, no earlier energy
    # is ever cheaper, so the battery finds nothing worth moving.
    without_solar = replace(FIXTURES["household_b"], has_solar=False, solar_kw=None,
                            annual_solar_export_kwh=None)
    result = run(year_of(without_solar))
    assert result.kwh_shifted == 0
    assert payback_years(without_solar, CONFIG.tariffs[without_solar.tariff_ref]) == np.inf
