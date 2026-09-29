"""The generator: the year's shape from the billing periods across the year
and, within the day, from the bill's window split where it has one and the
form where it has not. Run against all three fixtures."""

from datetime import timedelta

import numpy as np
import pytest

from src.config import load_config
from src.generator import DAYS, INTERVALS, household_year
from src.profile import load_fixtures
from src.tariff import SLOTS_PER_DAY, window_by_slot

CONFIG = load_config()
FIXTURES = {profile.name: profile for profile in load_fixtures()}
NAMES = ["reference_household", "household_b", "household_c"]
DAYTIME = slice(18, 32)  # 09:00-16:00
EVENING = slice(32, 44)  # 16:00-22:00


def generate(name):
    profile = FIXTURES[name]
    return profile, household_year(profile, CONFIG.tariffs[profile.tariff_ref], CONFIG.assumptions)


def by_day(values):
    return values.reshape(DAYS, SLOTS_PER_DAY)


def first_weekday(period, weekend=False):
    days = (period.start + timedelta(days=d) for d in range(7))
    return next(day for day in days if (day.weekday() >= 5) == weekend)


@pytest.mark.parametrize("name", NAMES)
def test_every_window_of_every_billing_period_keeps_its_billed_total(name):
    profile, year = generate(name)
    tariff = CONFIG.tariffs[profile.tariff_ref]
    load = by_day(year.load)

    assert year.load.shape == (INTERVALS,)
    for period in profile.billing_periods:
        days = load[(period.start - year.start).days:][:period.days]
        for window, kwh in period.kwh_by_window.items():
            in_window = [slot.name == window for slot in window_by_slot(tariff)]
            assert days[:, in_window].sum() == pytest.approx(kwh, rel=1e-9)


@pytest.mark.parametrize("name, peak_months", [
    ("reference_household", {6, 7, 8}),  # winter: overnight shoulder load
    ("household_b", {12, 1, 2}),          # summer: air conditioning
    ("household_c", {12, 1, 2}),
])
def test_the_seasonal_peak_comes_from_the_households_own_billing_periods(name, peak_months):
    profile, year = generate(name)
    daily = by_day(year.load).sum(axis=1)
    busiest = max(profile.billing_periods, key=lambda period: period.daily_kwh)

    busiest_day = year.start + timedelta(days=int(np.argmax(daily)))
    assert busiest.start <= busiest_day < busiest.end
    assert busiest.start.month in peak_months
    # Each day carries its period's daily use, so the year swings as the bills do.
    assert daily.max() / daily.min() == pytest.approx(
        busiest.daily_kwh / min(period.daily_kwh for period in profile.billing_periods))


@pytest.mark.parametrize("name, weekends_differ", [
    ("household_b", True),           # both out on weekdays
    ("household_c", False),          # someone home most days
    ("reference_household", False),  # occupancy not stated
])
def test_weekdays_differ_from_weekends_only_where_occupancy_implies_it(name, weekends_differ):
    profile, year = generate(name)
    load = by_day(year.load)
    period = profile.billing_periods[0]
    weekday = load[(first_weekday(period) - year.start).days]
    weekend = load[(first_weekday(period, weekend=True) - year.start).days]
    if weekends_differ:
        assert weekday[DAYTIME].sum() < weekend[DAYTIME].sum()
    else:
        assert np.array_equal(weekday, weekend)


@pytest.mark.parametrize("name, source", [
    ("reference_household", "bill"),
    ("household_b", "form"),  # flat tariff: no window split on the bill
    ("household_c", "bill"),
])
def test_a_flat_bill_takes_its_whole_time_of_day_shape_from_the_form(name, source):
    assert generate(name)[1].time_of_day_source == source


def test_summer_evening_air_conditioning_moves_use_into_the_evening():
    profile, year = generate("household_b")
    load = by_day(year.load)
    quiet = min(profile.billing_periods, key=lambda period: period.daily_kwh)
    busy = max(profile.billing_periods, key=lambda period: period.daily_kwh)

    def evening_share(period):
        day = load[(first_weekday(period) - year.start).days]
        return day[EVENING].sum() / day.sum()

    assert evening_share(busy) > evening_share(quiet)


def test_solar_follows_a_seasonal_envelope_and_still_matches_the_bills_export():
    profile, year = generate("household_b")
    solar = by_day(year.solar)
    months = [(year.start + timedelta(days=d)).month for d in range(DAYS)]

    def in_month(month):
        return [d for d in range(DAYS) if months[d] == month]

    # The best day of a month is a clear one, so June against December is the envelope.
    best = {month: max(solar[d].sum() for d in in_month(month)) for month in (6, 12)}
    assert best[6] / best[12] == pytest.approx(
        CONFIG.assumptions.solar_june_to_december_ratio, abs=0.03)
    # Winter days are shorter: output starts later in June than in December.
    first_light = {month: min(np.flatnonzero(solar[d])[0] for d in in_month(month))
                   for month in (6, 12)}
    assert first_light[6] > first_light[12]
    # And the whole year is still anchored to the bill.
    export = np.maximum(year.solar - year.load, 0.0).sum()
    assert export == pytest.approx(profile.annual_solar_export_kwh, rel=1e-6)
