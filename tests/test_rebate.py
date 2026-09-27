"""The battery rebate: STCs from the deeming factor in effect on the install
date, tapered by capacity band, sold at the STC price. Checked against the
sourced figures in config/incentives.yaml: factor 6.8 to 31 December 2026,
then 5.7, at $38 per STC."""

from datetime import date

import pytest

from src.config import load_config, parse_incentives
from src.finance import rebate

INCENTIVES = load_config().incentives

BEFORE_STEP_DOWN = date(2026, 9, 26)
FULL_SHARE_PER_KWH = 6.8 * 38.00  # $258.40 for each kWh in the first band


def rebate_on(day: date, usable_kwh: float) -> float:
    return rebate(
        usable_kwh,
        deeming_factor=INCENTIVES.deeming_period_on(day).factor,
        stc_price_aud=INCENTIVES.stc_price_aud,
        taper=INCENTIVES.capacity_taper,
    ).rebate_aud


def test_reproduces_the_worked_check_in_incentives_yaml():
    assert rebate_on(BEFORE_STEP_DOWN, 10) == pytest.approx(2_584.00)  # 10 × 1.00 × 6.8 × $38


def test_the_14_kwh_taper_boundary():
    at_13, at_14, at_15 = (rebate_on(BEFORE_STEP_DOWN, kwh) for kwh in (13, 14, 15))
    assert at_14 == pytest.approx(14 * FULL_SHARE_PER_KWH)            # all 14 kWh at the full share
    assert at_14 - at_13 == pytest.approx(FULL_SHARE_PER_KWH)         # the 14th kWh earns 100%
    assert at_15 - at_14 == pytest.approx(0.60 * FULL_SHARE_PER_KWH)  # the 15th earns 60%


def test_capacity_in_all_three_bands():
    # 30 kWh = 14 × 1.00 + 14 × 0.60 + 2 × 0.15 = 22.7 full-share kWh
    assert rebate_on(BEFORE_STEP_DOWN, 30) == pytest.approx(22.7 * FULL_SHARE_PER_KWH)


def test_capacity_above_50_kwh_earns_nothing():
    assert rebate_on(BEFORE_STEP_DOWN, 60) == pytest.approx(rebate_on(BEFORE_STEP_DOWN, 50))


@pytest.mark.parametrize("day, factor", [
    (date(2026, 12, 31), 6.8),  # the last day before the step-down
    (date(2027, 1, 1), 5.7),    # the step-down itself
    (date(2027, 3, 15), 5.7),   # after it
], ids=str)
def test_the_1_january_2027_step_down(day, factor):
    assert INCENTIVES.deeming_period_on(day).factor == factor
    assert rebate_on(day, 10) == pytest.approx(10 * factor * 38.00)


@pytest.mark.parametrize("day", [date(2026, 4, 30), date(2028, 1, 1)], ids=str)
def test_a_date_outside_the_published_schedule_is_refused(day):
    with pytest.raises(ValueError, match="no deeming factor"):
        rebate_on(day, 10)


def test_a_schedule_with_overlapping_periods_is_refused():
    raw = {
        "stc_price_aud": 38.00,
        "stc_price_range_aud": [33.00, 40.00],
        "deeming_factor_schedule": [
            {"from": date(2026, 5, 1), "until": date(2026, 12, 31), "factor": 6.8},
            {"from": date(2026, 12, 1), "until": date(2027, 6, 30), "factor": 5.7},
        ],
        "capacity_taper": [{"up_to_kwh": 14, "share": 1.00}],
    }
    with pytest.raises(ValueError, match="no gap or overlap"):
        parse_incentives(raw)
