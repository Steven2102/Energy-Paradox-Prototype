"""One representative year, half-hour by half-hour: the input to dispatch.

PROVISIONAL until stage 3. The fixtures give annual consumption by window;
dispatch needs 17,520 half-hours. Until the stage 3 generator shapes
consumption properly, this module does the least that stays true to the bill:

  - each energy window's annual kWh is spread evenly over the 365 days and
    evenly across that window's half-hours, so a flat tariff gets the same
    load in every half-hour;
  - solar is the same half-sine every day, between the daylight hours in
    config/assumptions.yaml, scaled so that the export without a battery
    equals the export on the bill.

Both are averages. A dispatch run on averages sees no day-to-day variation,
so it overstates what a small battery captures (CLAUDE.md, the fallback
note). Stage 3 replaces the shapes; the totals stay anchored to the bill.
"""

from collections import Counter
from dataclasses import dataclass

import numpy as np

from src.profile import HouseholdProfile
from src.tariff import SLOTS_PER_DAY, Tariff, window_by_slot

DAYS = 365
INTERVALS = DAYS * SLOTS_PER_DAY  # 17,520


@dataclass(frozen=True)
class Year:
    """One year of half-hours, from 00:00 on the first day."""

    load: np.ndarray         # kWh of general consumption; controlled load excluded
    solar: np.ndarray        # kWh generated
    import_rate: np.ndarray  # $/kWh
    feed_in: float           # $/kWh exported; 0 where the tariff has no feed-in rate


def household_year(profile: HouseholdProfile, tariff: Tariff,
                   daylight_hours: tuple[float, float]) -> Year:
    windows = window_by_slot(tariff)
    slots_in = Counter(window.name for window in windows)
    day = [profile.annual_kwh_by_window[window.name] / DAYS / slots_in[window.name]
           for window in windows]
    load = np.tile(day, DAYS)
    solar = _solar(profile, load, daylight_hours) if profile.has_solar else np.zeros(INTERVALS)
    return Year(
        load=load,
        solar=solar,
        import_rate=np.tile([window.rate_aud_per_kwh for window in windows], DAYS),
        feed_in=tariff.feed_in_tariff_aud_per_kwh or 0.0,
    )


def _solar(profile: HouseholdProfile, load: np.ndarray,
           daylight_hours: tuple[float, float]) -> np.ndarray:
    """Generation scaled so that, without a battery, the year's export matches the bill."""
    target = profile.annual_solar_export_kwh
    if target is None:
        raise ValueError(f"{profile.name}: has solar, but no annual export to scale it to")
    shape = np.tile(_daylight_curve(daylight_hours), DAYS)  # sums to 1 each day
    # Export only grows with generation, so bisect on the daily total.
    low, high = 0.0, (target + load.sum()) / DAYS
    for _ in range(60):
        daily = (low + high) / 2
        if np.maximum(daily * shape - load, 0.0).sum() < target:
            low = daily
        else:
            high = daily
    return high * shape


def _daylight_curve(daylight_hours: tuple[float, float]) -> np.ndarray:
    """A half-sine from sunrise to sunset, one value per half-hour, summing to 1."""
    sunrise, sunset = daylight_hours
    midpoints = (np.arange(SLOTS_PER_DAY) + 0.5) / 2  # hours
    curve = np.sin(np.pi * (midpoints - sunrise) / (sunset - sunrise))
    curve[(midpoints <= sunrise) | (midpoints >= sunset)] = 0.0
    return curve / curve.sum()
