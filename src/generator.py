"""One representative year, half-hour by half-hour: the input to dispatch.

The year's shape has two halves, and they come from different evidence.

Across the year -- from the billing periods. Each day takes the daily average
of the billing period it falls in, window by window, so the seasonality is the
household's own, never an assumed profile. The year runs 365 days from the
first period's start; a day after the last period repeats that period.

Within the day -- from the tariff's window split where the bill has one. A
time-of-use bill says how much was used in each window, so only the shape
inside each window is generated. A flat bill has one window covering the whole
day, so the entire daily shape comes from the form: the occupancy answers,
which give each day a home or an away shape, and the air-conditioning answer,
which places use above the household's lowest period in that type's hours. There is no special case: the asymmetry follows
from how many windows the bill has, and Year.time_of_day_source reports it,
because a flat-tariff result carries materially more uncertainty.

Solar is a clear-sky day for each date under a seasonal envelope (southern
hemisphere: highest in December), with cloudy days, scaled so that the year's
export without a battery equals the export on the bill -- or, where the bill
shows none, to capacity × the assumed annual yield.

An EV the household does not have yet (the adding-an-EV sweep) is charging the
bills do not contain, so it is added on top of them, overnight.

Not modelled: the running hours of a pool pump on the main circuit, the
heating hours of electric hot water on the main circuit, and the charging
hours of an EV already owned. Their use is in the bill's totals and follows
the day's shape; the recommendation declares each.

The assumptions are in config/assumptions.yaml.
"""

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np

from src.config import Assumptions, ShapeBlock
from src.profile import BillingPeriod, HouseholdProfile
from src.tariff import SLOTS_PER_DAY, Tariff, window_by_slot

DAYS = 365
INTERVALS = DAYS * SLOTS_PER_DAY  # 17,520
DECEMBER_SOLSTICE = 355           # day of the year


@dataclass(frozen=True)
class Year:
    """One year of half-hours, from 00:00 on `start`."""

    start: date
    load: np.ndarray         # kWh of general consumption; controlled load excluded
    solar: np.ndarray        # kWh generated
    import_rate: np.ndarray  # $/kWh
    feed_in: float           # $/kWh exported; 0 where the tariff has no feed-in rate
    # "bill": the bill splits use by window, and only the shape inside each
    # window is generated. "form": the bill gives magnitude only, and the whole
    # time-of-day shape comes from the form answers.
    time_of_day_source: str


def household_year(profile: HouseholdProfile, tariff: Tariff, assumptions: Assumptions) -> Year:
    start = profile.billing_periods[0].start
    dates = [start + timedelta(days=d) for d in range(DAYS)]
    load = _load(profile, tariff, dates, assumptions)
    solar = _solar(profile, load, dates, assumptions) if profile.has_solar else np.zeros(INTERVALS)
    return Year(
        start=start,
        load=load,
        solar=solar,
        import_rate=np.tile([window.rate_aud_per_kwh for window in window_by_slot(tariff)], DAYS),
        feed_in=tariff.feed_in_tariff_aud_per_kwh or 0.0,
        time_of_day_source="bill" if len(tariff.energy_windows) > 1 else "form",
    )


# ------------------------------------------------------------------------ use

def _load(profile: HouseholdProfile, tariff: Tariff, dates: list[date],
          assumptions: Assumptions) -> np.ndarray:
    by_slot = window_by_slot(tariff)
    slots_in = {window.name: np.array([w.name == window.name for w in by_slot])
                for window in tariff.energy_windows}
    shape_on = _occupancy(profile, assumptions)
    aircon = _aircon(profile, assumptions)
    lowest = min(period.daily_kwh for period in profile.billing_periods)

    days = []
    for day in dates:
        period = _period_on(profile.billing_periods, day)
        shape = shape_on(day)
        if aircon is not None and period.daily_kwh > 0:
            # Use above the household's lowest period is air conditioning.
            excess = (period.daily_kwh - lowest) / period.daily_kwh
            shape = (1 - excess) * shape + excess * aircon
        load = np.zeros(SLOTS_PER_DAY)
        for window, slots in slots_in.items():
            # The bill fixes how much falls in each window; the shape only spreads it.
            load[slots] = period.kwh_by_window[window] / period.days * shape[slots] / shape[slots].sum()
        days.append(load)
    return np.concatenate(days) + _added_ev(profile, assumptions)


def _occupancy(profile: HouseholdProfile, assumptions: Assumptions):
    """The day shape for a date. Weekends, a weekday someone is home, and each
    work-from-home day (counted from Monday) take the home shape; other weekdays
    the away shape. Unanswered, every day takes the not_stated shape."""
    shapes = {kind: _spread_blocks(assumptions.day_shapes[name])
              for kind, name in vars(assumptions.occupancy).items()}
    at_home = profile.form.get("home_during_the_day")
    work_from_home = profile.form.get("work_from_home_days") or 0

    def shape_on(day: date) -> np.ndarray:
        if at_home is None:
            return shapes["not_stated"]
        weekday = day.weekday()
        return shapes["home"] if weekday >= 5 or at_home or weekday < work_from_home \
            else shapes["away"]

    return shape_on


def _aircon(profile: HouseholdProfile, assumptions: Assumptions) -> np.ndarray | None:
    """Half-hourly shares of air-conditioning use, or None where the form reports none."""
    kind = profile.form.get("aircon")
    if kind in (None, "none"):
        return None
    if kind not in assumptions.aircon_hours:
        raise ValueError(f"{profile.name}: config/assumptions.yaml has no air-conditioning "
                         f"hours for {kind!r}")
    start, end = assumptions.aircon_hours[kind]
    return _spread_blocks((ShapeBlock(hours=(start, end), share=1.0, share_range=None),))


def _added_ev(profile: HouseholdProfile, assumptions: Assumptions) -> np.ndarray:
    """EV charging the bills do not contain, spread evenly over the charging hours
    of every day."""
    if not profile.added_ev_kwh_per_year:
        return np.zeros(INTERVALS)
    start, end = (int(hour * 2) for hour in assumptions.ev_charging_hours)
    charging = np.zeros(SLOTS_PER_DAY, dtype=bool)
    if start < end:
        charging[start:end] = True
    else:  # past midnight
        charging[start:] = charging[:end] = True
    day = np.where(charging, profile.added_ev_kwh_per_year / DAYS / charging.sum(), 0.0)
    return np.tile(day, DAYS)


def _spread_blocks(blocks: tuple[ShapeBlock, ...]) -> np.ndarray:
    """Blocks of hours -> 48 half-hourly shares, even within each block."""
    shape = np.zeros(SLOTS_PER_DAY)
    for block in blocks:
        first, last = int(block.hours[0] * 2), int(block.hours[1] * 2)
        shape[first:last] = block.share / (last - first)
    return shape


def _period_on(periods: tuple[BillingPeriod, ...], day: date) -> BillingPeriod:
    """The billing period a day falls in; a day after the last period repeats it."""
    for period in periods:
        if period.start <= day < period.end:
            return period
    return periods[-1]


# ---------------------------------------------------------------------- solar

def _solar(profile: HouseholdProfile, load: np.ndarray, dates: list[date],
           assumptions: Assumptions) -> np.ndarray:
    """Generation for the year. Solar the bill shows is scaled so that, without a
    battery, the year's export matches the bill; solar the household does not
    have yet (a revisit sweep) is scaled to capacity × the assumed annual yield."""
    # Relative output: timing from the clear-sky day, the day's amount from the
    # seasonal envelope, less on cloudy days.
    relative = np.concatenate([
        _clear_sky_day(day, assumptions) * _seasonal_envelope(day, assumptions)
        * _cloudiness(index, assumptions)
        for index, day in enumerate(dates)
    ])
    target = profile.annual_solar_export_kwh
    if target is None:
        return relative * profile.solar_kw * assumptions.solar_yield_kwh_per_kw / relative.sum()
    # Export only grows with generation, so bisect on the scale.
    low, high = 0.0, (target + load.sum()) / relative.sum()
    for _ in range(60):
        scale = (low + high) / 2
        if np.maximum(scale * relative - load, 0.0).sum() < target:
            low = scale
        else:
            high = scale
    return high * relative


def _clear_sky_day(day: date, assumptions: Assumptions) -> np.ndarray:
    """A half-sine from sunrise to sunset on this date, one value per half-hour,
    summing to 1. Day length follows from latitude and the sun's declination."""
    declination = np.radians(23.44) * np.sin(2 * np.pi * (284 + _day_of_year(day)) / 365)
    latitude = np.radians(assumptions.solar_latitude_deg)
    half_day = np.degrees(np.arccos(-np.tan(latitude) * np.tan(declination))) / 15  # hours
    sunrise = assumptions.solar_noon_hour - half_day
    sunset = assumptions.solar_noon_hour + half_day
    midpoints = (np.arange(SLOTS_PER_DAY) + 0.5) / 2  # hours
    curve = np.sin(np.pi * (midpoints - sunrise) / (sunset - sunrise))
    curve[(midpoints <= sunrise) | (midpoints >= sunset)] = 0.0
    return curve / curve.sum()


def _seasonal_envelope(day: date, assumptions: Assumptions) -> float:
    """The day's output relative to the year's mean: highest at the December
    solstice, lowest at the June one, in the configured ratio."""
    ratio = assumptions.solar_june_to_december_ratio
    amplitude = (1 - ratio) / (1 + ratio)
    return 1 + amplitude * np.cos(2 * np.pi * (_day_of_year(day) - DECEMBER_SOLSTICE) / 365)


def _cloudiness(index: int, assumptions: Assumptions) -> float:
    """1 on a clear day, cloudy_day_output on a cloudy one. Cloudy days are spread
    evenly: day i is cloudy when the running count of share × days ticks over."""
    share = assumptions.cloudy_day_share
    cloudy = int((index + 1) * share) > int(index * share)
    return assumptions.cloudy_day_output if cloudy else 1.0


def _day_of_year(day: date) -> int:
    return day.timetuple().tm_yday
