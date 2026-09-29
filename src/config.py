"""Loads config/: market values, never household values (hard rules 8 and 10).

Values in config/ are time-sensitive. batteries.yaml and incentives.yaml each
carry a verified flag, and callers should warn when it is not true. Ranges
recorded alongside point values are kept: they feed the assumptions at stage 4.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.tariff import SLOTS_PER_DAY, Tariff, clock, parse_tariff

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

Range = tuple[float, float]  # (low, high), inclusive


@dataclass(frozen=True)
class CostModel:
    """Installed price before rebate = fixed_aud + variable_aud_per_kwh × kWh."""

    fixed_aud: float
    variable_aud_per_kwh: float
    valid_range_kwh: Range  # the fit holds only for sizes in this range


@dataclass(frozen=True)
class Batteries:
    """config/batteries.yaml."""

    cost_model: CostModel
    round_trip_efficiency: float
    round_trip_efficiency_range: Range
    default_size_kwh: float
    power_kw: float  # continuous charge and discharge
    power_kw_range: Range
    marginal_throughput_cost_aud_per_kwh: float  # wear from delivering one more kWh
    marginal_throughput_cost_range: Range
    warranty_years: float
    verified: bool
    sourced: date | None


@dataclass(frozen=True)
class DeemingPeriod:
    start: date
    until: date    # inclusive
    factor: float  # STCs per kWh of usable capacity, before the capacity taper


@dataclass(frozen=True)
class Incentives:
    """config/incentives.yaml: the federal Cheaper Home Batteries Program."""

    stc_price_aud: float
    stc_price_range_aud: Range
    deeming_schedule: tuple[DeemingPeriod, ...]
    capacity_taper: tuple[tuple[float, float], ...]  # (up_to_kwh, share), ascending
    verified: bool
    sourced: date | None

    def deeming_period_on(self, day: date) -> DeemingPeriod:
        """The period whose factor applies to an installation on this date.

        Refuses a date outside the schedule rather than extrapolating a factor
        nobody has published.
        """
        for period in self.deeming_schedule:
            if period.start <= day <= period.until:
                return period
        raise ValueError(
            f"no deeming factor in config/incentives.yaml covers {day}: the schedule runs "
            f"from {self.deeming_schedule[0].start} to {self.deeming_schedule[-1].until}"
        )


@dataclass(frozen=True)
class ShapeBlock:
    """A block of hours and its share of the day's use, spread evenly within it."""

    hours: tuple[float, float]  # [start, end)
    share: float
    share_range: Range | None


@dataclass(frozen=True)
class Occupancy:
    weekday: str  # names of day shapes
    weekend: str


@dataclass(frozen=True)
class IndicativeSolar:
    """The coarse solar comparison behind solar_first. Not a PV model."""

    kw: float
    kw_range: Range
    self_consumption: float  # share of generation used on site
    self_consumption_range: Range
    installed_aud_per_kw: float
    installed_aud_per_kw_range: Range
    payback_ratio: float     # solar_first outranks when solar's payback is at most this share
    payback_ratio_range: Range


@dataclass(frozen=True)
class RevisitRanges:
    """The plausible range each revisit_if sweep searches."""

    battery_cost_share: Range  # of today's installed price
    feed_in_aud_per_kwh: Range
    rate_spread_change_aud_per_kwh: Range
    added_solar_kw: Range


@dataclass(frozen=True)
class Assumptions:
    """config/assumptions.yaml: modelling choices, estimated rather than measured."""

    dispatch_horizon_hours: float
    dispatch_horizon_hours_range: Range
    day_shapes: dict[str, tuple[ShapeBlock, ...]]
    occupancy: dict[str, Occupancy]               # form answer -> shapes; "not_stated" if none
    aircon_hours: dict[str, tuple[float, float]]  # form answer -> [start, end)
    solar_latitude_deg: float
    solar_noon_hour: float
    solar_june_to_december_ratio: float
    solar_june_to_december_ratio_range: Range
    cloudy_day_share: float
    cloudy_day_share_range: Range
    cloudy_day_output: float
    cloudy_day_output_range: Range
    solar_yield_kwh_per_kw: float
    solar_yield_kwh_per_kw_range: Range
    new_solar_feed_in_aud_per_kwh: float
    new_solar_feed_in_aud_per_kwh_range: Range
    indicative_solar: IndicativeSolar
    revisit_ranges: RevisitRanges

    @property
    def dispatch_horizon_intervals(self) -> int:
        return round(self.dispatch_horizon_hours * 2)  # half-hours


@dataclass(frozen=True)
class Config:
    tariffs: dict[str, Tariff]
    batteries: Batteries
    incentives: Incentives
    assumptions: Assumptions


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    tariffs = _read(config_dir / "tariffs.yaml")
    return Config(
        tariffs={tariff_id: parse_tariff(tariff_id, entry) for tariff_id, entry in tariffs.items()},
        batteries=parse_batteries(_read(config_dir / "batteries.yaml")),
        incentives=parse_incentives(_read(config_dir / "incentives.yaml")),
        assumptions=parse_assumptions(_read(config_dir / "assumptions.yaml")),
    )


def parse_batteries(raw: dict) -> Batteries:
    cost = raw["cost_model"]
    batteries = Batteries(
        cost_model=CostModel(
            fixed_aud=float(cost["fixed_aud"]),
            variable_aud_per_kwh=float(cost["variable_aud_per_kwh"]),
            valid_range_kwh=_range("cost_model.valid_range_kwh", cost["valid_range_kwh"]),
        ),
        round_trip_efficiency=float(raw["round_trip_efficiency"]),
        round_trip_efficiency_range=_range(
            "round_trip_efficiency_range", raw["round_trip_efficiency_range"]
        ),
        default_size_kwh=float(raw["default_size_kwh"]),
        power_kw=float(raw["power_kw"]),
        power_kw_range=_range("power_kw_range", raw["power_kw_range"]),
        marginal_throughput_cost_aud_per_kwh=float(raw["marginal_throughput_cost_aud_per_kwh"]),
        marginal_throughput_cost_range=_range(
            "marginal_throughput_cost_range", raw["marginal_throughput_cost_range"]
        ),
        warranty_years=float(raw["warranty_years"]),
        verified=raw.get("verified") is True,
        sourced=raw.get("sourced"),
    )
    _check_within("round_trip_efficiency", batteries.round_trip_efficiency,
                  "round_trip_efficiency_range", batteries.round_trip_efficiency_range)
    _check_within("power_kw", batteries.power_kw, "power_kw_range", batteries.power_kw_range)
    _check_within("marginal_throughput_cost_aud_per_kwh",
                  batteries.marginal_throughput_cost_aud_per_kwh,
                  "marginal_throughput_cost_range", batteries.marginal_throughput_cost_range)
    _check_within("default_size_kwh", batteries.default_size_kwh,
                  "cost_model.valid_range_kwh", batteries.cost_model.valid_range_kwh)
    return batteries


def parse_incentives(raw: dict) -> Incentives:
    schedule = tuple(
        DeemingPeriod(start=period["from"], until=period["until"], factor=float(period["factor"]))
        for period in raw["deeming_factor_schedule"]
    )
    if not schedule:
        raise ValueError("deeming_factor_schedule: no periods given")
    for period in schedule:
        if period.start > period.until:
            raise ValueError(f"deeming_factor_schedule: {period.start} is after {period.until}")
    # Each period starts the day after the previous one ends, so every date in
    # the schedule has exactly one factor.
    for earlier, later in zip(schedule, schedule[1:]):
        if later.start != earlier.until + timedelta(days=1):
            raise ValueError(
                f"deeming_factor_schedule: the period from {later.start} should start the day "
                f"after the previous one ends ({earlier.until}), with no gap or overlap"
            )

    taper = tuple(
        (float(band["up_to_kwh"]), float(band["share"])) for band in raw["capacity_taper"]
    )
    limits = [up_to_kwh for up_to_kwh, _ in taper]
    if limits != sorted(set(limits)):
        raise ValueError("capacity_taper: bands must be in ascending order of up_to_kwh")
    if not all(0 <= share <= 1 for _, share in taper):
        raise ValueError("capacity_taper: each share must be between 0 and 1")

    incentives = Incentives(
        stc_price_aud=float(raw["stc_price_aud"]),
        stc_price_range_aud=_range("stc_price_range_aud", raw["stc_price_range_aud"]),
        deeming_schedule=schedule,
        capacity_taper=taper,
        verified=raw.get("verified") is True,
        sourced=raw.get("sourced"),
    )
    _check_within("stc_price_aud", incentives.stc_price_aud,
                  "stc_price_range_aud", incentives.stc_price_range_aud)
    return incentives


def parse_assumptions(raw: dict) -> Assumptions:
    horizon = float(raw["dispatch_horizon_hours"])
    if horizon <= 0 or not float(horizon * 2).is_integer():
        raise ValueError("dispatch_horizon_hours must be positive, in whole half-hours")

    day_shapes = {name: _parse_day_shape(name, blocks) for name, blocks in raw["day_shapes"].items()}
    occupancy = {answer: Occupancy(weekday=entry["weekday"], weekend=entry["weekend"])
                 for answer, entry in raw["occupancy"].items()}
    if "not_stated" not in occupancy:
        raise ValueError("occupancy: needs a not_stated entry, for forms that leave it blank")
    for answer, shapes in occupancy.items():
        for shape in (shapes.weekday, shapes.weekend):
            if shape not in day_shapes:
                raise ValueError(f"occupancy {answer!r}: no day shape called {shape!r}")

    assumptions = Assumptions(
        dispatch_horizon_hours=horizon,
        dispatch_horizon_hours_range=_range(
            "dispatch_horizon_hours_range", raw["dispatch_horizon_hours_range"]),
        day_shapes=day_shapes,
        occupancy=occupancy,
        aircon_hours={answer: _hours(f"aircon_hours {answer!r}", pair)
                      for answer, pair in raw["aircon_hours"].items()},
        solar_latitude_deg=float(raw["solar_latitude_deg"]),
        solar_noon_hour=float(raw["solar_noon_hour"]),
        solar_june_to_december_ratio=float(raw["solar_june_to_december_ratio"]),
        solar_june_to_december_ratio_range=_range(
            "solar_june_to_december_ratio_range", raw["solar_june_to_december_ratio_range"]),
        cloudy_day_share=float(raw["cloudy_day_share"]),
        cloudy_day_share_range=_range("cloudy_day_share_range", raw["cloudy_day_share_range"]),
        cloudy_day_output=float(raw["cloudy_day_output"]),
        cloudy_day_output_range=_range("cloudy_day_output_range", raw["cloudy_day_output_range"]),
        solar_yield_kwh_per_kw=float(raw["solar_yield_kwh_per_kw"]),
        solar_yield_kwh_per_kw_range=_range(
            "solar_yield_kwh_per_kw_range", raw["solar_yield_kwh_per_kw_range"]),
        new_solar_feed_in_aud_per_kwh=float(raw["new_solar_feed_in_aud_per_kwh"]),
        new_solar_feed_in_aud_per_kwh_range=_range(
            "new_solar_feed_in_aud_per_kwh_range", raw["new_solar_feed_in_aud_per_kwh_range"]),
        indicative_solar=_parse_indicative_solar(raw["indicative_solar"]),
        revisit_ranges=RevisitRanges(**{
            key: _range(f"revisit_ranges {key}", raw["revisit_ranges"][key])
            for key in ("battery_cost_share", "feed_in_aud_per_kwh",
                        "rate_spread_change_aud_per_kwh", "added_solar_kw")
        }),
    )
    _check_within("dispatch_horizon_hours", assumptions.dispatch_horizon_hours,
                  "dispatch_horizon_hours_range", assumptions.dispatch_horizon_hours_range)
    _check_within("solar_june_to_december_ratio", assumptions.solar_june_to_december_ratio,
                  "solar_june_to_december_ratio_range",
                  assumptions.solar_june_to_december_ratio_range)
    _check_within("cloudy_day_share", assumptions.cloudy_day_share,
                  "cloudy_day_share_range", assumptions.cloudy_day_share_range)
    _check_within("cloudy_day_output", assumptions.cloudy_day_output,
                  "cloudy_day_output_range", assumptions.cloudy_day_output_range)
    _check_within("solar_yield_kwh_per_kw", assumptions.solar_yield_kwh_per_kw,
                  "solar_yield_kwh_per_kw_range", assumptions.solar_yield_kwh_per_kw_range)
    _check_within("new_solar_feed_in_aud_per_kwh", assumptions.new_solar_feed_in_aud_per_kwh,
                  "new_solar_feed_in_aud_per_kwh_range",
                  assumptions.new_solar_feed_in_aud_per_kwh_range)
    if not abs(assumptions.solar_latitude_deg) < 66:
        raise ValueError("solar_latitude_deg must lie outside the polar circles")
    if not 0 < assumptions.solar_june_to_december_ratio:
        raise ValueError("solar_june_to_december_ratio must be positive")
    if not (0 <= assumptions.cloudy_day_share < 1 and 0 <= assumptions.cloudy_day_output <= 1):
        raise ValueError("cloudy_day_share must lie in [0, 1) and cloudy_day_output in [0, 1]")
    return assumptions


def _parse_indicative_solar(raw: dict) -> IndicativeSolar:
    where = "indicative_solar"
    solar = IndicativeSolar(
        kw=float(raw["kw"]),
        kw_range=_range(f"{where} kw_range", raw["kw_range"]),
        self_consumption=float(raw["self_consumption"]),
        self_consumption_range=_range(f"{where} self_consumption_range",
                                      raw["self_consumption_range"]),
        installed_aud_per_kw=float(raw["installed_aud_per_kw"]),
        installed_aud_per_kw_range=_range(f"{where} installed_aud_per_kw_range",
                                          raw["installed_aud_per_kw_range"]),
        payback_ratio=float(raw["payback_ratio"]),
        payback_ratio_range=_range(f"{where} payback_ratio_range", raw["payback_ratio_range"]),
    )
    _check_within(f"{where} kw", solar.kw, "kw_range", solar.kw_range)
    _check_within(f"{where} self_consumption", solar.self_consumption,
                  "self_consumption_range", solar.self_consumption_range)
    _check_within(f"{where} installed_aud_per_kw", solar.installed_aud_per_kw,
                  "installed_aud_per_kw_range", solar.installed_aud_per_kw_range)
    _check_within(f"{where} payback_ratio", solar.payback_ratio,
                  "payback_ratio_range", solar.payback_ratio_range)
    return solar


def _parse_day_shape(name: str, entries: list) -> tuple[ShapeBlock, ...]:
    where = f"day_shapes {name!r}"
    blocks = tuple(
        ShapeBlock(
            hours=_hours(where, entry["hours"]),
            share=float(entry["share"]),
            share_range=(_range(f"{where} share_range", entry["share_range"])
                         if "share_range" in entry else None),
        )
        for entry in entries
    )
    # Every half-hour in exactly one block, and the blocks make up the whole day.
    for slot in range(SLOTS_PER_DAY):
        hour = slot / 2
        if sum(1 for block in blocks if block.hours[0] <= hour < block.hours[1]) != 1:
            raise ValueError(f"{where}: the half-hour from {clock(hour)} must be in exactly one block")
    if abs(sum(block.share for block in blocks) - 1) > 1e-9 or min(b.share for b in blocks) < 0:
        raise ValueError(f"{where}: shares must be non-negative and add up to 1")
    for block in blocks:
        if block.share_range is not None:
            _check_within(f"{where} share", block.share, "share_range", block.share_range)
    return blocks


def _hours(where: str, pair: list) -> tuple[float, float]:
    if len(pair) != 2:
        raise ValueError(f"{where}: hours must be a [start, end] pair, got {pair!r}")
    start, end = float(pair[0]), float(pair[1])
    if not (0 <= start < end <= 24 and (start * 2).is_integer() and (end * 2).is_integer()):
        raise ValueError(f"{where}: hours {pair!r} must satisfy 0 <= start < end <= 24, "
                         "on the half hour")
    return (start, end)


def _range(where: str, pair: list) -> Range:
    if len(pair) != 2 or pair[0] > pair[1]:
        raise ValueError(f"{where}: expected [low, high], got {pair!r}")
    return (float(pair[0]), float(pair[1]))


def _check_within(name: str, value: float, range_name: str, bounds: Range) -> None:
    low, high = bounds
    if not low <= value <= high:
        raise ValueError(f"{name} = {value:g} lies outside {range_name} [{low:g}, {high:g}]")


def _read(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))
