"""Loads config/: market values, never household values (hard rules 8 and 10).

Values in config/ are time-sensitive. batteries.yaml and incentives.yaml each
carry a verified flag, and callers should warn when it is not true. Ranges
recorded alongside point values are kept: they feed the assumptions at stage 4.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.tariff import Tariff, parse_tariff

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
    warranty_years: float
    verified: bool


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
class Config:
    tariffs: dict[str, Tariff]
    batteries: Batteries
    incentives: Incentives


def load_config(config_dir: Path = CONFIG_DIR) -> Config:
    tariffs = _read(config_dir / "tariffs.yaml")
    return Config(
        tariffs={tariff_id: parse_tariff(tariff_id, entry) for tariff_id, entry in tariffs.items()},
        batteries=parse_batteries(_read(config_dir / "batteries.yaml")),
        incentives=parse_incentives(_read(config_dir / "incentives.yaml")),
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
        warranty_years=float(raw["warranty_years"]),
        verified=raw.get("verified") is True,
    )
    _check_within("round_trip_efficiency", batteries.round_trip_efficiency,
                  "round_trip_efficiency_range", batteries.round_trip_efficiency_range)
    _check_within("power_kw", batteries.power_kw, "power_kw_range", batteries.power_kw_range)
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
    )
    _check_within("stc_price_aud", incentives.stc_price_aud,
                  "stc_price_range_aud", incentives.stc_price_range_aud)
    return incentives


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
