"""A household profile: what the bill says, and what the form asks.

A fixture in fixtures/ is a saved profile. Stage 5 builds the bill half from
uploaded bills instead, and stage 7 the form half from the web form; both
must produce this same structure. The bill half is a run of consecutive
billing periods with consumption by tariff window, because that is what bills
supply; the annual figures are sums over them.
"""

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.tariff import Tariff

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# Bill quantities are printed to two decimals and stated annual figures are
# rounded to whole kWh, so sums may miss a stated figure by a rounding error,
# and never by more.
RECONCILE_TOLERANCE_KWH = 1.0


@dataclass(frozen=True)
class BillingPeriod:
    start: date
    days: int
    kwh_by_window: dict[str, float]    # general consumption in each energy window
    controlled_load_kwh: float | None  # separate circuit; None if the household has none

    @property
    def end(self) -> date:
        """The first day after the period."""
        return self.start + timedelta(days=self.days)

    @property
    def daily_kwh(self) -> float:
        """General consumption per day, controlled load excluded."""
        return sum(self.kwh_by_window.values()) / self.days


@dataclass(frozen=True)
class HouseholdProfile:
    name: str                                   # fixture file name, e.g. "household_b"
    source: str                                 # real bills or synthetic; shown with every result
    tariff_ref: str                             # key into config/tariffs.yaml
    billing_periods: tuple[BillingPeriod, ...]  # consecutive, oldest first
    has_solar: bool
    solar_kw: float | None
    annual_solar_export_kwh: float | None
    form: dict                                  # form answers as given

    @property
    def annual_kwh_by_window(self) -> dict[str, float]:
        totals = {}
        for period in self.billing_periods:
            for window, kwh in period.kwh_by_window.items():
                totals[window] = totals.get(window, 0.0) + kwh
        return totals

    @property
    def annual_controlled_load_kwh(self) -> float | None:
        if self.billing_periods[0].controlled_load_kwh is None:
            return None
        return sum(period.controlled_load_kwh for period in self.billing_periods)

    @property
    def annual_kwh_total(self) -> float:
        return sum(self.annual_kwh_by_window.values()) + (self.annual_controlled_load_kwh or 0.0)


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> list[HouseholdProfile]:
    return [load_fixture(path) for path in sorted(fixtures_dir.glob("*.yaml"))]


def load_fixture(path: Path) -> HouseholdProfile:
    """Read one fixture, refusing it if its billing periods do not add up to its
    stated annual figures."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bill = raw["from_bill"]
    name = path.stem

    periods = tuple(_parse_period(name, entry) for entry in bill["billing_periods"])
    _check_periods(name, periods)
    if bill["has_solar"] and not bill.get("solar_kw"):
        raise ValueError(f"{name}: has_solar is true but solar_kw is not given")

    profile = HouseholdProfile(
        name=name,
        source=raw["meta"]["source"],
        tariff_ref=bill["tariff_ref"],
        billing_periods=periods,
        has_solar=bool(bill["has_solar"]),
        solar_kw=bill.get("solar_kw"),
        annual_solar_export_kwh=bill.get("annual_solar_export_kwh"),
        form=raw.get("from_form") or {},
    )
    if bool(bill.get("has_controlled_load")) != (profile.annual_controlled_load_kwh is not None):
        raise ValueError(f"{name}: has_controlled_load does not match the billing periods")

    # The periods must add up to the stated figures, window by window. This is
    # the check that catches a bill read with one of its rate blocks missing
    # (required test 4).
    summed = dict(profile.annual_kwh_by_window)
    if profile.annual_controlled_load_kwh is not None:
        summed["controlled_load"] = profile.annual_controlled_load_kwh
    stated = {window: float(kwh) for window, kwh in bill["annual_kwh_by_window"].items()}
    for window in sorted(stated.keys() | summed.keys()):
        if abs(summed.get(window, 0.0) - stated.get(window, 0.0)) > RECONCILE_TOLERANCE_KWH:
            raise ValueError(
                f"{name}: billing periods add up to {summed.get(window, 0.0):,.1f} kWh of "
                f"{window}, but the stated annual figure is {stated.get(window, 0.0):,.1f} kWh"
            )
    printed_total = float(bill["annual_kwh_total"])
    if abs(profile.annual_kwh_total - printed_total) > RECONCILE_TOLERANCE_KWH:
        raise ValueError(
            f"{name}: consumption adds up to {profile.annual_kwh_total:,.1f} kWh, "
            f"but the printed total is {printed_total:,.1f} kWh"
        )
    return profile


def _parse_period(name: str, entry: dict) -> BillingPeriod:
    kwh = {key: float(value) for key, value in entry.items() if key not in ("start", "days")}
    # The bill lists controlled load alongside the windows, but it is a separate
    # circuit rather than a time of day, and a battery cannot serve it.
    controlled_load = kwh.pop("controlled_load", None)
    if not kwh or min(kwh.values()) < 0 or (controlled_load or 0.0) < 0:
        raise ValueError(f"{name}: the billing period from {entry['start']} needs "
                         "non-negative consumption in at least one window")
    return BillingPeriod(start=entry["start"], days=int(entry["days"]),
                         kwh_by_window=kwh, controlled_load_kwh=controlled_load)


def _check_periods(name: str, periods: tuple[BillingPeriod, ...]) -> None:
    if not periods:
        raise ValueError(f"{name}: no billing periods")
    first = periods[0]
    for period in periods:
        if period.days <= 0:
            raise ValueError(f"{name}: the billing period from {period.start} has no days")
        if set(period.kwh_by_window) != set(first.kwh_by_window):
            raise ValueError(f"{name}: every billing period must split use into the same windows")
        if (period.controlled_load_kwh is None) != (first.controlled_load_kwh is None):
            raise ValueError(f"{name}: controlled load must appear in every billing period or none")
    for earlier, later in zip(periods, periods[1:]):
        if later.start != earlier.end:
            raise ValueError(f"{name}: the billing period from {later.start} should start on "
                             f"{earlier.end}; billing periods must be consecutive")


def check_matches_tariff(profile: HouseholdProfile, tariff: Tariff) -> None:
    """Raise unless the tariff can price every kWh this profile says was used."""
    tariff_windows = {window.name for window in tariff.energy_windows}
    if set(profile.annual_kwh_by_window) != tariff_windows:
        raise ValueError(
            f"{profile.name}: consumption is split into windows "
            f"{sorted(profile.annual_kwh_by_window)}, but tariff {tariff.id!r} has "
            f"windows {sorted(tariff_windows)}"
        )
    if profile.annual_controlled_load_kwh is not None and tariff.controlled_load is None:
        raise ValueError(
            f"{profile.name}: uses controlled load, but tariff {tariff.id!r} has no rate for it"
        )
    has_export_rate = (
        tariff.feed_in_tariff_aud_per_kwh is not None or tariff.feed_in_windows is not None
    )
    if profile.has_solar and not has_export_rate:
        raise ValueError(f"{profile.name}: has solar, but tariff {tariff.id!r} has no feed-in rate")
