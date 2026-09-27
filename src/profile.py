"""A household profile: what the bill says, and what the form asks.

A fixture in fixtures/ is a saved profile. Stage 5 builds the bill half from
an uploaded bill instead, and stage 7 the form half from the web form; both
must produce this same structure.
"""

from dataclasses import dataclass
from pathlib import Path

import yaml

from src.tariff import Tariff

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# Fixture figures are rounded to whole kWh, so the parts may miss the printed
# total by a rounding error, and never by more.
RECONCILE_TOLERANCE_KWH = 1.0


@dataclass(frozen=True)
class HouseholdProfile:
    name: str                                  # fixture file name, e.g. "household_b"
    source: str                                # real bills or synthetic; shown with every result
    tariff_ref: str                            # key into config/tariffs.yaml
    annual_kwh_by_window: dict[str, float]     # general consumption in each energy window
    annual_controlled_load_kwh: float | None   # separate circuit; None if the household has none
    has_solar: bool
    solar_kw: float | None
    annual_solar_export_kwh: float | None
    form: dict                                 # form answers as given; not used by the engine yet

    @property
    def annual_kwh_total(self) -> float:
        return sum(self.annual_kwh_by_window.values()) + (self.annual_controlled_load_kwh or 0.0)


def load_fixtures(fixtures_dir: Path = FIXTURES_DIR) -> list[HouseholdProfile]:
    return [load_fixture(path) for path in sorted(fixtures_dir.glob("*.yaml"))]


def load_fixture(path: Path) -> HouseholdProfile:
    """Read one fixture, refusing it if its consumption does not add up."""
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    bill = raw["from_bill"]
    name = path.stem

    by_window = {window: float(kwh) for window, kwh in bill["annual_kwh_by_window"].items()}
    # The bill lists controlled load alongside the windows, but it is a
    # separate circuit rather than a time of day, and a battery cannot serve it.
    controlled_load_kwh = by_window.pop("controlled_load", None)
    if bool(bill.get("has_controlled_load")) != (controlled_load_kwh is not None):
        raise ValueError(f"{name}: has_controlled_load does not match annual_kwh_by_window")
    if bill["has_solar"] and not bill.get("solar_kw"):
        raise ValueError(f"{name}: has_solar is true but solar_kw is not given")

    profile = HouseholdProfile(
        name=name,
        source=raw["meta"]["source"],
        tariff_ref=bill["tariff_ref"],
        annual_kwh_by_window=by_window,
        annual_controlled_load_kwh=controlled_load_kwh,
        has_solar=bool(bill["has_solar"]),
        solar_kw=bill.get("solar_kw"),
        annual_solar_export_kwh=bill.get("annual_solar_export_kwh"),
        form=raw.get("from_form") or {},
    )

    # The parts must add up to the printed total. This is the check that
    # catches a bill read with one of its rate blocks missing (required test 4).
    printed_total = float(bill["annual_kwh_total"])
    if abs(profile.annual_kwh_total - printed_total) > RECONCILE_TOLERANCE_KWH:
        raise ValueError(
            f"{name}: consumption by window adds up to {profile.annual_kwh_total:,.1f} kWh, "
            f"but the printed total is {printed_total:,.1f} kWh"
        )
    return profile


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
