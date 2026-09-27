"""What a tariff is: a list of charge components, not a shape.

A flat tariff is an energy-window list of length one, not a special case.
Every component type has a slot here from the start, including the ones this
version does not price: a charge type with no slot cannot be detected, and an
undetected charge is silently dropped from someone's bill (hard rule 9).

    component         key in config/tariffs.yaml    this version
    energy_windows    windows                       modelled
    daily_charge      daily_charge_aud              modelled (a battery does not change it)
    controlled_load   controlled_load               modelled; not addressable by a battery
    feed_in           feed_in_tariff_aud_per_kwh    modelled as a single rate
                      feed_in_windows               recorded and declared (time-varying export)
    demand            demand                        recorded and declared
    block             block                         recorded and declared
    seasonal          seasonal                      recorded and declared

"Recorded and declared" means the component is parsed and kept, and if a
household's tariff has one, unmodelled_components() names it so the result
can say plainly that this version does not price it.
"""

from dataclasses import dataclass
from datetime import date

# [start, end) pairs in hours of the day, e.g. ((21, 24), (0, 9)).
Hours = tuple[tuple[float, float], ...]

METADATA_KEYS = {
    "retailer", "network", "plan", "gst_included",
    "effective_from", "effective_until", "observed_until",
}
COMPONENT_KEYS = {
    "windows", "daily_charge_aud", "controlled_load",
    "feed_in_tariff_aud_per_kwh", "feed_in_windows",
    "demand", "block", "seasonal",
}

# Component slots this version records but does not price. When one is
# implemented, remove it here -- required test 6 then expects household_c to
# stop triggering the declaration.
UNMODELLED = ("demand", "block", "seasonal", "feed_in_windows")


@dataclass(frozen=True)
class EnergyWindow:
    name: str
    hours: Hours
    rate_aud_per_kwh: float


@dataclass(frozen=True)
class ControlledLoad:
    """A separately metered circuit, e.g. hot water. A battery cannot serve it."""

    rate_aud_per_kwh: float
    availability: str | None


@dataclass(frozen=True)
class DemandCharge:
    """$/kW on the highest half-hourly demand within measured_over, each period."""

    charge_aud_per_kw_month: float
    measured_over: Hours
    reset: str


@dataclass(frozen=True)
class Tariff:
    id: str
    plan: str | None
    retailer: str | None
    network: str | None
    effective_from: date | None
    effective_until: date | None
    observed_until: date | None

    energy_windows: tuple[EnergyWindow, ...]
    daily_charge_aud: float
    controlled_load: ControlledLoad | None
    feed_in_tariff_aud_per_kwh: float | None

    # Recorded but not priced by this version -- see unmodelled_components().
    # block, seasonal and feed_in_windows have no example in a real bill yet,
    # so they are kept exactly as written rather than given a guessed structure.
    demand: DemandCharge | None
    block: dict | None
    seasonal: dict | None
    feed_in_windows: dict | None


def unmodelled_components(tariff: Tariff) -> dict[str, str]:
    """{component: description} for each unpriced component this tariff has.

    Every entry must be declared to the household by name, and any result for
    this tariff labelled a partial picture (hard rule 9).
    """
    found = {}
    for component in UNMODELLED:
        if getattr(tariff, component) is not None:
            found[component] = _describe_unmodelled(tariff, component)
    return found


def _describe_unmodelled(tariff: Tariff, component: str) -> str:
    if component == "demand":
        demand = tariff.demand
        return (
            f"${demand.charge_aud_per_kw_month:.2f}/kW/month on the highest half-hourly "
            f"demand in {describe_hours(demand.measured_over)}, reset {demand.reset}"
        )
    descriptions = {
        "block": "a different rate once consumption passes a threshold",
        "seasonal": "rates that differ by month",
        "feed_in_windows": "an export rate that varies by time of day",
    }
    return descriptions[component]


def clock(hour: float) -> str:
    """9 -> '09:00', 16.5 -> '16:30', 24 -> '24:00'."""
    return f"{int(hour):02d}:{30 if hour % 1 else 0:02d}"


def describe_hours(hours: Hours) -> str:
    """((21, 24), (0, 9)) -> '21:00–24:00, 00:00–09:00'."""
    return ", ".join(f"{clock(start)}–{clock(end)}" for start, end in hours)


# ------------------------------------------------------------------ parsing

def parse_tariff(tariff_id: str, entry: dict) -> Tariff:
    """Build a Tariff from one entry of config/tariffs.yaml.

    Refuses anything it does not recognise rather than skipping it: an
    unparsed key could be a charge, and a charge is never dropped silently.
    """
    where = f"tariff {tariff_id!r}"
    _check_keys(where, entry, METADATA_KEYS | COMPONENT_KEYS)
    if entry.get("gst_included") is not True:
        raise ValueError(
            f"{where}: rates must include GST, as printed on the bill, "
            "and say so with gst_included: true"
        )

    windows = tuple(
        _parse_window(f"{where}, window {name!r}", name, window)
        for name, window in entry["windows"].items()
    )
    _check_windows_cover_the_day(where, windows)

    controlled_load = entry.get("controlled_load")
    feed_in = entry.get("feed_in_tariff_aud_per_kwh")
    demand = entry.get("demand")
    return Tariff(
        id=tariff_id,
        plan=entry.get("plan"),
        retailer=entry.get("retailer"),
        network=entry.get("network"),
        effective_from=entry.get("effective_from"),
        effective_until=entry.get("effective_until"),
        observed_until=entry.get("observed_until"),
        energy_windows=windows,
        daily_charge_aud=_number(f"{where}, daily_charge_aud", entry["daily_charge_aud"]),
        controlled_load=(
            None if controlled_load is None
            else _parse_controlled_load(f"{where}, controlled_load", controlled_load)
        ),
        feed_in_tariff_aud_per_kwh=(
            None if feed_in is None
            else _number(f"{where}, feed_in_tariff_aud_per_kwh", feed_in)
        ),
        demand=None if demand is None else _parse_demand(f"{where}, demand", demand),
        block=entry.get("block"),
        seasonal=entry.get("seasonal"),
        feed_in_windows=entry.get("feed_in_windows"),
    )


def _parse_window(where: str, name: str, window: dict) -> EnergyWindow:
    _check_keys(where, window, {"hours", "rate_aud_per_kwh"})
    return EnergyWindow(
        name=name,
        hours=_parse_hours(where, window["hours"]),
        rate_aud_per_kwh=_number(f"{where}, rate_aud_per_kwh", window["rate_aud_per_kwh"]),
    )


def _parse_controlled_load(where: str, entry: dict) -> ControlledLoad:
    _check_keys(where, entry, {"rate_aud_per_kwh", "availability"})
    return ControlledLoad(
        rate_aud_per_kwh=_number(f"{where}, rate_aud_per_kwh", entry["rate_aud_per_kwh"]),
        availability=entry.get("availability"),
    )


def _parse_demand(where: str, entry: dict) -> DemandCharge:
    _check_keys(where, entry, {"charge_aud_per_kw_month", "measured_over", "reset"})
    return DemandCharge(
        charge_aud_per_kw_month=_number(
            f"{where}, charge_aud_per_kw_month", entry["charge_aud_per_kw_month"]
        ),
        measured_over=_parse_hours(where, entry["measured_over"]),
        reset=entry["reset"],
    )


def _parse_hours(where: str, hours: list) -> Hours:
    """[[21, 24], [0, 9]] -> ((21, 24), (0, 9)), each within the day, on the half hour."""
    ranges = []
    for pair in hours:
        if len(pair) != 2:
            raise ValueError(f"{where}: hours must be [start, end] pairs, got {pair!r}")
        start, end = pair
        on_half_hour = float(start * 2).is_integer() and float(end * 2).is_integer()
        if not (0 <= start < end <= 24 and on_half_hour):
            raise ValueError(
                f"{where}: hours {pair!r} must satisfy 0 <= start < end <= 24, on the half hour"
            )
        ranges.append((start, end))
    return tuple(ranges)


def _check_windows_cover_the_day(where: str, windows: tuple[EnergyWindow, ...]) -> None:
    """Every half-hour must fall in exactly one energy window. Otherwise some
    consumption would be priced twice, or not at all."""
    for slot in range(48):
        t = slot / 2
        containing = [w.name for w in windows for start, end in w.hours if start <= t < end]
        if len(containing) != 1:
            raise ValueError(
                f"{where}: the half-hour from {clock(t)} is in {len(containing)} energy "
                f"windows {containing}; every half-hour must be in exactly one"
            )


def _check_keys(where: str, entry: dict, allowed: set[str]) -> None:
    unknown = set(entry) - allowed
    if unknown:
        raise ValueError(
            f"{where}: unrecognised key(s) {sorted(unknown)}. If this is a charge, give it "
            "a component slot in src/tariff.py; a charge is never dropped silently "
            "(hard rule 9)."
        )


def _number(where: str, value) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError(f"{where}: expected a non-negative number, got {value!r}")
    return float(value)
