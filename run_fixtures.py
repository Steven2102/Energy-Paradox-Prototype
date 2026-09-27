"""Stage 1 console runner: every fixture through the headline equation.

    python run_fixtures.py [--install-date YYYY-MM-DD]

For each household in fixtures/, prints the tariff it is on, its annual
consumption by window, and the payback calculation with every term shown
separately, then a one-line-per-household summary.

The dispatch simulation does not exist yet (stage 2), so the three terms it
will produce -- kwh_shifted, r_out and r_in -- are hardcoded below. No payback
printed here is a finding.

The rebate depends on the install date, which defaults to today. Pass
--install-date to reproduce a run exactly.
"""

import argparse
import math
from datetime import date

from src import finance
from src.config import Config, load_config
from src.finance import Payback
from src.profile import HouseholdProfile, check_matches_tariff, load_fixtures
from src.tariff import UNMODELLED, Tariff, describe_hours, unmodelled_components

# TODO(stage 2): delete PLACEHOLDER_DISPATCH and take kwh_shifted, r_out and
# r_in from the dispatch simulation.
#
# These are the only household-specific numbers in the pipeline, and they
# exist only so that it runs end to end before dispatch does. Each is read off
# the fixture's own bill figures and tariff by the rule beside it, for the
# default 10 kWh battery; none was tuned to produce a particular answer. They
# are deliberately crude -- they ignore power limits and day-to-day variation,
# which is what the simulation is for -- and nothing recomputes them if the
# battery or a tariff in config/ changes.
PLACEHOLDER_DISPATCH = {
    "reference_household": {
        "kwh_shifted": 1569,  # all peak-window consumption, 4.3 kWh/day
        "r_out": 0.4378,      # peak rate: discharges 16:00-21:00
        "r_in": 0.2585,       # offpeak rate: charges from the grid 09:00-16:00
        "basis": "grid-charged in offpeak; covers all peak-window consumption",
    },
    "household_b": {
        "kwh_shifted": 2880,  # 60% of 4,800 kWh, assumed outside solar hours (out on weekdays)
        "r_out": 0.30,        # the flat rate
        "r_in": 0.05,         # feed-in tariff forgone: charges from solar surplus
        "basis": "solar-charged; covers the 60% of consumption assumed to fall outside solar hours",
    },
    "household_c": {
        "kwh_shifted": 2100,  # all peak-window consumption, 5.8 kWh/day
        "r_out": 0.22,        # peak rate: discharges 16:00-21:00
        "r_in": 0.14,         # offpeak rate: charges from the grid 09:00-16:00
        "basis": "grid-charged in offpeak; covers all peak-window consumption",
    },
}

RULE = "=" * 88


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config()
    print_banner(config, args.install_date)

    results = []
    for profile in load_fixtures():
        tariff = config.tariffs[profile.tariff_ref]
        check_matches_tariff(profile, tariff)
        print_household(profile)
        print_tariff(tariff)
        print_consumption(profile, tariff)
        result = print_payback(profile, tariff, config, args.install_date)
        if result is not None:
            results.append((profile, tariff, result))

    print_summary(results)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run every fixture through the engine.")
    parser.add_argument(
        "--install-date",
        type=date.fromisoformat,
        default=date.today(),
        metavar="YYYY-MM-DD",
        help="when the battery is installed; sets the rebate's deeming factor (default: today)",
    )
    return parser.parse_args(argv)


def print_banner(config: Config, install_date: date) -> None:
    print("Stage 1 engine skeleton")
    print("  kwh_shifted, r_out and r_in are hardcoded placeholders until the dispatch")
    print("  simulation exists (stage 2). No payback printed here is a finding.")
    print(f"  Install date {install_date}, which sets the rebate (change with --install-date).")
    unverified = []
    if not config.batteries.verified:
        unverified.append("config/batteries.yaml")
    if not config.incentives.verified:
        unverified.append("config/incentives.yaml")
    if unverified:
        print("  Unverified, indicative market values in:")
        for source in unverified:
            print(f"    {source}")


def print_household(profile: HouseholdProfile) -> None:
    if profile.has_solar:
        solar = f"{profile.solar_kw:g} kW"
        if profile.annual_solar_export_kwh is not None:
            solar += f", {profile.annual_solar_export_kwh:,.0f} kWh/yr exported"
    else:
        solar = "none"
    print()
    print(RULE)
    print(profile.name)
    print(f"  source  {profile.source}")
    print(f"  solar   {solar}")
    print(RULE)


def print_tariff(tariff: Tariff) -> None:
    details = [tariff.retailer, tariff.plan, tariff.network]
    if tariff.effective_from:
        details.append(f"effective from {tariff.effective_from}")
    if tariff.effective_until:
        details.append(f"until {tariff.effective_until}")
    print(f"\nTariff  {tariff.id}")
    print("  " + " · ".join(detail for detail in details if detail))

    for i, window in enumerate(tariff.energy_windows):
        label = "energy_windows" if i == 0 else ""
        print(f"  {label:<17}{window.name:<10}{describe_hours(window.hours):<26}"
              f"${window.rate_aud_per_kwh:.5f}/kWh")
    print(f"  {'daily_charge':<17}${tariff.daily_charge_aud:.5f}/day")
    cl = tariff.controlled_load
    print(f"  {'controlled_load':<17}"
          + (f"${cl.rate_aud_per_kwh:.5f}/kWh, separate circuit" if cl else "none"))
    fit = tariff.feed_in_tariff_aud_per_kwh
    print(f"  {'feed_in':<17}" + (f"${fit:.5f}/kWh" if fit is not None else "none"))

    unmodelled = unmodelled_components(tariff)
    for component in UNMODELLED:
        status = f"NOT MODELLED: {unmodelled[component]}" if component in unmodelled else "none"
        print(f"  {component:<17}{status}")


def print_consumption(profile: HouseholdProfile, tariff: Tariff) -> None:
    total = profile.annual_kwh_total
    print("\nAnnual consumption by window")
    for window in tariff.energy_windows:
        kwh = profile.annual_kwh_by_window[window.name]
        print(f"  {window.name:<17}{kwh:>7,.0f} kWh {kwh / total:>7.1%}")
    if profile.annual_controlled_load_kwh is not None:
        kwh = profile.annual_controlled_load_kwh
        print(f"  {'controlled_load':<17}{kwh:>7,.0f} kWh {kwh / total:>7.1%}"
              "   separate circuit: a battery cannot serve it")
    print(f"  {'total':<17}{total:>7,.0f} kWh")
    if len(tariff.energy_windows) == 1:
        print("  One window: a flat-tariff bill gives magnitude only, with no time-of-day")
        print("  shape, so results here carry more uncertainty than for a time-of-use bill.")


def print_payback(
    profile: HouseholdProfile, tariff: Tariff, config: Config, install_date: date
) -> Payback | None:
    batteries = config.batteries
    size_kwh = batteries.default_size_kwh
    print(f"\nPayback  {size_kwh:g} kWh battery (default_size_kwh), {batteries.power_kw:g} kW, "
          f"{batteries.round_trip_efficiency:.0%} round trip, installed {install_date}")

    placeholder = PLACEHOLDER_DISPATCH.get(profile.name)
    if placeholder is None:
        print("  skipped: no placeholder dispatch terms for this fixture (PLACEHOLDER_DISPATCH)")
        return None

    cost = batteries.cost_model
    battery_cost = finance.battery_cost_aud(size_kwh, cost.fixed_aud, cost.variable_aud_per_kwh)
    incentives = config.incentives
    period = incentives.deeming_period_on(install_date)
    rebate = finance.rebate(
        size_kwh,
        deeming_factor=period.factor,
        stc_price_aud=incentives.stc_price_aud,
        taper=incentives.capacity_taper,
    )

    # Demand charges are not modelled yet, so demand_saving is zero. On a
    # tariff that has one, that zero means "not calculated", not "nothing".
    demand_saving = 0.0
    if tariff.demand is None:
        demand_note = "no demand charge on this tariff"
    else:
        demand_note = "NOT MODELLED: not calculated, rather than nothing"

    r = finance.payback(
        kwh_shifted=placeholder["kwh_shifted"],
        r_out=placeholder["r_out"],
        r_in=placeholder["r_in"],
        efficiency=batteries.round_trip_efficiency,
        demand_saving=demand_saving,
        battery_cost=battery_cost,
        rebate=rebate.rebate_aud,
    )

    print_term("kwh_shifted", f"{r.kwh_shifted:,.0f}", "kWh/yr", "PLACEHOLDER")
    print_term("r_out", f"{r.r_out:.5f}", "$/kWh", "PLACEHOLDER")
    print_term("r_in", f"{r.r_in:.5f}", "$/kWh", "PLACEHOLDER")
    print_term("efficiency", f"{r.efficiency:.2f}", "", "config/batteries.yaml")
    print_term("demand_saving", f"{r.demand_saving:,.2f}", "$/yr", demand_note)
    print_term("battery_cost", f"{r.battery_cost:,.2f}", "$",
               f"{cost.fixed_aud:,.0f} + {cost.variable_aud_per_kwh:,.0f} × {size_kwh:g} kWh"
               "   config/batteries.yaml")
    print_term("rebate", f"{r.rebate:,.2f}", "$", "worked below   config/incentives.yaml")
    print(f"  placeholder basis: {placeholder['basis']}")

    k = f"{r.kwh_shifted:,.0f}"
    d = f"{r.demand_saving:,.2f}"
    print()
    print("  annual_saving = kwh_shifted × (r_out − r_in / efficiency) + demand_saving")
    print(f"                = {k} × ({r.r_out:.5f} − {r.r_in:.5f} / {r.efficiency:.2f}) + {d}")
    print(f"                = {k} × ({r.r_out:.5f} − {r.r_in_after_losses:.5f}) + {d}")
    print(f"                = {k} × {r.saving_per_kwh:.5f} + {d}")
    print(f"                = {money(r.annual_saving)} per year")

    bands = " + ".join(f"{kwh:g} × {share:.2f}" for kwh, share in rebate.bands if kwh > 0)
    print()
    print("  rebate        = Σ taper bands (kWh in band × share) × deeming_factor × stc_price")
    print(f"                = ({bands or '0'}) × {rebate.deeming_factor:g}"
          f" × {rebate.stc_price_aud:.2f}")
    print(f"                = {rebate.stcs:,.1f} STCs × {rebate.stc_price_aud:.2f}")
    print(f"                = {money(rebate.rebate_aud)}")
    print(f"  deeming factor {period.factor:g} applies to installs from {period.start}"
          f" to {period.until}")

    print()
    print("  payback_years = (battery_cost − rebate) / annual_saving")
    print(f"                = ({r.battery_cost:,.2f} − {r.rebate:,.2f}) / {r.annual_saving:,.2f}")
    print(f"                = {r.net_cost:,.2f} / {r.annual_saving:,.2f}")
    if math.isinf(r.payback_years):
        print("                = never: the battery does not save money on these terms")
    else:
        print(f"                = {r.payback_years:.1f} years")

    unmodelled = unmodelled_components(tariff)
    if unmodelled:
        print()
        print("  PARTIAL PICTURE. This tariff has charges this version does not price:")
        for component, description in unmodelled.items():
            print(f"    {component}: {description}")
        print("  The saving and payback above leave them out, so they are not the answer")
        print("  for this household.")
    return r


def print_term(name: str, value: str, unit: str, source: str) -> None:
    print(f"  {name:<15}{value:>11}  {unit:<8}{source}")


def print_summary(results: list[tuple[HouseholdProfile, Tariff, Payback]]) -> None:
    print()
    print(RULE)
    print("Summary (placeholder dispatch terms: not results)")
    print(f"  {'household':<21}{'tariff':<20}{'windows':>7}  {'solar':<8}"
          f"{'saving/yr':>11}{'payback':>10}")
    for profile, tariff, r in results:
        solar = f"{profile.solar_kw:g} kW" if profile.has_solar else "none"
        payback = "never" if math.isinf(r.payback_years) else f"{r.payback_years:.1f} yr"
        unmodelled = unmodelled_components(tariff)
        note = f"  PARTIAL: {', '.join(unmodelled)} not priced" if unmodelled else ""
        print(f"  {profile.name:<21}{tariff.id:<20}{len(tariff.energy_windows):>7}  {solar:<8}"
              f"{money(r.annual_saving):>11}{payback:>10}{note}")


def money(amount: float) -> str:
    return f"${amount:,.2f}"


if __name__ == "__main__":
    main()
