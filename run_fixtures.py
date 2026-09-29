"""Console runner: every fixture through the engine.

    python run_fixtures.py [--install-date YYYY-MM-DD]

For each household in fixtures/, prints the tariff it is on, its annual
consumption by window, the payback calculation with every term shown
separately and the recommendation built from it, then a one-line-per-household
summary.

Every figure comes from one run of the engine (src/engine.py): the year
src/generator.py builds from the billing periods, tariff windows and form
answers; the half-hourly dispatch over it; the headline equation; and the
decision rule. src/recommend.py adds the rest of the recommendation, including
the revisit_if sweeps, which re-run the engine.

The rebate depends on the install date, which defaults to today. Pass
--install-date to reproduce a run exactly.
"""

import argparse
import math
from datetime import date, timedelta

import numpy as np

from src.config import Config, load_config
from src.engine import SOLAR_FIRST, Evaluation, evaluate
from src.generator import Year
from src.profile import HouseholdProfile, load_fixtures
from src.recommend import Recommendation, recommend
from src.revisit import Revisit
from src.tariff import UNMODELLED, Tariff, describe_hours, unmodelled_components

RULE = "=" * 88


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config()
    print_banner(config, args.install_date)

    results = []
    for profile in load_fixtures():
        tariff = config.tariffs[profile.tariff_ref]
        evaluation = evaluate(profile, tariff, config, args.install_date)
        recommendation = recommend(evaluation)
        print_household(profile)
        print_tariff(tariff)
        print_consumption(profile, tariff, evaluation.year)
        print_payback(evaluation)
        print_recommendation(recommendation)
        results.append((evaluation, recommendation))

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
    print("Stage 4: recommendation")
    print("  Each household: a half-hourly dispatch of one year, shaped by its billing periods,")
    print("  tariff windows and form answers; the headline equation; and the recommendation")
    print("  built from them, with thresholds found by re-running the engine.")
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


def print_consumption(profile: HouseholdProfile, tariff: Tariff, year: Year) -> None:
    periods = profile.billing_periods
    total = profile.annual_kwh_total
    print(f"\nAnnual consumption by window  ({len(periods)} billing periods, "
          f"{periods[0].start} to {periods[-1].end - timedelta(days=1)})")
    for window in tariff.energy_windows:
        kwh = profile.annual_kwh_by_window[window.name]
        print(f"  {window.name:<17}{kwh:>7,.0f} kWh {kwh / total:>7.1%}")
    if profile.annual_controlled_load_kwh is not None:
        kwh = profile.annual_controlled_load_kwh
        print(f"  {'controlled_load':<17}{kwh:>7,.0f} kWh {kwh / total:>7.1%}"
              "   separate circuit: a battery cannot serve it")
    print(f"  {'total':<17}{total:>7,.0f} kWh")

    busiest = max(periods, key=lambda period: period.daily_kwh)
    quietest = min(periods, key=lambda period: period.daily_kwh)
    print(f"  Across the year: the billing periods. Busiest {busiest.daily_kwh:.1f} kWh/day "
          f"from {busiest.start},")
    print(f"  quietest {quietest.daily_kwh:.1f} from {quietest.start}, a ratio of "
          f"{busiest.daily_kwh / quietest.daily_kwh:.1f} (controlled load excluded).")
    if year.time_of_day_source == "form":
        print("  Within the day: a flat bill has no window split, so the whole daily shape comes")
        print("  from the form, and this result is materially less certain than a time-of-use one.")
    else:
        print("  Within the day: the bill splits use by window; only the shape inside each window")
        print("  is generated.")
    print(f"  Form answers used: {describe_form(profile)}")
    if profile.has_solar:
        imported = np.maximum(year.load - year.solar, 0.0).sum()
        exported = np.maximum(year.solar - year.load, 0.0).sum()
        print(f"  Solar: {year.solar.sum():,.0f} kWh generated. Without a battery the year exports "
              f"{exported:,.0f}")
        print(f"  (scaled to the bill) and imports {imported:,.0f}.")


def describe_form(profile: HouseholdProfile) -> str:
    occupancy = profile.form.get("occupancy_pattern") or "not stated"
    aircon = profile.form.get("aircon_use") if profile.form.get("has_aircon") else None
    return f"occupancy {occupancy}" + (f", air conditioning {aircon}" if aircon else "")


def print_payback(evaluation: Evaluation) -> None:
    config, tariff = evaluation.config, evaluation.tariff
    batteries = config.batteries
    size_kwh = evaluation.battery_kwh
    r, rebate, dispatched = evaluation.payback, evaluation.rebate, evaluation.dispatch
    cost = batteries.cost_model
    period = config.incentives.deeming_period_on(evaluation.install_date)
    print(f"\nPayback  {size_kwh:g} kWh battery (default_size_kwh), {batteries.power_kw:g} kW, "
          f"{batteries.round_trip_efficiency:.0%} round trip, installed {evaluation.install_date}")

    # Demand charges are not modelled yet, so demand_saving is zero. On a
    # tariff that has one, that zero means "not calculated", not "nothing".
    if tariff.demand is None:
        demand_note = "no demand charge on this tariff"
    else:
        demand_note = "NOT MODELLED: not calculated, rather than nothing"

    undefined = "undefined"
    print_term("kwh_shifted", f"{r.kwh_shifted:,.0f}", "kWh/yr", "dispatch")
    print_term("r_out", undefined if r.r_out is None else f"{r.r_out:.5f}", "$/kWh", "dispatch")
    print_term("r_in", undefined if r.r_in is None else f"{r.r_in:.5f}", "$/kWh", "dispatch")
    print_term("efficiency", f"{r.efficiency:.2f}", "", "config/batteries.yaml")
    print_term("demand_saving", f"{r.demand_saving:,.2f}", "$/yr", demand_note)
    print_term("battery_cost", f"{r.battery_cost:,.2f}", "$",
               f"{cost.fixed_aud:,.0f} + {cost.variable_aud_per_kwh:,.0f} × {size_kwh:g} kWh"
               "   config/batteries.yaml")
    print_term("rebate", f"{r.rebate:,.2f}", "$", "worked below   config/incentives.yaml")
    from_solar = dispatched.charge_solar.sum()
    from_grid = dispatched.charge_grid.sum()
    print(f"  dispatch: charged {from_solar + from_grid:,.0f} kWh ({from_solar:,.0f} from solar "
          f"surplus, {from_grid:,.0f} from the grid)")
    print(f"            to deliver {r.kwh_shifted:,.0f} kWh; at most "
          f"{dispatched.stored.max():.1f} kWh held at once")
    print(f"            moving only kWh whose margin beats the "
          f"{batteries.marginal_throughput_cost_aud_per_kwh * 100:.1f} c/kWh wear cost"
          "   config/batteries.yaml")

    k = f"{r.kwh_shifted:,.0f}"
    d = f"{r.demand_saving:,.2f}"
    print()
    print("  annual_saving = kwh_shifted × (r_out − r_in / efficiency) + demand_saving")
    if r.kwh_shifted > 0:
        print(f"                = {k} × ({r.r_out:.5f} − {r.r_in:.5f} / {r.efficiency:.2f}) + {d}")
        print(f"                = {k} × ({r.r_out:.5f} − {r.r_in_after_losses:.5f}) + {d}")
        print(f"                = {k} × {r.saving_per_kwh:.5f} + {d}")
    else:
        print(f"                = 0 + {d}   (no earlier energy was ever cheaper than importing)")
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

    if evaluation.unmodelled:
        print()
        print("  PARTIAL PICTURE. This tariff has charges this version does not price:")
        for component, description in evaluation.unmodelled.items():
            print(f"    {component}: {description}")
        print("  The saving and payback above leave them out, so they are not the answer")
        print("  for this household.")


def print_term(name: str, value: str, unit: str, source: str) -> None:
    print(f"  {name:<15}{value:>11}  {unit:<8}{source}")


def print_recommendation(rec: Recommendation) -> None:
    print(f"\nRecommendation  {rec.action}")
    print(f"  The battery on its own: {rec.battery_action}. "
          f"Payback {years(rec.basis['payback_years'])}, against:")
    for test in rec.rule_tests:
        if test.passed is None:
            print(f"    {test.limit:<14} not given on the form")
        else:
            where = "inside" if test.passed else "OUTSIDE"
            print(f"    {test.limit:<14} {test.limit_years:g} years: {where} it by "
                  f"{abs(test.margin_years):.1f} years")
    if rec.solar is not None:
        outranks = "; outranks the battery" if rec.action == SOLAR_FIRST else ""
        print(f"  Solar (INDICATIVE, not a precise payback): {rec.solar.kw:g} kW would pay back in "
              f"about {rec.solar.payback_years:.0f} years{outranks}")

    limit = rec.limited_by
    print(f"  Limited by: {limit.label}. At most {limit.max_daily_kwh:.1f} kWh delivered in a day "
          f"against {limit.capacity_kwh:g} kWh;")
    print(f"    full on {limit.days_full} days; {limit.unmet_kwh:,.0f} kWh worth serving left unmet")

    near, positions = rec.near_term, dict(rec.long_term.positions)
    last = max(positions)
    crossover = rec.long_term.crossover_year
    print(f"  Position: year 1 {money(near.position_after_year_1, 0)} "
          f"(outlay {money(near.net_outlay, 0)}, saving {money(near.first_year_saving, 0)}); "
          f"year {last} {money(positions[last], 0)}; "
          f"crossover {'year ' + str(crossover) if crossover else 'never'}")

    print("  Not priced:")
    for item in rec.not_priced:
        flag = f"   <- the household's stated priority: {item.stated_priority}" \
            if item.stated_priority else ""
        print(f"    {item.item}{flag}")
    print("  Drivers:")
    for driver in rec.drivers:
        print(f"    {driver.priority or '(none)'}: {driver.effect}")
    print("  Revisit if:")
    for revisit in rec.revisit_if:
        print(f"    {revisit.parameter:<15}{describe_revisit(revisit)}")
    print("  Assumptions:")
    for item in rec.assumptions:
        value = ("not given" if item.value is None
                 else f"{item.value:,g}" if isinstance(item.value, float) else item.value)
        dated = f", {item.date}" if item.date else ""
        ranged = f"  range {list(item.range)}" if item.range else ""
        print(f"    {item.name}: {value}  ({item.source}{dated}){ranged}")
        if item.effect:
            print(f"      -> {item.effect}")


def describe_revisit(revisit: Revisit) -> str:
    if revisit.not_applicable:
        return f"not applicable: {revisit.not_applicable}"
    low, high = (describe_value(revisit, value) for value in revisit.searched)
    tracked = "the battery answer" if revisit.tracks == "battery_action" else "the answer"
    if revisit.to_action is None:
        return f"no change in {tracked} within the range searched, {low} to {high}"
    return (f"{revisit.from_action} -> {revisit.to_action} at "
            f"{describe_value(revisit, revisit.threshold)} "
            f"(now {describe_value(revisit, revisit.current)})")


def describe_value(revisit: Revisit, value: float) -> str:
    if revisit.parameter == "battery cost":
        return f"${value:,.0f}"
    if revisit.parameter == "adding solar":
        return f"{value:.1f} kW"
    if revisit.parameter == "rate spread":
        return f"{value * 100:+.1f} c/kWh"
    return f"{value * 100:.1f} c/kWh"


def years(value: float) -> str:
    return "never" if math.isinf(value) else f"{value:.1f} years"


def print_summary(results: list[tuple[Evaluation, Recommendation]]) -> None:
    print()
    print(RULE)
    print("Summary")
    print(f"  {'household':<21}{'tariff':<20}{'solar':<8}{'saving/yr':>11}{'payback':>10}"
          f"   {'action'}")
    for evaluation, rec in results:
        profile, tariff, r = evaluation.profile, evaluation.tariff, evaluation.payback
        solar = f"{profile.solar_kw:g} kW" if profile.has_solar else "none"
        payback = "never" if math.isinf(r.payback_years) else f"{r.payback_years:.1f} yr"
        note = f"  (PARTIAL: {', '.join(rec.unmodelled)} not priced)" if rec.unmodelled else ""
        print(f"  {profile.name:<21}{tariff.id:<20}{solar:<8}"
              f"{money(r.annual_saving):>11}{payback:>10}   {rec.action}{note}")


def money(amount: float, decimals: int = 2) -> str:
    return f"-${-amount:,.{decimals}f}" if amount < 0 else f"${amount:,.{decimals}f}"


if __name__ == "__main__":
    main()
