"""The recommendation: the structured object from CLAUDE.md's "What a
recommendation is". Every field is computed here; the language model (stage 5)
writes prose from it and adds nothing.

    action        battery_now | battery_not_yet | solar_first | no_action | cannot_assess
    size_kwh      recommended usable capacity, or None
    basis         every term of the headline equation, by name
    rule_tests    each limit of the decision rule, and whether the payback falls inside it
                  and by how much -- the justification must name a failed one
    limited_by    the binding constraint: what caps the battery's value (required test 5)
    near_term     year 1: the outlay, the saving, the position at the end of it
    long_term     the running cumulative position, year by year, and the crossover year
    not_priced    what the number leaves out, flagged where the household said it matters
    assumptions   name, value, source, date -- the ones that shaped this result, and every
                  missing form answer with whether it would change the answer
    drivers       which stated priorities produced the answer; none invented
    revisit_if    thresholds at which the action changes (src/revisit.py)
    unmodelled    charges on the bill this version does not price (hard rule 9)

and, so the reasoning can be walked through: battery_action (the decision rule's
answer about a battery, before solar_first), solar (the indicative comparison
behind solar_first), time_of_day_source (whether the day's shape came from the
bill or the form) and evaluated_kwh (the capacity the figures are for).

A recommendation missing any required section is refused (required test 7).
"""

import math
from dataclasses import dataclass, replace
from datetime import date

import numpy as np

from src import finance
from src.dispatch import worth_serving
from src.engine import (BATTERY_NOW, CANNOT_ASSESS, NO_ACTION, Evaluation, RuleTest,
                        SolarComparison, evaluate)
from src.generator import DAYS
from src.revisit import Revisit, revisit_if
from src.tariff import SLOTS_PER_DAY, window_by_slot

BASIS_TERMS = ("kwh_shifted", "r_out", "r_in", "efficiency", "annual_saving", "demand_saving",
               "battery_cost", "rebate", "payback_years")
TOLERANCE_KWH = 0.01


@dataclass(frozen=True)
class NearTerm:
    installed_price: float
    rebate: float
    net_outlay: float
    first_year_saving: float
    position_after_year_1: float


@dataclass(frozen=True)
class LongTerm:
    positions: tuple[tuple[int, float], ...]  # (year, cumulative $ position) from year 1
    crossover_year: int | None                # first year the position is no longer negative


@dataclass(frozen=True)
class NotPriced:
    item: str
    detail: str
    stated_priority: str | None  # the household's own motivation this touches, if any


@dataclass(frozen=True)
class Assumption:
    name: str
    value: object               # None where the form gave nothing
    source: str
    date: date | None           # when the figure was sourced; None for an undated estimate
    range: tuple | None = None  # in the value's own unit
    effect: str | None = None   # for a missing answer: whether it would change the answer
    unit: str | None = None     # "$", "%", "c/kWh" (value in $/kWh), "years", "hours";
                                # None for a plain number or a description


def quantity(value: object, unit: str | None) -> str:
    """An assumption's value, or one end of its range, as it reads in its unit."""
    if value is None:
        return "not given"
    if isinstance(value, str):
        return value
    if unit == "$":
        return f"${value:,.0f}"
    if unit == "%":
        return f"{value:.0%}"
    if unit == "c/kWh":
        return f"{value * 100:.1f} c/kWh"
    if unit in ("years", "hours"):
        return f"{value:g} {unit}"
    return f"{value:,g}"


@dataclass(frozen=True)
class Driver:
    priority: str | None  # None where the form gave no priorities
    effect: str


@dataclass(frozen=True)
class Constraint:
    """What caps the battery's value, read off the dispatch."""

    label: str
    windows: tuple[str, ...]  # the tariff windows the battery discharged into
    max_daily_kwh: float      # the most it delivered in a day
    capacity_kwh: float
    days_full: int            # days it reached capacity
    unmet_kwh: float          # imports worth serving that it left unserved


@dataclass(frozen=True)
class Recommendation:
    household: str
    action: str
    size_kwh: float | None
    basis: dict[str, float | None]
    rule_tests: tuple[RuleTest, ...]
    limited_by: Constraint
    near_term: NearTerm
    long_term: LongTerm
    not_priced: tuple[NotPriced, ...]
    assumptions: tuple[Assumption, ...]
    drivers: tuple[Driver, ...]
    revisit_if: tuple[Revisit, ...]
    unmodelled: dict[str, str]
    # So the reasoning can be walked through.
    battery_action: str
    solar: SolarComparison | None
    time_of_day_source: str
    evaluated_kwh: float


class IncompleteRecommendation(ValueError):
    pass


def recommend(evaluation: Evaluation) -> Recommendation:
    payback = evaluation.payback
    recommendation = Recommendation(
        household=evaluation.profile.name,
        action=evaluation.action,
        size_kwh=evaluation.battery_kwh if evaluation.action == BATTERY_NOW else None,
        basis={term: getattr(payback, term) for term in BASIS_TERMS},
        rule_tests=evaluation.rule_tests,
        limited_by=_limited_by(evaluation),
        near_term=NearTerm(
            installed_price=payback.battery_cost,
            rebate=payback.rebate,
            net_outlay=payback.net_cost,
            first_year_saving=payback.annual_saving,
            position_after_year_1=payback.annual_saving - payback.net_cost,
        ),
        long_term=_long_term(evaluation),
        not_priced=_not_priced(evaluation),
        assumptions=_assumptions(evaluation),
        drivers=_drivers(evaluation),
        revisit_if=revisit_if(evaluation),
        unmodelled=evaluation.unmodelled,
        battery_action=evaluation.battery_action,
        solar=evaluation.solar,
        time_of_day_source=evaluation.year.time_of_day_source,
        evaluated_kwh=evaluation.battery_kwh,
    )
    check_complete(recommendation)
    return recommendation


def check_complete(recommendation: Recommendation) -> None:
    """Refuse a recommendation with an empty required section (required test 7).
    r_out and r_in may be absent only where the battery moved nothing."""
    missing = [name for name in ("revisit_if", "assumptions", "not_priced")
               if not getattr(recommendation, name)]
    nothing_moved = recommendation.basis.get("kwh_shifted") == 0
    for term in BASIS_TERMS:
        value = recommendation.basis.get(term)
        if value is None and not (nothing_moved and term in ("r_out", "r_in")):
            missing.append(f"basis.{term}")
    if missing:
        raise IncompleteRecommendation(
            f"{recommendation.household}: refusing a recommendation without {', '.join(missing)}")


# ------------------------------------------------------------------- sections

def _long_term(evaluation: Evaluation) -> LongTerm:
    """The running position: −net outlay + years × annual saving, through the
    later of the warranty and the stated stay."""
    payback = evaluation.payback
    stay = evaluation.profile.form.get("years_expected_in_home") or 0
    through = int(max(evaluation.config.batteries.warranty_years, stay))
    positions = tuple((year, year * payback.annual_saving - payback.net_cost)
                      for year in range(1, through + 1))
    crossover = (math.ceil(payback.net_cost / payback.annual_saving)
                 if payback.annual_saving > 0 else None)
    return LongTerm(positions=positions, crossover_year=crossover)


def _not_priced(evaluation: Evaluation) -> tuple[NotPriced, ...]:
    stated = set(_motivation(evaluation))
    batteries = evaluation.config.batteries
    payback = evaluation.payback
    items = [
        NotPriced("backup power in an outage",
                  "The payback counts bill savings only; keeping essentials running in a "
                  "blackout is not valued.",
                  "backup_power" if "backup_power" in stated else None),
        NotPriced("independence from retailer price rises",
                  "Prices are held at today's rates, so protection from future rises is not "
                  "valued.",
                  "independence" if "independence" in stated else None),
        NotPriced("evening peak relief",
                  "Discharging in the evening eases the network's peak; nothing in the "
                  "household's number pays for that.",
                  None),
        NotPriced("battery wear",
                  f"Wear decides which kWh are worth moving "
                  f"({batteries.marginal_throughput_cost_aud_per_kwh * 100:.1f} c/kWh) but is "
                  "not subtracted from the saving.",
                  None),
    ]
    if payback.payback_years > batteries.warranty_years:
        items.append(NotPriced(
            "outliving the warranty",
            f"The payback ({_years(payback.payback_years)}) runs past the "
            f"{batteries.warranty_years:g}-year warranty, so the battery may not last long "
            "enough to break even.",
            None))
    for component, description in evaluation.unmodelled.items():
        items.append(NotPriced(
            f"{component} charge",
            f"{description}. On the bill but not priced, so the saving and payback cover "
            "energy only.",
            None))
    return tuple(items)


def _drivers(evaluation: Evaluation) -> tuple[Driver, ...]:
    stated = _motivation(evaluation)
    if not stated:
        return (Driver(None, "No priorities were given on the form, so none shaped this answer."),)
    drivers = []
    for priority in stated:
        if priority == "lower_bill" and evaluation.action == CANNOT_ASSESS:
            effect = "the answer turns on the bill saving, which this version cannot complete"
        elif priority == "lower_bill":
            effect = "decided the answer: the bill saving is what the payback measures"
        elif priority in ("independence", "backup_power"):
            effect = "stated but not priced, so it did not move the answer (see not_priced)"
        else:
            effect = "stated, but this version has no measure for it, so it did not move the answer"
        drivers.append(Driver(priority, effect))
    return tuple(drivers)


def _limited_by(evaluation: Evaluation) -> Constraint:
    year, dispatched = evaluation.year, evaluation.dispatch
    batteries = evaluation.config.batteries
    worth = worth_serving(
        year,
        efficiency=batteries.round_trip_efficiency,
        marginal_throughput_cost_aud_per_kwh=batteries.marginal_throughput_cost_aud_per_kwh,
        horizon_intervals=evaluation.config.assumptions.dispatch_horizon_intervals,
    )
    need = np.maximum(year.load - year.solar, 0.0)
    unmet_by_day = _by_day(np.where(worth, need, 0.0) - dispatched.discharge).sum(axis=1)
    full = _by_day(dispatched.stored).max(axis=1) >= evaluation.battery_kwh - TOLERANCE_KWH
    short = unmet_by_day > TOLERANCE_KWH
    slots = window_by_slot(evaluation.tariff)
    discharged = _by_day(dispatched.discharge).sum(axis=0) > 0
    windows = tuple(dict.fromkeys(slots[i].name for i in range(SLOTS_PER_DAY) if discharged[i]))

    if evaluation.unmodelled:
        label = "an unpriced charge"
    elif not short.any():
        label = f"demand in the {' and '.join(windows)} window" if windows else "demand"
    else:
        parts = []
        if (short & full).any():
            parts.append("battery capacity")
        if (short & ~full).any():
            parts.append("available surplus" if dispatched.charge_solar.sum() > 0
                         else "available cheap energy")
        label = " and ".join(parts)
    return Constraint(
        label=label,
        windows=windows,
        max_daily_kwh=float(_by_day(dispatched.discharge).sum(axis=1).max()),
        capacity_kwh=evaluation.battery_kwh,
        days_full=int(full.sum()),
        unmet_kwh=float(max(unmet_by_day.sum(), 0.0)),
    )


def _assumptions(evaluation: Evaluation) -> tuple[Assumption, ...]:
    config, profile, payback = evaluation.config, evaluation.profile, evaluation.payback
    batteries, incentives, assumptions = config.batteries, config.incentives, config.assumptions
    cost = batteries.cost_model
    period = incentives.deeming_period_on(evaluation.install_date)

    def rebate_at(stc_price: float) -> float:
        return finance.rebate(evaluation.battery_kwh, deeming_factor=period.factor,
                              stc_price_aud=stc_price, taper=incentives.capacity_taper).rebate_aud

    stc_low, stc_high = incentives.stc_price_range_aud
    items = [
        Assumption("battery installed price", round(payback.battery_cost, 2),
                   f"config/batteries.yaml: ${cost.fixed_aud:,.0f} + ${cost.variable_aud_per_kwh:,.0f}"
                   f" × {evaluation.battery_kwh:g} kWh", batteries.sourced, unit="$"),
        Assumption("rebate", round(payback.rebate, 2),
                   f"config/incentives.yaml: deeming factor {period.factor:g} for installs from "
                   f"{period.start} to {period.until}, at ${incentives.stc_price_aud:.2f} per STC; "
                   f"the range is the rebate at ${stc_low:.2f} to ${stc_high:.2f} per STC",
                   incentives.sourced, range=(rebate_at(stc_low), rebate_at(stc_high)), unit="$"),
        Assumption("round-trip efficiency", batteries.round_trip_efficiency,
                   "config/batteries.yaml", batteries.sourced,
                   range=batteries.round_trip_efficiency_range, unit="%"),
        Assumption("marginal wear cost", batteries.marginal_throughput_cost_aud_per_kwh,
                   "config/batteries.yaml", batteries.sourced,
                   range=batteries.marginal_throughput_cost_range, unit="c/kWh"),
        Assumption("warranty", batteries.warranty_years, "config/batteries.yaml",
                   batteries.sourced, unit="years"),
        Assumption("electricity prices", "held at today's rates: no growth",
                   f"tariff {evaluation.tariff.id} (config/tariffs.yaml)",
                   evaluation.tariff.effective_from),
        Assumption("planning horizon", assumptions.dispatch_horizon_hours,
                   "config/assumptions.yaml", None,
                   range=assumptions.dispatch_horizon_hours_range, unit="hours"),
    ]
    if profile.has_solar:
        items += [
            Assumption("solar output, June against December",
                       assumptions.solar_june_to_december_ratio, "config/assumptions.yaml", None,
                       range=assumptions.solar_june_to_december_ratio_range),
            Assumption("cloudy days", f"{assumptions.cloudy_day_share:.0%} of days at "
                       f"{assumptions.cloudy_day_output:.0%} output, evenly spaced (optimistic)",
                       "config/assumptions.yaml", None),
        ]
    if evaluation.year.time_of_day_source == "form":
        occupancy = profile.form.get("occupancy_pattern") or "not_stated"
        weekday = assumptions.occupancy[occupancy].weekday
        daytime = next(b for b in assumptions.day_shapes[weekday] if b.share_range is not None)
        items.append(Assumption(
            "time of day, from the form (the bill has no window split)",
            f"{occupancy}: weekdays use the {weekday} shape, "
            f"{daytime.share:.0%} of the day between {daytime.hours[0]:g}:00 and "
            f"{daytime.hours[1]:g}:00",
            "config/assumptions.yaml", None, range=daytime.share_range, unit="%"))
    if evaluation.solar is not None:
        indicative = assumptions.indicative_solar
        items.append(Assumption(
            "indicative solar comparison",
            f"{indicative.kw:g} kW × {assumptions.solar_yield_kwh_per_kw:,.0f} kWh/kW, "
            f"{indicative.self_consumption:.0%} used on site, "
            f"${indicative.installed_aud_per_kw:,.0f}/kW installed, "
            f"{evaluation.solar.feed_in * 100:.1f} c/kWh feed-in",
            "config/assumptions.yaml", None))
    return tuple(items) + _missing_answers(evaluation)


def _missing_answers(evaluation: Evaluation) -> tuple[Assumption, ...]:
    """Every form answer the engine needed but did not get, with whether it would
    change the answer. Nothing is defaulted silently."""
    form = evaluation.profile.form
    missing = []
    if form.get("years_expected_in_home") is None:
        missing.append(Assumption(
            "years_expected_in_home", None, "form (not answered)", None,
            effect=_stay_effect(evaluation)))
    if form.get("occupancy_pattern") is None:
        missing.append(Assumption(
            "occupancy_pattern", None,
            "form (not answered): the generator used the not_stated day shapes", None,
            effect=_alternatives_effect(evaluation, [
                {"occupancy_pattern": answer}
                for answer in evaluation.config.assumptions.occupancy if answer != "not_stated"])))
    if form.get("has_aircon") is None:
        missing.append(Assumption(
            "has_aircon / aircon_use", None,
            "form (not answered): no air conditioning was placed", None,
            effect=_alternatives_effect(evaluation, [
                {"has_aircon": True, "aircon_use": use}
                for use in evaluation.config.assumptions.aircon_hours])))
    return tuple(missing)


def _stay_effect(evaluation: Evaluation) -> str:
    warranty = evaluation.rule_tests[0]
    if evaluation.unmodelled:
        return "would not change the answer: the unpriced charge already prevents an assessment"
    if evaluation.battery_action == NO_ACTION:
        return "would not change the answer: the battery saves nothing"
    if warranty.passed is False:
        return (f"would not change the answer: the payback ({_years(evaluation.payback.payback_years)}) "
                f"fails the {warranty.limit_years:g}-year warranty on its own, whatever the stay")
    return (f"decides the answer: the payback is inside the warranty, so a battery is worth it "
            f"only if the household expects to stay at least "
            f"{_years(evaluation.payback.payback_years)}")


def _alternatives_effect(evaluation: Evaluation, answers: list[dict]) -> str:
    """Re-run the engine with each answer the form could have given."""
    changed = []
    for answer in answers:
        profile = replace(evaluation.profile, form={**evaluation.profile.form, **answer})
        rerun = evaluate(profile, evaluation.tariff, evaluation.config, evaluation.install_date)
        if rerun.action != evaluation.action:
            changed.append(f"{_describe(answer)} gives {rerun.action}")
    tried = "; ".join(_describe(answer) for answer in answers)
    if changed:
        return f"could change the answer: {'; '.join(changed)}"
    return f"would not change the answer (re-run with {tried})"


def _motivation(evaluation: Evaluation) -> list[str]:
    return list(evaluation.profile.form.get("motivation") or [])


def _describe(answer: dict) -> str:
    return ", ".join(f"{key} {value}" for key, value in answer.items())


def _years(years: float) -> str:
    return "never" if math.isinf(years) else f"{years:.1f} years"


def _by_day(values: np.ndarray) -> np.ndarray:
    return values.reshape(DAYS, SLOTS_PER_DAY)
