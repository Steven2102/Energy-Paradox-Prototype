"""revisit_if: the thresholds at which a recommendation's action changes.

Computed, never authored. For each parameter in the sweep set the engine is
re-run across a plausible range (config/assumptions.yaml, revisit_ranges), and
the first value at which the action changes is located by bisection -- or the
sweep reports that nothing changes within the range searched. A change within
one tolerance of the range's edge counts as no change: the edge is where the
search stopped, not something it found. Required test 8 re-runs every reported
threshold, and both ends of every range reported unchanged, to check them.

The sweep set is four parameters: battery cost, feed-in tariff, rate spread and
adding solar. Adding solar tracks the battery answer rather than the final
action: any solar at all makes solar_first inapplicable, so the question it
answers is whether solar would make a battery worthwhile.

A threshold can move the label without moving the advice. Where the action flips
between solar_first and a battery answer that is the same on both sides, what
crossed was solar's indicative payback against payback_ratio of the battery's:
solar pays back in about that share of the time either side, and the battery
answer has not moved. Each threshold records the answer on both sides, and
advice_changes says whether the battery answer differs -- so a justification
never tells a household its answer changes where only the label does.
"""

from dataclasses import dataclass, replace
from functools import cache
from typing import Callable

from src.engine import Evaluation, assess, evaluate

# How finely each threshold is located, in the parameter's own unit.
TOLERANCE = {
    "battery cost": 10.0,       # $
    "feed-in tariff": 0.001,    # $/kWh
    "rate spread": 0.001,       # $/kWh
    "adding solar": 0.1,        # kW
}


@dataclass(frozen=True)
class Side:
    """The engine's answer at one value beside a threshold."""

    action: str
    battery_action: str
    payback_years: float                # the battery's
    solar_payback_years: float | None   # indicative; None where the household has solar


@dataclass(frozen=True)
class Revisit:
    parameter: str                         # a key of TOLERANCE
    unit: str
    tracks: str                            # "action", or "battery_action" for adding solar
    current: float | None
    searched: tuple[float, float] | None   # the plausible range -- where nothing changes, as far
                                           # as that was verified; None where not applicable
    from_action: str
    to_action: str | None                  # None where nothing changes within the range searched
    threshold: float | None                # the first value that gives to_action
    before: float | None                   # the last value beside it that still gives from_action
    not_applicable: str | None             # why the sweep does not apply, where it does not
    either_side: tuple[Side, Side] | None = None  # at before and at threshold, where there is one
    advice_changes: bool | None = None     # the battery answer differs either side (see above)


def revisit_if(evaluation: Evaluation) -> tuple[Revisit, ...]:
    return (
        _battery_cost(evaluation),
        _feed_in(evaluation),
        _rate_spread(evaluation),
        _adding_solar(evaluation),
    )


def _battery_cost(evaluation: Evaluation) -> Revisit:
    config = evaluation.config
    cost = config.batteries.cost_model
    price = cost.fixed_aud + cost.variable_aud_per_kwh * evaluation.battery_kwh
    low, high = config.assumptions.revisit_ranges.battery_cost_share

    def at(new_price: float) -> Evaluation:
        scale = new_price / price
        scaled = replace(cost, fixed_aud=cost.fixed_aud * scale,
                         variable_aud_per_kwh=cost.variable_aud_per_kwh * scale)
        repriced = replace(config, batteries=replace(config.batteries, cost_model=scaled))
        # Price enters only after the simulation, so re-assess rather than re-run it.
        return assess(evaluation.profile, evaluation.tariff, repriced, evaluation.install_date,
                      evaluation.year, evaluation.dispatch)

    return sweep(evaluation, "battery cost", "$ installed, before rebate", "action",
                 price, (low * price, high * price), at)


def _feed_in(evaluation: Evaluation) -> Revisit:
    if not evaluation.profile.has_solar:
        return _not_applicable(evaluation, "feed-in tariff", "$/kWh", "action",
                               "the household has no solar, so it exports nothing")
    tariff = evaluation.tariff

    def at(rate: float) -> Evaluation:
        return evaluate(evaluation.profile, replace(tariff, feed_in_tariff_aud_per_kwh=rate),
                        evaluation.config, evaluation.install_date)

    return sweep(evaluation, "feed-in tariff", "$/kWh", "action",
                 tariff.feed_in_tariff_aud_per_kwh,
                 evaluation.config.assumptions.revisit_ranges.feed_in_aud_per_kwh, at)


def _rate_spread(evaluation: Evaluation) -> Revisit:
    tariff = evaluation.tariff
    dearest = max(tariff.energy_windows, key=lambda window: window.rate_aud_per_kwh)

    def at(change: float) -> Evaluation:
        windows = tuple(replace(window, rate_aud_per_kwh=window.rate_aud_per_kwh + change)
                        if window is dearest else window for window in tariff.energy_windows)
        return evaluate(evaluation.profile, replace(tariff, energy_windows=windows),
                        evaluation.config, evaluation.install_date)

    rate = "flat" if len(tariff.energy_windows) == 1 else dearest.name
    return sweep(evaluation, "rate spread", f"$/kWh added to the {rate} rate", "action",
                 0.0, evaluation.config.assumptions.revisit_ranges.rate_spread_change_aud_per_kwh,
                 at)


def _adding_solar(evaluation: Evaluation) -> Revisit:
    profile = evaluation.profile
    if profile.has_solar:
        return _not_applicable(evaluation, "adding solar", "kW of panels", "battery_action",
                               f"the household already has {profile.solar_kw:g} kW")
    config = evaluation.config
    tariff = evaluation.tariff
    if tariff.feed_in_tariff_aud_per_kwh is None:
        tariff = replace(tariff,
                         feed_in_tariff_aud_per_kwh=config.assumptions.new_solar_feed_in_aud_per_kwh)

    def at(kw: float) -> Evaluation:
        with_solar = replace(profile, has_solar=True, solar_kw=kw, annual_solar_export_kwh=None)
        return evaluate(with_solar, tariff, config, evaluation.install_date)

    return sweep(evaluation, "adding solar", "kW of panels", "battery_action",
                 0.0, config.assumptions.revisit_ranges.added_solar_kw, at)


# ------------------------------------------------------------------ searching

def sweep(evaluation: Evaluation, parameter: str, unit: str, tracks: str, current: float,
          searched: tuple[float, float], at: Callable[[float], Evaluation]) -> Revisit:
    """Look towards both ends of the range; report the nearer change of the
    tracked action, or none.

    A change within one tolerance of the end it was found towards is not a
    threshold: the flip could sit anywhere in that last step, and the edge is
    only where the search stopped. It is reported as no change, over the range
    pulled in to the last value verified unchanged -- the answer can already
    differ at the edge itself, and a no-change report must hold at both ends."""
    at = cache(at)  # the sides of a threshold are read back, not re-run
    now = getattr(evaluation, tracks)
    tolerance = TOLERANCE[parameter]
    unchanged_over = list(searched)
    nearest = None
    for which, end in enumerate(searched):
        if end == current or getattr(at(end), tracks) == now:
            continue
        before, threshold = _bisect(lambda value: getattr(at(value), tracks) == now,
                                    current, end, tolerance)
        if abs(end - threshold) <= tolerance:
            unchanged_over[which] = before
        elif nearest is None or abs(threshold - current) < abs(nearest[1] - current):
            nearest = (before, threshold)
    if nearest is None:
        return Revisit(parameter, unit, tracks, current, tuple(unchanged_over), now,
                       to_action=None, threshold=None, before=None, not_applicable=None)
    before, threshold = nearest
    sides = (_side(at(before)), _side(at(threshold)))
    return Revisit(parameter, unit, tracks, current, tuple(searched), now,
                   to_action=getattr(at(threshold), tracks), threshold=threshold,
                   before=before, not_applicable=None, either_side=sides,
                   advice_changes=sides[0].battery_action != sides[1].battery_action)


def _side(evaluation: Evaluation) -> Side:
    solar = evaluation.solar
    return Side(action=evaluation.action, battery_action=evaluation.battery_action,
                payback_years=evaluation.payback.payback_years,
                solar_payback_years=solar.payback_years if solar is not None else None)


def _bisect(unchanged: Callable[[float], bool], inside: float, outside: float,
            tolerance: float) -> tuple[float, float]:
    """Narrow [inside, outside] -- the action unchanged at the first, changed at the
    second -- to within tolerance. Returns (last unchanged, first changed)."""
    while abs(outside - inside) > tolerance:
        middle = (inside + outside) / 2
        if unchanged(middle):
            inside = middle
        else:
            outside = middle
    return inside, outside


def _not_applicable(evaluation: Evaluation, parameter: str, unit: str, tracks: str,
                    why: str) -> Revisit:
    return Revisit(parameter, unit, tracks, current=None, searched=None,
                   from_action=getattr(evaluation, tracks), to_action=None, threshold=None,
                   before=None, not_applicable=why)


def describe_value(revisit: Revisit, value: float) -> str:
    """A value of the swept parameter as the household reads it."""
    if revisit.parameter == "battery cost":
        return f"${value:,.0f}"
    if revisit.parameter == "adding solar":
        return f"{value:.1f} kW"
    if revisit.parameter == "rate spread":
        return f"{value * 100:+.1f} c/kWh"
    return f"{value * 100:.1f} c/kWh"
