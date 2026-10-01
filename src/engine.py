"""The engine: one household through the whole calculation.

    generator -> dispatch -> finance -> the decision rule -> solar_first

A pure function of (profile, tariff, config, install date) with nothing kept
between runs, so the revisit sweeps and the follow-up chat can change one
input and run it again.

Before any of it, one guard: this version evaluates adding a battery to a home
without one. A household that already has one cannot be assessed, so it gets
cannot_assess with the reason (not_assessable), and evaluate() refuses it
rather than compute a purchase payback.

The decision rule (CLAUDE.md): battery_now needs the payback inside both the
battery's warranty and the household's stated years in the home; failing
either gives battery_not_yet. A failure decides the answer even where the
other limit is missing from the form; where nothing has failed but a limit is
missing, the battery cannot be assessed. Nor can it on a tariff with a charge
this version does not price: that charge could move the answer, and a demand
charge usually moves it towards a battery. solar_first outranks a battery
answer when the indicative solar comparison is materially better.
"""

import math
from dataclasses import dataclass, replace
from datetime import date

from src import finance
from src.config import Config
from src.dispatch import Dispatch, simulate
from src.finance import Payback, Rebate
from src.generator import Year, household_year
from src.profile import HouseholdProfile, check_matches_tariff
from src.tariff import SLOTS_PER_DAY, Tariff, unmodelled_components, window_by_slot

BATTERY_NOW = "battery_now"
BATTERY_NOT_YET = "battery_not_yet"
SOLAR_FIRST = "solar_first"
NO_ACTION = "no_action"
CANNOT_ASSESS = "cannot_assess"


@dataclass(frozen=True)
class RuleTest:
    """One limit of the decision rule: the payback must fall inside it."""

    limit: str                  # "warranty" or "expected stay"
    limit_years: float | None   # None where the form did not give it
    passed: bool | None         # None where it cannot be tested
    margin_years: float | None  # payback minus the limit: outside (+) or inside (−), by how much


@dataclass(frozen=True)
class SolarComparison:
    """INDICATIVE ONLY, not a PV model: capacity × annual yield × a
    self-consumption share, valued against the windows the bill already gives.
    It decides whether solar_first outranks a battery answer, and is never
    quoted as a precise payback."""

    kw: float
    generation_kwh: float
    self_consumed_kwh: float  # capped at the billed use in the window covering midday
    exported_kwh: float
    midday_window: str
    midday_rate: float        # $/kWh
    feed_in: float            # $/kWh
    annual_saving: float      # $/yr
    installed_cost: float     # $
    payback_years: float      # indicative
    payback_ratio: float      # solar_first at most this share of the battery's payback


@dataclass(frozen=True)
class Evaluation:
    """One run of the engine, with the inputs that produced it."""

    profile: HouseholdProfile
    tariff: Tariff
    config: Config
    install_date: date
    year: Year
    dispatch: Dispatch
    battery_kwh: float
    rebate: Rebate
    payback: Payback
    unmodelled: dict[str, str]
    rule_tests: tuple[RuleTest, ...]
    battery_action: str            # the decision rule's answer about a battery
    solar: SolarComparison | None  # None where the household already has solar
    action: str                    # battery_action, unless solar_first outranks it


def not_assessable(profile: HouseholdProfile) -> str | None:
    """Why this version cannot assess the household at all, or None."""
    if profile.form.get("has_battery"):
        size = profile.form.get("battery_kwh")
        battery = f"a {size:g} kWh battery" if size else "a battery"
        return (f"The household already has {battery}. This version evaluates adding a battery "
                "to a home without one, so it has no purchase payback to give.")
    return None


def tariff_for(profile: HouseholdProfile, tariff: Tariff, config: Config) -> Tariff:
    """The tariff as it applies to this household. Where it has solar but the
    tariff sets no feed-in rate, exports earn the rate assumed for new solar,
    which the recommendation declares."""
    if (profile.has_solar and tariff.feed_in_tariff_aud_per_kwh is None
            and tariff.feed_in_windows is None):
        return replace(tariff,
                       feed_in_tariff_aud_per_kwh=config.assumptions.new_solar_feed_in_aud_per_kwh)
    return tariff


def evaluate(profile: HouseholdProfile, tariff: Tariff, config: Config,
             install_date: date) -> Evaluation:
    reason = not_assessable(profile)
    if reason:
        raise ValueError(f"{profile.name}: {reason}")
    year, dispatched = simulate_household(profile, tariff, config)
    return assess(profile, tariff, config, install_date, year, dispatched)


def simulate_household(profile: HouseholdProfile, tariff: Tariff,
                       config: Config) -> tuple[Year, Dispatch]:
    """The expensive part: the year's half-hours, and the battery dispatched over them."""
    check_matches_tariff(profile, tariff)
    batteries = config.batteries
    year = household_year(profile, tariff, config.assumptions)
    dispatched = simulate(
        year,
        capacity_kwh=batteries.default_size_kwh,
        power_kw=batteries.power_kw,
        efficiency=batteries.round_trip_efficiency,
        marginal_throughput_cost_aud_per_kwh=batteries.marginal_throughput_cost_aud_per_kwh,
        horizon_intervals=config.assumptions.dispatch_horizon_intervals,
    )
    return year, dispatched


def assess(profile: HouseholdProfile, tariff: Tariff, config: Config, install_date: date,
           year: Year, dispatched: Dispatch) -> Evaluation:
    """Everything after the simulation. The battery's price enters only here, so a
    sweep over price re-assesses one simulation rather than repeating it."""
    batteries = config.batteries
    size = batteries.default_size_kwh
    cost = batteries.cost_model
    period = config.incentives.deeming_period_on(install_date)
    rebate = finance.rebate(size, deeming_factor=period.factor,
                            stc_price_aud=config.incentives.stc_price_aud,
                            taper=config.incentives.capacity_taper)
    # Demand charges are not priced yet: demand_saving is zero because it is not
    # calculated, and the unmodelled declaration says so.
    payback = finance.payback(
        kwh_shifted=dispatched.kwh_shifted,
        r_out=dispatched.r_out,
        r_in=dispatched.r_in,
        efficiency=batteries.round_trip_efficiency,
        demand_saving=0.0,
        battery_cost=finance.battery_cost_aud(size, cost.fixed_aud, cost.variable_aud_per_kwh),
        rebate=rebate.rebate_aud,
    )
    unmodelled = unmodelled_components(tariff)
    rule_tests = (
        rule_test("warranty", payback.payback_years, batteries.warranty_years),
        rule_test("expected stay", payback.payback_years,
                  profile.form.get("years_expected_in_home")),
    )
    battery_action = decide(payback, rule_tests, unmodelled)
    solar = compare_solar(profile, tariff, config)
    outranked = (solar is not None
                 and battery_action in (BATTERY_NOW, BATTERY_NOT_YET, NO_ACTION)
                 and _solar_outranks(solar, payback))
    return Evaluation(
        profile=profile,
        tariff=tariff,
        config=config,
        install_date=install_date,
        year=year,
        dispatch=dispatched,
        battery_kwh=size,
        rebate=rebate,
        payback=payback,
        unmodelled=unmodelled,
        rule_tests=rule_tests,
        battery_action=battery_action,
        solar=solar,
        action=SOLAR_FIRST if outranked else battery_action,
    )


def decide(payback: Payback, rule_tests: tuple[RuleTest, ...], unmodelled: dict[str, str]) -> str:
    """The decision rule, as the answer about a battery."""
    if unmodelled:
        return CANNOT_ASSESS    # an unpriced charge could move the answer
    if payback.annual_saving <= 0:
        return NO_ACTION        # the battery saves nothing to pay itself back from
    if any(test.passed is False for test in rule_tests):
        return BATTERY_NOT_YET  # a failed limit decides it, whatever else is missing
    if any(test.passed is None for test in rule_tests):
        return CANNOT_ASSESS    # nothing failed, but the answer turns on a missing limit
    return BATTERY_NOW


def compare_solar(profile: HouseholdProfile, tariff: Tariff, config: Config) -> SolarComparison | None:
    """The coarse comparison behind solar_first; None where there is solar already."""
    if profile.has_solar:
        return None
    assumptions = config.assumptions
    indicative = assumptions.indicative_solar
    midday = window_by_slot(tariff)[SLOTS_PER_DAY // 2]  # the window covering 12:00
    generation = indicative.kw * assumptions.solar_yield_kwh_per_kw
    self_consumed = min(generation * indicative.self_consumption,
                        profile.annual_kwh_by_window[midday.name])
    feed_in = tariff.feed_in_tariff_aud_per_kwh
    if feed_in is None:
        feed_in = assumptions.new_solar_feed_in_aud_per_kwh
    saving = self_consumed * midday.rate_aud_per_kwh + (generation - self_consumed) * feed_in
    installed = indicative.kw * indicative.installed_aud_per_kw
    return SolarComparison(
        kw=indicative.kw,
        generation_kwh=generation,
        self_consumed_kwh=self_consumed,
        exported_kwh=generation - self_consumed,
        midday_window=midday.name,
        midday_rate=midday.rate_aud_per_kwh,
        feed_in=feed_in,
        annual_saving=saving,
        installed_cost=installed,
        payback_years=installed / saving if saving > 0 else math.inf,
        payback_ratio=indicative.payback_ratio,
    )


def _solar_outranks(solar: SolarComparison, payback: Payback) -> bool:
    """Materially better, not merely better: solar's indicative payback is at most
    payback_ratio of the battery's."""
    return (math.isfinite(solar.payback_years)
            and solar.payback_years <= solar.payback_ratio * payback.payback_years)


def rule_test(limit: str, payback_years: float, limit_years) -> RuleTest:
    if limit_years is None:
        return RuleTest(limit=limit, limit_years=None, passed=None, margin_years=None)
    limit_years = float(limit_years)
    return RuleTest(limit=limit, limit_years=limit_years, passed=payback_years <= limit_years,
                    margin_years=payback_years - limit_years)
