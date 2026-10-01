"""A household profile: what the bill says, and what the form asks.

A fixture in fixtures/ is a saved profile. Bill parsing (stage 6) will build
the bill half from uploaded bills, and the web app (app.py) builds the form
half; both produce this same structure. The bill half is a run of consecutive
billing periods with consumption by tariff window, because that is what bills
supply; the annual figures are sums over them.

The form half is declared here, in QUESTIONS, and nowhere else: the web app
draws its fields from that list, and every set of answers -- a fixture's or
the form's -- is checked against it. Each question names the computed output
its answer moves. A question that moves nothing computed does not belong on
the form, which is why there is no household size: the bill fixes the
magnitude, and occupancy and the appliances carry the shape.
"""

from dataclasses import dataclass, replace
from datetime import date, timedelta
from pathlib import Path

import yaml

from src.tariff import Tariff

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"

# Bill quantities are printed to two decimals and stated annual figures are
# rounded to whole kWh, so sums may miss a stated figure by a rounding error,
# and never by more.
RECONCILE_TOLERANCE_KWH = 1.0


# ----------------------------------------------------------------- the form

@dataclass(frozen=True)
class Question:
    """One question on the form, and the computed output its answer moves."""

    key: str                    # the answer's name in HouseholdProfile.form
    ask: str
    kind: str                   # "yes_no", "choice", "choices", "number" or "whole"
    moves: str                  # what the answer changes in the result; never empty
    choices: tuple[tuple[str, str], ...] = ()  # (value, label), for "choice" and "choices"
    unit: str | None = None
    bounds: tuple[float, float] | None = None  # inclusive, for "number" and "whole"
    asked_if: tuple[str, object] | None = None  # (key, answer): asked only after that answer
    required: bool = False      # the engine cannot run without it
    note: str | None = None     # shown with the question

    def problem_with(self, value) -> str | None:
        """Why the form could not have given this answer, or None."""
        values = [choice for choice, _ in self.choices]
        if self.kind == "yes_no":
            return None if isinstance(value, bool) else f"expected yes or no, got {value!r}"
        if self.kind == "choice":
            return None if value in values else f"expected one of {values}, got {value!r}"
        if self.kind == "choices":
            if not isinstance(value, (list, tuple)) or len(set(value)) != len(value):
                return f"expected a list of distinct choices, got {value!r}"
            unknown = [item for item in value if item not in values]
            return f"expected choices from {values}, got {unknown}" if unknown else None
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"expected a number, got {value!r}"
        if self.kind == "whole" and value != int(value):
            return f"expected a whole number, got {value!r}"
        low, high = self.bounds
        return None if low <= value <= high else f"{value!r} is outside {low:g} to {high:g}"


QUESTIONS: tuple[Question, ...] = (
    Question("has_solar", "Do you have solar panels?", "yes_no", required=True,
             moves="where the battery charges from: solar surplus as well as the grid. It also "
                   "takes solar first off the table."),
    Question("solar_kw", "How big is the solar system?", "number", unit="kW", bounds=(0.5, 40),
             asked_if=("has_solar", True),
             moves="how much the panels generate, where the bill shows no export to scale them to"),
    Question("has_battery", "Do you already have a home battery?", "yes_no",
             moves="whether this version can assess the household at all: it evaluates adding a "
                   "battery, so one already installed gives cannot_assess"),
    Question("battery_kwh", "How big is it?", "number", unit="kWh", bounds=(1, 100),
             asked_if=("has_battery", True),
             moves="the reason stated with cannot_assess"),
    Question("home_during_the_day", "Is someone usually home during the day on weekdays?",
             "yes_no",
             moves="the shape of each weekday's use: home all day, or out during the day"),
    Question("work_from_home_days", "How many weekdays does someone work from home?", "whole",
             bounds=(0, 5), asked_if=("home_during_the_day", False),
             moves="how many weekdays take the at-home shape"),
    Question("aircon", "Air conditioning", "choice",
             choices=(("none", "None"), ("split", "Split system"), ("ducted", "Ducted")),
             moves="the hours that seasonal use above the household's quietest period is "
                   "placed in"),
    Question("pool_pump", "Pool pump", "choice",
             choices=(("none", "No pool pump"), ("main_circuit", "Yes, on the main circuit"),
                      ("controlled_load", "Yes, on the controlled load")),
             moves="a declaration: a pump on the main circuit is in the bill's totals, but its "
                   "running hours are not modelled"),
    Question("hot_water", "Hot water", "choice",
             choices=(("electric_controlled_load", "Electric, on the controlled load"),
                      ("electric_main_circuit", "Electric, on the main circuit"),
                      ("gas", "Gas"), ("solar", "Solar")),
             note="The controlled load is the separately metered circuit on the bill. A battery "
                  "cannot serve it.",
             moves="a declaration: electric hot water on the main circuit is in the bill's "
                   "totals, but its heating hours are not modelled"),
    Question("ev", "Electric vehicle", "choice",
             choices=(("none", "No"), ("have", "Yes, we have one"),
                      ("planning", "We're planning one")),
             moves="the adding-an-EV revisit: not applicable with an EV already, and a named, "
                   "dated condition with one planned. A planned EV is also declared as not "
                   "priced; one already owned, as charging whose hours are not modelled."),
    Question("ev_planned_year", "Roughly when?", "whole", unit="year", bounds=(2020, 2060),
             asked_if=("ev", "planning"),
             moves="the date on the planned-EV condition, and whether it lands inside the stay"),
    Question("years_expected_in_home", "How many more years do you expect to live here?",
             "number", unit="years", bounds=(0, 60),
             moves="the decision rule: a battery pays back within both the warranty and this"),
    Question("motivation", "What matters to you?", "choices",
             choices=(("lower_bill", "A lower bill"),
                      ("independence", "Independence from retailer price rises"),
                      ("backup_power", "Backup power in an outage")),
             note="Named, never weighted: nothing here enters the arithmetic.",
             moves="drivers, and which items not_priced flags as mattering to the household"),
    Question("max_upfront_aud", "The most you'd be comfortable spending upfront", "number",
             unit="$", bounds=(0, 200_000),
             note="Only compared with a recommended battery's price. Nothing is asked about "
                  "income, savings or assets.",
             moves="a declaration when a recommended battery's net outlay is more than this"),
)
QUESTION = {question.key: question for question in QUESTIONS}


def check_answers(answers: dict) -> None:
    """Refuse answers the form could not have given: a question it does not ask, a
    value outside a question's choices or bounds, a follow-up answered without the
    answer that asks it, or a required question left blank."""
    problems = [f"{key!r} is not a question on the form" for key in answers if key not in QUESTION]
    for question in QUESTIONS:
        value = answers.get(question.key)
        if value is None:
            if question.required:
                problems.append(f"{question.key} must be answered")
            continue
        if question.asked_if is not None:
            parent, trigger = question.asked_if
            if answers.get(parent) != trigger:
                problems.append(f"{question.key} is only asked when {parent} is {trigger!r}")
        problem = question.problem_with(value)
        if problem:
            problems.append(f"{question.key}: {problem}")
    if problems:
        raise ValueError("; ".join(problems))


# ------------------------------------------------------------------ the profile


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
    annual_solar_export_kwh: float | None       # from the bill; None where it shows none
    form: dict                                  # answers to QUESTIONS, as given
    added_ev_kwh_per_year: float = 0.0          # EV charging the bills do not contain (a sweep)

    @property
    def has_solar(self) -> bool:
        return self.form.get("has_solar") is True

    @property
    def solar_kw(self) -> float | None:
        return self.form.get("solar_kw")

    def with_answers(self, **answers) -> "HouseholdProfile":
        """The same household with some answers changed, checked like any other."""
        form = {**self.form, **answers}
        check_answers(form)
        return replace(self, form=form)

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
    form = {key: value for key, value in (raw.get("from_form") or {}).items() if value is not None}
    try:
        check_answers(form)
    except ValueError as error:
        raise ValueError(f"{name}: {error}") from None

    profile = HouseholdProfile(
        name=name,
        source=raw["meta"]["source"],
        tariff_ref=bill["tariff_ref"],
        billing_periods=periods,
        annual_solar_export_kwh=bill.get("annual_solar_export_kwh"),
        form=form,
    )
    problems = profile_problems(profile)
    if problems:
        raise ValueError(f"{name}: {'; '.join(problems)}")
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


def profile_problems(profile: HouseholdProfile) -> list[str]:
    """Where the form's answers and the bill cannot both be right, or leave the
    engine without what it needs."""
    problems = []
    if profile.has_solar and profile.solar_kw is None and profile.annual_solar_export_kwh is None:
        problems.append("there is solar, but neither its size nor its export is given")
    if not profile.has_solar and profile.annual_solar_export_kwh:
        problems.append("the bill shows solar export, but the form says there is no solar")
    return problems


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
