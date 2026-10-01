"""The follow-up chat: one question about a recommendation, one of three outcomes.

    explain         answered from the recommendation as it stands; nothing re-run
    counterfactual  one parameter changed, the whole path re-run (recommend_for),
                    and the answer given from the recommendation before and after
    out_of_scope    declined, with the reason, in words fixed here

The model does two things and decides nothing. It classifies, proposing an
intent -- and for a counterfactual a parameter and a value -- as JSON; and it
writes the answer. Python checks the proposal against the closed set below: a
reply that does not parse, or names anything outside the set, is out of scope.
Every answer passes the same untraceable-figure check as the justification.

The closed set is three counterfactuals: adding an EV, adding solar, and a
different battery size. Two more are declined with the reason, never
approximated: changing tariff type, because a time-of-use bill gives use by
window and a flat bill gives none, so switching means re-bucketing consumption
the bill does not describe; and price growth, because prices are held flat and
growth was cut from the sweep set.
"""

import json
import re
from dataclasses import dataclass, replace
from datetime import date

from src import llm
from src.config import Config
from src.explain import answer, compare, comparison_request, question_request
from src.profile import QUESTION, HouseholdProfile
from src.recommend import NotAssessed, Recommendation, recommend_for

COUNTERFACTUALS = ("add_ev", "add_solar", "battery_size")

DECLINED = {
    "tariff_type": "Changing tariff type can't be re-run. A time-of-use bill gives your use by "
                   "window and a flat bill gives none, so switching would mean re-bucketing "
                   "consumption the bill does not describe.",
    "price_growth": "Price growth can't be re-run. This version holds electricity prices at "
                    "today's rates, and growth over time is not modelled.",
    "retailer_or_brand": "Retailers, brands and products are outside what this assistant covers. "
                         "It assesses a battery against your own bill; it doesn't compare "
                         "products or say where to buy.",
    "other": "That isn't something this assessment can answer. It can explain the result, or "
             "re-run it with an EV added, solar added, or a different battery size.",
}

CLASSIFIER = """\
You sort one question a household asks about its home-battery assessment into one of three \
kinds. Reply with a single JSON object and nothing else: no code fence, no explanation.

{"intent": "explain"}
  The question asks about the assessment as it stands: why, how, what a figure means, what \
was assumed, what is left out.

{"intent": "counterfactual", "parameter": P, "value": V}
  The question asks what would happen if one of these changed. P is one of:
  "add_ev": the household gets an electric vehicle. V is the EV's charging in kWh a year, only \
if the question states it in kWh a year; otherwise null.
  "add_solar": the household adds solar panels. V is the system's size in kW, only if the \
question states it in kW; otherwise null.
  "battery_size": a battery of a different size. V is its usable capacity in kWh, only if the \
question states it in kWh; otherwise null.

{"intent": "out_of_scope", "topic": T}
  Anything else. T is one of:
  "tariff_type": moving to a different kind of tariff, such as flat, time-of-use or demand.
  "price_growth": electricity prices rising or falling over time.
  "retailer_or_brand": retailers, brands, products, installers, or where to buy.
  "other": anything else.

Copy a number exactly as the question states it; never convert or calculate one. If the \
question fits none of these, or you are not sure, reply {"intent": "out_of_scope", "topic": \
"other"}."""


@dataclass(frozen=True)
class Proposal:
    """What the classifier proposed, as far as Python accepts it."""

    intent: str                   # "explain", "counterfactual" or "out_of_scope"
    parameter: str | None = None  # a counterfactual's: one of COUNTERFACTUALS
    value: float | None = None    # the number the question states, if any
    topic: str | None = None      # out of scope: a key of DECLINED


@dataclass(frozen=True)
class Reply:
    question: str
    intent: str
    text: str                                # the model's answer, checked; or the decline
    changed: str | None = None               # a counterfactual: what was changed and re-run
    sheet: str | None = None                 # what the model answered from; None if declined
    after: Recommendation | None = None      # a counterfactual: the re-run recommendation


def ask(question: str, before: Recommendation, profile: HouseholdProfile, config: Config,
        install_date: date, live: bool = False) -> Reply:
    """One question about `before`, the recommendation for `profile`. `live` lets the
    two model calls reach the provider when their replies are not cached."""
    proposal = classify(question, live=live)
    if proposal.intent == "explain":
        return Reply(question, "explain", answer(before, question, live=live),
                     sheet=question_request(before, question))
    if proposal.intent == "counterfactual":
        return _counterfactual(question, proposal, before, profile, config, install_date, live)
    return Reply(question, "out_of_scope", DECLINED[proposal.topic])


def classify(question: str, live: bool = False) -> Proposal:
    reply = llm.complete(CLASSIFIER, question, live=live, about=f"classifying {question!r}")
    return accepted(reply)


def accepted(reply: str) -> Proposal:
    """The classifier's reply, checked against the closed set. Anything that does not
    parse, or does not fit, is out of scope: the chat fails closed."""
    other = Proposal("out_of_scope", topic="other")
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", reply.strip(), re.DOTALL)
    try:
        proposal = json.loads(fenced.group(1) if fenced else reply)
    except ValueError:
        return other
    if not isinstance(proposal, dict):
        return other
    intent = proposal.get("intent")
    if intent == "explain":
        return Proposal("explain")
    if intent == "counterfactual":
        parameter, value = proposal.get("parameter"), proposal.get("value")
        if parameter in ("tariff_type", "price_growth"):
            return Proposal("out_of_scope", topic=parameter)
        numeric = value is None or (isinstance(value, (int, float)) and not isinstance(value, bool))
        if parameter not in COUNTERFACTUALS or not numeric:
            return other
        return Proposal("counterfactual", parameter=parameter, value=value)
    if intent == "out_of_scope":
        topic = proposal.get("topic")
        return Proposal("out_of_scope", topic=topic if topic in DECLINED else "other")
    return other


def _counterfactual(question: str, proposal: Proposal, before: Recommendation,
                    profile: HouseholdProfile, config: Config, install_date: date,
                    live: bool) -> Reply:
    change = changed_inputs(proposal, profile, config)
    if isinstance(change, str):
        return Reply(question, "out_of_scope", change)
    changed, new_profile, new_config = change
    after = recommend_for(new_profile, new_config, install_date)
    if isinstance(after, NotAssessed):
        return Reply(question, "out_of_scope", after.reason)
    return Reply(question, "counterfactual", compare(before, after, changed, question, live=live),
                 changed=changed, sheet=comparison_request(before, after, changed, question),
                 after=after)


def changed_inputs(proposal: Proposal, profile: HouseholdProfile,
                   config: Config) -> tuple[str, HouseholdProfile, Config] | str:
    """(what changed, the profile and config to re-run with), or why it cannot run."""
    value = proposal.value
    if proposal.parameter == "add_ev":
        if profile.form.get("ev") == "have":
            return ("You already have an EV, so its charging is in your bills. Adding another "
                    "isn't something this version re-runs.")
        kwh = config.assumptions.ev_kwh_per_year if value is None else float(value)
        low, high = config.assumptions.revisit_ranges.added_ev_kwh_per_year
        if not low < kwh <= high:
            return (f"EV charging can be re-run up to {high:,.0f} kWh a year; {kwh:,.0f} kWh a "
                    "year is outside that.")
        typical = " (a typical EV)" if value is None else ""
        return (f"an EV added: {kwh:,.0f} kWh a year of overnight charging the bills do not "
                f"contain{typical}", replace(profile, added_ev_kwh_per_year=kwh), config)
    if proposal.parameter == "add_solar":
        if profile.has_solar:
            return ("You already have solar, scaled to the export on your bill. Adding more "
                    "panels isn't something this version re-runs.")
        kw = config.assumptions.indicative_solar.kw if value is None else float(value)
        low, high = QUESTION["solar_kw"].bounds
        if not low <= kw <= high:
            return f"Solar can be re-run from {low:g} to {high:g} kW; {kw:g} kW is outside that."
        with_solar = replace(profile, form={**profile.form, "has_solar": True, "solar_kw": kw},
                             annual_solar_export_kwh=None)
        return (f"{kw:g} kW of solar added, generating an assumed "
                f"{config.assumptions.solar_yield_kwh_per_kw:,.0f} kWh per kW a year "
                "(INDICATIVE: an assumed yield, not a PV model)", with_solar, config)
    low, high = config.batteries.cost_model.valid_range_kwh
    if value is None:
        return (f"Which size? A different battery can be re-run for any usable capacity from "
                f"{low:g} to {high:g} kWh.")
    if not low <= value <= high:
        return (f"The battery price model covers {low:g} to {high:g} kWh of usable capacity; "
                f"{value:g} kWh is outside it.")
    resized = replace(config, batteries=replace(config.batteries, default_size_kwh=float(value)))
    return (f"a {value:g} kWh battery instead of {config.batteries.default_size_kwh:g} kWh",
            profile, resized)
