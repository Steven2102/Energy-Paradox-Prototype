"""The justification: prose for the household, written by a language model from a
Recommendation and nothing else (hard rules 1 and 3).

The model never calculates. fact_sheet() turns the recommendation into text with
every figure already computed and formatted, here in Python; the system prompt
forbids any number that is not in it; and before a reply is returned, every
figure in it is checked against its input. A reply using a figure the input
does not contain is refused, never shown.

Nor does the model decide anything. The recommendation, the limits the payback
failed, the thresholds -- including whether a threshold changes the advice or
only the label -- and what is left unpriced all arrive decided.

explain() writes the justification. For the follow-up chat (src/followup.py),
answer() replies to one question about the same recommendation, and compare()
to a what-if, from the recommendation before and after the change -- refusing
a reply that leaves out the payback either side. All go through src/llm.py,
which caches every reply on disk.
"""

import math
import re

from src import llm
from src.engine import BATTERY_NOT_YET, BATTERY_NOW, CANNOT_ASSESS, NO_ACTION, SOLAR_FIRST
from src.recommend import Recommendation, quantity
from src.revisit import Revisit, describe_value

LABELS = {
    BATTERY_NOW: "Battery now",
    BATTERY_NOT_YET: "Battery not yet",
    SOLAR_FIRST: "Solar first",
    NO_ACTION: "No action",
    CANNOT_ASSESS: "Cannot assess",
}

PRIORITIES = {
    "lower_bill": "a lower bill",
    "independence": "independence from retailer price rises",
    "backup_power": "backup power in an outage",
}

# What leaving an unpriced charge out does to a battery's value, where that is known.
UNPRICED_EFFECT = {
    "demand": "A battery can lower a demand charge, so leaving it out may understate what a "
              "battery is worth.",
}

SYSTEM_PROMPT = """\
You explain a home-battery assessment to the household it was made for, in South East \
Queensland, Australia. A separate model computed the assessment from the household's bills and \
form answers. The user message gives you its fact sheet, which is everything you know about this \
household.

Rules. They are not negotiable.
1. Use only the numbers provided. Quote each figure exactly as the fact sheet writes it, with its \
unit. Do not calculate, estimate, round, convert or combine figures, and introduce no number of \
your own: no monthly amounts, totals, percentages, times of day or costs for other battery sizes \
that the fact sheet does not state.
2. If a figure is not in the fact sheet, say that it is not available, rather than supplying one.
3. Report the assessment; do not make it. The recommendation, the limits the payback failed, the \
thresholds, what is left unpriced and how certain the result is are all decided in the fact sheet: \
add no advice or judgement of your own. Name no brands, products or retailers and no file names or \
codes, and do not mention the fact sheet itself.
4. Where the fact sheet says ADVICE CHANGES: NO, only the label moves. Never say that the answer \
or the recommendation changes there. Say that the label would move but the advice would stay the \
same, and give the reason the fact sheet gives.
5. Anything marked INDICATIVE is a rough comparison, not a precise payback. Say so whenever you \
use it.
6. Anything marked PARTIAL leaves out a charge on the bill. Say so, and never present a partial \
figure as the answer.

To write the justification: plain English for a homeowner, about 400 words and never more than \
500. Write \
prose only, in short paragraphs of full sentences: no bullet points, no numbered lists, no bold, \
no headings and no tables. Where you mention several items, write them into a sentence, joined by \
commas, never set out one to a line. Name the recommendation by its label, in quotation marks, \
exactly as the fact sheet writes it. Cover, in this order:
- the recommendation and the main reason for it;
- the decision rule: which limit the payback failed and by how much, or why a limit could not \
be tested;
- how the saving is worked out, walking through the calculation with the fact sheet's figures;
- what it costs now against where the household stands later: year 1, a later year, and the \
crossover year;
- what the payback leaves out, starting with anything the household said matters to it;
- what would change the assessment, and what was checked and would not;
- the two or three assumptions that matter most, any unanswered questions, and how certain the \
result is.

To answer a question: answer from the fact sheet alone, in at most four sentences."""

JUSTIFY = "Write the justification for the household this fact sheet describes."

NUMBER = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")


class UntraceableFigure(ValueError):
    """A reply used a figure its input does not contain."""


class IncompleteAnswer(ValueError):
    """A what-if reply left out a figure it must give."""


def explain(recommendation: Recommendation) -> str:
    """The justification, for the household the recommendation was computed for."""
    message = justification_request(recommendation)
    return _checked(llm.complete(SYSTEM_PROMPT, message, about=recommendation.household), message)


def answer(recommendation: Recommendation, question: str, live: bool = False) -> str:
    """A reply to one question, from the recommendation alone."""
    message = question_request(recommendation, question)
    return _checked(llm.complete(SYSTEM_PROMPT, message, live=live,
                                 about=f"{recommendation.household}, asked {question!r}"),
                    message)


def compare(before: Recommendation, after: Recommendation, changed: str, question: str,
            live: bool = False) -> str:
    """A reply to a what-if question, from the recommendation before the change and
    after it. It must give the battery payback both before and after."""
    message = comparison_request(before, after, changed, question)
    reply = _checked(llm.complete(SYSTEM_PROMPT, message, live=live,
                                  about=f"{before.household}, asked {question!r}"), message)
    missing = [figure for figure in dict.fromkeys(_payback_figure(rec) for rec in (before, after))
               if figure not in reply]
    if missing:
        raise IncompleteAnswer(f"refusing an answer that leaves out the payback before or after "
                               f"the change: {', '.join(missing)}")
    return reply


def comparison_request(before: Recommendation, after: Recommendation, changed: str,
                       question: str) -> str:
    return (f"{comparison_sheet(before, after, changed)}\n\nTHE HOUSEHOLD ASKS\n{question}\n\n"
            "Answer the question from this comparison, briefly: at most four sentences and under "
            "100 words, giving the battery payback before and after the change, both exactly as "
            "written.")


def comparison_sheet(before: Recommendation, after: Recommendation, changed: str) -> str:
    """The figures a what-if changes, before and after, every one formatted here."""
    def charged(rec: Recommendation) -> str:
        split = rec.charged_from
        return (f"{_kwh(split.total_kwh)}, {split.solar_share:.0%} solar surplus and "
                f"{split.grid_share:.0%} grid import")

    def crossover(rec: Recommendation) -> str:
        year = rec.long_term.crossover_year
        return f"year {year}" if year else "never"

    rows = [
        ("Recommendation", f'"{LABELS[before.action]}"', f'"{LABELS[after.action]}"'),
        ("Battery size assessed", f"{before.evaluated_kwh:g} kWh", f"{after.evaluated_kwh:g} kWh"),
        ("Battery payback", _years(before.basis["payback_years"]),
         _years(after.basis["payback_years"])),
        ("Annual saving", _money(before.basis["annual_saving"]), _money(after.basis["annual_saving"])),
        ("Net outlay, the installed price less the rebate", _money(before.near_term.net_outlay),
         _money(after.near_term.net_outlay)),
        ("kWh the battery moves in a year", _kwh(before.basis["kwh_shifted"]),
         _kwh(after.basis["kwh_shifted"])),
        ("What the battery charged with in a year", charged(before), charged(after)),
        ("Crossover, the first year the position is no longer negative", crossover(before),
         crossover(after)),
    ]
    rows += [(_limit(old), _verdict(old), _verdict(new))
             for old, new in zip(before.rule_tests, after.rule_tests)]
    lines = [f"BEFORE AND AFTER: {changed}. Everything was re-run through the whole model."]
    lines += [f"- {name}: {old} before; {new} after." for name, old, new in rows]
    if after.unmodelled:
        lines.append("- PARTIAL, before and after: a charge on the bill is not priced, so these "
                     "figures cover energy only and cannot decide the answer.")
    return "\n".join(lines)


def _payback_figure(rec: Recommendation) -> str:
    """The payback as a reply must carry it: the number, or "never"."""
    years = rec.basis["payback_years"]
    return "never" if math.isinf(years) else f"{years:.1f}"


def _limit(test) -> str:
    return "Warranty" if test.limit == "warranty" else "Expected years in the home"


def _verdict(test) -> str:
    if test.passed is None:
        return "not given, so not tested"
    where = "inside" if test.passed else "outside"
    return f"the payback is {where} its {test.limit_years:g} years by {_years(abs(test.margin_years))}"


def justification_request(recommendation: Recommendation) -> str:
    return f"{fact_sheet(recommendation)}\n\n{JUSTIFY}"


def question_request(recommendation: Recommendation, question: str) -> str:
    return (f"{fact_sheet(recommendation)}\n\nTHE HOUSEHOLD ASKS\n{question}\n\n"
            "Answer the question briefly, answering only what was asked: at most four sentences "
            "and under 100 words.")


def untraceable(reply: str, source: str) -> list[str]:
    """Figures in the reply that appear nowhere in its source, in order of first use."""
    known = {_number(token) for token in NUMBER.findall(source)}
    return list(dict.fromkeys(token for token in NUMBER.findall(reply)
                              if _number(token) not in known))


def _checked(reply: str, source: str) -> str:
    figures = untraceable(reply, source)
    if figures:
        raise UntraceableFigure(f"refusing a reply with figures its input does not contain: "
                                f"{', '.join(figures)}")
    return reply


def _number(token: str) -> float:
    return float(token.replace(",", ""))


# ----------------------------------------------------------------- fact sheet

def fact_sheet(recommendation: Recommendation) -> str:
    """Everything the model may say about this household, every figure formatted."""
    rec = recommendation
    sections = [
        _recommendation(rec),
        _decision_rule(rec),
        _arithmetic(rec),
        _position(rec),
        _not_priced(rec),
        _revisits(rec),
        _assumptions(rec),
        _unanswered(rec),
        _priorities_and_certainty(rec),
    ]
    return "FACT SHEET\n\n" + "\n\n".join(section for section in sections if section)


def _recommendation(rec: Recommendation) -> str:
    lines = [f'RECOMMENDATION: "{LABELS[rec.action]}"']
    battery = rec.basis["payback_years"]
    if rec.action == SOLAR_FIRST:
        lines.append(f'- The battery on its own: "{LABELS[rec.battery_action]}".')
        lines.append(f"- Solar comes first because it looks materially better than the battery: "
                     f"solar's INDICATIVE payback is {_about(rec.solar.payback_years)}, against "
                     f"the battery's {_years(battery)}. Solar comes first only where its payback "
                     f"is at most {rec.solar.payback_ratio:g} of the battery's.")
    elif rec.action == BATTERY_NOW:
        lines.append("- The payback falls within both limits of the decision rule.")
    elif rec.action == BATTERY_NOT_YET:
        lines.append("- The payback falls outside at least one limit of the decision rule.")
    elif rec.action == NO_ACTION:
        lines.append("- The battery would save nothing: no energy is cheaper to store than to buy "
                     "when it would be used.")
    elif rec.unmodelled:
        for component, description in rec.unmodelled.items():
            lines.append(f"- The bill has a {component} charge this version does not price: "
                         f"{description}. {UNPRICED_EFFECT.get(component, '')}".rstrip())
        lines.append("- Every battery figure below covers energy only. It is PARTIAL and cannot "
                     "decide the answer.")
    else:
        lines.append("- The payback is inside the warranty, but whether a battery is worth it "
                     "turns on how long the household expects to stay, which the form did not give.")
    size = rec.size_kwh if rec.size_kwh is not None else rec.evaluated_kwh
    lines.append(f"- Battery size assessed: {size:g} kWh of usable capacity.")
    if rec.comfortable_spend is not None:
        spend = rec.comfortable_spend
        lines.append(f"- Its net outlay, {_money(spend.net_outlay)}, is {_money(spend.over_by)} more "
                     f"than the {_money(spend.stated)} the household said it is comfortable "
                     "spending upfront. Declared, not weighed: it does not change the "
                     "recommendation.")
    if rec.solar is not None and rec.action != SOLAR_FIRST:
        solar = f"- Solar, for comparison: INDICATIVE payback {_about(rec.solar.payback_years)}."
        if rec.battery_action == CANNOT_ASSESS:
            lines.append(f"{solar} It is not weighed against the battery here, because the "
                         "battery could not be assessed.")
        else:
            lines.append(f"{solar} It does not come first: that needs its payback to be at most "
                         f"{rec.solar.payback_ratio:g} of the battery's.")
    return "\n".join(lines)


def _decision_rule(rec: Recommendation) -> str:
    partial = ", energy only (PARTIAL)" if rec.unmodelled else ""
    lines = ["THE DECISION RULE",
             "A battery is recommended now only if its payback falls within both the battery's "
             "warranty and the years the household expects to stay in its home.",
             f"- The battery's payback{partial}: {_years(rec.basis['payback_years'])}."]
    names = {"warranty": "Warranty", "expected stay": "Expected years in the home"}
    for test in rec.rule_tests:
        if test.passed is None:
            effect = _missing_effect(rec, "years_expected_in_home")
            lines.append(f"- {names[test.limit]}: not given on the form, so this limit could not "
                         f"be tested. Effect: {effect}.")
        else:
            verdict = "PASSED" if test.passed else "FAILED"
            where = "inside" if test.passed else "outside"
            lines.append(f"- {names[test.limit]}: {test.limit_years:g} years. {verdict}: the payback "
                         f"is {where} it by {_years(abs(test.margin_years))}.")
    if rec.unmodelled:
        lines.append("- While a charge on the bill is unpriced, these limits cannot decide the "
                     "answer.")
    return "\n".join(lines)


def _arithmetic(rec: Recommendation) -> str:
    basis, limit = rec.basis, rec.limited_by
    lines = ["HOW THE SAVING AND THE PAYBACK ARE WORKED OUT",
             "annual saving = kWh moved × (value on release − cost to store ÷ efficiency) "
             "+ demand saving",
             "payback = (installed price − rebate) ÷ annual saving",
             f"- kWh the battery moves in a year: {_kwh(basis['kwh_shifted'])}"]
    charged = rec.charged_from
    if charged.total_kwh > 0:
        lines.append(f"- what the battery charged with in a year: {_kwh(charged.total_kwh)}, of "
                     f"which {_kwh(charged.solar_kwh)} ({charged.solar_share:.0%}) was solar "
                     f"surplus that would otherwise have been exported and "
                     f"{_kwh(charged.grid_kwh)} ({charged.grid_share:.0%}) was imported from "
                     "the grid")
    if basis["kwh_shifted"] > 0:
        per_kwh_delivered = basis["r_in"] / basis["efficiency"]
        lines += [
            f"- value on release, the average rate where the battery discharged: "
            f"{_cents(basis['r_out'])}",
            f"- cost to store, the average cost of what it charged with (the import rate when "
            f"charging from the grid, the feed-in tariff forgone when charging from solar): "
            f"{_cents(basis['r_in'])}",
            f"- round-trip efficiency: {basis['efficiency']:.0%}",
            f"- cost to store per kWh delivered, after losses (cost to store ÷ efficiency): "
            f"{_cents(per_kwh_delivered)}",
            f"- margin on each kWh moved (value on release − cost per kWh delivered): "
            f"{_cents(basis['r_out'] - per_kwh_delivered)}",
        ]
    demand = " (not calculated: the demand charge is not priced, so this is PARTIAL)" \
        if "demand" in rec.unmodelled else ""
    lines += [
        f"- demand saving: {_money(basis['demand_saving'])}{demand}",
        f"- annual saving: {_money(basis['annual_saving'])}",
        f"- installed price: {_money(basis['battery_cost'])}; rebate: {_money(basis['rebate'])}; "
        f"net outlay: {_money(rec.near_term.net_outlay)}",
        f"- payback: {_years(basis['payback_years'])}",
    ]
    # The most delivered in a day shows a battery bigger than the household's use. Above
    # capacity it only means a day caught the end of one charge and the next, so it is left out.
    held = (f"full on {limit.days_full} days of the year, with {limit.unmet_kwh:,.0f} kWh worth "
            f"serving left unmet")
    if limit.max_daily_kwh < limit.capacity_kwh:
        detail = (f"The most it delivered in a day was {limit.max_daily_kwh:.1f} kWh, against "
                  f"{limit.capacity_kwh:g} kWh of capacity; it was {held}.")
    else:
        detail = f"It was {held}."
    lines.append(f"- What limits the battery's value: {limit.label}. {detail}")
    return "\n".join(lines)


def _position(rec: Recommendation) -> str:
    near, long_term = rec.near_term, rec.long_term
    lines = ["NOW AGAINST LATER: the running position, savings to date minus the net outlay",
             f"- Year 1: {_money(near.position_after_year_1)} (the net outlay of "
             f"{_money(near.net_outlay)}, less the first year's saving of "
             f"{_money(near.first_year_saving)})"]
    lines += [f"- Year {year}: {_money(position)}" for year, position in long_term.positions[1:]]
    crossover = (f"year {long_term.crossover_year}" if long_term.crossover_year
                 else "never: the battery saves nothing")
    lines += [f"- Crossover, the first year the position is no longer negative: {crossover}",
              "- Every year is at today's prices: no growth."]
    return "\n".join(lines)


def _not_priced(rec: Recommendation) -> str:
    lines = ["WHAT THE PAYBACK LEAVES OUT (not priced)"]
    for item in sorted(rec.not_priced, key=lambda item: item.stated_priority is None):
        flag = " THE HOUSEHOLD SAID THIS MATTERS TO IT." if item.stated_priority else ""
        lines.append(f"- {item.item}.{flag} {item.detail}")
    return "\n".join(lines)


def _revisits(rec: Recommendation) -> str:
    lines = ["WHAT WOULD CHANGE THE ASSESSMENT: each found by re-running the model"]
    if rec.unmodelled:
        lines.append(f'While a charge on the bill is unpriced, none of these can move the answer '
                     f'from "{LABELS[CANNOT_ASSESS]}".')
    lines += [f"- {_revisit(rec, revisit)}" for revisit in rec.revisit_if]
    return "\n".join(lines)


def _revisit(rec: Recommendation, revisit: Revisit) -> str:
    rate = revisit.unit.split("added to the ")[-1]  # "peak rate", "flat rate"
    names = {"battery cost": "Battery price (installed, before the rebate)",
             "feed-in tariff": "Feed-in tariff",
             "rate spread": f"{rate.capitalize()} (c/kWh added to it)",
             "adding solar": "Adding solar (kW of panels)",
             "adding an EV": "Adding an EV (overnight charging the bills do not contain)"}
    name = names[revisit.parameter]
    if revisit.not_applicable:
        return f"{name}: not applicable, because {revisit.not_applicable}."
    name += f", now {describe_value(revisit, revisit.current)}"
    if revisit.to_action is None:
        low, high = (describe_value(revisit, value) for value in revisit.searched)
        tracked = "the battery answer" if revisit.tracks == "battery_action" else "the answer"
        span = (f"with anything up to {high}" if revisit.searched[0] == revisit.current
                else f"anywhere from {low} to {high}")
        return (f"{name}: no change in {tracked} {span}, the range searched; it stays "
                f'"{LABELS[revisit.from_action]}".{_planned(revisit)}')
    at = describe_value(revisit, revisit.threshold)
    before, after = revisit.either_side
    moves = f'"{LABELS[revisit.from_action]}" to "{LABELS[revisit.to_action]}"'
    if revisit.advice_changes:
        return (f"{name}: at {at} the answer changes from {moves}. ADVICE CHANGES: YES. The "
                f"battery's payback at {at} is {_years(after.payback_years)}.{_planned(revisit)}")
    text = (f'{name}: at {at} the label moves from {moves}. ADVICE CHANGES: NO. The battery answer '
            f'is "{LABELS[after.battery_action]}" on both sides; its payback at {at} is '
            f"{_years(after.payback_years)}, and the warranty is "
            f"{rec.rule_tests[0].limit_years:g} years.")
    if before.solar_payback_years is not None and after.solar_payback_years is not None:
        shares = [side.solar_payback_years / side.payback_years for side in (before, after)]
        share = (f"{shares[0]:.2f} of the battery's on both sides"
                 if f"{shares[0]:.2f}" == f"{shares[1]:.2f}"
                 else f"{shares[0]:.2f} and then {shares[1]:.2f} of the battery's")
        text += (f" Only solar's lead moves. Solar's INDICATIVE payback, "
                 f"{_about(after.solar_payback_years)}, is {share}; solar comes first only while "
                 f"that is {rec.solar.payback_ratio:g} or less. The label moves because solar's lead "
                 f"narrows past that cut-off, not because the answer about a battery changes.")
    return text + _planned(revisit)


def _planned(revisit: Revisit) -> str:
    """The household's own plan, as a named, dated condition, where it states one."""
    plan = revisit.planned
    if plan is None:
        return ""
    when = f"roughly {plan.year}" if plan.year is not None else "when is not given"
    then = "stays" if plan.action_then == revisit.from_action else "becomes"
    text = (f" THE HOUSEHOLD PLANS AN EV ({when}). With a typical EV's charging, "
            f'{describe_value(revisit, plan.at)}, the answer {then} "{LABELS[plan.action_then]}".')
    if plan.inside_stay is not None:
        where = "inside" if plan.inside_stay else "after the end of"
        text += (f" {plan.year} lands {where} the stated stay, which runs to about "
                 f"{int(plan.stay_ends)}.")
    elif plan.year is None:
        text += " With no year given, whether it lands inside the stay cannot be said."
    else:
        text += " The stay was not given, so whether it lands inside it cannot be said."
    return text


def _assumptions(rec: Recommendation) -> str:
    lines = ["ASSUMPTIONS THAT SHAPED THE RESULT"]
    for item in rec.assumptions:
        if item.value is None:
            continue
        dated = f"; dated {item.date.isoformat()}" if item.date else ""
        ranged = (f"; plausible range {quantity(item.range[0], item.unit)} to "
                  f"{quantity(item.range[1], item.unit)}" if item.range else "")
        lines.append(f"- {item.name}: {quantity(item.value, item.unit)} "
                     f"(source: {item.source}{dated}){ranged}")
    return "\n".join(lines)


def _unanswered(rec: Recommendation) -> str:
    missing = [item for item in rec.assumptions
               if item.value is None and item.name != "years_expected_in_home"]
    if not missing:
        return ""
    lines = ["FORM QUESTIONS NOT ANSWERED (the stay is under THE DECISION RULE)"]
    lines += [f"- {item.name}: {item.source}. Effect: {item.effect}." for item in missing]
    return "\n".join(lines)


def _priorities_and_certainty(rec: Recommendation) -> str:
    lines = ["THE HOUSEHOLD'S STATED PRIORITIES"]
    for driver in rec.drivers:
        if driver.priority is None:
            lines.append(f"- {driver.effect}")
        else:
            lines.append(f"- {PRIORITIES.get(driver.priority, driver.priority)}: {driver.effect}")
    lines += ["", "HOW CERTAIN THIS IS"]
    if rec.time_of_day_source == "form":
        lines.append("- The bill has one flat rate and no time-of-day split, so how this "
                     "household's use falls across the day comes from its form answers, not its "
                     "bills. This result is materially less certain than one built from a "
                     "time-of-use bill.")
    else:
        lines.append("- The bill splits use into time-of-use windows, so how much falls in each "
                     "window is billed, not estimated; only the shape within each window is "
                     "modelled.")
    lines.append("- The figures come from one representative year, carried forward at today's "
                 "prices. A different year's use or prices would give different figures: they are "
                 "not a forecast.")
    if rec.unmodelled:
        lines.append("- PARTIAL: a charge on the bill is not priced (see RECOMMENDATION).")
    return "\n".join(lines)


def _missing_effect(rec: Recommendation, name: str) -> str:
    return next(item.effect for item in rec.assumptions if item.name == name)


# ------------------------------------------------------------------ formatting

def _money(amount: float) -> str:
    return f"{'-' if amount < 0 else ''}${abs(amount):,.0f}"


def _years(years: float) -> str:
    return "never" if math.isinf(years) else f"{years:.1f} years"


def _about(years: float) -> str:
    return "never" if math.isinf(years) else f"about {years:.0f} years"


def _cents(aud_per_kwh: float) -> str:
    return f"{aud_per_kwh * 100:.2f} c/kWh"


def _kwh(kwh: float) -> str:
    return f"{kwh:,.0f} kWh"
