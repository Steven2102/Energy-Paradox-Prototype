"""The web app, step 1: the inputs, and the result they produce.

    streamlit run app.py

A bill upload, a fixture whose bill values stand in for it, and the form's
questions -- drawn from QUESTIONS in src/profile.py, never listed here, so the
form and the schema cannot drift apart. Bill parsing (stage 6) is not wired
up: an uploaded bill is acknowledged, not read, and the page says which values
are used instead.
"""

from datetime import date

import streamlit as st

from src.config import load_config
from src.explain import LABELS, IncompleteAnswer, UntraceableFigure, explain
from src.finance import show_equation
from src import followup, llm
from src.llm import LLMError
from src.profile import QUESTIONS, HouseholdProfile, load_fixtures, profile_problems
from src.recommend import NotAssessed, Recommendation, quantity, recommend_for
from src.revisit import Revisit, describe_value
from src.tariff import describe_hours

FLAT_TARIFF = (
    "Your bill has one flat rate and no time-of-day split, so it says how much you use but "
    "not when. How your use falls across the day comes almost entirely from your answers "
    "above. Treat this result as materially less certain than one from a time-of-use bill.")


@st.cache_resource
def fixtures() -> dict[str, HouseholdProfile]:
    return {profile.name: profile for profile in load_fixtures()}


@st.cache_resource
def config():
    return load_config()


@st.cache_data(show_spinner=False)
def assess(name: str, answers: tuple, install_date: date) -> Recommendation | NotAssessed:
    profile = fixtures()[name].with_answers(**dict(answers))
    return recommend_for(profile, config(), install_date)


def main() -> None:
    st.set_page_config(page_title="Home battery check", page_icon="🔋")
    live = st.sidebar.toggle(
        "Answer new questions live", value=False, key="live_answers",
        help="Follow-up questions not answered ahead of time go to the language-model "
             "provider, which can take a while. The explanation stays as it was written ahead "
             "of time either way.")
    st.title("Home battery check")
    st.write("Whether a home battery is worth it for your household, from your bill and a few "
             "questions. Every figure is calculated; the written explanation only reports them.")

    profile = bill_section()
    answers = question_section(profile)
    problems = profile_problems(profile.with_answers(**answers))
    for problem in problems:
        st.error(f"Before this can be assessed: {problem[0].upper()}{problem[1:]}.")

    install_date = date.today()
    key = (profile.name, tuple(sorted(answers.items(), key=lambda item: item[0])), install_date)
    if st.button("Assess", key="assess", type="primary", disabled=bool(problems)):
        st.session_state["assessed"] = key
    if st.session_state.get("assessed") is None:
        return
    if st.session_state["assessed"] != key:
        st.info("Your answers have changed since the last assessment. Press Assess to update it.")
        return
    with st.spinner("Running the model: a year of half-hours, and each threshold re-run…"):
        result = assess(*key)
    results_section(result, install_date)
    if isinstance(result, Recommendation):
        followup_section(result, profile.with_answers(**answers), key, live)


# ----------------------------------------------------------------------- bill

def bill_section() -> HouseholdProfile:
    st.header("1. Your bill")
    uploaded = st.file_uploader("Electricity bill (PDF)", type=["pdf"])
    names = list(fixtures())
    name = st.radio("Bill values to use", names, key="fixture",
                    index=names.index("reference_household"),
                    format_func=lambda name: f"{name}: {fixtures()[name].source}")
    profile = fixtures()[name]
    if uploaded is not None:
        st.warning(f"Received **{uploaded.name}** ({uploaded.size / 1024:,.0f} KB), but automatic "
                   f"reading of bills is not enabled yet, so it has not been read. The values "
                   f"below, from the {name} bills, are used instead. The file is not stored or "
                   f"sent anywhere.")
    else:
        st.info(f"Automatic reading of bills is not enabled yet. The values below, from the "
                f"{name} bills, are used.")
    with st.expander(f"Bill values in use: {name}", expanded=uploaded is not None):
        show_bill(profile)
    return profile


def show_bill(profile: HouseholdProfile) -> None:
    tariff = config().tariffs[profile.tariff_ref]
    periods = profile.billing_periods
    st.markdown(f"**Tariff** {tariff.id}, daily charge ${tariff.daily_charge_aud:.2f}")
    rows = [{"window": window.name, "hours": describe_hours(window.hours),
             "rate (c/kWh)": f"{window.rate_aud_per_kwh * 100:.2f}",
             "use in the year (kWh)": f"{profile.annual_kwh_by_window[window.name]:,.0f}"}
            for window in tariff.energy_windows]
    if profile.annual_controlled_load_kwh is not None:
        rows.append({"window": "controlled load", "hours": "separate circuit",
                     "rate (c/kWh)": f"{tariff.controlled_load.rate_aud_per_kwh * 100:.2f}",
                     "use in the year (kWh)": f"{profile.annual_controlled_load_kwh:,.0f}"})
    st.table(rows)
    st.markdown(f"**Billing periods** {len(periods)}, from {periods[0].start} to "
                f"{periods[-1].end}")
    if tariff.feed_in_tariff_aud_per_kwh is not None:
        st.markdown(f"**Feed-in tariff** {tariff.feed_in_tariff_aud_per_kwh * 100:.1f} c/kWh")
    if profile.annual_solar_export_kwh:
        st.markdown(f"**Solar export on the bill** {profile.annual_solar_export_kwh:,.0f} kWh "
                    "a year")


# ------------------------------------------------------------------ questions

def question_section(profile: HouseholdProfile) -> dict:
    """Every question in QUESTIONS, asked in order, each follow-up only after the
    answer that asks it. Defaults are the chosen fixture's answers."""
    st.header("2. Your household")
    answers = {}
    for question in QUESTIONS:
        if question.asked_if is not None:
            parent, trigger = question.asked_if
            if answers.get(parent) != trigger:
                answers[question.key] = None
                continue
        answers[question.key] = ask(question, profile.form.get(question.key),
                                    key=f"{profile.name}.{question.key}")
    return answers


def ask(question, default, key: str):
    label = question.ask + (f" ({question.unit})" if question.unit else "")
    help_text = f"{question.note + ' ' if question.note else ''}Moves: {question.moves}."
    if question.kind == "yes_no":
        options = [True, False] if question.required else [None, True, False]
        words = {None: "Not answered", True: "Yes", False: "No"}
        return st.radio(label, options, index=options.index(default) if default in options else 0,
                        key=key, horizontal=True, format_func=words.get, help=help_text)
    if question.kind == "choice":
        options = [None] + [value for value, _ in question.choices]
        words = {None: "Not answered", **dict(question.choices)}
        return st.selectbox(label, options, index=options.index(default), key=key,
                            format_func=words.get, help=help_text)
    if question.kind == "choices":
        words = dict(question.choices)
        chosen = st.multiselect(label, list(words), default=list(default or []), key=key,
                                format_func=words.get, help=help_text)
        return chosen or None
    number = int if question.kind == "whole" else float  # Streamlit refuses mixed types
    low, high = question.bounds
    value = st.number_input(label, min_value=number(low), max_value=number(high),
                            value=None if default is None else number(default),
                            step=number(1), placeholder="Not answered", key=key, help=help_text)
    return None if value is None else number(value)


# -------------------------------------------------------------------- results

def results_section(result: Recommendation | NotAssessed, install_date: date) -> None:
    st.header("3. Result")
    if isinstance(result, NotAssessed):
        st.error(f"**{LABELS[result.action]}.** {result.reason}")
        return
    rec = result
    st.subheader(f"{LABELS[rec.action]}")
    st.code(show_equation(rec.basis), language=None)
    if rec.time_of_day_source == "form":
        st.warning(FLAT_TARIFF)
    if rec.unmodelled:
        charges = "; ".join(f"{component}: {detail}" for component, detail in rec.unmodelled.items())
        st.warning(f"Partial picture: this bill has a charge this version does not price "
                   f"({charges}). The figures below cover energy only.")

    basis, near, long_term = rec.basis, rec.near_term, rec.long_term
    columns = st.columns(4)
    columns[0].metric("Battery payback", years(basis["payback_years"]))
    columns[1].metric("Net outlay", f"${near.net_outlay:,.0f}")
    columns[2].metric("Saving a year", f"${near.first_year_saving:,.0f}")
    columns[3].metric("Crossover", f"year {long_term.crossover_year}"
                      if long_term.crossover_year else "never")
    st.caption(f"A {rec.evaluated_kwh:g} kWh battery installed {install_date}; the battery on its "
               f"own: {LABELS[rec.battery_action]}.")

    for test in rec.rule_tests:
        limit = "warranty" if test.limit == "warranty" else "expected stay in the home"
        if test.passed is None:
            st.write(f"- **{limit.capitalize()}:** not given, so not tested.")
        else:
            where = "inside" if test.passed else "outside"
            st.write(f"- **{limit.capitalize()} ({test.limit_years:g} years):** the payback is "
                     f"{where} it by {abs(test.margin_years):.1f} years.")
    if rec.comfortable_spend is not None:
        spend = rec.comfortable_spend
        st.info(f"The net outlay, ${spend.net_outlay:,.0f}, is ${spend.over_by:,.0f} more than "
                f"the ${spend.stated:,.0f} you said you're comfortable spending upfront.")

    st.markdown("**What would change the assessment**")
    for revisit in rec.revisit_if:
        st.write(f"- {describe(revisit)}")
    st.markdown("**What the payback leaves out**")
    for item in sorted(rec.not_priced, key=lambda item: item.stated_priority is None):
        flag = " *(you said this matters)*" if item.stated_priority else ""
        st.write(f"- **{item.item}**{flag}: {item.detail}")
    with st.expander("Assumptions, and questions not answered"):
        for item in rec.assumptions:
            ranged = (f"; plausible range {quantity(item.range[0], item.unit)} to "
                      f"{quantity(item.range[1], item.unit)}" if item.range else "")
            effect = f" → {item.effect}" if item.effect else ""
            st.write(f"- **{item.name}**: {quantity(item.value, item.unit)} "
                     f"({item.source}{ranged}){effect}")

    st.markdown("**In words**")
    if llm.LIVE:
        st.caption("Live mode: an explanation not yet cached is written now, from the figures "
                   "above (never your bill), by the language-model provider.")
    else:
        st.caption("Offline: shown only if it was written ahead of time. Every figure in it is "
                   "checked against the figures above.")
    if st.button("Show the explanation", key="explain"):
        try:
            st.write(explain(rec))
        except (LLMError, UntraceableFigure) as error:
            st.error(str(error))


def followup_section(rec: Recommendation, profile: HouseholdProfile, key: tuple,
                     live: bool) -> None:
    """One question at a time about the result above: explained, re-run or declined."""
    st.header("4. Ask about this result")
    with st.form("followup"):
        question = st.text_input("Your question", key="question",
                                 placeholder="Why so long? What if we got an EV? What about a "
                                             "13 kWh battery?")
        asked = st.form_submit_button("Ask")
    if asked and question.strip():
        try:
            with st.spinner("Answering…" + (" New questions go to the provider live." if live
                                            else "")):
                reply = followup.ask(question.strip(), rec, profile, config(), key[2],
                                     live=live)
        except (LLMError, UntraceableFigure, IncompleteAnswer) as error:
            hint = "" if live else " Or turn on “Answer new questions live” in the sidebar."
            st.error(f"{error}{hint}")
            return
        st.session_state["reply"] = (key, reply)
    shown = st.session_state.get("reply")
    if shown is None or shown[0] != key:
        return
    reply = shown[1]
    st.markdown(f"**You asked:** {reply.question}")
    if reply.changed:
        st.caption(f"Changed and re-run through the whole model: {reply.changed}.")
    st.write(reply.text)


def describe(revisit: Revisit) -> str:
    if revisit.not_applicable:
        return f"{revisit.parameter.capitalize()}: not applicable, as {revisit.not_applicable}."
    if revisit.to_action is None:
        low, high = (describe_value(revisit, value) for value in revisit.searched)
        text = f"{revisit.parameter.capitalize()}: no change from {low} to {high}"
    else:
        label = "the label" if revisit.advice_changes is False else "the answer"
        text = (f"{revisit.parameter.capitalize()}: at {describe_value(revisit, revisit.threshold)}"
                f" {label} moves to {LABELS[revisit.to_action]}")
        if revisit.advice_changes is False:
            text += ", but the advice does not change"
    plan = revisit.planned
    if plan is not None:
        when = f"roughly {plan.year}" if plan.year is not None else "no year given"
        text += (f". **You plan an EV ({when}):** with a typical EV's charging, the answer is "
                 f"{LABELS[plan.action_then]}")
        if plan.inside_stay is not None:
            text += (f"; that lands {'inside' if plan.inside_stay else 'after'} your stated stay "
                     f"(to about {int(plan.stay_ends)})")
    return text + "."


def years(value: float) -> str:
    return "never" if value == float("inf") else f"{value:.1f} years"


main()
