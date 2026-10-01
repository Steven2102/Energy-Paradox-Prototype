"""The web app (app.py), run headless with Streamlit's AppTest.

The form must come from QUESTIONS, the page must say the bill was not read,
and a flat-tariff result must say the form carries the shape of the day. An
assessment here returns the fixture's shared recommendation: the engine path is
tested elsewhere, and this checks what the page shows. The explanation button
is never pressed, so no test calls a model. A file upload cannot be driven
headless, so the page's reply to one is not tested here.
"""

import re
from datetime import date

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

import run_fixtures
from src import followup, llm, recommend
from src.engine import not_assessable
from src.followup import Reply
from src.profile import QUESTIONS
from src.revisit import Revisit
from tests.shared import recommendation


@pytest.fixture
def app(monkeypatch):
    real = recommend.recommend_for

    def shared(profile, config, install_date):
        if not_assessable(profile):
            return real(profile, config, install_date)  # computes nothing
        return recommendation(profile.name)

    monkeypatch.setattr(recommend, "recommend_for", shared)
    st.cache_data.clear()
    return AppTest.from_file("app.py", default_timeout=60).run()


def labels(at):
    return [widget.label for kind in (at.radio, at.selectbox, at.number_input, at.multiselect)
            for widget in kind if widget.label != "Bill values to use"]


def assess(at, fixture=None):
    if fixture:
        at.radio(key="fixture").set_value(fixture).run()
    at.button(key="assess").click().run()
    assert not at.exception
    return at


def test_the_form_asks_what_the_schema_declares_and_follow_ups_only_when_asked(app):
    assert not app.exception
    # The reference household's answers ask no follow-up.
    asked = [question.ask for question in QUESTIONS if question.asked_if is None]
    assert len(labels(app)) == len(asked)
    assert all(any(label.startswith(ask) for label in labels(app)) for ask in asked)
    app.radio(key="reference_household.has_solar").set_value(True).run()
    assert "How big is the solar system? (kW)" in labels(app)


def test_the_page_says_the_bill_is_not_read_and_whose_values_are_used(app):
    notice = " ".join(info.value for info in app.info)
    assert "Automatic reading of bills is not enabled yet" in notice
    assert "reference_household" in notice


def test_a_flat_tariff_result_says_the_form_carries_the_shape_of_the_day(app):
    flat = [warning.value for warning in assess(app, "household_b").warning]
    assert any("one flat rate" in text and "answers above" in text for text in flat)
    time_of_use = [warning.value for warning in assess(app, "reference_household").warning]
    assert not any("one flat rate" in text for text in time_of_use)


def test_an_existing_battery_shows_cannot_assess_and_the_reason(app):
    app.radio(key="reference_household.has_battery").set_value(True).run()
    app.number_input(key="reference_household.battery_kwh").set_value(13.5).run()
    errors = [error.value for error in assess(app).error]
    assert any("Cannot assess" in text and "already has a 13.5 kWh battery" in text
               for text in errors)


def test_the_app_and_the_runner_look_for_the_same_cached_reply(monkeypatch, tmp_path):
    # A cache warmed by run_fixtures.py is evidence the app will hit it only if
    # both build the same key. Each runs its real path to the explanation, offline
    # into an empty cache, and fails naming the key it looked for. The sweeps are
    # stubbed alike in both, for time: the key still depends on everything else --
    # the profile each builds, the tariff, the install date and the evaluation.
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(llm, "LIVE", False)
    monkeypatch.setattr(recommend, "revisit_if", lambda evaluation: (Revisit(
        "battery cost", "$", "action", None, None, evaluation.action, None, None, None,
        "stubbed: this test compares cache keys, not sweeps"),))
    st.cache_data.clear()
    at = AppTest.from_file("app.py", default_timeout=60).run()
    at.radio(key="fixture").set_value("household_b").run()
    at.button(key="assess").click().run()
    at.button(key="explain").click().run()
    shown = " ".join(error.value for error in at.error)
    app_key = re.search(r"cache key ([0-9a-f]{64})", shown).group(1)
    assert "household_b" in shown

    with pytest.raises(llm.NotCached) as missing:
        run_fixtures.main(["--install-date", date.today().isoformat(), "--explain"])
    assert (missing.value.about, missing.value.key) == ("household_b", app_key)


# ------------------------------------------------------------ the follow-up chat

def ask_in(at, question):
    at.text_input(key="question").input(question)
    next(button for button in at.button if button.label == "Ask").click().run()
    assert not at.exception
    return at


def test_a_follow_up_shows_the_question_the_answer_and_what_was_re_run(app, monkeypatch):
    calls = []

    def stand_in(question, before, profile, config, install_date, live=False):
        calls.append(live)
        return Reply(question, "counterfactual", "It goes from 37.2 years to 42.9 years.",
                     changed="a 13 kWh battery instead of 10 kWh")

    monkeypatch.setattr(followup, "ask", stand_in)
    ask_in(assess(app), "What about a 13 kWh battery?")
    shown = " ".join(element.value for element in [*app.markdown, *app.caption])
    assert "What about a 13 kWh battery?" in shown
    assert "It goes from 37.2 years to 42.9 years." in shown
    assert "re-run through the whole model: a 13 kWh battery instead of 10 kWh" in shown
    assert calls == [False]  # offline, as the switch starts


def test_offline_an_unanswered_follow_up_is_an_error_saying_how_to_answer_it(
        app, monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)  # nothing cached
    monkeypatch.setattr(llm, "LIVE", False)
    errors = " ".join(error.value for error in ask_in(assess(app), "Why so long?").error)
    assert "LLM_LIVE=1 python run_fixtures.py" in errors and "--questions" in errors
    assert "Answer new questions live" in errors


def test_the_live_switch_reaches_follow_ups_and_nothing_else(app, monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(followup, "ask", lambda *args, live=False, **kwargs: (
        calls.append(live) or Reply(args[0], "explain", "An answer.")))
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path)  # nothing cached
    monkeypatch.setattr(llm, "LIVE", False)
    app.toggle(key="live_answers").set_value(True).run()
    ask_in(assess(app), "Why so long?")
    assert calls == [True]
    # The explanation stays as written ahead of time: uncached, it is an error, not a call.
    next(button for button in app.button if button.label == "Show the explanation").click().run()
    assert any("No cached reply" in error.value for error in app.error)
