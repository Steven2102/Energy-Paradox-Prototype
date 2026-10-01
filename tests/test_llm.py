"""src/llm.py, the provider wrapper: its cache, its settings and its failures,
with a stand-in provider. Nothing here reaches the network."""

import json
import os
import subprocess
import sys

import pytest

from src import llm


@pytest.fixture
def settings(monkeypatch, tmp_path):
    """No .env to read, an empty cache, live calls on, no provider settings."""
    monkeypatch.setattr(llm, "ROOT", tmp_path)
    monkeypatch.setattr(llm, "CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(llm, "LIVE", True)
    for name in ("LLM_PROVIDER", "LLM_MODEL", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def stand_in(settings, monkeypatch):
    """A provider that answers every message and remembers each one it was sent."""
    sent = []

    def provider(system, user, *, model, api_key):
        sent.append(user)
        return f"reply to {user}"

    monkeypatch.setitem(llm.PROVIDERS, "stand_in", provider)
    monkeypatch.setitem(llm.KEY_NAMES, "stand_in", "STAND_IN_API_KEY")
    monkeypatch.setitem(llm.DEFAULT_MODELS, "stand_in", "stand-in-1")
    monkeypatch.setenv("LLM_PROVIDER", "stand_in")
    monkeypatch.setenv("STAND_IN_API_KEY", "not a real key")
    return sent


def test_a_reply_is_cached_on_its_exact_input(stand_in):
    assert llm.complete("system", "one") == "reply to one"
    assert llm.complete("system", "one") == "reply to one"
    assert llm.complete("system", "two") == "reply to two"
    assert stand_in == ["one", "two"]  # the repeated input was not sent again
    cached = json.loads((llm.CACHE_DIR / f"{llm.cache_key('system', 'one')}.json").read_text())
    assert (cached["model"], cached["user"], cached["reply"]) == ("stand-in-1", "one", "reply to one")


def test_a_missing_key_fails_with_a_clear_message_before_anything_is_sent(settings, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    with pytest.raises(llm.LLMError, match="ANTHROPIC_API_KEY is not set"):
        llm.complete("system", "user")


def test_a_provider_the_file_does_not_implement_says_where_to_add_it(settings, monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(llm.LLMError, match="LLM_PROVIDER=openai .* src/llm.py"):
        llm.complete("system", "user")


def test_offline_an_uncached_input_raises_at_once_naming_it_its_key_and_the_warm_up(
        stand_in, monkeypatch):
    monkeypatch.setattr(llm, "LIVE", False)
    with pytest.raises(llm.NotCached) as missing:
        llm.complete("system", "never recorded", about="household_x")
    key = llm.cache_key("system", "never recorded")
    assert (missing.value.about, missing.value.key) == ("household_x", key)
    assert key in str(missing.value)
    assert "LLM_LIVE=1 python run_fixtures.py --explain --install-date" in str(missing.value)
    assert stand_in == []  # nothing was sent


def test_offline_is_the_default():
    environment = {name: value for name, value in os.environ.items()
                   if name not in ("LLM_LIVE", "LLM_RECORD")}
    live = subprocess.run([sys.executable, "-c", "from src import llm; print(llm.LIVE)"],
                          cwd=llm.ROOT, env=environment, capture_output=True, text=True,
                          check=True)
    assert live.stdout.strip() == "False"
