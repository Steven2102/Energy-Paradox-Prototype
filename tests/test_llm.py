"""src/llm.py, the provider wrapper: its cache, its settings and its failures,
with a stand-in provider. Nothing here reaches the network."""

import json

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


def test_with_live_calls_off_an_unrecorded_input_is_an_error(stand_in, monkeypatch):
    monkeypatch.setattr(llm, "LIVE", False)
    with pytest.raises(llm.LLMError, match="LLM_RECORD=1"):
        llm.complete("system", "never recorded")
    assert stand_in == []
