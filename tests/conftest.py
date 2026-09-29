"""Applies to every test: no test calls a language model unless it asks to."""

import os

import pytest

from src import llm
from tests.shared import RECORDINGS


@pytest.fixture(autouse=True)
def replayed_replies(monkeypatch):
    """Model replies come from tests/recordings/, never from a provider, and a reply
    that was not recorded is an error. With LLM_RECORD=1 set, a missing reply is
    fetched live and recorded instead."""
    monkeypatch.setattr(llm, "CACHE_DIR", RECORDINGS)
    monkeypatch.setattr(llm, "LIVE", os.environ.get("LLM_RECORD") == "1")
