"""The one module that talks to a language-model provider.

Everything else calls complete(system, user). The provider, the model and the
key come from .env (see .env.example), so moving to whichever provider the
university's ethics process approves is a change to this file and to .env,
and to nothing else: add a function beside _anthropic, and list it in
PROVIDERS, KEY_NAMES and DEFAULT_MODELS.

Replies are cached on disk, keyed on the exact input -- the system prompt and
the user message -- so a rehearsed demo never waits on the network, and tests
replay recorded replies without calling anyone. The key leaves out the provider
and the model: after switching either, clear the cache (or re-record) to get
fresh replies. Each cached reply records the model that wrote it.

Offline by default: only cached replies, and a miss raises NotCached at once,
naming what was asked about and the cache key. The provider is reached only
when the command itself sets LLM_LIVE=1 (read from the environment, never from
.env) -- which is how the cache is warmed for a demo:

    LLM_LIVE=1 python run_fixtures.py --explain --install-date <the demo's date>

A caller can also go live for one call (live=True): the app does that for
follow-up questions when its "Answer new questions live" switch is on, and for
nothing else. The tests replay tests/recordings/ instead; LLM_RECORD=1 records
what is missing.
"""

import hashlib
import json
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / ".llm_cache"  # gitignored; the tests point it at tests/recordings/
LIVE = os.environ.get("LLM_LIVE") == "1"  # off: only cached replies, and a miss raises
WARM = ("LLM_LIVE=1 python run_fixtures.py --explain --install-date <the demo's date>, adding "
        "--questions <file> for rehearsed follow-up questions")

MAX_TOKENS = 8192  # the reply and the model's thinking, which counts against it
TIMEOUT_SECONDS = 60


class LLMError(RuntimeError):
    """The model could not be asked, or its reply cannot be used."""


class NotCached(LLMError):
    """Offline, and no reply is cached for this input."""

    def __init__(self, about: str, key: str):
        self.about, self.key = about, key
        super().__init__(
            f"No cached reply for {about} (cache key {key}) in {CACHE_DIR}, and offline mode "
            f"is on, so the provider was not called. Warm the cache: {WARM}. (The tests "
            "replay tests/recordings/: LLM_RECORD=1 python -m pytest tests/test_explain.py)")


def complete(system: str, user: str, *, about: str = "this input", live: bool = False) -> str:
    """The model's reply to one system prompt and one user message. `about` names
    what was asked, for the error when offline and nothing is cached; `live` lets
    this one call reach the provider when LLM_LIVE is not set."""
    key = cache_key(system, user)
    path = CACHE_DIR / f"{key}.json"
    if path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached["system"] == system and cached["user"] == user:
            return cached["reply"]
    if not (LIVE or live):
        raise NotCached(about, key)
    provider, model, api_key = _settings()
    reply = PROVIDERS[provider](system, user, model=model, api_key=api_key)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    record = {"provider": provider, "model": model, "recorded": date.today().isoformat(),
              "system": system, "user": user, "reply": reply}
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return reply


def _settings() -> tuple[str, str, str]:
    """(provider, model, key) from .env; environment variables already set win."""
    load_dotenv(ROOT / ".env")
    provider = os.environ.get("LLM_PROVIDER", "").strip()
    if provider not in PROVIDERS:
        found = f"LLM_PROVIDER={provider}" if provider else "LLM_PROVIDER is not set, and"
        raise LLMError(f"{found} is not a provider src/llm.py implements "
                       f"({', '.join(PROVIDERS)}). Set it in .env, or add the provider to "
                       "src/llm.py: that file is the only one a new provider changes.")
    key_name = KEY_NAMES[provider]
    api_key = os.environ.get(key_name, "").strip()
    if not api_key:
        raise LLMError(f"{key_name} is not set, so {provider} cannot be called. Copy .env.example "
                       f"to .env and put the key there.")
    model = os.environ.get("LLM_MODEL", "").strip() or DEFAULT_MODELS[provider]
    return provider, model, api_key


def cache_key(system: str, user: str) -> str:
    """The name a reply to exactly this input is cached under."""
    return hashlib.sha256(json.dumps([system, user]).encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ providers

def _anthropic(system: str, user: str, *, model: str, api_key: str) -> str:
    import anthropic  # here, not at the top: only the chosen provider's SDK need be installed

    # gzip only: the SDK's HTTP client offers brotli whenever a Brotli package is installed,
    # and cannot decode it with Brotli older than 1.2 (Anaconda ships 1.0.9).
    client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS,
                                 default_headers={"Accept-Encoding": "gzip"})
    message = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if message.stop_reason == "max_tokens":
        raise LLMError(f"the reply was cut off at {MAX_TOKENS} tokens")
    return "".join(block.text for block in message.content if block.type == "text")


PROVIDERS = {"anthropic": _anthropic}
KEY_NAMES = {"anthropic": "ANTHROPIC_API_KEY"}
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5"}
