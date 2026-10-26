"""Provider table: where the model call goes, and how it is addressed.

A provider is a record, not a code fork. Everything that differs between them
— endpoint, key variable, default model — is declared here, and
``providers/base.py`` does the transport for all of them.

``coral`` is the default: it is the service this harness was built against and
the one the default numbers were measured on. Any OpenAI-compatible endpoint
(OpenAI itself, vLLM, Ollama, LiteLLM, a company gateway) works the same way
via ``openai``, with ``--base-url`` and ``--model``.

To add one, append a ``Provider`` here. If it speaks the Responses API
(tools, ``previous_response_id``), ``--agent`` mode works unchanged; if it
only speaks chat completions, the one-turn path works and ``--agent`` should
stay off.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from wrasse.providers.base import DEFAULT_BASE_URL


@dataclass(frozen=True)
class Provider:
    """How to reach one model endpoint."""

    name: str
    base_url: str
    api_key_env: str
    default_model: str | None = None
    supports_responses: bool = True


CORAL = Provider(
    name="coral",
    base_url=DEFAULT_BASE_URL,
    api_key_env="CORALBRICKS_API_KEY",
    default_model="deepseek-v4.1-flash-fast",
)

OPENAI = Provider(
    name="openai",
    base_url="https://api.openai.com",
    api_key_env="OPENAI_API_KEY",
    default_model=os.environ.get("WRASSE_MODEL", "gpt-4.1"),
)

PROVIDERS: dict[str, Provider] = {p.name: p for p in (CORAL, OPENAI)}


def resolve_provider(name: str | None) -> Provider:
    """Look up a provider by name; ``$WRASSE_PROVIDER`` is the fallback."""
    key = (name or os.environ.get("WRASSE_PROVIDER") or CORAL.name).strip().lower()
    try:
        return PROVIDERS[key]
    except KeyError:
        known = ", ".join(sorted(PROVIDERS))
        raise SystemExit(
            f"unknown provider {key!r} — known providers: {known}"
        ) from None


__all__ = ["Provider", "PROVIDERS", "CORAL", "OPENAI", "resolve_provider"]
