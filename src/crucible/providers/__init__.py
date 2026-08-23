"""Model providers.

`resolve` is the only function the rest of the package uses. It exists so that
choosing a provider is a lookup rather than a chain of conditionals scattered
through the audit, and so adding a provider is one entry in one dictionary.

`detect_provider` exists for the opposite direction. A user who has a key does
not necessarily know, or care, which of four vendors this tool would call it.
Keys announce their own issuer in their first few characters, so one field can
take whatever the user has and route it, instead of asking them to choose a
vendor from a list before they can begin.
"""

import os

from .anthropic import AnthropicProvider
from .base import MAX_OUTPUT_TOKENS, Provider, ProviderError, QuotaExhausted, output_budget
from .featherless import FeatherlessProvider
from .gemini import FALLBACK_MODEL, GeminiProvider
from .keypool import KeyPool
from .openai import OpenAIProvider
from .openai_compatible import OpenAICompatibleProvider
from .vertex import VertexProvider

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "featherless": FeatherlessProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "vertex": VertexProvider,
}

# Vertex serves the same Gemini models as the key pool, so it is a transport
# rather than a separate entry in the catalogue. Setting this sends every
# `gemini` call through the billed project instead of the shared free-tier
# keys, which is what a demo anybody else can reach needs: the pool has a
# daily cap, and the day it runs out every visitor is told no column order
# produced a usable answer.
TRANSPORT_VARIABLE = "CRUCIBLE_GEMINI_TRANSPORT"

# Checked in this order, because an Anthropic key also starts with the OpenAI
# prefix. Matching in the wrong order would send it to the wrong vendor, and
# with it the user's column names.
#
# Two Google prefixes, because Google changed the format: keys minted in AI
# Studio now begin `AQ.` where they used to begin `AIza`. That change is the
# reason `detect_provider` returning None must never mean "refuse the key".
# A prefix table is a convenience that goes stale on somebody else's schedule,
# so an unrecognized key is a key whose vendor the user gets asked about, not a
# key that is turned away.
KEY_PREFIXES = (
    ("sk-ant-", "anthropic"),
    ("AQ.", "gemini"),
    ("AIza", "gemini"),
    ("rc_", "featherless"),
    ("sk-", "openai"),
)

# Which environment variable each provider reads when no key is passed in.
KEY_VARIABLES = {
    "anthropic": "ANTHROPIC_API_KEY",
    "featherless": "FEATHERLESS_API_KEY",
    "gemini": "CRUCIBLE_GEMINI_KEYS",
    "openai": "OPENAI_API_KEY",
    # Credentials reach Vertex by several routes; the project is the one
    # thing that must be named.
    "vertex": "GOOGLE_CLOUD_PROJECT",
}


def detect_provider(api_key: str) -> str | None:
    """Which vendor issued this key, or None when the shape is not recognized.

    Returning None rather than guessing is deliberate: an unrecognized key sent
    to a guessed vendor is a rejected call at best, and at worst it is the
    user's schema delivered to a company they did not choose.

    None means "ask", not "refuse". Callers must let the user name the provider
    themselves, because this table describes what key formats looked like when
    it was written and vendors change them without notice.
    """
    key = (api_key or "").strip()
    for prefix, provider in KEY_PREFIXES:
        if key.startswith(prefix):
            return provider
    return None


def effective_provider(name: str) -> str:
    """Which provider will actually answer for this one.

    A caller that asked for Gemini and a deployment configured for Vertex get
    the same models down a different pipe. This is a function rather than two
    lines inside `resolve` because the CLI checks for a credential before it
    calls anything, and a check that disagreed with the call refused to run an
    audit the transport was perfectly able to serve.
    """
    if name == "gemini" and os.environ.get(
            TRANSPORT_VARIABLE, "").strip().lower() == "vertex":
        return "vertex"
    return name


def resolve(name: str, **options) -> Provider:
    """Build a provider by name. Unknown names fail loudly rather than falling
    back to a default, because silently auditing with a model the caller did
    not ask for is worse than not auditing at all."""
    name = effective_provider(name)
    try:
        factory = PROVIDERS[name]
    except KeyError:
        raise ProviderError(
            f"unknown provider {name!r}; available: {', '.join(sorted(PROVIDERS))}")
    return factory(**options)


__all__ = ["Provider", "ProviderError", "QuotaExhausted", "KeyPool",
           "AnthropicProvider", "FeatherlessProvider", "GeminiProvider",
           "OpenAIProvider", "OpenAICompatibleProvider",
           "PROVIDERS", "KEY_PREFIXES", "KEY_VARIABLES", "FALLBACK_MODEL",
           "VertexProvider", "TRANSPORT_VARIABLE",
           "detect_provider", "resolve", "effective_provider", "output_budget", "MAX_OUTPUT_TOKENS"]
