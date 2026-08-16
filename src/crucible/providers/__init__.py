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

from .anthropic import AnthropicProvider
from .base import MAX_OUTPUT_TOKENS, Provider, ProviderError, QuotaExhausted, output_budget
from .featherless import FeatherlessProvider
from .gemini import FALLBACK_MODEL, GeminiProvider
from .keypool import KeyPool
from .openai import OpenAIProvider
from .openai_compatible import OpenAICompatibleProvider

PROVIDERS = {
    "anthropic": AnthropicProvider,
    "featherless": FeatherlessProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
}

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


def resolve(name: str, **options) -> Provider:
    """Build a provider by name. Unknown names fail loudly rather than falling
    back to a default, because silently auditing with a model the caller did
    not ask for is worse than not auditing at all."""
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
           "detect_provider", "resolve", "output_budget", "MAX_OUTPUT_TOKENS"]
