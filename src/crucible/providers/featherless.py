"""Featherless.ai, which serves the open-weight models in the catalogue.

One key, taken from the environment or supplied per call by a user who would
rather bill their own account. The transport, retry policy and reasoning-model
budget handling are shared with every other OpenAI-shaped provider.
"""

import os

from .openai_compatible import OpenAICompatibleProvider

BASE = os.environ.get("FEATHERLESS_BASE", "https://api.featherless.ai/v1")
KEY_ENVIRONMENT_VARIABLE = "FEATHERLESS_API_KEY"


class FeatherlessProvider(OpenAICompatibleProvider):
    name = "featherless"
    label = "Featherless"
    default_base = BASE
    key_variable = KEY_ENVIRONMENT_VARIABLE
