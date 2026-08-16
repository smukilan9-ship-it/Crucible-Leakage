"""OpenAI, for a user who brings their own key.

The strongest cell in the benchmark's condition ladder belongs to a model
served here, so this is not a courtesy entry: a user with an OpenAI key gets a
better audit than the pooled default provides.
"""

import os

from .openai_compatible import OpenAICompatibleProvider

BASE = os.environ.get("OPENAI_BASE", "https://api.openai.com/v1")
KEY_ENVIRONMENT_VARIABLE = "OPENAI_API_KEY"


class OpenAIProvider(OpenAICompatibleProvider):
    name = "openai"
    label = "OpenAI"
    default_base = BASE
    key_variable = KEY_ENVIRONMENT_VARIABLE
