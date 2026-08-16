"""Any provider that serves the OpenAI `/chat/completions` shape.

Featherless and OpenAI itself differ in a base URL, an environment variable and
a name. Everything else, including the retry policy and the doubling budget for
a reasoning model that spends its whole allowance thinking, is identical, so it
lives here once and each provider is four class attributes.
"""

import asyncio
import json
import os

import httpx

from .base import MAX_OUTPUT_TOKENS, Provider, ProviderError

TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.TransportError)
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class OpenAICompatibleProvider(Provider):
    """Subclass and set the four attributes below."""

    name = "openai-compatible"
    label = "This provider"
    default_base = ""
    key_variable = ""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 max_retries: int = 4):
        key = api_key or os.environ.get(self.key_variable)
        if not key:
            raise ProviderError(f"{self.key_variable} is not set")
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url or os.environ.get(
                f"{self.name.upper()}_BASE", self.default_base),
            headers={"Authorization": f"Bearer {key}"},
            timeout=httpx.Timeout(180.0, connect=15.0),
        )

    async def chat(self, model: str, prompt: str, max_tokens: int | None = None) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        delay_seconds = 2.0
        last_error: Exception | None = None

        for _ in range(self._max_retries + 1):
            try:
                response = await self._client.post("/chat/completions", json=payload)
                if response.status_code in (401, 403):
                    raise ProviderError(
                        f"{self.label} rejected the API key. Check it is correct "
                        f"and still active, or enter a different one.")
                if response.status_code == 404:
                    raise ProviderError(
                        f"{self.label} does not serve {model!r} on this account.")
                if response.status_code in RETRYABLE_STATUS:
                    last_error = ProviderError(
                        f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    response.raise_for_status()
                    body = response.json()
                    text = body["choices"][0]["message"]["content"]
                    if text and text.strip():
                        return text
                    finish = body["choices"][0].get("finish_reason")
                    last_error = ProviderError(
                        f"empty completion (finish_reason={finish})")
                    # A reasoning model spends the output budget thinking before
                    # it writes anything. When the whole budget goes on reasoning
                    # the reply is empty with finish_reason "length", and
                    # repeating the same request repeats the same failure.
                    if finish == "length" and payload.get("max_tokens"):
                        payload["max_tokens"] = min(
                            payload["max_tokens"] * 2, MAX_OUTPUT_TOKENS)
            except TRANSPORT_ERRORS as error:
                last_error = error
            except (KeyError, IndexError, json.JSONDecodeError) as error:
                last_error = ProviderError(f"malformed response: {error}")
            await asyncio.sleep(delay_seconds)
            delay_seconds *= 2

        raise ProviderError(
            f"call failed after {self._max_retries + 1} attempts: {last_error}")

    async def aclose(self) -> None:
        await self._client.aclose()
