"""Anthropic, for a user who brings their own key.

A different wire shape from the OpenAI-compatible providers, so it does not
share their transport: the messages endpoint takes an `x-api-key` header rather
than a bearer token, requires `max_tokens` on every call, and returns a list of
content blocks instead of a single string. The behavior Crucible depends on is
the same, and so are the two rules that matter: never return an empty answer,
and never record a failed call as one.
"""

import asyncio
import json
import os

import httpx

from .base import MAX_OUTPUT_TOKENS, Provider, ProviderError

BASE = os.environ.get("ANTHROPIC_BASE", "https://api.anthropic.com/v1")
KEY_ENVIRONMENT_VARIABLE = "ANTHROPIC_API_KEY"
API_VERSION = "2023-06-01"

TRANSPORT_ERRORS = (httpx.TimeoutException, httpx.TransportError)
RETRYABLE_STATUS = {429, 500, 502, 503, 504, 529}


class AnthropicProvider(Provider):
    name = "anthropic"
    label = "Anthropic"

    def __init__(self, api_key: str | None = None, base_url: str = BASE,
                 max_retries: int = 4):
        key = api_key or os.environ.get(KEY_ENVIRONMENT_VARIABLE)
        if not key:
            raise ProviderError(f"{KEY_ENVIRONMENT_VARIABLE} is not set")
        self._max_retries = max_retries
        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers={"x-api-key": key, "anthropic-version": API_VERSION,
                     "content-type": "application/json"},
            timeout=httpx.Timeout(180.0, connect=15.0),
        )

    async def chat(self, model: str, prompt: str, max_tokens: int | None = None) -> str:
        # Required by this API rather than optional, so there is always a budget
        # to double when a reasoning model spends it all before writing.
        budget = min(max_tokens or 8192, MAX_OUTPUT_TOKENS)
        delay_seconds = 2.0
        last_error: Exception | None = None

        for _ in range(self._max_retries + 1):
            payload = {
                "model": model,
                "max_tokens": budget,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": prompt}],
            }
            try:
                response = await self._client.post("/messages", json=payload)
                if response.status_code in (401, 403):
                    raise ProviderError(
                        "Anthropic rejected the API key. Check it is correct and "
                        "still active, or enter a different one.")
                if response.status_code == 404:
                    raise ProviderError(
                        f"Anthropic does not serve {model!r} on this account.")
                if response.status_code in RETRYABLE_STATUS:
                    last_error = ProviderError(
                        f"HTTP {response.status_code}: {response.text[:200]}")
                else:
                    response.raise_for_status()
                    body = response.json()
                    text = _first_text(body)
                    if text:
                        return text
                    stop = body.get("stop_reason")
                    last_error = ProviderError(f"empty completion (stop_reason={stop})")
                    if stop == "max_tokens":
                        budget = min(budget * 2, MAX_OUTPUT_TOKENS)
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


def _first_text(body: dict) -> str:
    """The first text block. A reply may lead with other block types, so the
    list is walked rather than indexed."""
    for block in body.get("content", []):
        if block.get("type") == "text":
            text = block.get("text", "")
            if text and text.strip():
                return text
    return ""
