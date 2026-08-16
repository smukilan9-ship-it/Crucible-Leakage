"""Google Gemini, served from a pool of free-tier keys.

Free-tier keys allow a small number of requests per day each, so a public
demonstration needs a pool and needs to be careful with it. Everything about
that carefulness lives in `keypool.py`; this file is the transport.

A quota rejection here is not retried against the same key. The whole point of
the pool is that a key which says "no more today" should be set aside and a
different one tried, and repeating the request against the exhausted key is how
a free tier becomes a banned account.
"""

import asyncio
import json

import httpx

from .base import MAX_OUTPUT_TOKENS, Provider, ProviderError, QuotaExhausted
from .keypool import KeyPool

BASE = "https://generativelanguage.googleapis.com/v1beta"
KEYS_ENVIRONMENT_VARIABLE = "CRUCIBLE_GEMINI_KEYS"
DAILY_LIMIT_ENVIRONMENT_VARIABLE = "CRUCIBLE_GEMINI_DAILY_LIMIT"

QUOTA_STATUS = {429}
RETRYABLE_STATUS = {500, 502, 503, 504}

# Gemini identifiers move: a model is retired, renamed, or not enabled on a
# particular account, and the API answers 404 rather than serving something
# close. Falling back one generation keeps an audit running, and the substitute
# is a model the benchmark also measured, so the report can still say what the
# figure behind its verdicts was. The substitution is announced, never silent:
# `chat` records it on the provider and the report carries it.
FALLBACK_MODEL = {
    "gemini-3.7-flash": "gemini-3.5-flash",
}


class GeminiProvider(Provider):
    name = "gemini"

    def __init__(self, pool: KeyPool | None = None, base_url: str = BASE):
        self._pool = pool or KeyPool.from_environment(
            KEYS_ENVIRONMENT_VARIABLE, _daily_limit(), name="gemini")
        self._client = httpx.AsyncClient(
            base_url=base_url, timeout=httpx.Timeout(180.0, connect=15.0))
        # Set when a requested model was not served and a substitute was used,
        # as {asked: served}. Read by the audit so the report names what
        # actually answered rather than what was requested.
        self.substituted: dict[str, str] = {}

    def status(self) -> dict:
        return {"provider": self.name, **self._pool.status(),
                **({"substituted": dict(self.substituted)} if self.substituted else {})}

    async def chat(self, model: str, prompt: str, max_tokens: int | None = None) -> str:
        budget = max_tokens or 8192
        # Enough attempts to ride out a busy model, not so many that a genuinely
        # broken request takes a minute to say so.
        attempts = (min(len(self._pool), 6) or 1) + 4
        delay_seconds = 2.0
        last_error: Exception | None = None
        asked_for = model
        overloads = 0

        for attempt in range(attempts):
            try:
                _, key = self._pool.acquire()
            except LookupError as error:
                raise QuotaExhausted(str(error)) from error

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": min(budget, MAX_OUTPUT_TOKENS),
                },
            }
            try:
                response = await self._client.post(
                    f"/models/{model}:generateContent",
                    params={"key": key}, json=payload,
                )
                if response.status_code in (401, 403):
                    # One bad key in a pool should not stop the audit; the next
                    # key may be fine. Only a pool with no working key at all is
                    # a failure, and `acquire` reports that separately.
                    last_error = ProviderError("a pooled key was rejected")
                    continue
                if response.status_code in QUOTA_STATUS:
                    # This key is spent. Do not sleep and do not retry it; take
                    # the next key instead.
                    last_error = ProviderError("a key reached its quota")
                    continue
                if response.status_code == 404:
                    substitute = FALLBACK_MODEL.get(model)
                    if not substitute:
                        raise ProviderError(
                            f"Gemini does not serve {model!r}, and no substitute "
                            f"is configured for it.")
                    self.substituted[asked_for] = substitute
                    model = substitute
                    last_error = ProviderError(
                        f"{asked_for} is not served; using {substitute}")
                    continue
                if response.status_code in RETRYABLE_STATUS:
                    # 503 means the model is busy, not that the request is
                    # wrong. Waiting helps; waiting forever does not. After a
                    # couple of these the substitute takes over, because a
                    # completed audit on a measured second choice beats a
                    # failed one on the first.
                    overloads += 1
                    last_error = ProviderError(
                        f"HTTP {response.status_code} from {model}")
                    substitute = FALLBACK_MODEL.get(model)
                    if overloads >= 2 and substitute:
                        self.substituted[asked_for] = substitute
                        model = substitute
                        overloads = 0
                        delay_seconds = 2.0
                        continue
                else:
                    response.raise_for_status()
                    text = _first_text(response.json())
                    if text:
                        return text
                    last_error = ProviderError("empty completion")
                    if _finish_reason(response.json()) == "MAX_TOKENS":
                        budget = min(budget * 2, MAX_OUTPUT_TOKENS)
            except (httpx.TimeoutException, httpx.TransportError) as error:
                last_error = error
            except (KeyError, IndexError, json.JSONDecodeError) as error:
                last_error = ProviderError(f"malformed response: {error}")

            if attempt < attempts - 1:          # no point sleeping before giving up
                await asyncio.sleep(delay_seconds)
                delay_seconds = min(delay_seconds * 2, 20)

        raise ProviderError(
            f"Gemini could not answer: {last_error}. HTTP 503 means the model is "
            f"busy rather than that anything is wrong with the request; trying "
            f"again in a minute usually works, and a key of your own in Settings "
            f"avoids the shared pool's contention entirely.")

    async def aclose(self) -> None:
        await self._client.aclose()


def _first_text(body: dict) -> str:
    for candidate in body.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            text = part.get("text", "")
            if text and text.strip():
                return text
    return ""


def _finish_reason(body: dict) -> str | None:
    candidates = body.get("candidates") or [{}]
    return candidates[0].get("finishReason")


def _daily_limit() -> int:
    import os
    try:
        return int(os.environ.get(DAILY_LIMIT_ENVIRONMENT_VARIABLE, "20"))
    except ValueError:
        return 20
