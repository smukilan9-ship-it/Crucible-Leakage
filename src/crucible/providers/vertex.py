"""The same Gemini models, reached through Vertex instead of an API key.

This exists because the free tier does not survive an audience. A pooled
AI Studio key is capped per day, and the day it runs out every visitor sees
"no column order produced a usable answer", which reads as a broken tool
rather than an empty wallet. Vertex bills a project and has no such cliff.

It is a transport, not a new model. The identifiers, the prompts, the shuffle
counts and the criterion are all unchanged, so a verdict produced here and a
verdict produced through the key pool are the same verdict.

`google-auth` is an optional extra rather than a dependency. Somebody who
installs this package to audit a CSV with their own key should not be made to
download Google's auth stack to do it.
"""

import asyncio
import json
import os

import httpx

from .base import MAX_OUTPUT_TOKENS, Provider, ProviderError, QuotaExhausted

HOST = "https://aiplatform.googleapis.com"
PROJECT_VARIABLE = "GOOGLE_CLOUD_PROJECT"
LOCATION_VARIABLE = "CRUCIBLE_VERTEX_LOCATION"
# Gemini 3 models are not served regionally: asking us-central1 for
# gemini-3.7-flash returns 404, and the 404 names the region rather than the
# reason, which costs an afternoon if you have not seen it before.
DEFAULT_LOCATION = "global"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
RETRYABLE = {429, 500, 502, 503, 504}


class VertexProvider(Provider):
    name = "vertex"

    def __init__(self, project: str | None = None, location: str | None = None):
        self.project = project or os.environ.get(PROJECT_VARIABLE, "").strip()
        if not self.project:
            raise ProviderError(
                f"no Google Cloud project, so Vertex cannot be called. Set "
                f"{PROJECT_VARIABLE} to the project that should be billed.")
        self.location = location or os.environ.get(
            LOCATION_VARIABLE, "").strip() or DEFAULT_LOCATION
        self._client = httpx.AsyncClient(
            base_url=HOST, timeout=httpx.Timeout(180.0, connect=15.0))
        self._credentials = None
        self.calls = 0
        self.tokens = {"prompt": 0, "answer": 0, "thoughts": 0}

    def _token(self) -> str:
        """An access token from whatever credential the environment holds.

        On a server that is a service account in GOOGLE_APPLICATION_CREDENTIALS.
        On a laptop it is whatever `gcloud auth application-default login` left
        behind. Neither is ever read by this package: google-auth finds them
        and hands back a token with an hour on it.
        """
        try:
            import google.auth
            import google.auth.transport.requests
        except ImportError as error:
            raise ProviderError(
                "Vertex needs the google-auth package, which this one does not "
                "install by default. `pip install \"crucible-leakage[vertex]\"`."
            ) from error

        if self._credentials is None:
            try:
                self._credentials, _ = google.auth.default(scopes=[SCOPE])
            except Exception as error:                      # noqa: BLE001
                raise ProviderError(
                    f"no usable Google credential: {error}. On a server set "
                    f"GOOGLE_APPLICATION_CREDENTIALS to a service account key; "
                    f"locally run `gcloud auth application-default login`."
                ) from error
        if not self._credentials.valid:
            self._credentials.refresh(
                google.auth.transport.requests.Request())
        return self._credentials.token

    def status(self) -> dict:
        return {"provider": self.name, "project": self.project,
                "location": self.location, "calls": self.calls,
                "tokens": dict(self.tokens)}

    async def chat(self, model: str, prompt: str,
                   max_tokens: int | None = None) -> str:
        path = (f"/v1/projects/{self.project}/locations/{self.location}"
                f"/publishers/google/models/{model}:generateContent")
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "maxOutputTokens": min(max_tokens or 8192, MAX_OUTPUT_TOKENS),
            },
        }
        delay, last = 2.0, None
        for _ in range(6):
            try:
                token = await asyncio.to_thread(self._token)
                response = await self._client.post(path, json=payload, headers={
                    "Authorization": f"Bearer {token}",
                    # Application default credentials carry no billing project
                    # of their own, and without this header Vertex answers 403
                    # with a link to the docs rather than a verdict.
                    "x-goog-user-project": self.project,
                })
            except httpx.HTTPError as error:
                last = ProviderError(f"vertex could not be reached: {error}")
                await asyncio.sleep(delay); delay *= 2
                continue

            if response.status_code in RETRYABLE:
                last = ProviderError(f"vertex returned {response.status_code}")
                await asyncio.sleep(delay); delay *= 2
                continue
            if response.status_code >= 400:
                raise ProviderError(f"vertex returned {response.status_code}: "
                                    f"{response.text[:300]}")

            body = response.json()
            candidates = body.get("candidates") or []
            if not candidates:
                # A prompt stopped by a safety filter comes back with no
                # candidate at all. Reporting that as an empty answer would let
                # the screen score the table as entirely clean.
                raise ProviderError(
                    f"vertex returned no candidate "
                    f"({json.dumps(body.get('promptFeedback', {}))[:200]})")
            parts = candidates[0].get("content", {}).get("parts", [])
            text = "".join(part.get("text", "") for part in parts)
            usage = body.get("usageMetadata", {})
            self.calls += 1
            self.tokens["prompt"] += usage.get("promptTokenCount", 0)
            self.tokens["answer"] += usage.get("candidatesTokenCount", 0)
            self.tokens["thoughts"] += usage.get("thoughtsTokenCount", 0)
            if not text.strip():
                raise ProviderError(
                    f"vertex answered nothing; finish reason "
                    f"{candidates[0].get('finishReason')}")
            return text
        raise QuotaExhausted(f"vertex would not answer after six attempts: {last}")

    async def aclose(self) -> None:
        await self._client.aclose()
