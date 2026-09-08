"""LLM abstraction: structured extraction + explanation, with deterministic fallbacks.

Design decisions
----------------
* The LLM is used ONLY for tasks where it genuinely adds value:
  - structured resume/JD extraction (hybrid with deterministic extraction)
  - natural-language explanation of computed scores
  - the RAG assistant's answer generation
* Scores/skill matching are computed deterministically and NEVER by the LLM.
* All calls are guarded: timeouts, truncated context, JSON-schema validated output,
  prompt-injection resistant (documents are passed as data, never as instructions).
* If no API key is configured the system degrades gracefully to the heuristic
  pipeline and still produces a fully working product (important for reviewers
  without keys).
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

from app.core.config import settings

logger = logging.getLogger("app.llm")


class LLMUnavailableError(Exception):
    """Raised when an LLM call is attempted without a configured provider."""


class CircuitBreaker:
    """Trips after N consecutive retryable provider failures; while open, all
    LLM calls fail fast (no multi-second backoff waits). Half-opens after the
    cooldown to probe recovery with a single real call."""

    def __init__(self, threshold: int = 3, cooldown_s: int = 600) -> None:
        self.threshold = threshold
        self.cooldown_s = cooldown_s
        self.failures = 0
        self.opened_at = 0.0

    @property
    def is_open(self) -> bool:
        """True when the breaker should block calls. Non-mutating: peeking
        does not trigger the half-open probe."""
        if self.failures < self.threshold:
            return False
        return (time.time() - self.opened_at) < self.cooldown_s

    def allow_probe(self) -> bool:
        """Called by chat(): if the cooldown expired, permit exactly one trial
        call (half-open) and reset the clock."""
        if self.failures < self.threshold:
            return True  # circuit closed — normal operation
        if (time.time() - self.opened_at) >= self.cooldown_s:
            self.opened_at = time.time()  # start a fresh probe window
            logger.info("LLM circuit half-open — probing provider")
            return True
        return False

    @property
    def state(self) -> str:
        if self.failures < self.threshold:
            return "closed"
        if (time.time() - self.opened_at) < self.cooldown_s:
            return "open"
        return "half-open"

    def record_success(self) -> None:
        self.failures = 0

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures == self.threshold:
            self.opened_at = time.time()
            logger.warning("LLM circuit OPEN for %ds after %d consecutive failures", self.cooldown_s, self.failures)


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat completions API.

    Works with OpenAI (paid) AND Google Gemini's OpenAI-compatible endpoint
    (free tier) — provider is resolved in config, this client doesn't care.
    """

    def __init__(self) -> None:
        self._client: Any = None
        self.breaker = CircuitBreaker(threshold=3, cooldown_s=600)

    @property
    def available(self) -> bool:
        """Non-mutating: should callers attempt LLM features at all?
        False while the circuit is hard-open (recent consecutive failures)."""
        return settings.llm_available and not self.breaker.is_open

    @property
    def provider(self) -> str:
        return settings.llm_provider

    def _ensure_client(self) -> Any:
        if not self.available:
            raise LLMUnavailableError("No LLM API key configured.")
        if self._client is None:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": settings.effective_llm_api_key}
            if settings.effective_llm_base_url:
                kwargs["base_url"] = settings.effective_llm_base_url
            self._client = OpenAI(**kwargs)
        return self._client

    def chat_stream(self, system: str, user: str, *, json_mode: bool = False):
        """Token-streaming completion. Yields content deltas; raises on failure.
        Respects the circuit breaker (fail fast when provider is down)."""
        client = self._ensure_client()
        if len(user) > settings.llm_guard_max_chars:
            user = user[: settings.llm_guard_max_chars]
            logger.warning("LLM user payload truncated to %d chars", settings.llm_guard_max_chars)
        try:
            stream = client.chat.completions.create(
                model=settings.effective_llm_model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
                max_tokens=settings.llm_max_tokens,
                timeout=settings.llm_timeout_s,
                stream=True,
                **({"response_format": {"type": "json_object"}} if json_mode else {}),
            )
            self.breaker.record_success()
            for chunk in stream:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as exc:  # noqa: BLE001
            if self._is_retryable(exc):
                self.breaker.record_failure()
            logger.error("LLM stream failed: %s", exc)
            raise

    _RETRYABLE_SUBSTRINGS = ("429", "503", "high demand", "overloaded", "rate limit", "timeout")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in LLMClient._RETRYABLE_SUBSTRINGS)

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Single-turn completion with retry/backoff for transient provider
        errors (429 rate limits, 503 load spikes). A circuit breaker fails fast
        once the provider is clearly down (e.g. exhausted free-tier quota), so
        requests never hang on multi-second retry chains. Raises on final
        failure; callers handle fallback."""
        client = self._ensure_client()
        if not self.breaker.allow_probe():
            raise LLMUnavailableError("LLM circuit open — provider recently failed repeatedly.")
        # Truncate user payload defensively (prompt-injection / cost guard)
        if len(user) > settings.llm_guard_max_chars:
            user = user[: settings.llm_guard_max_chars]
            logger.warning("LLM user payload truncated to %d chars", settings.llm_guard_max_chars)
        delays = (2, 5)  # two retries: wait 2s, then 5s
        last_exc: Exception | None = None
        for delay_before in (0,) + delays:
            if delay_before:
                time.sleep(delay_before)
            try:
                resp = client.chat.completions.create(
                    model=settings.effective_llm_model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    temperature=0.2,
                    max_tokens=settings.llm_max_tokens,
                    timeout=settings.llm_timeout_s,
                    **({"response_format": {"type": "json_object"}} if json_mode else {}),
                )
                content = resp.choices[0].message.content or ""
                logger.info("LLM call ok: %d chars out", len(content))
                self.breaker.record_success()
                return content
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_retryable(exc):
                    # trip counter + fail fast if the breaker just opened
                    self.breaker.record_failure()
                    if self.breaker.is_open:
                        logger.error("LLM circuit open — failing fast: %s", str(exc)[:160])
                        raise
                    logger.warning("LLM transient error (will retry): %s", str(exc)[:160])
                    continue
                logger.error("LLM call failed: %s", exc)
                raise
        logger.error("LLM call failed after retries: %s", last_exc)
        raise last_exc  # type: ignore[misc]
        logger.error("LLM call failed after retries: %s", last_exc)
        raise last_exc  # type: ignore[misc]


_JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{.*?}|\[.*?])\s*```", re.DOTALL)
# Gemini occasionally emits lenient JSON (unquoted keys / trailing commas);
# fix the common cases before parsing.
_UNQUOTED_KEY_RE = re.compile(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)")
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _lenient_json(candidate: str) -> Any:
    """Parse JSON, tolerating Gemini's occasional unquoted-key/trailing-comma output."""
    for attempt_text in (candidate, _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', candidate),
                         _TRAILING_COMMA_RE.sub(r"\1", candidate),
                         _TRAILING_COMMA_RE.sub(r"\1", _UNQUOTED_KEY_RE.sub(r'\1"\2"\3', candidate))):
        try:
            return json.loads(attempt_text)
        except json.JSONDecodeError:
            continue
    raise json.JSONDecodeError("unparseable LLM JSON", candidate, 0)


def extract_json(text: str) -> Any:
    """Best-effort parse of a JSON object/array, tolerating fenced code blocks."""
    if not text:
        raise ValueError("Empty LLM response")
    candidate = text.strip()
    fenced = _JSON_BLOCK_RE.search(candidate)
    if fenced:
        candidate = fenced.group(1)
    try:
        return _lenient_json(candidate)
    except json.JSONDecodeError:
        # last resort: find outermost braces
        start, end = candidate.find("{"), candidate.rfind("}")
        if start != -1 and end > start:
            return _lenient_json(candidate[start : end + 1])
        raise


llm = LLMClient()
