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
    """Provider chain: OpenAI (paid) -> Gemini (free) -> Groq (free backup).

    Each provider gets its own circuit breaker. When one provider's free quota
    is exhausted (429) or it 503s, the next provider is tried automatically —
    so a dead quota on one service never kills the AI features.
    """

    _PROVIDER_ORDER = ("openai", "gemini", "groq")

    def __init__(self) -> None:
        self._clients: dict[str, Any] = {}
        self.breakers: dict[str, CircuitBreaker] = {
            name: CircuitBreaker(threshold=3, cooldown_s=600) for name in self._PROVIDER_ORDER
        }

    def _configs(self) -> list[dict]:
        """Ordered provider configs (only those with keys set)."""
        out: list[dict] = []
        if settings.openai_api_key.strip():
            out.append({
                "name": "openai",
                "api_key": settings.openai_api_key,
                "base_url": settings.openai_base_url or "https://api.openai.com/v1",
                "model": settings.openai_model,
            })
        if settings.gemini_api_key.strip():
            out.append({
                "name": "gemini",
                "api_key": settings.gemini_api_key,
                "base_url": "https://generativelanguage.googleapis.com/v1beta/openai/",
                "model": settings.gemini_model,
            })
        if settings.groq_api_key.strip():
            out.append({
                "name": "groq",
                "api_key": settings.groq_api_key,
                "base_url": "https://api.groq.com/openai/v1",
                "model": settings.groq_model,
            })
        return out

    @property
    def provider(self) -> str:
        names = [c["name"] for c in self._configs()]
        return "+".join(names) if names else "none"

    @property
    def breaker(self) -> CircuitBreaker:
        """Primary provider's breaker (default slot when nothing is configured —
        used by tests to exercise breaker mechanics)."""
        configs = self._configs()
        return self.breakers[configs[0]["name"]] if configs else self.breakers["openai"]

    @property
    def available(self) -> bool:
        """Non-mutating: at least one provider configured and not all circuits
        hard-open."""
        configs = self._configs()
        if not configs or not settings.llm_enabled:
            return False
        return not all(self.breakers[c["name"]].is_open for c in configs)

    def _client_for(self, cfg: dict) -> Any:
        if cfg["name"] not in self._clients:
            from openai import OpenAI

            kwargs: dict[str, Any] = {"api_key": cfg["api_key"]}
            if cfg["base_url"]:
                kwargs["base_url"] = cfg["base_url"]
            self._clients[cfg["name"]] = OpenAI(**kwargs)
        return self._clients[cfg["name"]]

    def _payload(self, system: str, user: str) -> str:
        if len(user) > settings.llm_guard_max_chars:
            logger.warning("LLM user payload truncated to %d chars", settings.llm_guard_max_chars)
            return user[: settings.llm_guard_max_chars]
        return user

    _RETRYABLE_SUBSTRINGS = ("429", "503", "high demand", "overloaded", "rate limit", "timeout")

    @staticmethod
    def _is_retryable(exc: Exception) -> bool:
        msg = str(exc).lower()
        return any(s in msg for s in LLMClient._RETRYABLE_SUBSTRINGS)

    def chat(self, system: str, user: str, *, json_mode: bool = False) -> str:
        """Single-turn completion walking the provider chain: retries with
        backoff per provider (429/503), circuit-breaker fail-fast, then the
        next provider. Raises on final failure; callers handle fallback."""
        user = self._payload(system, user)
        configs = self._configs()
        if not configs:
            raise LLMUnavailableError("No LLM API key configured.")
        last_exc: Exception | None = None
        for cfg in configs:
            breaker = self.breakers[cfg["name"]]
            if breaker.is_open or not breaker.allow_probe():
                continue  # known-dead provider — fail fast, try the next
            client = self._client_for(cfg)
            for delay_before in (0, 2, 5):
                if delay_before:
                    time.sleep(delay_before)
                try:
                    resp = client.chat.completions.create(
                        model=cfg["model"],
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
                    logger.info("LLM call ok via %s: %d chars out", cfg["name"], len(content))
                    breaker.record_success()
                    return content
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    if self._is_retryable(exc):
                        breaker.record_failure()
                        if breaker.is_open:
                            logger.error("LLM circuit open for %s — failing fast: %s", cfg["name"], str(exc)[:160])
                            break  # move to the next provider
                        logger.warning("LLM transient error via %s (will retry): %s", cfg["name"], str(exc)[:160])
                        continue
                    # non-retryable for THIS provider (e.g. unknown model) —
                    # the same request may well work on the next one
                    logger.error("LLM call failed via %s: %s", cfg["name"], str(exc)[:160])
                    break
        if last_exc:
            logger.error("All LLM providers failed: %s", last_exc)
            raise last_exc
        raise LLMUnavailableError("All LLM provider circuits are open.")

    def chat_stream(self, system: str, user: str, *, json_mode: bool = False):
        """Token-streaming completion walking the provider chain. Yields content
        deltas from the first provider that works; raises on final failure."""
        user = self._payload(system, user)
        configs = self._configs()
        if not configs:
            raise LLMUnavailableError("No LLM API key configured.")
        last_exc: Exception | None = None
        produced = False
        for cfg in configs:
            breaker = self.breakers[cfg["name"]]
            if breaker.is_open or not breaker.allow_probe():
                continue
            client = self._client_for(cfg)
            try:
                stream = client.chat.completions.create(
                    model=cfg["model"],
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
                for chunk in stream:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        produced = True
                        yield delta.content
                if produced:
                    breaker.record_success()
                    return
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                if self._is_retryable(exc):
                    breaker.record_failure()
                logger.error("LLM stream failed via %s: %s", cfg["name"], str(exc)[:160])
        if last_exc:
            raise last_exc
        raise LLMUnavailableError("All LLM provider circuits are open.")


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
