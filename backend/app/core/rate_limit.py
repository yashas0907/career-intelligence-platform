"""Lightweight in-process rate limiter (no external deps).

Fixed-window per client IP. Good enough for a small-scale deployment;
swap for redis-based limiting if this ever runs at real load.
"""

from __future__ import annotations

import time
from collections import defaultdict

from fastapi import Request
from fastapi.responses import JSONResponse


class RateLimiter:
    def __init__(self, requests_per_minute: int = 30, window_s: int = 60) -> None:
        self.limit = requests_per_minute
        self.window = window_s
        self._hits: dict[str, list[float]] = defaultdict(list)

    def check(self, key: str) -> tuple[bool, float]:
        """Returns (allowed, retry_after_seconds)."""
        now = time.time()
        window_start = now - self.window
        hits = [t for t in self._hits.get(key, ()) if t > window_start]
        if len(hits) >= self.limit:
            self._hits[key] = hits
            retry = self.window - (now - hits[0])
            return False, max(0.0, retry)
        hits.append(now)
        self._hits[key] = hits
        return True, 0.0

    def prune(self) -> None:
        cutoff = time.time() - self.window
        for key in list(self._hits.keys()):
            self._hits[key] = [t for t in self._hits[key] if t > cutoff]
            if not self._hits[key]:
                del self._hits[key]
