"""Simple in-memory token-bucket rate limiter.

Per-client (API key or IP) limiting to protect the expensive agent endpoints.
In-memory is fine for a single instance; M6's scaling notes cover moving the bucket
to Redis so limits hold across horizontally-scaled workers."""
from __future__ import annotations

import time
from collections import defaultdict

from fastapi import HTTPException, Request


class TokenBucket:
    def __init__(self, rate: float, capacity: int):
        self.rate = rate          # tokens refilled per second
        self.capacity = capacity  # burst size
        self._tokens: dict[str, float] = defaultdict(lambda: capacity)
        self._last: dict[str, float] = defaultdict(time.monotonic)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        elapsed = now - self._last[key]
        self._last[key] = now
        self._tokens[key] = min(self.capacity, self._tokens[key] + elapsed * self.rate)
        if self._tokens[key] >= 1.0:
            self._tokens[key] -= 1.0
            return True
        return False


# ~1 run/sec sustained, burst of 5. Tune per deployment.
_bucket = TokenBucket(rate=1.0, capacity=5)


def rate_limit(request: Request) -> None:
    key = request.headers.get("x-api-key") or (request.client.host if request.client else "anon")
    if not _bucket.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


from app.core.config import settings  # noqa: E402


def require_passcode(request: Request) -> None:
    """Simple passcode gate for the WIP hosted microsite (default 0000).
    Accepts X-Cortex-Passcode header or ?passcode= query. Disabled when unset."""
    if not settings.passcode:
        return
    supplied = request.headers.get("x-cortex-passcode") or request.query_params.get("passcode", "")
    if supplied != settings.passcode:
        raise HTTPException(status_code=401, detail="passcode required")


# Per-IP daily run cap (protects the gated free-tier live runs). In-memory; resets on
# restart. Production would back this with Redis + a real date key.
_daily: dict[str, int] = {}
_daily_day: dict[str, str] = {}


def check_daily_cap(request: Request) -> None:
    import datetime

    day = datetime.date.today().isoformat()
    ip = request.client.host if request.client else "anon"
    if _daily_day.get(ip) != day:
        _daily_day[ip] = day
        _daily[ip] = 0
    if _daily[ip] >= settings.daily_run_cap:
        raise HTTPException(status_code=429, detail="daily run limit reached, see the instant demo")
    _daily[ip] += 1
