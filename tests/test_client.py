"""LLM adapter internals: rate limiting and retry-delay parsing."""

import time

from src.extract.client import _RateLimiter, _retry_delay_seconds


def test_retry_delay_parsing():
    assert _retry_delay_seconds("Please retry in 55.07s") == 55.07
    assert _retry_delay_seconds("{'retryDelay': '30s'}") == 30.0
    assert _retry_delay_seconds("no delay here") is None


def test_rate_limiter_paces_calls():
    limiter = _RateLimiter(per_minute=600)  # 0.1s minimum interval
    start = time.monotonic()
    for _ in range(3):
        limiter.wait()
    elapsed = time.monotonic() - start
    assert elapsed >= 0.2  # at least two gaps of 0.1s


def test_rate_limiter_disabled_when_zero():
    limiter = _RateLimiter(per_minute=0)
    start = time.monotonic()
    for _ in range(5):
        limiter.wait()
    assert time.monotonic() - start < 0.05
