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


# --- wait-and-resume on quota errors ---------------------------------------

class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def __init__(self, fail_times: int) -> None:
        self.fail_times = fail_times
        self.calls = 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError(
                "429 RESOURCE_EXHAUSTED Quota exceeded; retryDelay': '1s'"
            )
        return _FakeResponse('{"candidate_name": "X", "skills": [], "projects": [], "experience_highlights": []}')


class _FakeClient:
    def __init__(self, fail_times: int) -> None:
        self.models = _FakeModels(fail_times)


def _client_with(fail_times: int, **kwargs):
    from src.cache import NullCache
    from src.extract.client import GeminiClient

    c = GeminiClient("fake-key", cache=NullCache(), **kwargs)
    c._client = _FakeClient(fail_times)
    return c


def test_quota_error_waits_then_resumes():
    client = _client_with(1, max_wait_seconds=2, wait_budget_seconds=10)
    data = client.generate_json(system="s", prompt="p")
    assert data["candidate_name"] == "X"
    assert client.rate_limit_waits == 1
    assert client.total_wait_seconds >= 1.0


def test_quota_error_fails_fast_when_disabled():
    import pytest

    from src.extract.client import LLMError

    client = _client_with(1, on_rate_limit="fail", max_retries=0)
    with pytest.raises(LLMError):
        client.generate_json(system="s", prompt="p")
    assert client.rate_limit_waits == 0


def test_quota_error_stops_when_budget_exhausted():
    import pytest

    from src.extract.client import LLMError

    # Budget too small to allow even one 1s wait.
    client = _client_with(5, max_wait_seconds=1, wait_budget_seconds=0, max_retries=0)
    with pytest.raises(LLMError):
        client.generate_json(system="s", prompt="p")
