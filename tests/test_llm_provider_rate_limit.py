import httpx

from src.llm.configured_provider import _retry_after_seconds


def test_retry_after_seconds_accepts_delta_seconds():
    request = httpx.Request("POST", "https://api.example.test")
    response = httpx.Response(429, headers={"retry-after": "2.5"}, request=request)
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    assert _retry_after_seconds(exc) == 2.5


def test_retry_after_seconds_ignores_invalid_header():
    request = httpx.Request("POST", "https://api.example.test")
    response = httpx.Response(429, headers={"retry-after": "not-a-date"}, request=request)
    exc = httpx.HTTPStatusError("rate limited", request=request, response=response)

    assert _retry_after_seconds(exc) is None
