import json
import socket
import urllib.error

from src.semantic_governance.client import RestSemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption
from src.semantic_governance.telemetry import InMemorySemanticTelemetry, redact_secrets


def request() -> SemanticDecisionRequest:
    return SemanticDecisionRequest(
        run_id="run-1", step=20, actor="governor", question_id="profile-v1",
        state={"population": 10}, question="Choose",
        options=(SemanticOption("keep_previous", "Keep"), SemanticOption("unknown", "Unknown")),
    )


def test_timeout_is_explicit_and_does_not_escape():
    calls = []

    def timeout(url, payload, headers, seconds):
        calls.append((url, seconds))
        raise socket.timeout()

    provider = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", transport=timeout, max_retries=1
    )
    result = provider.decide(request())
    assert len(calls) == 2
    assert result.selected_option == "unknown"
    assert result.route == "fallback"
    assert result.fallback_reason == "timeout"


def test_invalid_response_fails_closed_without_retrying():
    calls = 0

    def invalid(url, payload, headers, seconds):
        nonlocal calls
        calls += 1
        return 200, b'{"selected_option":"keep_previous"}', {}

    result = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", transport=invalid, max_retries=4
    ).decide(request())
    assert calls == 1
    assert result.fallback_reason == "invalid_response"


def test_http_error_preserves_status_without_copying_response_details():
    calls = 0

    def bad_gateway(url, payload, headers, seconds):
        nonlocal calls
        calls += 1
        raise urllib.error.HTTPError(
            url, 502, "private upstream detail", hdrs={}, fp=None
        )

    result = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", transport=bad_gateway, max_retries=1
    ).decide(request())
    assert calls == 2
    assert result.selected_option == "unknown"
    assert result.fallback_reason == "http_status_502"
    assert "private upstream detail" not in json.dumps(result.to_dict())


def test_valid_response_preserves_request_identity_and_request_id():
    def valid(url, payload, headers, seconds):
        sent = json.loads(payload)
        assert sent["state"] == {"population": 10}
        return 200, json.dumps({
            "selected_option": "keep_previous",
            "probabilities": {"keep_previous": 0.8, "unknown": 0.2},
            "confidence": 0.7,
            "inference_time_ms": 3.5,
        }).encode(), {"X-Request-ID": "req-7"}

    req = request()
    result = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", transport=valid
    ).decide(req)
    assert result.input_hash == req.input_hash
    assert result.selected_option == "keep_previous"
    assert result.request_id == "req-7"


def test_low_confidence_is_an_explicit_fallback():
    def uncertain(url, payload, headers, seconds):
        return 200, json.dumps({
            "selected_option": "keep_previous",
            "probabilities": {"keep_previous": 0.51, "unknown": 0.49},
            "confidence": 0.1,
            "inference_time_ms": 2.0,
        }).encode(), {}

    result = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", transport=uncertain, min_confidence=0.65
    ).decide(request())
    assert result.selected_option == "unknown"
    assert result.route == "fallback"
    assert result.fallback_reason == "low_confidence"


def test_secret_is_only_read_from_environment_and_never_in_telemetry(monkeypatch):
    monkeypatch.setenv("JEV_TEST_KEY", "top-secret-value")
    telemetry = InMemorySemanticTelemetry()

    def failure(url, payload, headers, seconds):
        assert headers["Authorization"] == "Bearer top-secret-value"
        raise OSError("top-secret-value must not be copied")

    result = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", api_key_env="JEV_TEST_KEY",
        transport=failure, max_retries=0, telemetry=telemetry,
    ).decide(request())
    rendered = json.dumps(telemetry.rows)
    assert result.fallback_reason == "transport_error"
    assert "top-secret-value" not in rendered
    assert redact_secrets({"authorization": "Bearer x"}) == {"authorization": "[REDACTED]"}


def test_circuit_breaker_opens_after_the_configured_failures():
    calls = 0

    def failure(url, payload, headers, seconds):
        nonlocal calls
        calls += 1
        raise OSError("offline")

    provider = RestSemanticDecisionProvider(
        "http://127.0.0.1:8008", transport=failure, max_retries=0,
        circuit_failure_threshold=2, circuit_cooldown_seconds=60,
    )
    assert provider.decide(request()).fallback_reason == "transport_error"
    assert provider.decide(request()).fallback_reason == "transport_error"
    assert provider.decide(request()).fallback_reason == "circuit_open"
    assert calls == 2
