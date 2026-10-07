"""Provider Laya: protocollo /v1/systemone di Jev, senza chiave ne' spesa."""

import json

import pytest

from src.semantic_governance.batch import decide_many
from src.semantic_governance.factory import build_semantic_provider
from src.semantic_governance.schemas import (
    MAX_OPTIONS_TYPESAFE,
    SemanticDecisionRequest,
    SemanticOption,
)


def _req(i=0):
    return SemanticDecisionRequest(
        run_id="r", step=1, actor=f"agent:a{i}", question_id="q", state={"h": i},
        question="Which?",
        options=(SemanticOption("x", "X"), SemanticOption("y", "Y"), SemanticOption("unknown", "U")),
        max_options=MAX_OPTIONS_TYPESAFE,
    )


class Recorder:
    def __init__(self):
        self.calls = []

    def __call__(self, url, payload, headers, timeout):
        body = json.loads(payload)
        self.calls.append((url, body, dict(headers)))
        answers = {
            k: {"choice": "o0", "probabilities": {"o0": 0.7, "o1": 0.2, "o2": 0.1},
                "confidence": 0.9}
            for k in body["questions"]
        }
        return 200, json.dumps({"answers": answers, "usage": {"input_tokens": 5}}).encode(), {}


def _provider(transport, **extra):
    config = {"provider": "laya", "endpoint": "http://127.0.0.1:8123", **extra}
    provider = build_semantic_provider(config)
    provider._transport = transport
    return provider


def test_endpoint_is_required():
    with pytest.raises(ValueError, match="endpoint"):
        build_semantic_provider({"provider": "laya"})


def test_no_key_no_ledger_and_the_state_is_not_packed():
    rec = Recorder()
    decision = _provider(rec).decide(_req())
    url, body, headers = rec.calls[0]
    assert url == "http://127.0.0.1:8123/v1/systemone"
    assert "Authorization" not in headers
    assert body["state"] == {"h": 0}          # stato del caso, non impacchettato
    assert len(body["questions"]) == 1
    assert decision.selected_option == "x"
    assert decision.runtime == "laya"


def test_many_requests_are_one_question_each_and_keep_order():
    rec = Recorder()
    out = decide_many(_provider(rec), [_req(i) for i in range(6)], max_workers=3)
    assert len(rec.calls) == 6
    assert all(len(body["questions"]) == 1 for _, body, _ in rec.calls)
    assert [d.actor for d in out] == [f"agent:a{i}" for i in range(6)]


def test_remote_http_host_is_allowed_without_a_key():
    provider = build_semantic_provider({"provider": "laya", "endpoint": "http://10.0.0.5:8000"})
    assert provider is not None


def test_optional_key_is_sent_when_configured(monkeypatch):
    monkeypatch.setenv("LAYA_API_KEY", "sk-laya")
    rec = Recorder()
    _provider(rec, endpoint="https://farm.example:8443", api_key_env="LAYA_API_KEY").decide(_req())
    assert rec.calls[0][2]["Authorization"] == "Bearer sk-laya"


def test_typesafe_still_requires_budget_and_key_name():
    with pytest.raises(ValueError, match="budget_usd"):
        build_semantic_provider({"provider": "typesafe"})


def test_cli_section_supports_laya():
    from scripts.run_governor_experiment import _semantic_section

    with pytest.raises(ValueError, match="endpoint"):
        _semantic_section("laya", run_id="x")
    section = _semantic_section("laya", run_id="x", endpoint="http://127.0.0.1:8123")
    assert section["provider"] == "laya" and section["endpoint"] == "http://127.0.0.1:8123"
