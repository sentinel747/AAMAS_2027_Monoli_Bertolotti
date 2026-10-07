import json

import pytest

from src.semantic_governance.budget import SpendingLedger
from src.semantic_governance.schemas import (
    MAX_OPTIONS_TYPESAFE,
    SemanticDecisionRequest,
    SemanticOption,
)
from src.semantic_governance.typesafe import TypeSafeSemanticDecisionProvider


def _req(step, actor="agent:a1"):
    return SemanticDecisionRequest(
        run_id="r", step=step, actor=actor, question_id="jev-semif-agent-v1:L1",
        state={"hydration": 0.1}, question="What should this colonist prioritise?",
        options=(
            SemanticOption("sustenance", "Food and water"),
            SemanticOption("build", "Build"),
            SemanticOption("unknown", "Insufficient information"),
        ),
        max_options=MAX_OPTIONS_TYPESAFE,
    )


class Recorder:
    def __init__(self, responder):
        self.calls = []
        self.responder = responder

    def __call__(self, url, payload, headers, timeout):
        body = json.loads(payload)
        self.calls.append((url, body, dict(headers)))
        return self.responder(body)


def _answer_all(body, winner="o0", p=0.9, tokens=100):
    answers = {}
    for key, q in body["questions"].items():
        keys = list(q["criteria"])
        rest = (1.0 - p) / (len(keys) - 1)
        answers[key] = {
            "type": "choice", "choice": winner,
            "probabilities": {k: (p if k == winner else rest) for k in keys},
            "confidence": 0.8,
        }
    return 200, json.dumps({"answers": answers, "usage": {"input_tokens": tokens}}).encode(), {}


@pytest.fixture
def env_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_AI", "sk-test-secret")


def _provider(tmp_path, transport, cap=0.5, **kw):
    return TypeSafeSemanticDecisionProvider(
        ledger=SpendingLedger(tmp_path / "ledger.json", cap_usd=cap),
        transport=transport, sleep=lambda s: None, **kw,
    )


def test_wire_format_and_mapping_back(tmp_path, env_key):
    rec = Recorder(_answer_all)
    decision = _provider(tmp_path, rec).decide(_req(1))
    url, body, headers = rec.calls[0]
    assert url == "https://api.typesafe.ai/v1/systemone"
    assert headers["Authorization"] == "Bearer sk-test-secret"
    assert body["model"] == "jev-latest"
    question = next(iter(body["questions"].values()))
    assert question["type"] == "choice"
    assert question["criteria"] == {
        "o0": "Food and water", "o1": "Build", "o2": "Insufficient information",
    }
    assert decision.selected_option == "sustenance"
    assert set(decision.probabilities) == {"sustenance", "build", "unknown"}
    assert decision.runtime == "jev-typesafe"


def test_many_requests_travel_in_one_body_and_keep_order(tmp_path, env_key):
    rec = Recorder(_answer_all)
    decisions = _provider(tmp_path, rec).decide_many(
        [_req(1, "agent:a1"), _req(1, "agent:a2"), _req(1, "agent:a3")]
    )
    assert len(rec.calls) == 1
    body = rec.calls[0][1]
    assert set(body["state"]) == {"q0", "q1", "q2"}
    assert body["questions"]["q1"]["instructions"].startswith("Answer only about state.q1.")
    assert [d.actor for d in decisions] == ["agent:a1", "agent:a2", "agent:a3"]


def test_usage_tokens_are_recorded_in_the_ledger(tmp_path, env_key):
    rec = Recorder(lambda b: _answer_all(b, tokens=1_000_000))
    provider = _provider(tmp_path, rec, cap=0.5)
    provider.decide(_req(1))
    assert provider.ledger.summary()["input_tokens"] == 1_000_000


def test_budget_already_exhausted_sends_nothing(tmp_path, env_key):
    (tmp_path / "ledger.json").write_text(json.dumps({"input_tokens": 50_000_000, "requests": 1}))
    rec = Recorder(_answer_all)
    decision = _provider(tmp_path, rec).decide(_req(1))
    assert rec.calls == []
    assert decision.route == "fallback"
    assert decision.fallback_reason == "budget_exceeded"


def test_missing_key_sends_nothing(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_AI", raising=False)
    rec = Recorder(_answer_all)
    decision = _provider(tmp_path, rec).decide(_req(1))
    assert rec.calls == []
    assert decision.fallback_reason == "missing_api_key_env"


def test_429_waits_retry_after_once_then_succeeds(tmp_path, env_key):
    waits = []
    state = {"n": 0}

    def responder(body):
        state["n"] += 1
        if state["n"] == 1:
            return 429, b"{}", {"Retry-After": "2"}
        return _answer_all(body)

    provider = TypeSafeSemanticDecisionProvider(
        ledger=SpendingLedger(tmp_path / "l.json", cap_usd=0.5),
        transport=Recorder(responder), sleep=waits.append,
    )
    decision = provider.decide(_req(1))
    assert waits == [2.0]
    assert decision.route != "fallback"


def test_http_error_gives_explicit_fallback_for_every_request(tmp_path, env_key):
    rec = Recorder(lambda b: (500, b"{}", {}))
    decisions = _provider(tmp_path, rec).decide_many([_req(1, "agent:a"), _req(1, "agent:b")])
    assert [d.fallback_reason for d in decisions] == ["http_status_500", "http_status_500"]


def test_low_confidence_falls_back(tmp_path, env_key):
    def responder(body):
        status, raw, headers = _answer_all(body)
        data = json.loads(raw)
        for answer in data["answers"].values():
            answer["confidence"] = 0.1
        return status, json.dumps(data).encode(), headers

    decision = _provider(tmp_path, Recorder(responder)).decide(_req(1))
    assert decision.fallback_reason == "low_confidence"


def test_large_batches_are_split_under_the_token_budget(tmp_path, env_key):
    rec = Recorder(_answer_all)
    provider = _provider(tmp_path, rec, max_prompt_tokens=300)
    decisions = provider.decide_many([_req(1, f"agent:a{i}") for i in range(12)])
    assert len(rec.calls) > 1
    assert len(decisions) == 12


def test_secret_never_appears_in_decisions(tmp_path, env_key):
    decision = _provider(tmp_path, Recorder(lambda b: (500, b"sk-test-secret", {}))).decide(_req(1))
    assert "sk-test-secret" not in json.dumps(decision.to_dict())


def test_factory_refuses_typesafe_without_budget():
    from src.semantic_governance.factory import build_semantic_provider
    with pytest.raises(ValueError, match="budget_usd"):
        build_semantic_provider({"provider": "typesafe"})


def test_factory_builds_typesafe_with_explicit_budget(tmp_path):
    from src.semantic_governance.factory import build_semantic_provider
    provider = build_semantic_provider({
        "provider": "typesafe", "budget_usd": 0.5,
        "ledger_path": str(tmp_path / "l.json"),
    })
    assert isinstance(provider, TypeSafeSemanticDecisionProvider)


def test_token_estimate_is_conservative_for_numeric_json():
    # Misura dal vivo (2026-09-23): token reali ~1,3 volte caratteri/3 sugli
    # stati numerici; blocchi stimati a 60k erano ~78k e il server rispondeva
    # 400. La stima deve stare sopra il reale: caratteri/2.
    body = {"state": {f"q{i}": {"hydration": 0.12, "satiety": 0.8, "cell_occupants": 190}
                      for i in range(200)}}
    raw = len(json.dumps(body, ensure_ascii=False))
    assert TypeSafeSemanticDecisionProvider._estimate_tokens(body) >= raw // 2


def test_packed_state_stays_under_the_per_question_limit(tmp_path, env_key):
    # TypeSafe: stato + domanda piu' lunga <= ~32k token. Impacchettando, lo
    # stato e' l'unione dei casi: con stati grandi (governo) va spezzato prima
    # del limite totale (misura del 2026-09-23: 400 su quasi tutti i blocchi).
    big = {"indicators": {f"k{i}": {"mean": 1.2345, "std": 0.1234} for i in range(60)}}
    rec = Recorder(_answer_all)
    provider = _provider(tmp_path, rec, max_prompt_tokens=60_000)

    def req(i):
        return SemanticDecisionRequest(
            run_id="r", step=i, actor="governor", question_id="g", state=big,
            question="Which area?",
            options=(SemanticOption("x", "X"), SemanticOption("unknown", "U")),
            max_options=MAX_OPTIONS_TYPESAFE,
        )

    provider.decide_many([req(i) for i in range(40)])
    for _, body, _ in rec.calls:
        state_tokens = TypeSafeSemanticDecisionProvider._estimate_tokens({"state": body["state"]})
        from src.semantic_governance.typesafe import MAX_STATE_TOKENS
        assert state_tokens <= MAX_STATE_TOKENS
    assert len(rec.calls) > 1
