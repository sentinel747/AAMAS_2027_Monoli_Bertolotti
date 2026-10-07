import threading

from src.semantic_governance.batch import decide_many
from src.semantic_governance.client import FakeSemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption


def _req(i):
    return SemanticDecisionRequest(
        run_id="r", step=1, actor=f"agent:{i}", question_id="q",
        state={"i": i}, question="Which?",
        options=(SemanticOption("a", "A"), SemanticOption("unknown", "U")),
    )


def test_order_is_preserved_with_a_thread_pool():
    out = decide_many(FakeSemanticDecisionProvider(), [_req(i) for i in range(40)], max_workers=8)
    assert [d.actor for d in out] == [f"agent:{i}" for i in range(40)]


def test_native_batch_is_preferred():
    class Native:
        def __init__(self):
            self.batches = []
        def decide(self, r):
            raise AssertionError("decide must not be used")
        def decide_many(self, rs):
            self.batches.append(len(rs))
            return [FakeSemanticDecisionProvider().decide(r) for r in rs]
    native = Native()
    decide_many(native, [_req(i) for i in range(5)])
    assert native.batches == [5]


def test_exception_on_one_request_becomes_a_fallback():
    class Flaky:
        def decide(self, r):
            if r.actor == "agent:3":
                raise RuntimeError("boom sk-secret")
            return FakeSemanticDecisionProvider().decide(r)
    out = decide_many(Flaky(), [_req(i) for i in range(5)], max_workers=2)
    assert out[3].fallback_reason == "provider_exception"
    assert "sk-secret" not in str(out[3].to_dict())
    assert out[2].route != "fallback"


def test_empty_input_returns_empty_list():
    assert decide_many(FakeSemanticDecisionProvider(), []) == []


def test_rest_breaker_counts_failures_consistently_under_threads():
    from src.semantic_governance.client import RestSemanticDecisionProvider

    def always_fails(url, payload, headers, timeout):
        raise TimeoutError()

    provider = RestSemanticDecisionProvider(
        "http://jev.test", transport=always_fails, max_retries=0,
        circuit_failure_threshold=1000, circuit_cooldown_seconds=60,
    )
    out = decide_many(provider, [_req(i) for i in range(200)], max_workers=32)
    assert all(d.fallback_reason == "timeout" for d in out)
    assert provider._consecutive_failures == 200
