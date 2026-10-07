import json

import pytest

from src.semantic_governance.client import FakeSemanticDecisionProvider
from src.semantic_governance.replay import ReplaySemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption


def request(state=None) -> SemanticDecisionRequest:
    return SemanticDecisionRequest(
        run_id="run-1", step=20, actor="governor", question_id="profile-v1",
        state=state or {"population": 10}, question="Choose",
        options=(SemanticOption("keep_previous", "Keep"), SemanticOption("unknown", "Unknown")),
    )


def test_fake_is_deterministic_and_records_calls():
    provider = FakeSemanticDecisionProvider({"profile-v1": "keep_previous"})
    first = provider.decide(request())
    second = provider.decide(request())
    assert first == second
    assert len(provider.requests) == 2


def test_replay_is_deterministic_and_uses_no_transport(tmp_path):
    req = request()
    recorded = FakeSemanticDecisionProvider({"profile-v1": "keep_previous"}).decide(req)
    path = tmp_path / "semantic_decisions.jsonl"
    path.write_text(json.dumps(recorded.to_dict()) + "\n", encoding="utf-8")
    replay = ReplaySemanticDecisionProvider.from_jsonl(path)
    first = replay.decide(req)
    second = replay.decide(req)
    assert first == second
    assert first.route == "replay"
    assert first.rationale["replayed_route"] == "filter"


def test_replay_requires_an_exact_input_hash():
    original = request()
    replay = ReplaySemanticDecisionProvider([
        FakeSemanticDecisionProvider({"profile-v1": "keep_previous"}).decide(original)
    ])
    with pytest.raises(KeyError, match="exact match"):
        replay.decide(request({"population": 11}))
