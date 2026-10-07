import pytest

from src.semantic_governance.schemas import (
    SCHEMA_VERSION,
    SemanticDecision,
    SemanticDecisionRequest,
    SemanticOption,
)


def request() -> SemanticDecisionRequest:
    return SemanticDecisionRequest(
        run_id="run-1",
        step=20,
        actor="governor",
        question_id="policy-profile-v1",
        state={"population": 10, "food": 4.5},
        question="Which bounded policy should be used?",
        options=(
            SemanticOption("keep_previous", "Keep the policy already in force"),
            SemanticOption("unknown", "Insufficient evidence"),
        ),
    )


def test_request_hash_is_canonical_and_versioned():
    first = request()
    second = request()
    assert first.schema_version == SCHEMA_VERSION
    assert first.input_hash == second.input_hash
    assert first.wire_payload()["options"][1]["id"] == "unknown"


def test_request_requires_a_fallback_option():
    with pytest.raises(ValueError, match="unknown"):
        SemanticDecisionRequest(
            run_id="run-1", step=0, actor="governor", question_id="q",
            state={"x": 1}, question="Choose",
            options=(SemanticOption("a", "A"), SemanticOption("b", "B")),
        )


def test_decision_rejects_a_partial_distribution():
    req = request()
    with pytest.raises(ValueError, match="sum to 1"):
        SemanticDecision(
            run_id=req.run_id, step=req.step, actor=req.actor,
            question_id=req.question_id, input_hash=req.input_hash,
            selected_option="unknown", probabilities={"unknown": 0.5},
            confidence=0.0, route="fallback",
        )


# --- Limite di opzioni per richiesta (piano 2026-09-22, Task 1) -------------


def _options(n):
    from src.semantic_governance.schemas import SemanticOption
    ids = [f"o{i}" for i in range(n - 1)] + ["unknown"]
    return tuple(SemanticOption(i, f"option {i}") for i in ids)


def _request(n, **extra):
    from src.semantic_governance.schemas import SemanticDecisionRequest
    return SemanticDecisionRequest(
        run_id="r", step=1, actor="a", question_id="q",
        state={"k": 1}, question="Which?", options=_options(n), **extra,
    )


def test_default_limit_is_still_sixteen():
    import pytest
    _request(16)
    with pytest.raises(ValueError, match="options must contain"):
        _request(17)


def test_typesafe_limit_accepts_more_options():
    from src.semantic_governance.schemas import MAX_OPTIONS_TYPESAFE
    assert MAX_OPTIONS_TYPESAFE == 255
    assert len(_request(29, max_options=MAX_OPTIONS_TYPESAFE).options) == 29


def test_limit_above_typesafe_is_rejected():
    import pytest
    with pytest.raises(ValueError, match="max_options"):
        _request(3, max_options=256)


def test_max_options_does_not_change_the_input_hash():
    assert _request(5).input_hash == _request(5, max_options=255).input_hash
