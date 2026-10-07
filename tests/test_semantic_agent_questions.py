import numpy as np
import pytest

from src.agents import pillars
from src.semantic_governance import agent_questions as aq
from src.semantic_governance.schemas import SemanticDecision


def _decision(request, probabilities, route="filter"):
    selected = max(probabilities, key=probabilities.get)
    return SemanticDecision(
        run_id=request.run_id, step=request.step, actor=request.actor,
        question_id=request.question_id, input_hash=request.input_hash,
        selected_option=selected, probabilities=probabilities,
        confidence=0.5, route=route,
    )


COLONY = {"population": 10, "mean_hydration": 0.8}
RECORD = {"hydration": 0.1, "satiety": 0.9}


def test_level1_has_five_pillars_plus_unknown_and_no_social():
    r = aq.level1_request("r", 3, "agent_001", COLONY, RECORD)
    assert r.option_ids == ("sustenance", "resources", "build", "life", "explore", "unknown")
    assert r.actor == "agent:agent_001"
    assert r.question_id == aq.L1_QUESTION_ID


def test_uniform_distribution_is_neutral():
    r = aq.level1_request("r", 3, "a", COLONY, RECORD)
    f = aq.pillar_factors(_decision(r, {o: 1 / 6 for o in r.option_ids}))
    assert np.allclose(f, 1.0)


def test_unknown_certainty_is_exactly_neutral():
    r = aq.level1_request("r", 3, "a", COLONY, RECORD)
    probs = {o: 0.0 for o in r.option_ids}
    probs["unknown"] = 1.0
    assert np.array_equal(aq.pillar_factors(_decision(r, probs)), np.ones(pillars.N_PILLARS))


def test_error_fallbacks_are_exactly_neutral():
    # Revisione finale (C1): il neutro non dipende da `route` ma dalla
    # distribuzione. I fallback d'errore mettono massa 1 su `unknown`.
    from src.semantic_governance.client import fallback_decision

    r = aq.level1_request("r", 3, "a", COLONY, RECORD)
    for reason in ("timeout", "low_confidence", "http_status_500", "budget_exceeded"):
        f = aq.pillar_factors(fallback_decision(r, reason))
        assert np.array_equal(f, np.ones(pillars.N_PILLARS))


def test_confident_choice_boosts_its_pillar_and_keeps_social_neutral():
    r = aq.level1_request("r", 3, "a", COLONY, RECORD)
    probs = {o: 0.0 for o in r.option_ids}
    probs["sustenance"] = 1.0
    f = aq.pillar_factors(_decision(r, probs))
    assert f[pillars.P_SUSTENANCE] == pytest.approx(5.0)
    assert f[pillars.P_BUILD] == pytest.approx(0.0)
    assert f[pillars.P_SOCIAL] == 1.0


def test_level2_exists_only_for_resources_build_explore():
    assert aq.LEVEL2_PILLARS == ("resources", "build", "explore")
    r = aq.level2_request("r", 3, "a", "build", COLONY, RECORD)
    assert len(r.option_ids) == 13 and r.option_ids[-1] == "unknown"
    with pytest.raises(ValueError):
        aq.level2_request("r", 3, "a", "life", COLONY, RECORD)


def test_action_factors_touch_only_the_pillar_actions():
    r = aq.level2_request("r", 3, "a", "resources", COLONY, RECORD)
    probs = {o: 0.0 for o in r.option_ids}
    probs["collect_minerals"] = 1.0
    f = aq.action_factors(_decision(r, probs), "resources")
    assert f.shape == (pillars.N_ACTIONS,)
    from src.agents.action_space import ActionType
    assert f[pillars.ACTION_INDEX[ActionType.COLLECT_MINERALS]] == pytest.approx(2.0)
    assert f[pillars.ACTION_INDEX[ActionType.COLLECT_MATERIALS]] == pytest.approx(0.0)
    assert f[pillars.ACTION_INDEX[ActionType.REST]] == 1.0


def test_requests_are_deterministic_for_replay():
    a = aq.level1_request("r", 3, "a", COLONY, RECORD)
    b = aq.level1_request("r", 3, "a", dict(COLONY), dict(RECORD))
    assert a.input_hash == b.input_hash
