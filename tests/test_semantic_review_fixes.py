"""Regressioni dalla revisione finale del piano SemIf per agente (2026-09-22).

Ogni test riproduce un difetto trovato dal revisore indipendente: C1, I1-I4 e
i tre promossi per effetto (chiave in chiaro, tetto che si azzera, timeout non
contabilizzati).
"""

import json

import numpy as np
import pytest

from src.agents import pillars
from src.semantic_governance import agent_questions as aq
from src.semantic_governance.batch import decide_many
from src.semantic_governance.budget import SpendingLedger
from src.semantic_governance.client import FakeSemanticDecisionProvider, fallback_decision
from src.semantic_governance.replay import ReplaySemanticDecisionProvider
from src.semantic_governance.schemas import (
    MAX_OPTIONS_TYPESAFE,
    SemanticDecision,
    SemanticDecisionRequest,
    SemanticOption,
)
from src.semantic_governance.typesafe import TypeSafeSemanticDecisionProvider

COLONY = {"population": 10}
RECORD = {"hydration": 0.5}


# --- C1: replay di una run live -------------------------------------------------

def test_live_unknown_top_answer_and_its_replay_give_the_same_factors():
    request = aq.level1_request("r", 3, "a1", COLONY, RECORD)
    probabilities = {"sustenance": 0.1, "resources": 0.1, "build": 0.25,
                     "life": 0.1, "explore": 0.1, "unknown": 0.35}
    live = SemanticDecision(
        run_id="r", step=3, actor=request.actor, question_id=request.question_id,
        input_hash=request.input_hash, selected_option="unknown",
        probabilities=probabilities, confidence=0.1, route="fallback",
        fallback_reason="selected_fallback",
    )
    replayed = ReplaySemanticDecisionProvider([live]).decide(request)
    assert np.array_equal(aq.pillar_factors(live), aq.pillar_factors(replayed))


def test_error_fallback_is_still_exactly_neutral():
    request = aq.level1_request("r", 3, "a1", COLONY, RECORD)
    factors = aq.pillar_factors(fallback_decision(request, "timeout"))
    assert np.array_equal(factors, np.ones(pillars.N_PILLARS))


# --- I3: un replay mancato non va inghiottito -----------------------------------

def _req(i):
    return SemanticDecisionRequest(
        run_id="r", step=1, actor=f"agent:{i}", question_id="q", state={"i": i},
        question="Which?", options=(SemanticOption("a", "A"), SemanticOption("unknown", "U")),
    )


def test_replay_miss_is_not_swallowed_by_the_batch():
    with pytest.raises(KeyError):
        decide_many(ReplaySemanticDecisionProvider([]), [_req(0), _req(1)], max_workers=2)


# --- I2: nessuna eccezione del provider arriva alla simulazione ------------------

def test_native_batch_exception_becomes_fallback_for_everyone():
    class Broken:
        def decide_many(self, requests):
            raise RuntimeError("boom sk-secret")

    out = decide_many(Broken(), [_req(0), _req(1)])
    assert [d.fallback_reason for d in out] == ["provider_exception"] * 2
    assert "sk-secret" not in json.dumps([d.to_dict() for d in out])


def test_native_batch_with_wrong_length_becomes_fallback():
    class Short:
        def decide_many(self, requests):
            return [FakeSemanticDecisionProvider().decide(requests[0])]

    out = decide_many(Short(), [_req(0), _req(1), _req(2)])
    assert [d.fallback_reason for d in out] == ["provider_exception"] * 3


def _ts_req(step=1):
    return SemanticDecisionRequest(
        run_id="r", step=step, actor="agent:a", question_id="q", state={"h": 0.1},
        question="Which?",
        options=(SemanticOption("x", "X"), SemanticOption("unknown", "U")),
        max_options=MAX_OPTIONS_TYPESAFE,
    )


def _ts(tmp_path, transport, **kw):
    return TypeSafeSemanticDecisionProvider(
        ledger=kw.pop("ledger", None) or SpendingLedger(tmp_path / "l.json", cap_usd=0.5),
        transport=transport, sleep=lambda s: None, **kw,
    )


@pytest.fixture
def key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_AI", "sk-test")


@pytest.mark.parametrize("body", [b"null", b"[1, 2]", b'{"answers": {}, "usage": 7}'])
def test_malformed_200_bodies_fall_back_without_exception(tmp_path, key, body):
    decision = _ts(tmp_path, lambda *a: (200, body, {})).decide(_ts_req())
    assert decision.route == "fallback"
    assert decision.fallback_reason == "invalid_response"


def test_corrupt_ledger_falls_back_without_exception(tmp_path, key):
    (tmp_path / "l.json").write_text("{not json", encoding="utf-8")
    calls = []
    decision = _ts(tmp_path, lambda *a: calls.append(a) or (500, b"", {})).decide(_ts_req())
    assert calls == []
    assert decision.fallback_reason == "ledger_error"


def test_ledger_write_error_does_not_escape(tmp_path, key):
    class Failing(SpendingLedger):
        def record(self, input_tokens):
            raise PermissionError("locked by antivirus")

    ok = lambda url, payload, headers, timeout: (200, json.dumps({  # noqa: E731
        "answers": {"q0": {"choice": "o0", "probabilities": {"o0": 0.9, "o1": 0.1},
                           "confidence": 0.9}},
        "usage": {"input_tokens": 10}}).encode(), {})
    provider = _ts(tmp_path, ok, ledger=Failing(tmp_path / "l.json", cap_usd=0.5))
    decision = provider.decide(_ts_req())
    assert decision.route == "fallback"
    assert decision.fallback_reason == "ledger_error"


# --- promossi per effetto: soldi e sicurezza ------------------------------------

def test_timeout_after_authorization_is_charged_conservatively(tmp_path, key):
    def timeout(*a):
        raise TimeoutError()

    provider = _ts(tmp_path, timeout)
    provider.decide(_ts_req())
    assert provider.ledger.summary()["input_tokens"] > 0


def test_http_endpoint_is_refused_except_localhost(tmp_path):
    ledger = SpendingLedger(tmp_path / "l.json", cap_usd=0.5)
    with pytest.raises(ValueError, match="https"):
        TypeSafeSemanticDecisionProvider(ledger=ledger, endpoint="http://api.typesafe.ai")
    TypeSafeSemanticDecisionProvider(ledger=ledger, endpoint="http://127.0.0.1:4000")


def test_default_ledger_path_is_absolute_under_the_repository(tmp_path, monkeypatch):
    from pathlib import Path

    from src.semantic_governance.factory import build_semantic_provider

    monkeypatch.chdir(tmp_path)
    provider = build_semantic_provider({"provider": "typesafe", "budget_usd": 0.5})
    path = provider.ledger.path
    assert path.is_absolute()
    assert path.parts[-3:] == ("runs", "jev_semif_experiments", "_typesafe_spend.json")
    assert Path(__file__).resolve().parents[1] in path.parents


# --- I4: fattori che sopravvivono al confine ------------------------------------

def _layer(tmp_path, answer_by_step, **over):
    from scripts.parity_harness import build_config
    from src.semantic_governance.agent_layer import AgentLayerSettings, SemanticAgentLayer
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    def factory(request):
        chosen = answer_by_step(request)
        probabilities = {o: float(o == chosen) for o in request.option_ids}
        return SemanticDecision(
            run_id=request.run_id, step=request.step, actor=request.actor,
            question_id=request.question_id, input_hash=request.input_hash,
            selected_option=chosen, probabilities=probabilities, confidence=1.0,
            route="fallback" if chosen == "unknown" else "filter",
        )

    base = {"mode": "all", "cadence_steps": 1, "levels": 2, "workers": 2,
            "semantic": {"provider": "fake"}}
    base.update(over)
    settings = AgentLayerSettings.from_config({"semantic_agents": base})
    state = AgentCoupledRunner(build_config("golden", 20, 5, 0, "softmax")).core
    layer = SemanticAgentLayer(settings, run_id="r",
                               provider=FakeSemanticDecisionProvider(factory=factory))
    layer.attach_output(tmp_path)
    return layer, state


def test_level2_factors_do_not_survive_a_new_level1_winner(tmp_path):
    def answer(request):
        if request.question_id == aq.L1_QUESTION_ID:
            return "resources" if request.step == 1 else "life"
        return request.option_ids[0]

    layer, state = _layer(tmp_path, answer)
    layer.advance(1, state)
    assert state.semantic_action_factors
    layer.advance(2, state)
    assert state.semantic_action_factors == {}
    layer.close()


def test_hard_mode_resets_agents_that_are_no_longer_ambiguous(tmp_path):
    layer, state = _layer(tmp_path, lambda r: "explore", mode="hard", levels=1)
    ids = list(state.agents.ids)
    layer.advance(1, state)                       # nessun margine: nessuno
    state.pillar_margin_out = {ids[0]: 0.01}
    layer.advance(2, state)
    assert ids[0] in state.semantic_pillar_factors
    state.pillar_margin_out = {ids[0]: 0.9}
    layer.advance(3, state)
    assert ids[0] not in state.semantic_pillar_factors
    layer.close()


# --- I1: il ricalcolo dopo i claim non deve perdere il fattore --------------------

def test_factor_survives_the_claim_recompute_path():
    import src.core.kernel_decision as kd
    import src.simulation.agent_coupled_runner as runner_module
    from scripts.parity_harness import build_config
    from src.core import state_digest
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    strong = np.ones(pillars.N_PILLARS)
    strong[pillars.P_EXPLORE] = 50.0

    def run(force_recompute):
        records = []
        inner_step = runner_module._core_step
        inner_bind = kd.decide_preferences_precomputed

        def bind(*args, **kwargs):
            if force_recompute:
                kwargs["precomputed_scores"] = None
            return inner_bind(*args, **kwargs)

        def step(state, index, dt, config, rng):
            state.semantic_pillar_factors = {a: strong for a in state.agents.ids}
            out = inner_step(state, index, dt, config, rng)
            records.append(state_digest.step_digest(state, out))
            return out

        runner_module._core_step = step
        kd.decide_preferences_precomputed = bind
        try:
            AgentCoupledRunner(build_config("golden", 20, 6, 0, "greedy")).run(
                days=6, output_dir=None)
        finally:
            runner_module._core_step = inner_step
            kd.decide_preferences_precomputed = inner_bind
        return records

    assert run(force_recompute=True) == run(force_recompute=False)


# --- shell web: lo strato non e' supportato e non va ignorato in silenzio ------

def test_web_shell_refuses_an_active_agent_layer():
    from src.api.state_store import SimulationController

    config = {
        "name": "web_semantic_guard", "seed": 1,
        "world": {"width": 20, "height": 20, "map_profile": "balanced"},
        "agents": {"count": 3, "llm_count": 0, "decision_mode": "preferences"},
        "population": {"enabled": False},
        "semantic_agents": {"mode": "all", "semantic": {"provider": "fake"}},
    }
    with pytest.raises(ValueError, match="semantic_agents"):
        SimulationController(config)
