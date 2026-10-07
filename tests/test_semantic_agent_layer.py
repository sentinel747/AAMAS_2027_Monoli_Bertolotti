"""Strato SemIf dei singoli agenti (piano 2026-09-22, Task 8)."""

from scripts.parity_harness import build_config
from src.simulation.agent_coupled_runner import AgentCoupledRunner


def _golden_state(seed):
    """Il `CoreState` iniziale della config golden, dal runner vero."""
    config = build_config("golden", 20, 5, seed, "softmax")
    runner = AgentCoupledRunner(config)
    return runner.core, config


import json

import numpy as np

from src.agents import pillars
from src.semantic_governance.agent_layer import AgentLayerSettings, SemanticAgentLayer
from src.semantic_governance.client import FakeSemanticDecisionProvider
from src.semantic_governance.agent_questions import L1_QUESTION_ID


def _settings(**over):
    base = {"mode": "all", "cadence_steps": 2, "levels": 1, "workers": 4,
            "hard_margin": 0.15, "semantic": {"provider": "fake"}}
    base.update(over)
    return AgentLayerSettings.from_config({"semantic_agents": base})


def test_absent_or_off_section_builds_nothing():
    assert AgentLayerSettings.from_config({}) is None
    assert AgentLayerSettings.from_config({"semantic_agents": {"mode": "off"}}) is None


def test_unknown_fake_gives_all_neutral_factors(tmp_path):
    state, _ = _golden_state(0)
    layer = SemanticAgentLayer(_settings(), run_id="r",
                               provider=FakeSemanticDecisionProvider({L1_QUESTION_ID: "unknown"}))
    layer.attach_output(tmp_path)
    layer.advance(1, state)
    assert state.semantic_pillar_factors
    for vector in state.semantic_pillar_factors.values():
        assert np.array_equal(vector, np.ones(pillars.N_PILLARS))


def test_cadence_keeps_factors_between_boundaries(tmp_path):
    state, _ = _golden_state(0)
    fake = FakeSemanticDecisionProvider({L1_QUESTION_ID: "explore"})
    layer = SemanticAgentLayer(_settings(cadence_steps=3), run_id="r", provider=fake)
    layer.attach_output(tmp_path)
    layer.advance(1, state)
    first = len(fake.requests)
    layer.advance(2, state)
    layer.advance(3, state)
    assert len(fake.requests) == first            # nessuna richiesta fuori confine
    layer.advance(4, state)
    assert len(fake.requests) == 2 * first         # nuovo confine


def test_hard_mode_asks_only_ambiguous_agents(tmp_path):
    state, _ = _golden_state(0)
    ids = list(state.agents.ids)
    fake = FakeSemanticDecisionProvider({L1_QUESTION_ID: "explore"})
    layer = SemanticAgentLayer(_settings(mode="hard", cadence_steps=1), run_id="r", provider=fake)
    layer.attach_output(tmp_path)
    layer.advance(1, state)                        # nessun margine ancora: nessuna richiesta
    assert fake.requests == []
    assert state.pillar_margin_out == {}           # lo strato chiede i margini al kernel
    state.pillar_margin_out = {ids[0]: 0.05, ids[1]: 0.9}
    layer.advance(2, state)
    assert [r.actor for r in fake.requests] == [f"agent:{ids[0]}"]


def test_decisions_stream_to_jsonl_and_telemetry_is_written(tmp_path):
    state, _ = _golden_state(0)
    layer = SemanticAgentLayer(_settings(), run_id="r",
                               provider=FakeSemanticDecisionProvider({L1_QUESTION_ID: "build"}))
    layer.attach_output(tmp_path)
    layer.advance(1, state)
    summary = layer.close()
    rows = (tmp_path / "semantic_agent_decisions.jsonl").read_text().splitlines()
    assert len(rows) == summary["decisions"] > 0
    assert json.loads(rows[0])["question_id"] == L1_QUESTION_ID
    assert (tmp_path / "semantic_agent_telemetry.json").exists()


def test_level2_is_asked_only_for_eligible_pillars(tmp_path):
    state, _ = _golden_state(0)
    fake = FakeSemanticDecisionProvider({L1_QUESTION_ID: "life"})
    layer = SemanticAgentLayer(_settings(levels=2), run_id="r", provider=fake)
    layer.attach_output(tmp_path)
    layer.advance(1, state)
    assert all(r.question_id == L1_QUESTION_ID for r in fake.requests)
    assert state.semantic_action_factors == {}


def test_rerun_in_the_same_folder_does_not_append_old_decisions(tmp_path):
    for _ in range(2):
        state, _ = _golden_state(0)
        layer = SemanticAgentLayer(_settings(), run_id="r",
                                   provider=FakeSemanticDecisionProvider({L1_QUESTION_ID: "build"}))
        layer.attach_output(tmp_path)
        layer.advance(1, state)
        summary = layer.close()
    rows = (tmp_path / "semantic_agent_decisions.jsonl").read_text().splitlines()
    assert len(rows) == summary["decisions"]


def test_agent_layer_never_thresholds_confidence():
    # Spec: a livello di agente nessuna soglia, la distribuzione intera e' il
    # fattore. Lo smoke del 2026-09-22 sulla farm con soglia 0,65 ereditata dal
    # governo ha scartato 249 decisioni su 250.
    settings = AgentLayerSettings.from_config({"semantic_agents": {
        "mode": "all",
        "semantic": {"provider": "rest", "endpoint": "http://x", "min_confidence": 0.65},
    }})
    assert settings.semantic["min_confidence"] == 0.0
