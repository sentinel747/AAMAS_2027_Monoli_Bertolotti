"""Innesto dei fattori semantici per agente nel kernel (piano 2026-09-22, Task 6).

Il test usa lo stesso oracolo della procedura di parita': la run vera del
runner headless, intercettando `_core_step` come fa `scripts/parity_harness.py`.
Prima di ogni passo `prepare(state, step)` deposita i campi semantici, come
fara' lo strato SemIf degli agenti.
"""

import numpy as np
import pytest

import src.simulation.agent_coupled_runner as runner_module
from scripts.parity_harness import build_config
from src.agents import pillars
from src.core import state_digest
from src.simulation.agent_coupled_runner import AgentCoupledRunner


def _run(prepare=None, *, seed=0, steps=12, agents=20, engine=None):
    records: list[dict] = []
    captured: dict = {}
    inner = runner_module._core_step

    def recording_step(state, step_index, dt_days, config, rng):
        captured["state"] = state
        if prepare is not None:
            prepare(state, step_index)
        outcome = inner(state, step_index, dt_days, config, rng)
        records.append(state_digest.step_digest(state, outcome))
        return outcome

    config = build_config("golden", agents, steps, seed, "softmax")
    if engine is not None:
        config["agents"]["decision_engine"] = engine
    runner_module._core_step = recording_step
    try:
        AgentCoupledRunner(config).run(days=steps, output_dir=None)
    finally:
        runner_module._core_step = inner
    return records, captured["state"]


def _ones_for_everyone(state, _step):
    ids = list(state.agents.ids)
    state.semantic_pillar_factors = {a: np.ones(pillars.N_PILLARS) for a in ids}
    state.semantic_action_factors = {a: np.ones(pillars.N_ACTIONS) for a in ids}
    if state.semantic_counters is None:
        state.semantic_counters = {}


def test_the_oracle_is_deterministic():
    first, _ = _run()
    second, _ = _run()
    assert first == second


def test_all_ones_factors_are_identical_to_no_factors():
    base, _ = _run()
    neutral, state = _run(_ones_for_everyone)
    assert neutral == base
    assert state.semantic_counters["pillar_changed"] == 0
    assert state.semantic_counters["agent_decisions"] > 0


def test_missing_agent_entry_is_neutral():
    base, _ = _run()

    def empty(state, _step):
        state.semantic_pillar_factors = {}
        state.semantic_action_factors = {}

    assert _run(empty)[0] == base


def test_strong_factor_changes_choices():
    strong = np.ones(pillars.N_PILLARS)
    strong[pillars.P_EXPLORE] = 50.0

    def push(state, _step):
        state.semantic_pillar_factors = {a: strong for a in state.agents.ids}
        if state.semantic_counters is None:
            state.semantic_counters = {}

    base, _ = _run()
    pushed, state = _run(push)
    assert state.semantic_counters["pillar_changed"] > 0
    assert pushed != base


def test_margin_is_written_only_when_requested():
    _, untouched = _run(steps=2)
    assert untouched.pillar_margin_out is None

    def ask(state, _step):
        state.pillar_margin_out = {}

    base, _ = _run(steps=2)
    asked, state = _run(ask, steps=2)
    assert asked == base  # scrivere i margini non cambia la simulazione
    assert state.pillar_margin_out
    assert all(0.0 <= m <= 1.0 for m in state.pillar_margin_out.values())


def test_reference_engine_with_factors_is_refused():
    def empty(state, _step):
        state.semantic_pillar_factors = {}

    with pytest.raises(ValueError, match="vectorized"):
        _run(empty, steps=1, engine="reference")
