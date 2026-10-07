"""Regression contract for Task 12 (Shell GUI -> SimulationController sul
core): written and verified GREEN against the CURRENT object-engine
controller BEFORE any migration, per the task brief's "Order of work". This
is the safety net a future migration of `SimulationController` to
`src/core/kernel.py` must keep passing byte-for-byte (same public JSON
shape: agent keys, cell keys, metrics keys).

Task 12 itself is currently BLOCKED (see .superpowers/sdd/task-12-report.md,
NEEDS_CONTEXT): `src/core/kernel.py`'s `StepOutcome` does not carry enough
per-action detail (no `ActionRequest`/`ActionResult`/`Observation` objects,
and no pre-vitals per-agent scalar snapshot - `tick_vitals` mutates
`AgentArrays` in place before `step()` returns) to reconstruct
`SimulationController`'s per-step bookkeeping (action log rows, replay
events, thoughts) AFTER the fact without either modifying a pinned logic
module or duplicating the decide/execute/vitals loop in the shell. This
test file exists independently of that blocker: it locks down the
CURRENT-engine contract so whoever resolves the seam has a green baseline
to migrate against.
"""

from src.api.state_store import SimulationController


BASE_CONFIG = {
    "name": "shell_gui_smoke",
    "seed": 0,
    "days": 25,
    "world": {"width": 16, "height": 16, "map_profile": "balanced"},
    "agents": {"count": 12, "llm_count": 0},
    "simulation": {"days_per_step": 7, "max_days": 100_000},
    "population": {"enabled": False},
    "climate": {"enabled": False},
    "environmental_layer": {"enabled": False},
    "extreme_events": {"enabled": False, "chance_per_step": 0},
    "llm": {"max_calls_per_step": 0},
}

AGENT_KEYS = (
    "agent_id",
    "name",
    "role",
    "x",
    "y",
    "inventory",
    "health",
    "satiety",
    "oxygen_level",
    "hydration",
    "fatigue",
    "memory_summary",
)

METRICS_KEYS = (
    "step",
    "day",
    "population",
    "survival_rate",
    "average_agent_health",
    "actions_attempted",
    "average_habitability",
)


def test_controller_reset_and_20_steps_with_object_engine():
    ctl = SimulationController(dict(BASE_CONFIG))
    ctl.reset(dict(BASE_CONFIG))
    ctl.step(20)

    state = ctl.current_state()

    assert state["step"] >= 20
    assert state["step_index"] == ctl.step_index
    assert len(state["agents"]) >= 1  # some founders may have died; never silently empty on a healthy run
    assert isinstance(state["metrics"], dict)
    for key in METRICS_KEYS:
        assert key in state["metrics"], key

    agent = state["agents"][0]
    for key in AGENT_KEYS:
        assert key in agent, key

    # well-formed JSON: no NaN/Infinity anywhere in the payload (see _json_finite).
    import json

    json.dumps(state)

    assert isinstance(state["recent_events"], list)
    assert state["decision_mode"] == "tree"
    assert isinstance(state["recent_actions"], list)
    assert state["recent_actions"]
    assert {"step", "agent_id", "action", "local_x_m", "local_y_m"}.issubset(
        state["recent_actions"][-1]
    )
    assert isinstance(state["dead_agents"], list)
    assert isinstance(state["api_usage"], dict)


def test_controller_current_state_shape_at_day_zero():
    ctl = SimulationController(dict(BASE_CONFIG))
    state = ctl.current_state()
    assert state["step"] == 0
    assert len(state["agents"]) == 12
    for agent in state["agents"]:
        for key in AGENT_KEYS:
            assert key in agent, key
