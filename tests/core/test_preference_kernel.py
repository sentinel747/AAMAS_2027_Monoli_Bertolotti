import numpy as np

from src.agents.population import spawn_initial_agents, vision_radius_cells
from src.core import kernel
from src.core.shell_common import build_core_state
from src.world.world_generator import WorldGenerator


def _make_state(seed=0, agents=8, decision_mode="preferences"):
    world = WorldGenerator(seed=seed).generate(32, 20)
    object_agents = spawn_initial_agents(agents, world, seed=seed)
    core, _world_view = build_core_state(world, object_agents)
    config = {
        "seed": seed,
        "agents": {
            "decision_mode": decision_mode,
            "decision_sampling": "softmax",
        },
        "headless": {"fast_observation": True, "rule_based_observation": True},
    }
    return core, config


def test_vision_radius_cells_default_is_historical():
    assert vision_radius_cells(5_000.0, 180) == 1
    assert vision_radius_cells(5_000.0, 20) == 1
    assert vision_radius_cells(100.0, 180) == 1
    assert vision_radius_cells(59_000.0, 180) == 1
    # Values outside the user-facing range are clamped before conversion.
    assert vision_radius_cells(120_000.0, 180) == 1


def test_preferences_step_runs_and_is_deterministic():
    core_a, config = _make_state()
    core_b, _ = _make_state()
    for step_index in range(1, 11):
        kernel.step(core_a, step_index, 7.0, config, None)
        kernel.step(core_b, step_index, 7.0, config, None)
    count = core_a.agents.n
    assert np.array_equal(core_a.agents.x[:count], core_b.agents.x[:count])
    assert np.array_equal(core_a.agents.y[:count], core_b.agents.y[:count])
    assert np.array_equal(core_a.agents.health[:count], core_b.agents.health[:count])


def test_missing_decision_mode_key_defaults_to_tree():
    core, config = _make_state()
    config["agents"].pop("decision_mode")
    outcome = kernel.step(core, 1, 7.0, config, None)
    assert all(
        "[pref]" not in (record.request.message or "")
        for record in outcome.agent_step_records
    )


def test_preferences_and_tree_diverge():
    core_preferences, config_preferences = _make_state(decision_mode="preferences")
    core_tree, config_tree = _make_state(decision_mode="tree")
    for step_index in range(1, 16):
        kernel.step(
            core_preferences, step_index, 7.0, config_preferences, None
        )
        kernel.step(core_tree, step_index, 7.0, config_tree, None)
    count = core_preferences.agents.n
    same = np.array_equal(
        core_preferences.agents.x[:count], core_tree.agents.x[:count]
    ) and np.array_equal(
        core_preferences.agents.inv[:count], core_tree.agents.inv[:count]
    )
    assert not same, "le due modalita non possono coincidere su 15 step"


def test_preferences_share_one_full_vision_scan_per_occupied_cell(monkeypatch):
    core, config = _make_state(seed=81, agents=4)
    config["headless"]["fast_observation"] = False
    rows = core.agents.alive_rows()
    core.agents.x[rows] = 10
    core.agents.y[rows] = 8
    for agent_id in core.agents.ids:
        core.side[agent_id].perception_radius_m = 59_000.0

    calls = 0
    real_observe = kernel.observe

    def counted_observe(*args, **kwargs):
        nonlocal calls
        calls += 1
        return real_observe(*args, **kwargs)

    monkeypatch.setattr(kernel, "observe", counted_observe)
    outcome = kernel.step(core, 1, 7.0, config, None)

    assert calls == 1
    counts = {
        record.action_log_row["visible_cells_count"]
        for record in outcome.agent_step_records
    }
    modes = {
        record.action_log_row["observation_mode"]
        for record in outcome.agent_step_records
    }
    assert len(counts) == 1 and next(iter(counts)) > 0
    assert modes == {"preferences_shared_full"}
    assert np.count_nonzero(core.cells.explored) > 0


def test_preferences_fast_vision_marks_current_cell_explored():
    core, config = _make_state(seed=82, agents=1)
    row = int(core.agents.alive_rows()[0])
    x, y = int(core.agents.x[row]), int(core.agents.y[row])

    outcome = kernel.step(core, 1, 7.0, config, None)

    assert core.cells.explored[y, x]
    assert outcome.agent_step_records[0].action_log_row["observation_mode"] == (
        "preferences_shared_fast"
    )


def test_cell_only_recovery_is_exact_at_end_of_week():
    core, config = _make_state(seed=83, agents=1)
    config["agents"].update(
        {
            "decision_sampling": "greedy",
            "cell_proposal_top_k": 0,
            "individual_survival_priority_enabled": False,
        }
    )
    config["model"] = {"psychosocial_enabled": True}
    row = int(core.agents.alive_rows()[0])
    core.agents.health[row] = 0.2
    core.agents.satiety[row] = 0.3
    core.agents.hydration[row] = 0.1
    core.agents.oxygen[row] = 0.4
    core.agents.fatigue[row] = 0.9
    core.agents.stress[row] = 0.8
    core.agents.morale[row] = 0.2
    core.agents.pref[row] = 0.0
    core.agents.pref[row, 3] = 1.0

    outcome = kernel.step(core, 1, 7.0, config, None)

    assert outcome.agent_step_records[0].request.action.value == (
        "physiological_recovery"
    )
    assert core.agents.health[row] == 1.0
    assert core.agents.satiety[row] == 1.0
    assert core.agents.hydration[row] == 1.0
    assert core.agents.oxygen[row] == 1.0
    assert core.agents.fatigue[row] == 0.0
    assert core.agents.stress[row] == 0.0
    assert core.agents.morale[row] == 1.0
