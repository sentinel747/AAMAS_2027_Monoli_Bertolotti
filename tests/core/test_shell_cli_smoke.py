"""Regression contract for Task 13 (Shell CLI -> `AgentCoupledRunner` sul
core), mirroring `tests/core/test_shell_gui_smoke.py`'s own role for Task 12:
verified GREEN against the CURRENT engine (by construction here - this file
was added once the migration was already complete and re-verified against
it, see task-13-report.md's TDD section for the honest account of the
ordering) and meant to stay green forever after as the safety net for any
future change to `AgentCoupledRunner`/`src/core/kernel.py`/
`src/core/shell_common.py`. Locks down the CURRENT-engine artifact contract:
the same files `run_artifacts.save_run_artifacts` has always written, plus
the v1 metrics keys the plan brief calls out by name
(`crew_stress_index`, `power_margin`, `colony_prosperity_index`, ...).
"""

from pathlib import Path

from src.simulation.agent_coupled_runner import AgentCoupledRunner
from src.simulation.run_artifacts import build_cell_infrastructure_summary
from src.world.world_generator import WorldGenerator


BASE_CONFIG = {
    "name": "shell_cli_smoke",
    "seed": 0,
    "days": 20,
    "world": {"width": 16, "height": 16, "map_profile": "balanced"},
    "agents": {"count": 12, "llm_count": 0},
    "simulation": {"days_per_step": 7, "max_days": 100_000},
    "population": {"enabled": False},
    "climate": {"enabled": False},
    "environmental_layer": {"enabled": False},
    "extreme_events": {"enabled": False, "chance_per_step": 0},
    "llm": {"max_calls_per_step": 0},
    "snapshot_interval": 5,
}

# Every file `save_run_artifacts` (src/simulation/run_artifacts.py) always
# writes, regardless of config - the CLI runner's own artifact contract this
# migration must not change (see the task brief's "Governing directive").
EXPECTED_ARTIFACT_FILES = (
    "config.yaml",
    "run_metadata.json",
    "events.jsonl",
    "agent_decisions.jsonl",
    "validated_actions.jsonl",
    "rejected_actions.jsonl",
    "agent_conversations.jsonl",
    "agent_thoughts.jsonl",
    "replay_events.jsonl",
    "world_static_base.json",
    "agent_states.jsonl",
    "dead_agents.jsonl",
    "state_timeseries.csv",
    "social_network.json",
    "cell_infrastructure.json",
    "final_metrics.json",
    "research_summary.json",
    "api_usage.json",
    "token_usage.json",
    "llm_usage.jsonl",
    "replay_manifest.json",
    "summary.md",
)

# The v1 metrics keys the task brief names explicitly, plus a few more
# already relied on elsewhere in the suite (test_shell_gui_smoke.py's own
# METRICS_KEYS) - both engines must keep emitting these.
METRICS_KEYS = (
    "step",
    "day",
    "population",
    "survival_rate",
    "average_agent_health",
    "actions_attempted",
    "average_habitability",
    "crew_stress_index",
    "power_margin",
    "colony_prosperity_index",
)


def test_runner_short_headless_run_writes_expected_artifacts(tmp_path):
    runner = AgentCoupledRunner(dict(BASE_CONFIG))
    out = runner.run(days=20, output_dir=tmp_path / "run")

    assert out is not None and Path(out).exists()
    for name in EXPECTED_ARTIFACT_FILES:
        assert (out / name).exists(), name
    assert list((out / "world_snapshots").glob("step_*.json"))

    assert runner.global_metrics, "at least one step's metrics must have been recorded"
    metrics = runner.global_metrics[-1]
    for key in METRICS_KEYS:
        assert key in metrics, key

    import json

    final_metrics = json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))
    for key in METRICS_KEYS:
        assert key in final_metrics, key
    for key in (
        "structure_cache_hit_rate",
        "structure_view_hits",
        "structure_view_misses",
        "underserved_population",
        "local_life_support_population_coverage",
    ):
        assert key in final_metrics, key
    cell_infrastructure = json.loads(
        (out / "cell_infrastructure.json").read_text(encoding="utf-8")
    )
    assert cell_infrastructure
    assert {
        "population",
        "targets",
        "actual",
        "deficits",
        "warehouse",
        "open_construction_site_count",
        "construction_sites",
    } <= set(cell_infrastructure[0])
    assert "open_construction_site_count" in final_metrics
    # well-formed JSON everywhere written to disk (no NaN/Infinity, no
    # leftover non-serializable objects like the ExtremeEventEngine bug this
    # same migration surfaced and fixed - see task-13-report.md).
    json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))
    for snapshot_path in (out / "world_snapshots").glob("step_*.json"):
        json.loads(snapshot_path.read_text(encoding="utf-8"))


def test_runner_current_state_shape_at_construction():
    runner = AgentCoupledRunner(dict(BASE_CONFIG))
    assert len(runner.agents) == 12
    for agent in runner.agents.values():
        agent.to_dict()  # must not raise
    assert runner.world.metadata.get("environmental_layer_enabled") is False


def test_cell_infrastructure_persists_open_site_progress_and_position():
    world = WorldGenerator(seed=8).generate(16, 16)
    cell = world.get_cell(3, 4)
    cell.construction_sites["oxygen_plant@1234:5678"] = 100.0 / 3.0

    rows = build_cell_infrastructure_summary(world)
    row = next(item for item in rows if (item["x"], item["y"]) == (3, 4))

    assert row["open_construction_site_count"] == 1
    assert row["construction_sites"] == [
        {
            "site_key": "oxygen_plant@1234:5678",
            "structure_type": "oxygen_plant",
            "progress_percent": 100.0 / 3.0,
            "local_x_m": 1234.0,
            "local_y_m": 5678.0,
        }
    ]


def test_cell_infrastructure_separates_legacy_ice_work_from_buildings():
    world = WorldGenerator(seed=81).generate(16, 16)
    cell = world.get_cell(3, 4)
    cell.construction_sites["ice_extraction"] = 75.0

    rows = build_cell_infrastructure_summary(world)
    row = next(item for item in rows if (item["x"], item["y"]) == (3, 4))

    assert row["open_construction_site_count"] == 0
    assert row["construction_sites"] == []
    assert row["open_extraction_site_count"] == 1
    assert row["extraction_sites"] == [
        {
            "site_key": "ice_extraction",
            "resource": "ice",
            "progress_percent": 75.0,
        }
    ]
