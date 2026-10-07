import json

from src.experiments.runner import ExperimentRunner
from src.llm.provider import LLMResponse
from src.simulation.agent_coupled_runner import AgentCoupledRunner


def test_experiment_runner_smoke():
    out = ExperimentRunner("configs/test/smoke_rule_based.yaml").run(output_root="outputs/test_artifacts/experiments")
    assert (out / "final_metrics.json").exists()
    assert (out / "summary.md").exists()


def test_experiment_runner_llm_count_configured_produces_artifacts_without_real_calls(tmp_path, monkeypatch):
    """Task 13 adapted this test (was
    `test_experiment_runner_records_async_llm_attempts`): `AgentCoupledRunner`
    now drives its per-step physics through `src.core.kernel.step`, which -
    per that module's own docstring - decides EVERY agent through the
    rule-based path regardless of `mode` ("the LLM-governor branch
    (`_decide_with_llm_governor`, `agent.async_decide`) ... is deliberately
    out of scope"), identical to `SimulationController`'s own Task-12
    concession (see task-12-report.md's "Concerns"). A configured
    `FailingProvider` can therefore never actually be called any more - there
    is no live `LLMAgent` object with a bound `async_decide` left after
    `__init__` converts to the vectorized `CoreState` (see
    `src/core/shell_common.py`'s `build_core_state`), only a `mode` label -
    so the original intent ("a provider failure gets tracked end to end
    through cost_tracker/agent_decisions/events") can no longer hold: this is
    not a missing `StepOutcome` field, it is the entire async decision
    mechanism, which would need live per-agent objects kept in sync with the
    arrays every step (a real, currently out-of-scope design addition - see
    task-13-report.md's "Concerns" for the full reasoning). What's preserved:
    an `agents.llm_count` config does not crash the pipeline, the provider is
    never actually invoked (deterministic, API-credit-free by construction -
    see CLAUDE.md's zero-external-API policy), and every artifact this
    engine's own `save()`/`run_artifacts.py` always produces still exists.
    """
    class FailingProvider:
        provider_id = "fake"
        model = "fake-model"

        async def async_complete_json(self, prompt, schema_hint=None):
            return LLMResponse(
                text='{"thought_summary":"fallback after error","public_message":"","chosen_action":"observe","target":{"type":"self"}}',
                provider="fallback",
                api_call_attempted=True,
                error="HTTPStatusError: 401 test",
            )

    monkeypatch.setattr("src.simulation.agent_coupled_runner.create_agent_providers", lambda config, llm_count: [FailingProvider()])
    monkeypatch.setattr("src.experiments.runner.plot_metric_csv", lambda csv_path, output_dir: None)
    config_path = tmp_path / "llm_attempt.yaml"
    config_path.write_text(
        "\n".join(
            [
                "name: llm_attempt",
                "seed: 3",
                "days: 1",
                "simulation: {days_per_step: 1, max_days: 10}",
                "snapshot_interval: 1",
                "world: {width: 8, height: 8}",
                "agents: {count: 2, llm_count: 1}",
                "llm: {provider: fake, max_calls_per_run: 3, max_calls_per_day: 3}",
                "population: {enabled: false}",
            ]
        ),
        encoding="utf-8",
    )

    out = ExperimentRunner(config_path).run(output_root=tmp_path / "runs")
    api_usage = json.loads((out / "api_usage.json").read_text(encoding="utf-8"))
    # agent_decisions.jsonl/events.jsonl still exist (save() always writes
    # them) but carry no LLM content: `decision_history` only exists on
    # `LLMAgent`, never wired into the kernel-driven `AgentView`, so
    # `self.agent_decisions` stays permanently empty (same as the GUI shell,
    # Task 12) and the provider's own "HTTPStatusError: 401 test" never gets
    # a chance to be logged since it is never called.
    assert (out / "agent_decisions.jsonl").exists()
    assert (out / "events.jsonl").exists()
    assert (out / "final_metrics.json").exists()
    assert api_usage["calls"] == 0
    assert api_usage["failed_calls"] == 0


def test_action_jsonl_contains_mars_society_abm_context(tmp_path):
    config = {
        "seed": 12,
        "days": 1,
        "simulation": {"days_per_step": 1, "max_days": 10},
        "world": {"width": 12, "height": 8},
        "agents": {"count": 2, "llm_count": 0},
        "population": {"enabled": False},
    }
    out = tmp_path / "action_context_run"

    AgentCoupledRunner(config).run(days=1, output_dir=out)
    rows = [
        json.loads(line)
        for line in (out / "validated_actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    replay_rows = [
        json.loads(line)
        for line in (out / "replay_events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert replay_rows
    assert {"step", "day", "agent_id", "action", "x", "y", "target"}.issubset(replay_rows[0])
    static_base = json.loads((out / "world_static_base.json").read_text(encoding="utf-8"))
    assert static_base["width"] == 12
    assert static_base["height"] == 8
    assert static_base["cells"]
    row = rows[0]
    assert {"step", "day", "agent_id", "agent_name", "role", "mode", "action", "accepted", "target"}.issubset(row)
    assert {"x", "y", "local_x_m", "local_y_m", "perception_radius_m", "movement_distance_m_per_step"}.issubset(row)
    assert {"health", "hydration", "satiety", "oxygen_level", "fatigue", "inventory"}.issubset(row)
    assert {"visible_cells_count", "partial_visible_cells_count", "nearby_agents_count"}.issubset(row)
    assert row["observation_mode"] in {"rule_based", "fast", "full"}


def test_action_summary_is_persisted_when_memory_logs_are_disabled(tmp_path):
    config = {
        "name": "persistent_rejection_summary",
        "seed": 44,
        "world": {"width": 12, "height": 8},
        "agents": {
            "count": 3,
            "llm_count": 0,
            "decision_mode": "preferences",
        },
        "simulation": {"days_per_step": 7, "max_days": 30},
        "population": {"enabled": False},
        "headless": {"store_memory_logs": False, "fast_observation": True},
    }
    out = tmp_path / "summary_without_rows"

    runner = AgentCoupledRunner(config)
    runner.run(days=3, output_dir=out)
    summary = json.loads((out / "action_summary.json").read_text(encoding="utf-8"))
    replay_manifest = json.loads((out / "replay_manifest.json").read_text(encoding="utf-8"))
    final_metrics = json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))
    research_summary = json.loads((out / "research_summary.json").read_text(encoding="utf-8"))

    assert (out / "rejected_actions.jsonl").read_text(encoding="utf-8") == ""
    assert summary["attempted"] == runner.actions_attempted_count == 9
    assert summary["rejected"] == runner.actions_rejected_count
    assert sum(row["attempted"] for row in summary["by_action"].values()) == 9
    assert replay_manifest["memory_logs_enabled"] is False
    assert replay_manifest["action_log_rows"] == 0
    assert replay_manifest["replay_rows"] > 0
    assert replay_manifest["replay_last_step"] == 3
    assert "logistics_cooperation_index" in final_metrics
    assert research_summary["social_dynamics"]["cooperation_index"] == final_metrics["cooperation_index"]
    assert research_summary["social_dynamics"]["resource_sharing_events"] == int(
        final_metrics["automatic_redistribution_active_steps"]
    )


def test_rule_based_observation_keeps_replay_artifacts_without_fast_observer(tmp_path):
    config = {
        "seed": 13,
        "days": 2,
        "simulation": {"days_per_step": 1, "max_days": 10},
        "world": {"width": 12, "height": 8},
        "agents": {"count": 3, "llm_count": 0},
        "population": {"enabled": False},
        "snapshot_interval": 1,
        "headless": {
            "fast_observation": False,
            "rule_based_observation": True,
            "output_flush_interval_steps": 10,
            "biology_update_scope": "active",
        },
    }
    out = tmp_path / "rule_observer_run"

    AgentCoupledRunner(config).run(days=2, output_dir=out)
    rows = [
        json.loads(line)
        for line in (out / "validated_actions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert rows
    assert {row["observation_mode"] for row in rows} == {"rule_based"}
    assert (out / "replay_events.jsonl").exists()
    assert (out / "state_timeseries.csv").exists()
    assert list((out / "world_snapshots").glob("step_*.json"))


def test_headless_llm_governor_config_is_inert_under_core_kernel(tmp_path, monkeypatch):
    """Task 13 adapted this test (was
    `test_headless_llm_governor_limits_calls_per_step`): with the per-step
    physics running through `src.core.kernel.step` (rule-based decisions for
    every agent, regardless of `mode` - see the sibling test above and
    task-13-report.md's "Concerns" for the full rationale),
    `_decide_with_llm_governor`/the `llm.max_calls_per_step` cap it used to
    enforce are never invoked by `run_async` any more, so a `CountingProvider`
    is never called and no agent ever gets "governed" (skipped) in the old
    sense. What's preserved: `agents.llm_count`/`llm.max_calls_per_step`
    configured together still produce a complete, valid run (no crash, real
    artifacts) - `usage["calls"]` and `metrics["llm_governor_skips"]` are
    both `0` (not the old cap-driven `2`/`6.0`), an honest reflection of "no
    governance happens because no LLM decisions happen," not a cap being hit.
    """
    class CountingProvider:
        provider_id = "fake"
        model = "fake-model"

        async def async_complete_json(self, prompt, schema_hint=None):
            return LLMResponse(
                text='{"thought_summary":"governed decision","public_message":"","chosen_action":"observe","target":{"type":"self"}}',
                provider="fake:fake-model",
                api_call_attempted=True,
            )

    monkeypatch.setattr(
        "src.simulation.agent_coupled_runner.create_agent_providers",
        lambda config, llm_count: [CountingProvider() for _ in range(llm_count)],
    )
    out = tmp_path / "governor_run"
    config = {
        "seed": 5,
        "days": 2,
        "simulation": {"days_per_step": 1, "max_days": 10},
        "world": {"width": 8, "height": 8},
        "agents": {"count": 4, "llm_count": 4},
        "llm": {"max_calls_per_step": 1, "max_calls_per_run": 10, "max_calls_per_day": 10},
        "population": {"enabled": False},
        "headless": {"fast_observation": True, "log_interval_steps": 1},
    }

    runner = AgentCoupledRunner(config)
    runner.run(days=2, output_dir=out)

    usage = json.loads((out / "api_usage.json").read_text(encoding="utf-8"))
    metrics = json.loads((out / "final_metrics.json").read_text(encoding="utf-8"))
    assert usage["calls"] == 0
    assert metrics["llm_governor_skips"] == 0.0
    assert metrics["population"] == 4
