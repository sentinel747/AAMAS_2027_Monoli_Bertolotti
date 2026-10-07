"""Riga di comando e hash del trattamento per lo strato SemIf degli agenti
(piano 2026-09-22, Task 10)."""

import json

import pytest

from scripts import run_governor_experiment as rge

AGENTS = {"mode": "all", "cadence_steps": 4, "levels": 1, "workers": 64,
          "hard_margin": 0.15, "semantic": {"provider": "fake", "run_id": "a"}}


def test_hash_without_agents_is_unchanged():
    section = rge._semantic_section("fake", run_id="x")
    assert rge._semantic_treatment_hash("semif", "", section) == \
        rge._semantic_treatment_hash("semif", "", section, None)
    assert rge._semantic_treatment_hash("none", "", None) == ""


def test_agents_make_historical_arms_semantic():
    assert rge._semantic_treatment_hash("none", "", None, AGENTS) != ""


def test_run_id_and_workers_do_not_change_the_agent_treatment():
    other = json.loads(json.dumps(AGENTS))
    other["semantic"]["run_id"] = "b"
    other["workers"] = 8
    assert rge._semantic_treatment_hash("none", "", None, AGENTS) == \
        rge._semantic_treatment_hash("none", "", None, other)


def test_cadence_changes_the_agent_treatment():
    other = dict(AGENTS, cadence_steps=1)
    assert rge._semantic_treatment_hash("none", "", None, AGENTS) != \
        rge._semantic_treatment_hash("none", "", None, other)


def test_typesafe_section_requires_a_budget():
    with pytest.raises(ValueError, match="budget"):
        rge._semantic_section("typesafe", run_id="x")
    section = rge._semantic_section("typesafe", run_id="x", budget_usd=0.5)
    assert section["budget_usd"] == 0.5
    assert section["api_key_env"] == "TYPESAFE_AI"
    assert section["model"] == "jev-latest"


def test_fake_agent_decision_is_written_only_when_requested():
    assert "jev-semif-agent-v1:L1" not in rge._semantic_section("fake", run_id="x")["decisions"]
    section = rge._semantic_section("fake", run_id="x", agent_decision="explore")
    assert section["decisions"]["jev-semif-agent-v1:L1"] == "explore"


def test_label_carries_the_agent_layer():
    assert rge._etichetta("none", "", False, semantic_agents="all:fake") == "none+ag:all:fake"
    assert rge._etichetta("none", "", False) == "none"


STATE_FILES = ("action_summary.json", "agent_states.jsonl", "events.jsonl",
               "replay_events.jsonl", "state_timeseries.csv")


def _same_state(left, right, name):
    """Confronto byte per byte, salvo `wall_time` di `events.jsonl`, che e'
    l'orario di parete al secondo e non uno stato della simulazione."""
    if name != "events.jsonl":
        return (left / name).read_bytes() == (right / name).read_bytes()
    import json as _json

    def righe(path):
        out = []
        for line in (path / name).read_text(encoding="utf-8").splitlines():
            row = _json.loads(line)
            row.pop("wall_time", None)
            out.append(row)
        return out

    return righe(left) == righe(right)


def _cli(out, *extra):
    import sys

    argv = ["run_governor_experiment.py", "--arms", "none", "--seeds", "101",
            "--steps", "4", "--agents", "8", "--semantic-agents", "all",
            "--semantic-agents-cadence", "1", "--semantic-run-id", "rid",
            "--out", str(out), *extra]
    old = sys.argv
    sys.argv = argv
    try:
        assert rge.main() == 0
    finally:
        sys.argv = old
    runs = [p for p in out.iterdir() if p.is_dir()]
    assert len(runs) == 1
    return runs[0]


def test_fake_then_replay_reproduces_the_run(tmp_path):
    fake = _cli(tmp_path / "a", "--semantic-provider", "fake",
                "--semantic-agent-decision", "explore")
    replay = _cli(tmp_path / "b", "--semantic-provider", "replay",
                  "--semantic-replay-from", str(fake / "semantic_agent_decisions.jsonl"))
    for name in STATE_FILES:
        assert _same_state(fake, replay, name), name
    results = json.loads((tmp_path / "a" / "results.json").read_text(encoding="utf-8"))
    assert results[0]["braccio"] == "none+ag:all:fake"
    assert results[0]["semantic_treatment_hash"]


def test_offline_agents_yaml_runs_without_network(tmp_path, monkeypatch):
    from pathlib import Path

    from src.experiments.runner import ExperimentRunner

    monkeypatch.setattr("src.experiments.runner.plot_metric_csv", lambda *args: None)
    root = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "jev_semif"
    output = ExperimentRunner(root / "offline_agents_fake.yaml").run(output_root=tmp_path)
    telemetry = json.loads((output / "semantic_agent_telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["decisions"] > 0
    assert telemetry["fallbacks"] == 0
    assert telemetry["kernel_counters"]["agent_decisions"] > 0


@pytest.mark.parametrize("name", ["farm_agents_l1.yaml", "typesafe_agents_smoke.yaml"])
def test_live_yamls_declare_an_explicit_layer(name):
    from pathlib import Path

    import yaml

    from src.semantic_governance.agent_layer import AgentLayerSettings

    root = Path(__file__).resolve().parents[2] / "configs" / "experiments" / "jev_semif"
    config = yaml.safe_load((root / name).read_text(encoding="utf-8"))
    settings = AgentLayerSettings.from_config(config)
    assert settings is not None
    assert config["agents"]["decision_mode"] == "preferences"
    if "typesafe" in name:
        assert 0 < settings.semantic["budget_usd"] <= 5.0


def test_cli_agent_section_has_no_confidence_threshold(tmp_path):
    import yaml

    from scripts.run_governor_experiment import run_arm

    out = tmp_path / "run"
    run_arm("none", 101, 2, 8, out, cadence_steps=1, semantic_provider="fake",
            semantic_agents="all", semantic_agents_cadence=1,
            semantic_min_confidence=0.65)
    config = yaml.safe_load((out / "config.yaml").read_text(encoding="utf-8"))
    assert config["semantic_agents"]["semantic"]["min_confidence"] == 0.0
