import json
from pathlib import Path

from src.experiments.runner import ExperimentRunner
from scripts.run_governor_experiment import run_arm
from src.governors.record import load_records
from src.semantic_governance.artifacts import (
    sanitized_semantic_config,
    semantic_config_fingerprint,
    write_semantic_artifacts,
)
from src.semantic_governance.schemas import SemanticDecision
from src.simulation.agent_coupled_runner import AgentCoupledRunner


ROOT = Path(__file__).resolve().parents[2]


def test_offline_yaml_writes_replayable_semantic_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr("src.experiments.runner.plot_metric_csv", lambda *args: None)
    output = ExperimentRunner(
        ROOT / "configs" / "experiments" / "jev_semif" / "offline_semif_fake.yaml"
    ).run(output_root=tmp_path)

    manifest = json.loads((output / "semantic_manifest.json").read_text(encoding="utf-8"))
    telemetry = json.loads((output / "semantic_telemetry.json").read_text(encoding="utf-8"))
    rows = [
        SemanticDecision.from_dict(json.loads(line))
        for line in (output / "semantic_decisions.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    assert telemetry["decisions"] == len(rows)
    assert telemetry["deep_escalations"] == 0
    assert telemetry["avoided_deep_generations"] == len(rows)
    assert manifest["manifest_version"] == "1.0"
    assert manifest["schema_version"] == "1.0"
    assert manifest["semantic_provider"] == "fake"
    assert manifest["candidate_profiles"]["governor"] == "jev-semif-governor-v1"
    assert len(manifest["semantic_config_hash"]) == 64


def test_non_semantic_run_does_not_change_its_artifact_set(tmp_path):
    output = tmp_path / "historical-arm"
    AgentCoupledRunner({
        "seed": 1,
        "days": 1,
        "world": {"width": 8, "height": 8},
        "agents": {"count": 2, "llm_count": 0},
        "population": {"enabled": False},
    }).run(days=1, output_dir=output)
    assert not (output / "semantic_manifest.json").exists()
    assert not (output / "semantic_telemetry.json").exists()
    assert not (output / "semantic_decisions.jsonl").exists()


def test_manifest_redaction_removes_credentials_and_endpoint_query():
    safe = sanitized_semantic_config({
        "governors": {
            "arm": "semif",
            "semantic": {
                "provider": "rest",
                "endpoint": "http://user:pass@127.0.0.1:3000/decide?api_key=do-not-store",
                "api_key": "do-not-store",
                "api_key_env": "JEV_PRIVATE_KEY",
            },
        }
    })
    encoded = json.dumps(safe)
    assert "do-not-store" not in encoded
    assert "user:pass" not in encoded
    assert "?" not in safe["governors"]["semantic"]["endpoint"]
    assert safe["governors"]["semantic"]["endpoint"] == "http://127.0.0.1:3000/decide"


def test_admin_only_semantic_manifest_uses_provider_and_stable_treatment_hash(tmp_path):
    first = {
        "seed": 5,
        "governors": {
            "arm": "scripted",
            "administrators": {
                "enabled": True,
                "follow_governor_arm": False,
                "arm": "semif",
                "semantic": {"provider": "fake", "run_id": "admin-run-a"},
            },
        },
    }
    second = json.loads(json.dumps(first))
    second["governors"]["administrators"]["semantic"]["run_id"] = "admin-run-b"

    assert semantic_config_fingerprint(first) == semantic_config_fingerprint(second)
    write_semantic_artifacts(tmp_path, first)
    manifest = json.loads((tmp_path / "semantic_manifest.json").read_text(encoding="utf-8"))
    assert manifest["semantic_provider"] == "fake"


def test_experimental_runner_fake_then_replay_is_offline_and_deterministic(tmp_path):
    root = tmp_path / "runs" / "jev_semif_experiments"
    fake_dir = root / "fake"
    replay_dir = root / "replay"
    common = {
        "arm": "semif",
        "seed": 9,
        "steps": 4,
        "agents": 40,
        "cadence_steps": 1,
        "wait_seconds": 1,
        "log_agents": False,
        "semantic_run_id": "phase4-runner-replay",
    }
    fake = run_arm(
        output_dir=fake_dir,
        semantic_provider="fake",
        semantic_governor_decision="select_candidate:mean_food_per_occupant",
        **common,
    )
    replay = run_arm(
        output_dir=replay_dir,
        semantic_provider="replay",
        semantic_replay_from=str(fake_dir / "semantic_decisions.jsonl"),
        **common,
    )
    fake_policies = [row["policy"] for row in load_records(fake_dir / "governor_decisions.jsonl")]
    replay_policies = [row["policy"] for row in load_records(replay_dir / "governor_decisions.jsonl")]
    assert replay_policies == fake_policies
    assert replay["population"] == fake["population"]
    assert replay["structures_total"] == fake["structures_total"]
    replay_manifest = json.loads((replay_dir / "semantic_manifest.json").read_text(encoding="utf-8"))
    assert replay_manifest["semantic_provider"] == "replay"
