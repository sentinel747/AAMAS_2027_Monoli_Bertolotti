"""Gate Fase 2: piccola run SemIf fake e replay senza rete."""

import json

from scripts.parity_harness import realistic_config
from src.core.state_digest import step_digest
from src.governors.record import load_records
from src.simulation.agent_coupled_runner import AgentCoupledRunner


def config(provider: str, replay_from=None) -> dict:
    value = realistic_config(40, 6, 0)
    semantic = {
        "provider": provider,
        "run_id": "phase2-replay-fixture",
    }
    if provider == "fake":
        semantic["decisions"] = {
            "jev-semif-governor-v1": "select_candidate:mean_food_per_occupant"
        }
    if replay_from is not None:
        semantic["replay_from"] = str(replay_from)
    value["governors"] = {
        "arm": "semif",
        "cadence_steps": 2,
        "wait_seconds": 1,
        "semantic": semantic,
    }
    return value


def policies(path):
    return [row["policy"] for row in load_records(path)]


def test_fake_run_and_semantic_replay_are_deterministic(tmp_path):
    root = tmp_path / "runs" / "jev_semif_experiments"
    fake_dir = root / "fake"
    replay_dir = root / "replay"

    fake = AgentCoupledRunner(config("fake"))
    fake.run(days=6, output_dir=fake_dir)
    fake_digest = step_digest(fake.core)["overall"]
    fake_record = fake_dir / "governor_decisions.jsonl"
    rows = load_records(fake_record)
    assert rows and all(
        row["governor"]["semantic_decision"]["schema_version"] == "1.0"
        for row in rows
    )

    semantic_replay = root / "semantic_decisions.jsonl"
    semantic_replay.write_text(
        "".join(
            json.dumps(row["governor"]["semantic_decision"], ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )

    replay = AgentCoupledRunner(config("replay", semantic_replay))
    replay.run(days=6, output_dir=replay_dir)

    assert policies(replay_dir / "governor_decisions.jsonl") == policies(fake_record)
    assert step_digest(replay.core)["overall"] == fake_digest
    assert all(
        row["governor"]["semantic_decision"]["route"] == "replay"
        for row in load_records(replay_dir / "governor_decisions.jsonl")
    )
