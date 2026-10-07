"""Strato SemIf degli agenti collegato alla shell headless (piano 2026-09-22, Task 9).

Il test decisivo: con lo strato acceso ma una decisione fake `unknown` (fattori
tutti neutri) i file di stato della run sono identici byte per byte a quelli
della stessa run senza strato.
"""

import json

import pytest

from scripts.parity_harness import build_config
from src.simulation.agent_coupled_runner import AgentCoupledRunner

STEPS = 4
#: I file che due esecuzioni identiche riproducono byte per byte (misurato il
#: 2026-09-22: final_metrics, research_summary, run_metadata, summary e
#: replay_manifest contengono tempi di parete). `validated_actions.jsonl` e'
#: vuoto con la config golden (`max_action_log_rows: 0`), quindi non e' un
#: oracolo: le azioni stanno in `replay_events.jsonl` e `action_summary.json`.
STATE_FILES = (
    "action_summary.json", "agent_states.jsonl", "events.jsonl",
    "replay_events.jsonl", "state_timeseries.csv", "cell_infrastructure.json",
    "social_network.json",
)


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


def _run(tmp_path, name, extra=None, agents_extra=None):
    config = build_config("golden", 8, STEPS, 101, "softmax")
    config.update(extra or {})
    config["agents"].update(agents_extra or {})
    out = tmp_path / name
    AgentCoupledRunner(config).run(days=STEPS, output_dir=out)
    return out


def _layer(decision, **over):
    section = {
        "mode": "all", "cadence_steps": 1, "levels": 1, "workers": 2,
        "semantic": {"provider": "fake", "run_id": "t",
                     "decisions": {"jev-semif-agent-v1:L1": decision}},
    }
    section.update(over)
    return {"semantic_agents": section}


def test_layer_with_unknown_fake_is_byte_identical_to_no_layer(tmp_path):
    base = _run(tmp_path, "base")
    neutral = _run(tmp_path, "neutral", _layer("unknown", levels=2))
    for name in STATE_FILES:
        assert (base / name).stat().st_size > 0, name
        assert _same_state(base, neutral, name), name
    assert (neutral / "semantic_agent_decisions.jsonl").exists()
    assert not (base / "semantic_agent_decisions.jsonl").exists()


def test_layer_with_a_real_preference_changes_the_run(tmp_path):
    base = _run(tmp_path, "base")
    pushed = _run(tmp_path, "pushed", _layer("explore"))
    assert (base / "replay_events.jsonl").read_bytes() != (
        pushed / "replay_events.jsonl"
    ).read_bytes()
    telemetry = json.loads((pushed / "semantic_agent_telemetry.json").read_text())
    assert telemetry["kernel_counters"]["pillar_changed"] > 0
    manifest = json.loads((pushed / "semantic_manifest.json").read_text())
    assert manifest["agent_layer"]["mode"] == "all"


def test_incompatible_decision_mode_fails_at_start(tmp_path):
    with pytest.raises(ValueError, match="preferences"):
        _run(tmp_path, "bad", _layer("unknown"), {"decision_mode": "tree"})


def test_run_without_layer_writes_no_semantic_files(tmp_path):
    base = _run(tmp_path, "base")
    assert not (base / "semantic_manifest.json").exists()
    assert not (base / "semantic_agent_telemetry.json").exists()
