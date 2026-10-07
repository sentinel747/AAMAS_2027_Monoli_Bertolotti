"""Etichettatura delle domande raccolte con un provider (fake nei test)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _harvest(tmp_path, n_agents=30, n_gov=5):
    rows = []
    for i in range(n_agents):
        rows.append({"kind": "agent_l1", "question_id": "jev-semif-agent-v1:L1",
                     "actor": f"agent:a{i}", "step": 1, "input_hash": "", "state": {"i": i},
                     "question": "Which?", "options": [
                         {"id": "sustenance", "description": "s"},
                         {"id": "unknown", "description": "u"}]})
    for i in range(n_gov):
        rows.append({"kind": "governor_l1", "question_id": "jev-semif-governor-v2h:L1",
                     "actor": "governor", "step": i, "input_hash": "", "state": {"g": i},
                     "question": "Which area?", "options": [
                         {"id": "keep_previous", "description": "k"},
                         {"id": "area:life", "description": "l"},
                         {"id": "unknown", "description": "u"}]})
    path = tmp_path / "harvest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def _run(*args):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "semif_label_cases.py"), *args],
                          cwd=ROOT, capture_output=True, text=True, timeout=120)


def test_fake_labelling_samples_per_kind_and_keeps_distributions(tmp_path):
    harvest = _harvest(tmp_path)
    out = tmp_path / "fake.jsonl"
    r = _run("--harvest", str(harvest), "--provider", "fake", "--per-kind", "10",
             "--seed", "7", "--out", str(out))
    assert r.returncode == 0, r.stderr
    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    kinds = [row["kind"] for row in rows]
    assert kinds.count("agent_l1") == 10 and kinds.count("governor_l1") == 5
    assert all(abs(sum(row["probabilities"].values()) - 1) < 1e-9 for row in rows)
    assert all(row["input_hash"] for row in rows)


def test_same_seed_samples_the_same_cases(tmp_path):
    harvest = _harvest(tmp_path)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    for out in (a, b):
        assert _run("--harvest", str(harvest), "--provider", "fake", "--per-kind", "10",
                    "--seed", "7", "--out", str(out)).returncode == 0
    ids = lambda p: [json.loads(l)["input_hash"] for l in p.read_text().splitlines()]  # noqa: E731
    assert ids(a) == ids(b)


def test_live_providers_need_confirmation(tmp_path):
    harvest = _harvest(tmp_path)
    r = _run("--harvest", str(harvest), "--provider", "typesafe", "--typesafe-budget-usd", "0.1",
             "--out", str(tmp_path / "x.jsonl"))
    assert r.returncode == 2
    assert "confirm-live" in r.stderr
