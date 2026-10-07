"""Confronto su casi etichettati fra provider (piano 2026-09-22, Task 11)."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "semif_case_benchmark.py"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=ROOT,
                          capture_output=True, text=True, timeout=120)


def test_fake_provider_scores_the_sustenance_cases(tmp_path):
    result = _run("--provider", "fake", "--out", str(tmp_path))
    assert result.returncode == 0, result.stderr
    rows = (tmp_path / "case_results.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 20
    summary = json.loads((tmp_path / "case_summary.json").read_text(encoding="utf-8"))
    assert summary["agreement"] == 4 and summary["cases"] == 20


def test_live_providers_require_explicit_confirmation(tmp_path):
    for provider in ("typesafe", "rest"):
        out = tmp_path / provider
        result = _run("--provider", provider, "--out", str(out),
                      "--endpoint", "http://127.0.0.1:1", "--typesafe-budget-usd", "0.01")
        assert result.returncode == 2
        assert not out.exists() or not any(out.iterdir())
