"""Calibrazione a temperatura di un provider su un riferimento (2026-09-23)."""

import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

from src.semantic_governance.calibration import (
    fit_temperature,
    temper,
    temperature_for,
)

ROOT = Path(__file__).resolve().parents[1]


def test_temper_with_one_is_the_identity():
    p = {"a": 0.5, "b": 0.3, "unknown": 0.2}
    assert temper(p, 1.0) == p


def test_temper_below_one_sharpens_and_keeps_the_argmax():
    p = {"a": 0.5, "b": 0.3, "unknown": 0.2}
    q = temper(p, 0.5)
    assert max(q, key=q.get) == "a"
    assert q["a"] > p["a"]
    assert math.isclose(sum(q.values()), 1.0)


def test_temper_handles_zero_probabilities():
    q = temper({"a": 1.0, "b": 0.0}, 0.3)
    assert q == {"a": 1.0, "b": 0.0}


def test_fit_recovers_a_sharpening_temperature():
    # lo studente ha l'argmax giusto ma distribuzioni piatte: T ottimo < 1
    rows = [({"a": 0.4, "b": 0.3, "unknown": 0.3}, "a")] * 50
    t = fit_temperature(rows)
    assert t < 1.0


def test_fit_on_confidently_wrong_student_softens():
    rows = [({"a": 0.95, "b": 0.05}, "b")] * 10 + [({"a": 0.95, "b": 0.05}, "a")] * 10
    assert fit_temperature(rows) > 1.0


def test_temperature_for_matches_the_most_specific_key():
    table = {"agent-v1:L1": 0.5, "governor-v2h:L1": 0.3, "governor-v2h:L2": 0.7}
    assert temperature_for("jev-semif-agent-v1:L1", table) == 0.5
    assert temperature_for("jev-semif-governor-v2h:L2:build", table) == 0.7
    assert temperature_for("jev-semif-admin-v2h:L1", table) == 1.0


def test_rest_provider_applies_the_temperature():
    from src.semantic_governance.client import RestSemanticDecisionProvider
    from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption

    request = SemanticDecisionRequest(
        run_id="r", step=1, actor="a", question_id="jev-semif-agent-v1:L1",
        state={"x": 1}, question="Which?",
        options=(SemanticOption("a", "A"), SemanticOption("b", "B"), SemanticOption("unknown", "U")),
    )

    def transport(url, payload, headers, timeout):
        body = {"selected_option": "a", "probabilities": {"a": 0.5, "b": 0.3, "unknown": 0.2},
                "confidence": 0.1, "inference_time_ms": 5.0}
        return 200, json.dumps(body).encode(), {}

    plain = RestSemanticDecisionProvider("http://x", transport=transport, min_confidence=0.0)
    sharp = RestSemanticDecisionProvider("http://x", transport=transport, min_confidence=0.0,
                                         temperature={"agent-v1:L1": 0.5})
    p, q = plain.decide(request), sharp.decide(request)
    assert p.probabilities["a"] == pytest.approx(0.5)
    assert q.probabilities["a"] > 0.5
    assert q.confidence > p.confidence or p.confidence == 0.1


def test_calibrate_script_writes_a_table(tmp_path):
    teacher = tmp_path / "t.jsonl"
    student = tmp_path / "s.jsonl"
    rows_t, rows_s = [], []
    for i in range(40):
        h = f"h{i}"
        rows_t.append({"kind": "agent_l1", "input_hash": h, "question_id": "jev-semif-agent-v1:L1",
                       "selected": "a", "probabilities": {"a": 0.9, "b": 0.05, "unknown": 0.05},
                       "route": "filter", "fallback_reason": ""})
        rows_s.append({"kind": "agent_l1", "input_hash": h, "question_id": "jev-semif-agent-v1:L1",
                       "selected": "a", "probabilities": {"a": 0.4, "b": 0.3, "unknown": 0.3},
                       "route": "filter", "fallback_reason": ""})
    teacher.write_text("".join(json.dumps(r) + "\n" for r in rows_t), encoding="utf-8")
    student.write_text("".join(json.dumps(r) + "\n" for r in rows_s), encoding="utf-8")
    out = tmp_path / "calib.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "semif_calibrate.py"),
                        "--teacher", str(teacher), "--student", str(student), "--out", str(out)],
                       cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr
    table = json.loads(out.read_text(encoding="utf-8"))
    assert table["temperature"]["agent-v1:L1"] < 1.0
    assert table["report"]["agent-v1:L1"]["agreement"] == 1.0
