"""Stima la temperatura di un provider (studente) su un riferimento (maestro).

Legge due file di `semif_label_cases.py` etichettati sugli stessi casi, li
unisce per `input_hash`, scarta i casi in cui uno dei due ha avuto un errore
(timeout, HTTP, budget: non un'astensione scelta), e per ogni tipo di domanda
stima la temperatura che rende la distribuzione dello studente piu' verosimile
rispetto alla scelta del maestro. Riporta anche quanto spesso la regola di
decisione v2h scatterebbe prima e dopo.

    python scripts/semif_calibrate.py --teacher jev.jsonl --student qwen.jsonl --out calib.json

L'uscita `temperature` va nella sezione del provider rest come
`temperature: {chiave: T}`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.governors.policy_candidates import decisive_probabilities  # noqa: E402
from src.semantic_governance.calibration import (  # noqa: E402
    calibration_key,
    fit_temperature,
    temper,
)

_OK = ("", "selected_fallback")


def _load(path: Path) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["input_hash"]] = row
    return rows


def _nll(pairs, t):
    return sum(-math.log(temper(p, t).get(target, 0.0) + 1e-12) for p, target in pairs) / len(pairs)


def _argmax(p):
    return max(p, key=p.get)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher", type=Path, required=True)
    parser.add_argument("--student", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    teacher, student = _load(args.teacher), _load(args.student)
    groups: dict[str, list] = {}
    skipped = 0
    for h, t_row in teacher.items():
        s_row = student.get(h)
        if s_row is None or t_row["fallback_reason"] not in _OK or s_row["fallback_reason"] not in _OK:
            skipped += 1
            continue
        groups.setdefault(calibration_key(t_row["question_id"]), []).append(
            (s_row["probabilities"], t_row["selected"], t_row["probabilities"])
        )

    temperatures, report = {}, {}
    for key, rows in sorted(groups.items()):
        pairs = [(p, target) for p, target, _ in rows]
        t = fit_temperature(pairs)
        temperatures[key] = round(t, 4)
        report[key] = {
            "cases": len(rows),
            "agreement": round(sum(_argmax(p) == target for p, target in pairs) / len(pairs), 4),
            "nll_before": round(_nll(pairs, 1.0), 4),
            "nll_after": round(_nll(pairs, t), 4),
            "decisive_teacher": round(
                sum(decisive_probabilities(tp, target) for _, target, tp in rows) / len(rows), 4),
            "decisive_student_before": round(
                sum(decisive_probabilities(p, _argmax(p)) for p, _, _ in rows) / len(rows), 4),
            "decisive_student_after": round(
                sum(decisive_probabilities(temper(p, t), _argmax(p)) for p, _, _ in rows) / len(rows), 4),
        }
    out = {"temperature": temperatures, "report": report, "skipped_error_cases": skipped,
           "teacher": str(args.teacher), "student": str(args.student)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
