"""Confronto su casi etichettati fra provider SemIf (livello 1 degli agenti).

Stessi 20 casi, stessa domanda, stesse opzioni: cambia solo chi decide. Le
etichette sono scritte a mano (configs/experiments/jev_semif/agent_cases_v1.json):
il confronto dice se un provider concorda con un giudizio umano esplicito, non
se ha "ragione". I casi 19-20 attendono `unknown`: misurano se sa astenersi.

    python scripts/semif_case_benchmark.py --provider fake --out <dir>
    python scripts/semif_case_benchmark.py --provider rest --endpoint http://127.0.0.1:8008 --confirm-live --out <dir>
    python scripts/semif_case_benchmark.py --provider typesafe --typesafe-budget-usd 0.5 --confirm-live --out <dir>

`rest` e `typesafe` contattano la rete: senza `--confirm-live` lo script esce con
codice 2 e non scrive nulla. Nessuna variabile d'ambiente viene mai stampata.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_governor_experiment import _semantic_section  # noqa: E402
from src.semantic_governance import agent_questions as aq  # noqa: E402
from src.semantic_governance.batch import decide_many  # noqa: E402
from src.semantic_governance.factory import build_semantic_provider  # noqa: E402
from src.semantic_governance.schemas import MAX_OPTIONS, MAX_OPTIONS_TYPESAFE  # noqa: E402

DEFAULT_CASES = REPO_ROOT / "configs" / "experiments" / "jev_semif" / "agent_cases_v1.json"
LIVE = ("rest", "typesafe")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("fake", "rest", "typesafe"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--api-key-env", default="")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--typesafe-budget-usd", type=float, default=0.0)
    parser.add_argument("--typesafe-model", default="jev-latest")
    parser.add_argument("--typesafe-question-type", default="choice")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--confirm-live", action="store_true",
        help="obbligatorio per rest e typesafe: contattano la rete",
    )
    args = parser.parse_args(argv)
    if args.provider in LIVE and not args.confirm_live:
        print(f"provider {args.provider} contatta la rete: serve --confirm-live", file=sys.stderr)
        return 2

    data = json.loads(args.cases.read_text(encoding="utf-8"))
    section = _semantic_section(
        args.provider,
        run_id="cases",
        endpoint=args.endpoint,
        api_key_env=args.api_key_env,
        timeout_seconds=args.timeout,
        # Soglia a zero: qui si misura la distribuzione, non la si taglia.
        min_confidence=0.0,
        budget_usd=args.typesafe_budget_usd,
        typesafe_model=args.typesafe_model,
        question_type=args.typesafe_question_type,
    )
    provider = build_semantic_provider(section)
    limit = MAX_OPTIONS_TYPESAFE if args.provider == "typesafe" else MAX_OPTIONS
    cases = data["cases"]
    requests = [
        aq.level1_request("cases", index + 1, case["id"], data["colony"], case["colonist"], limit)
        for index, case in enumerate(cases)
    ]
    decisions = decide_many(provider, requests, max_workers=args.workers)

    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    for case, decision in zip(cases, decisions):
        rows.append({
            "id": case["id"],
            "expected": case["expected"],
            "selected": decision.selected_option,
            "agrees": decision.selected_option == case["expected"],
            "confidence": decision.confidence,
            "probabilities": dict(decision.probabilities),
            "latency_ms": decision.latency_ms,
            "route": decision.route,
            "fallback_reason": decision.fallback_reason,
            "runtime": decision.runtime,
        })
    (args.out / "case_results.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    latencies = sorted(r["latency_ms"] for r in rows)
    by_label: dict[str, list[float]] = {}
    for r in rows:
        by_label.setdefault(r["expected"], []).append(r["confidence"])
    summary = {
        "provider": args.provider,
        "cases": len(rows),
        "agreement": sum(r["agrees"] for r in rows),
        "fallbacks": sum(r["route"] == "fallback" for r in rows),
        "fallback_reasons": sorted({r["fallback_reason"] for r in rows if r["fallback_reason"]}),
        "mean_confidence_by_expected": {
            k: round(statistics.fmean(v), 4) for k, v in sorted(by_label.items())
        },
        "latency_ms_p50": round(latencies[len(latencies) // 2], 3) if latencies else None,
        "latency_ms_p95": (
            round(latencies[int(0.95 * (len(latencies) - 1))], 3) if latencies else None
        ),
    }
    ledger = getattr(provider, "ledger", None)
    if ledger is not None:
        summary["typesafe_ledger"] = ledger.summary()
    (args.out / "case_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
