"""Etichetta le domande SemIf raccolte con un provider, senza soglia.

Legge il JSONL di `semif_harvest_cases.py`, campiona fino a `--per-kind` domande
per tipo (agent_l1, governor_l1, governor_l2, admin_l1, admin_l2) con un seme
fisso, le fa decidere al provider scelto e salva la distribuzione completa.
La stessa lista di casi, con lo stesso seme, si etichetta con piu' provider
(Jev come riferimento, Qwen per la calibrazione, Laya per la distillazione) e
le righe si confrontano per `input_hash`.

    python scripts/semif_label_cases.py --harvest <harvest.jsonl> --provider typesafe \\
        --typesafe-budget-usd 1.0 --per-kind 400 --confirm-live --out <jev.jsonl>

`rest`, `typesafe` e `laya` contattano la rete: servono `--confirm-live`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.run_governor_experiment import _semantic_section  # noqa: E402
from src.semantic_governance.batch import decide_many  # noqa: E402
from src.semantic_governance.factory import build_semantic_provider  # noqa: E402
from src.semantic_governance.schemas import (  # noqa: E402
    MAX_OPTIONS,
    MAX_OPTIONS_TYPESAFE,
    SemanticDecisionRequest,
    SemanticOption,
)

LIVE = ("rest", "typesafe", "laya")


def _requests(rows, max_options):
    for row in rows:
        yield SemanticDecisionRequest(
            run_id="label",
            step=int(row["step"]),
            actor=str(row["actor"]),
            question_id=str(row["question_id"]),
            state=row["state"],
            question=str(row["question"]),
            options=tuple(SemanticOption(o["id"], o["description"]) for o in row["options"]),
            max_options=max_options,
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harvest", type=Path, required=True)
    parser.add_argument("--provider", choices=("fake", "rest", "typesafe", "laya"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-kind", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--endpoint", default="")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--typesafe-budget-usd", type=float, default=0.0)
    parser.add_argument("--confirm-live", action="store_true")
    args = parser.parse_args(argv)
    if args.provider in LIVE and not args.confirm_live:
        print(f"provider {args.provider} contatta la rete: serve --confirm-live", file=sys.stderr)
        return 2

    rows = [json.loads(line) for line in args.harvest.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_kind: dict[str, list] = {}
    for row in rows:
        by_kind.setdefault(row["kind"], []).append(row)
    rng = random.Random(args.seed)
    sample = []
    for kind in sorted(by_kind):
        group = sorted(by_kind[kind], key=lambda r: (r["actor"], int(r["step"]), r["question_id"]))
        sample.extend(rng.sample(group, min(args.per_kind, len(group))))

    section = _semantic_section(
        "rest" if args.provider == "laya" else args.provider,
        run_id="label",
        endpoint=args.endpoint,
        timeout_seconds=args.timeout,
        min_confidence=0.0,
        budget_usd=args.typesafe_budget_usd,
    )
    if args.provider == "laya":
        section = {"provider": "laya", "endpoint": args.endpoint,
                   "timeout_seconds": args.timeout, "min_confidence": 0.0}
    provider = build_semantic_provider(section)
    limit = MAX_OPTIONS_TYPESAFE if args.provider in ("typesafe", "laya") else MAX_OPTIONS
    requests = list(_requests(sample, limit))
    decisions = decide_many(provider, requests, max_workers=args.workers)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for row, request, decision in zip(sample, requests, decisions):
            sink.write(json.dumps({
                "kind": row["kind"],
                "input_hash": request.input_hash,
                "question_id": request.question_id,
                "provider": args.provider,
                "selected": decision.selected_option,
                "probabilities": dict(decision.probabilities),
                "confidence": decision.confidence,
                "route": decision.route,
                "fallback_reason": decision.fallback_reason,
                "latency_ms": decision.latency_ms,
            }, ensure_ascii=False) + "\n")
    counts: dict[str, int] = {}
    for row in sample:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    errors = sum(1 for d in decisions if d.route == "fallback" and d.fallback_reason not in ("", "selected_fallback"))
    summary = {"provider": args.provider, "cases": len(sample), "by_kind": counts, "errors": errors}
    ledger = getattr(provider, "ledger", None)
    if ledger is not None:
        summary["ledger"] = ledger.summary()
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
