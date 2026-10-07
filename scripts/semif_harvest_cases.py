"""Raccoglie domande SemIf reali dalla simulazione, per etichettarle e calibrare.

Le domande sono quelle che governatore (v2h), amministratori (v2h) e agenti
(livello 1) fanno davvero in una run: stesso stato, stesse opzioni, stessa
forma. Il "raccoglitore" risponde in modo da non cambiare la simulazione:

- agenti: `unknown` (fattori neutri, run identica a quella rule-based);
- governatore e amministratori, livello 1: ruota fra le aree disponibili, cosi'
  il livello 2 viene chiesto per ogni area nel corso della run;
- livello 2: `unknown` (il governatore conserva, l'amministratore accetta).

Due run per seme: (A) governatore SemIf v2h + strato agenti; (B) governatore
scripted (una legge sempre in vigore) + amministratori SemIf v2h, perche' in v2h
gli amministratori deliberano solo su una legge del governo.

Nessuna rete: il raccoglitore sostituisce il provider in memoria, in questo
processo. Uscita: un JSONL con una riga per domanda (dedotta per input_hash).

    python scripts/semif_harvest_cases.py --seeds 1 2 3 --steps 600 --agents 300 \\
        --out runs/jev_semif_experiments/labelled_20260923/harvest.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.semantic_governance.schemas import SemanticDecision  # noqa: E402


class Harvester:
    """Provider che registra ogni domanda e risponde senza cambiare la run."""

    def __init__(self) -> None:
        self.rows: dict[str, dict] = {}
        self._turno = 0

    def _kind(self, request) -> str:
        if request.actor.startswith("agent:"):
            return "agent_l1"
        chi = "governor" if request.actor == "governor" else "admin"
        livello = "l2" if ":L2:" in request.question_id else "l1"
        return f"{chi}_{livello}"

    def decide(self, request) -> SemanticDecision:
        kind = self._kind(request)
        self.rows.setdefault(request.input_hash, {
            "kind": kind,
            "question_id": request.question_id,
            "actor": request.actor,
            "step": request.step,
            "input_hash": request.input_hash,
            "state": request.state,
            "question": request.question,
            "options": [o.to_dict() for o in request.options],
        })
        aree = [o for o in request.option_ids if o.startswith("area:")]
        if kind in ("governor_l1", "admin_l1") and aree:
            scelta = aree[self._turno % len(aree)]
            self._turno += 1
        else:
            scelta = "unknown"
        probabilities = {o: float(o == scelta) for o in request.option_ids}
        return SemanticDecision(
            run_id=request.run_id, step=request.step, actor=request.actor,
            question_id=request.question_id, input_hash=request.input_hash,
            selected_option=scelta, probabilities=probabilities, confidence=1.0,
            route="fallback" if scelta == "unknown" else "filter",
            rationale={"source": "harvest"}, runtime="harvest",
        )


def _install(harvester: Harvester) -> None:
    import src.semantic_governance.agent_layer as agent_layer
    import src.semantic_governance.factory as factory

    def build(_config, telemetry=None):
        return harvester

    factory.build_semantic_provider = build
    agent_layer.build_semantic_provider = build


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--agents", type=int, default=300)
    parser.add_argument("--cadence", type=int, default=20)
    parser.add_argument("--agents-cadence", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    from scripts.run_governor_experiment import run_arm

    harvester = Harvester()
    _install(harvester)
    work = args.out.parent / "harvest_runs"
    for seed in args.seeds:
        # (A) governatore v2h + strato agenti
        run_arm(
            "semif", seed, args.steps, args.agents, work / f"A_seed{seed}",
            cadence_steps=args.cadence, wait_seconds=30, log_agents=False,
            semantic_provider="fake", semantic_run_id=f"harvest-A-{seed}",
            semantic_candidate_profile="v2h",
            semantic_agents="all", semantic_agents_cadence=args.agents_cadence,
        )
        # (B) legge scripted sempre in vigore + amministratori v2h
        run_arm(
            "scripted", seed, args.steps, args.agents, work / f"B_seed{seed}",
            cadence_steps=args.cadence, wait_seconds=30, log_agents=False,
            administrators=True, admin_arm="semif",
            semantic_provider="fake", semantic_run_id=f"harvest-B-{seed}",
            semantic_candidate_profile="v2h",
        )
        conteggio: dict[str, int] = {}
        for row in harvester.rows.values():
            conteggio[row["kind"]] = conteggio.get(row["kind"], 0) + 1
        print(f"[harvest] dopo il seme {seed}: {conteggio}", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as sink:
        for row in harvester.rows.values():
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[harvest] {len(harvester.rows)} domande distinte in {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
