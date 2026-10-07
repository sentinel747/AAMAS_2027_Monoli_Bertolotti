# -*- coding: utf-8 -*-
"""La prova a vuoto: quanto costa una campagna, prima di lanciarla.

**Che cosa misura, e che cosa NON misura.** Non misura chi governa meglio: le
run sono troppo corte e i semi troppo pochi. Misura le grandezze che decidono
se una campagna e' eseguibile, e che si scoprono altrimenti solo spendendola:

- il **tempo di orologio** per braccio, che nel regime bloccante e' dominato
  dalla latenza del modello e non dalla simulazione;
- il **costo** in chiamate e in denaro, per braccio e per gradino di contesto;
- che le direttive **entrino davvero in vigore** --- una run `llm` che chiude
  come la baseline per una cadenza sbagliata o un provider muto e' il modo
  piu' caro di non misurare niente;
- che tutti e quattro i gradini di contesto producano **policy valide**, e in
  particolare che il gradino cieco non venga scartato dalla traduzione inversa;
- **quando nascono i distretti**, cioe' da che passo in poi lo strato
  amministrativo comincia a costare chiamate.

**Perche' esiste come script e non come comando a mano.** Le stesse grandezze
vanno rimisurate ogni volta che cambia il modello, la cadenza o lo scenario, e
una misura che si rifa' a mano si rifa' diversa.

Uso (il consenso alle chiamate resta esplicito e a carico di chi lancia)::

    MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_a_vuoto.py \\
        --provider gpu_farm --model gpt-oss:20b --passi 150 --seme 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.parity_harness import realistic_config  # noqa: E402
from src.governors.context import LIVELLI  # noqa: E402
from src.llm.provider_registry import LLM_CONSENT_ENV, real_calls_allowed  # noqa: E402
from src.simulation.agent_coupled_runner import AgentCoupledRunner  # noqa: E402


def _config(args, arm: str, contesto: str, amministratori: bool) -> dict:
    config = realistic_config(agents=args.agenti, steps=args.passi, seed=args.seme)
    config["headless"]["progress_every"] = 10**9
    blocco: dict = {
        "arm": arm,
        "cadence_steps": args.cadenza,
        "priority_multiplier_range": [0.25, 4.0],
        "context_level": contesto,
        # **Regime bloccante, e non e' un dettaglio.** Con una latenza di venti
        # secondi e un passo da un decimo, una cadenza non bloccante dovrebbe
        # valere piu' di duecento passi perche' una risposta faccia in tempo:
        # su una run corta nessuna direttiva entrerebbe mai in vigore e la
        # prova misurerebbe la baseline al prezzo del modello.
        "wait_seconds": args.attesa if arm == "llm" else 0.0,
    }
    if arm == "llm":
        blocco["assignments"] = [{
            "provider": args.provider,
            "model": args.model,
            "thinking": args.thinking,
            "temperature": args.temperatura,
        }]
    if amministratori:
        blocco["administrators"] = {
            "enabled": True,
            "cells_per_district": args.celle_per_distretto,
            "follow_governor_arm": True,
        }
    config["governors"] = blocco
    return config


def _esegui(args, etichetta: str, arm: str, contesto: str, amministratori: bool) -> dict:
    cartella = Path(args.uscita) / etichetta.replace(" ", "_").replace("+", "e")
    inizio = time.perf_counter()
    runner = AgentCoupledRunner(_config(args, arm, contesto, amministratori))
    runner.run(args.passi, output_dir=cartella)
    durata = time.perf_counter() - inizio

    colpi = dict(runner.core.governor_policy_hits or {})
    scattate = sum(v for k, v in colpi.items() if not k.startswith("- nessuna"))
    amm = getattr(runner, "_administration", None)

    decisioni = []
    registro = cartella / "governor_decisions.jsonl"
    if registro.exists():
        decisioni = [json.loads(r) for r in registro.read_text(encoding="utf-8").splitlines()]
    con_regole = sum(1 for d in decisioni if (d.get("policy") or {}).get("rules"))

    uso = {}
    percorso_uso = cartella / "api_usage.json"
    if percorso_uso.exists():
        uso = json.loads(percorso_uso.read_text(encoding="utf-8"))

    return {
        "etichetta": etichetta,
        "secondi": durata,
        "vivi": len(runner.core.agents.alive_rows()),
        "celle": int((runner.core.cells.occupancy > 0).sum()),
        "tick": len(decisioni),
        "tick_con_policy": con_regole,
        "celle_passo_governate": int(scattate),
        "chiamate": int(uso.get("calls", 0) or 0),
        "costo": float(uso.get("estimated_cost_usd", 0.0) or 0.0),
        "distretti": amm.distretti.numero_distretti() if amm else 0,
        "interventi": amm.interventi if amm else 0,
        "astensioni": amm.astensioni if amm else 0,
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", default="gpu_farm")
    p.add_argument("--model", default="gpt-oss:20b")
    p.add_argument("--thinking", default="off", choices=["off", "low", "medium", "high", "dynamic"])
    p.add_argument("--temperatura", type=float, default=0.0)
    p.add_argument("--passi", type=int, default=150)
    p.add_argument("--agenti", type=int, default=120)
    p.add_argument("--seme", type=int, default=7)
    p.add_argument("--cadenza", type=int, default=25)
    p.add_argument("--attesa", type=float, default=90.0)
    p.add_argument("--celle-per-distretto", type=int, default=3)
    p.add_argument("--uscita", default="outputs/prova_a_vuoto")
    p.add_argument("--con-amministratori", action="store_true",
                   help="aggiunge una configurazione con lo strato amministrativo")
    args = p.parse_args()

    if not real_calls_allowed():
        print(f"{LLM_CONSENT_ENV} non e' impostata: i bracci `llm` girerebbero sul "
              "fallback deterministico e la prova misurerebbe l'impianto, non il "
              "modello. Esco senza eseguire nulla.")
        return 2

    configurazioni = [
        ("baseline (nessun governo)", "none", "completo", False),
        ("scripted", "scripted", "completo", False),
        ("random", "random", "completo", False),
    ]
    configurazioni += [(f"llm {livello}", "llm", livello, False) for livello in LIVELLI]
    if args.con_amministratori:
        configurazioni.append(("llm completo + amministratori", "llm", "completo", True))

    print(f"Prova a vuoto: {args.provider}/{args.model}, thinking={args.thinking}, "
          f"{args.agenti} coloni, {args.passi} passi, seme {args.seme}, cadenza "
          f"{args.cadenza}, attesa {args.attesa:g}s\n", flush=True)

    esiti = []
    for etichetta, arm, contesto, amm in configurazioni:
        print(f"  ... {etichetta}", flush=True)
        esiti.append(_esegui(args, etichetta, arm, contesto, amm))

    print(f"\n{'configurazione':>30} {'secondi':>8} {'vivi':>5} {'celle':>6} "
          f"{'tick':>5} {'con policy':>11} {'gov.te':>7} {'chiam.':>7} {'costo $':>9}")
    for e in esiti:
        print(f"{e['etichetta']:>30} {e['secondi']:8.1f} {e['vivi']:5d} {e['celle']:6d} "
              f"{e['tick']:5d} {e['tick_con_policy']:11d} {e['celle_passo_governate']:7d} "
              f"{e['chiamate']:7d} {e['costo']:9.4f}")

    llm = [e for e in esiti if e["etichetta"].startswith("llm")]
    muti = [e for e in llm if e["tick_con_policy"] == 0]
    inerti = [e for e in llm if e["tick_con_policy"] and e["celle_passo_governate"] == 0]
    print()
    if muti:
        print("  ATTENZIONE: nessuna policy valida in " + ", ".join(e["etichetta"] for e in muti))
        print("  Una campagna su questi gradini misurerebbe la baseline al prezzo del modello.")
    if inerti:
        print("  ATTENZIONE: policy valide ma zero celle catturate in "
              + ", ".join(e["etichetta"] for e in inerti))
    if not muti and not inerti and llm:
        print("  Tutti i gradini producono policy valide che catturano celle.")

    totale_chiamate = sum(e["chiamate"] for e in esiti)
    totale_costo = sum(e["costo"] for e in esiti)
    secondi_llm = sum(e["secondi"] for e in llm)
    per_tick = secondi_llm / max(1, sum(e["tick"] for e in llm))
    print(f"\n  {totale_chiamate} chiamate, ${totale_costo:.4f}, "
          f"{per_tick:.1f} s per tick sui bracci llm.")
    print(f"  Proiezione per una campagna: passi x bracci x semi / cadenza x {per_tick:.1f} s.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
