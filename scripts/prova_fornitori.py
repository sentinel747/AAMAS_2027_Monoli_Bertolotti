# -*- coding: utf-8 -*-
"""Una chiamata per modello, con il prompt vero del governatore.

**Perche' non basta un "ciao, rispondi ok".** `check_provider.py` verifica che
l'endpoint risponda; questo verifica che risponda *una politica eseguibile*. Fra
le due cose c'e' tutto cio' che in questo progetto si e' rotto in silenzio:

- un modello che parla ma non sa produrre JSON valido sotto un prompt da
  duemila token;
- un modello che produce JSON ma nomina indicatori fuori vocabolario, e
  `parse_policy` scarta ogni regola --- la run gira, il governo tace, l'esito e'
  identico alla baseline;
- un endpoint che accetta il corpo minimo e rifiuta `response_format`, o il
  campo del ragionamento, o un `max_tokens` grande.

Nessuno dei tre da' un errore visibile durante una campagna: danno una policy
vuota, che e' indistinguibile da un governo che sceglie di non intervenire. Per
questo la prova costruisce un **quadro di colonia vero** --- da una simulazione
breve, non da valori inventati --- e passa dal proponente reale, `LLMProposer`,
fino a `parse_policy`. Cio' che si legge in tabella e' il numero di regole che
sarebbero entrate in vigore.

Uso (il consenso resta esplicito e a carico di chi lancia)::

    MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_fornitori.py
    MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_fornitori.py --provider gpu_farm4
    MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_fornitori.py --solo-gratuiti
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.parity_harness import realistic_config  # noqa: E402
from src.governors.llm_arm import LLMProposer  # noqa: E402
from src.governors.observation import build_picture  # noqa: E402
from src.governors.policy import Bounds  # noqa: E402
from src.llm.provider_registry import (  # noqa: E402
    LLM_CONSENT_ENV,
    create_llm_provider,
    load_provider_registry,
    real_calls_allowed,
    resolve_offered_model,
)
from src.simulation.agent_coupled_runner import AgentCoupledRunner  # noqa: E402

#: Le farm sono nostre e non costano niente; il resto e' a consumo.
GRATUITI = ("gpu_farm", "gpu_farm4", "ollama")

BOUNDS = Bounds(0.25, 4.0)


def quadro_vero(agenti: int = 60, passi: int = 30):
    """Un quadro di colonia preso da una simulazione, non inventato.

    Un prompt costruito a mano avrebbe numeri tondi e distribuzioni piatte,
    cioe' il caso piu' facile. Quello che serve sapere e' se il modello regge il
    prompt che ricevera' davvero.
    """
    config = realistic_config(agents=agenti, steps=passi, seed=3)
    config.setdefault("headless", {})["log_interval_steps"] = 10**9
    runner = AgentCoupledRunner(config)
    runner.run(days=passi, output_dir=None)
    return build_picture(
        passi, runner._last_metrics, runner.core.cells, runner.core.agents,
        runner.core.agents.alive_rows(),
    )


def prova(provider_id: str, config: dict, model: str, registro: dict, picture) -> dict:
    inizio = time.perf_counter()
    try:
        nome = resolve_offered_model(provider_id, config, model)
        provider = create_llm_provider(provider_id, nome or None, registro)
        proposta = asyncio.run(LLMProposer(provider, nome).propose(picture, BOUNDS))
    except Exception as errore:  # noqa: BLE001 - una prova non deve mai fermarsi
        return {
            "provider": provider_id, "model": model, "esito": "ERRORE",
            "dettaglio": f"{type(errore).__name__}: {errore}"[:90],
            "secondi": time.perf_counter() - inizio,
            "regole": 0, "tokens_in": 0, "tokens_out": 0, "costo": 0.0,
        }

    regole = len(proposta.policy.rules) if proposta.policy else 0
    motivo = (proposta.rationale or "").strip()
    if regole:
        esito, dettaglio = "ok", motivo[:90]
    elif proposta.policy is not None:
        # Policy presente ma vuota: il modello ha risposto e ha scelto di non
        # intervenire, oppure ogni regola e' stata scartata da `parse_policy`.
        esito, dettaglio = "VUOTA", (motivo or "nessuna regola sopravvissuta")[:90]
    else:
        esito, dettaglio = "MUTO", (motivo or "nessuna proposta")[:90]
    return {
        "provider": provider_id, "model": model, "esito": esito, "dettaglio": dettaglio,
        "secondi": proposta.latency_s or (time.perf_counter() - inizio),
        "regole": regole,
        "tokens_in": int(getattr(proposta, "tokens_in", 0) or 0),
        "tokens_out": int(getattr(proposta, "tokens_out", 0) or 0),
        "costo": float(getattr(proposta, "cost", 0.0) or 0.0),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", action="append", default=[],
                   help="limita a questi fornitori; ripetibile")
    p.add_argument("--solo-gratuiti", action="store_true",
                   help="salta i fornitori a consumo")
    p.add_argument("--agenti", type=int, default=60)
    p.add_argument("--passi", type=int, default=30)
    args = p.parse_args()

    if not real_calls_allowed():
        print(f"{LLM_CONSENT_ENV} non e' impostata: ogni provider sarebbe il "
              "fallback deterministico e la prova non direbbe niente. Esco.")
        return 2

    registro = load_provider_registry()
    scelti = [n for n in registro if not args.provider or n in args.provider]
    if args.solo_gratuiti:
        scelti = [n for n in scelti if n in GRATUITI]

    print("Costruisco un quadro di colonia vero...", flush=True)
    picture = quadro_vero(args.agenti, args.passi)
    print(f"  passo {picture.step}, {picture.population} coloni, "
          f"{picture.n_cells} celle, {len(picture.indicators)} indicatori\n", flush=True)

    esiti = []
    for provider_id in scelti:
        config = registro[provider_id]
        chiave = config.get("api_key_env") or ""
        modelli = config.get("available_models") or list((config.get("model_options") or {}).keys())
        if chiave and provider_id != "ollama" and not (os.getenv(chiave) or "").strip():
            print(f"{provider_id:<11} {chiave} assente nell'ambiente: salto "
                  f"{len(modelli)} modelli", flush=True)
            continue
        for model in modelli:
            print(f"  ... {provider_id}/{model}", flush=True)
            esiti.append(prova(provider_id, config, model, registro, picture))

    print(f"\n{'fornitore':<11} {'modello':<24} {'esito':>6} {'reg':>4} "
          f"{'sec':>7} {'tok in':>7} {'tok out':>8} {'costo $':>9}  motivo/errore")
    for e in esiti:
        print(f"{e['provider']:<11} {e['model']:<24} {e['esito']:>6} {e['regole']:>4} "
              f"{e['secondi']:7.1f} {e['tokens_in']:7d} {e['tokens_out']:8d} "
              f"{e['costo']:9.4f}  {e['dettaglio']}")

    utili = [e for e in esiti if e["esito"] == "ok"]
    print(f"\n  {len(utili)}/{len(esiti)} modelli hanno prodotto una politica eseguibile.")
    print(f"  costo totale: ${sum(e['costo'] for e in esiti):.4f}")
    rotti = [e for e in esiti if e["esito"] != "ok"]
    if rotti:
        print("\n  NON usabili come governatori:")
        for e in rotti:
            print(f"    {e['provider']}/{e['model']}: {e['esito']} — {e['dettaglio']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
