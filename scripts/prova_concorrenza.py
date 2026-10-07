# -*- coding: utf-8 -*-
"""Quante chiamate insieme regge davvero un fornitore, e da dove smette di guadagnare.

**Il malinteso da togliere di mezzo.** La finestra di contesto (128k, 256k) e' un
limite PER RICHIESTA: cinquanta chiamate da 5k token non ne fanno 250k contro un
tetto di 128k, sono cinquanta richieste indipendenti da 5k. Cio' che limita
quante ne girano davvero insieme e' la memoria che il server dedica alla cache
delle chiavi e quante sequenze elabora in parallelo; il resto lo mette in coda.

**Perche' misurarlo conta stanotte.** Una nostra run, a colonia matura, interroga
fino a CINQUANTA amministratori nello stesso istante (`asyncio.gather` senza
limite). Se il fornitore ne serve otto per volta, una run sola lo sta gia'
saturando e affiancarne una seconda non guadagna niente: accoda soltanto. Se ne
serve cinquanta, allora due simulazioni insieme rendono davvero.

Si misura mandando K chiamate della NOSTRA dimensione tipica (~5k token) tutte
insieme, per K crescente, e guardando la resa: chiamate al secondo. Finche' la
resa sale, c'e' margine; quando si appiattisce, quello e' il tetto.

Uso:
  MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_concorrenza.py
  MARSABM_ALLOW_LLM_CALLS=1 python scripts/prova_concorrenza.py \\
      --provider gpu_farm4 --model Qwen/Qwen3.8-27B-FP8 --gradini 1 10 25 50
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.llm.provider_registry import create_llm_provider  # noqa: E402

#: La dimensione tipica di un nostro prompt: 4.400 token in mediana sulle 73.159
#: chiamate d'archivio, 5.712 al massimo. Quattro caratteri per token.
TOKEN_TIPICI = 4400
RIEMPITIVO = ("Colony telemetry line {n}: nominal, no operational content, "
              "nothing here changes any decision.\n")


def prompt(n_token: int) -> str:
    coda = ('\n\nReply with a JSON object with exactly one key "ok" whose value '
            'is the integer 1. Nothing else.\n')
    pezzi = []
    lunghezza = 0
    n = 0
    obiettivo = n_token * 4 - len(coda)
    while lunghezza < obiettivo:
        riga = RIEMPITIVO.format(n=n)
        pezzi.append(riga)
        lunghezza += len(riga)
        n += 1
    return "".join(pezzi) + coda


def una(provider, testo: str) -> tuple[float, bool]:
    inizio = time.perf_counter()
    try:
        risposta = provider.complete_json(testo)
        ok = not (getattr(risposta, "error", "") or "")
    except Exception:  # noqa: BLE001 - un fallimento e' un dato, non un incidente
        ok = False
    return time.perf_counter() - inizio, ok


def gradino(provider, testo: str, k: int) -> dict:
    inizio = time.perf_counter()
    with ThreadPoolExecutor(max_workers=k) as pool:
        esiti = list(pool.map(lambda _: una(provider, testo), range(k)))
    muro = time.perf_counter() - inizio
    latenze = [t for t, _ in esiti]
    falliti = sum(1 for _, ok in esiti if not ok)
    return {
        "k": k, "muro": muro, "resa": k / muro if muro else 0.0,
        "mediana": statistics.median(latenze), "max": max(latenze),
        "falliti": falliti,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="gpu_farm")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--gradini", type=int, nargs="+", default=[1, 5, 10, 25, 50])
    ap.add_argument("--token", type=int, default=TOKEN_TIPICI)
    args = ap.parse_args()

    provider = create_llm_provider(args.provider, args.model)
    if type(provider).__name__ == "FallbackProvider":
        print("ATTENZIONE: provider di ripiego, non sta chiamando niente.")
        return 1

    testo = prompt(args.token)
    print(f"fornitore: {args.provider} / {args.model}   prompt ~{args.token} token")
    print("%4s %9s %10s %12s %10s %8s" %
          ("K", "muro (s)", "resa/s", "latenza med", "lat. max", "falliti"))
    migliore = 0.0
    for k in args.gradini:
        r = gradino(provider, testo, k)
        migliore = max(migliore, r["resa"])
        print("%4d %9.1f %10.2f %12.1f %10.1f %8d" %
              (r["k"], r["muro"], r["resa"], r["mediana"], r["max"], r["falliti"]))
    print()
    print("La resa e' chiamate al secondo: finche' sale, il fornitore ha margine;")
    print("quando si appiattisce, quello e' il numero di richieste che serve davvero")
    print("insieme --- oltre, accoda soltanto, e la latenza massima cresce.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
