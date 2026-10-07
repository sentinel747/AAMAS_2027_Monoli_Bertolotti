# -*- coding: utf-8 -*-
"""Le risposte non interpretabili degli amministratori, per modello.

Verifica l'affermazione della sezione sulle condizioni di ammissione: quante
risposte degli amministratori il parser non capisce, quante richieste ci sono
per esecuzione, quante esecuzioni ne hanno almeno una. Legge `results.jsonl`,
lo stesso registro da cui `qualita_run.py` prende i suoi conteggi.

Uso:
    python scripts/verifica_malformate.py
"""
from __future__ import annotations

import io
import json
import statistics
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

#: Tutte le campagne con amministratori linguistici che entrano nei risultati.
GRUPPI = {
    # Le due campagne a protocollo intero: nove bracci su undici hanno lo strato
    # amministrativo, per cinque semi, cioe' quarantacinque esecuzioni ciascuna.
    "gpt-oss:20b (OSS-55)": ["runs/base_pulita"],
    "Qwen3.8-27B (QWEN-55)": ["runs/base_qwen"],
    # Le campagne su mondi, modelli e livelli di contesto.
    "gpt-oss:20b (mondi)": [
        "runs/campagna_scarsa_v3/llm_completo_amm",
        "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
        "runs/campagna_scarsa_v3/llm_completo_amm_rep3",
        "runs/livelli/senza_aiuti/llm_senza_aiuti_amm",
        "runs/livelli/nomi_veri/llm_nomi_veri_amm",
        "runs/livelli/cieco/llm_cieco_amm",
        "runs/mondi/ice_rich/llm_completo_amm",
        "runs/mondi/fragmented/llm_completo_amm",
        "runs/mondi/high_hazard/llm_completo_amm",
        "runs/mondi/scarce_resources/llm_completo_amm_v2",
    ],
    "gpt-oss:120b": ["runs/modelli/gptoss120b/llm_completo_amm"],
    "Qwen3.8-27B (modelli)": ["runs/modelli/qwen38_27b/llm_completo_amm"],
    "qwen3.6:27b": ["runs/modelli/qwen36_27b/llm_completo_amm"],
}


def righe(cartella: Path) -> list[dict]:
    """Tutti i `results.jsonl` sotto la cartella, anche nei bracci figli."""
    fuori = []
    if (cartella / "results.jsonl").exists():
        percorsi = [cartella / "results.jsonl"]
    else:
        percorsi = sorted(cartella.glob("*/results.jsonl"))
    for f in percorsi:
        for r in io.open(f, encoding="utf-8", errors="replace"):
            r = r.strip()
            if not r:
                continue
            try:
                fuori.append(json.loads(r))
            except json.JSONDecodeError:
                pass
    return fuori


def main() -> int:
    print("Risposte degli amministratori non interpretabili, per modello.\n")
    print(f"{'campagna':22s} {'esec.':>6s} {'richieste/esec.':>16s} {'malformate':>11s} "
          f"{'media':>7s} {'coda':>5s} {'esec. con >=1':>14s}")
    print("-" * 80)
    for nome, bracci in GRUPPI.items():
        per_esec: list[tuple[int, int]] = []   # (malformate, richieste)
        for rel in bracci:
            for r in righe(RADICE / rel):
                amm = r.get("amministrazione") or {}
                if not amm:
                    continue
                richieste = int(amm.get("interventions") or 0) + int(amm.get("abstentions") or 0)
                if richieste == 0:
                    continue
                per_esec.append((int(amm.get("malformed") or 0), richieste))
        if not per_esec:
            print(f"{nome:22s}  (nessun registro con strato amministrativo)")
            continue
        mal = [m for m, _ in per_esec]
        ric = [q for _, q in per_esec]
        con = sum(1 for m in mal if m > 0)
        print(f"{nome:22s} {len(per_esec):6d} {statistics.fmean(ric):16.0f} {sum(mal):11d} "
              f"{statistics.fmean(mal):7.2f} {max(mal):5d} {con:8d}/{len(per_esec):<5d}")
    print("\nRichieste per esecuzione: media sulle esecuzioni del gruppo.")
    print("Coda: il massimo su una singola esecuzione.")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
