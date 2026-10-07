# -*- coding: utf-8 -*-
"""La quota di riscritture degli amministratori, con una definizione sola.

**Perche' questo script esiste.** La quota di tornate in cui un amministratore
sostituisce la legge del governatore invece di accettarla compare in piu' punti
del lavoro, e si puo' contare in due modi che danno numeri diversi. Il registro
porta due contatori aggregati per tornata, `interventions` e `abstentions`, e
porta anche la decisione distretto per distretto in `last_round`. I due non
coincidono, perche' una riscrittura che non cambia nulla, cioe' che riproduce
l'effetto della legge in vigore, e' una riscrittura per il secondo conteggio e
non per il primo.

La definizione usata in tutto il lavoro e' quella di `analisi_decentramento.py`:
un distretto ha accettato se il registro dice `accepted_government_policy`, e in
ogni altro caso ha riscritto, a vuoto o no. E' la definizione giusta perche' la
domanda e' che cosa il modello ha FATTO, e scrivere una legge che risulta
equivalente e' comunque averla scritta. Questo script applica quella definizione
e basta, cosi' che ogni numero sulla quota di riscritture nel lavoro venga da
qui.

Uso:
    python scripts/quota_riscritture.py
    python scripts/quota_riscritture.py --quartili   # la deriva dentro la run
"""
from __future__ import annotations

import argparse
import io
import json
import statistics
import sys
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RADICE))

from scripts.analisi_decentramento import stato_del_distretto  # noqa: E402

BRACCI = [
    ("contesto pieno, es. 1", "runs/campagna_scarsa_v3/llm_completo_amm"),
    ("contesto pieno, es. 2", "runs/campagna_scarsa_v3/llm_completo_amm_rep2"),
    ("contesto pieno, es. 3", "runs/campagna_scarsa_v3/llm_completo_amm_rep3"),
    ("senza aiuti", "runs/livelli/senza_aiuti/llm_senza_aiuti_amm"),
    ("nomi veri", "runs/livelli/nomi_veri/llm_nomi_veri_amm"),
    ("cieco", "runs/livelli/cieco/llm_cieco_amm"),
    ("gpt-oss:120b", "runs/modelli/gptoss120b/llm_completo_amm"),
    ("Qwen3.8-27B", "runs/modelli/qwen38_27b/llm_completo_amm"),
    ("qwen3.6:27b", "runs/modelli/qwen36_27b/llm_completo_amm"),
]


def seme_di(nome: str) -> int:
    return int(nome.rsplit("seed", 1)[1])


def decisioni(cartella: Path) -> list[tuple[int, str]]:
    """[(passo, stato)] per ogni decisione di distretto, in ordine di passo."""
    f = cartella / "administrator_decisions.jsonl"
    if not f.exists():
        return []
    fuori: list[tuple[int, str]] = []
    for riga in io.open(f, encoding="utf-8", errors="replace"):
        riga = riga.strip()
        if not riga:
            continue
        try:
            t = json.loads(riga)
        except json.JSONDecodeError:
            continue
        passo = t.get("step")
        for d in t.get("last_round") or []:
            if "district" in d:
                fuori.append((int(passo or 0), stato_del_distretto(d)))
    return fuori


def quota(dec: list[tuple[int, str]]) -> float | None:
    if not dec:
        return None
    return sum(1 for _, s in dec if s != "accetta") / len(dec) * 100


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quartili", action="store_true")
    args = ap.parse_args()

    print("Quota di riscritture (accettare contro sostituire), definizione di "
          "analisi_decentramento.py.\nMondo di riferimento, semi 3-7, dotazione 0,6.\n")
    print(f"{'braccio':22s} {'quota':>7s} {'decisioni':>10s}   per seme"
          + ("        quartili" if args.quartili else ""))
    print("-" * (52 + (34 if args.quartili else 0)))
    for nome, rel in BRACCI:
        per_seme: dict[int, float] = {}
        tutte: list[tuple[int, str]] = []
        for cartella in sorted((RADICE / rel).glob("*seed*")):
            d = decisioni(cartella)
            if not d:
                continue
            q = quota(d)
            if q is not None:
                per_seme[seme_di(cartella.name)] = q
            tutte += d
        if not tutte:
            print(f"{nome:22s}  (nessun registro)")
            continue
        riga = (f"{nome:22s} {quota(tutte):6.1f}% {len(tutte):10d}   "
                + " ".join(f"{per_seme[s]:.0f}" for s in sorted(per_seme)))
        if args.quartili:
            n = len(tutte)
            q4 = [quota(tutte[k * n // 4:(k + 1) * n // 4]) for k in range(4)]
            riga += "    " + " ".join(f"{x:.1f}" for x in q4) + f"  ({q4[3] - q4[0]:+.1f})"
        print(riga)

    print("\nConfronto appaiato per seme contro il contesto pieno "
          "(media delle tre esecuzioni ripetute):")
    rif: dict[int, list[float]] = {}
    for _, rel in BRACCI[:3]:
        for cartella in sorted((RADICE / rel).glob("*seed*")):
            q = quota(decisioni(cartella))
            if q is not None:
                rif.setdefault(seme_di(cartella.name), []).append(q)
    medie = {s: statistics.fmean(v) for s, v in rif.items()}
    scarto = max(max(v) - min(v) for v in rif.values())
    print(f"  contesto pieno: " + ", ".join(f"s{s}={medie[s]:.0f}%" for s in sorted(medie))
          + f"   (media {statistics.fmean(medie.values()):.1f}%)")
    print(f"  scarto massimo fra due esecuzioni identiche sullo stesso seme: {scarto:.1f} punti\n")
    for nome, rel in BRACCI[3:6]:
        diff = []
        for cartella in sorted((RADICE / rel).glob("*seed*")):
            s = seme_di(cartella.name)
            q = quota(decisioni(cartella))
            if q is not None and s in medie:
                diff.append(q - medie[s])
        if not diff:
            continue
        segni = "".join("+" if d > 0 else "-" for d in diff)
        fuori = sum(1 for d in diff if abs(d) > scarto)
        print(f"  {nome:14s} media {statistics.fmean(diff):+6.1f}   "
              + " ".join(f"{d:+6.1f}" for d in diff)
              + f"   {segni}   fuori dallo scarto {fuori}/{len(diff)}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
