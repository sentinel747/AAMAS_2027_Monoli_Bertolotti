# -*- coding: utf-8 -*-
"""Quali regole della politica scattano davvero, e non solo quali sono scritte.

**La domanda.** La tesi mostra che i modelli comprimono il territorio e che lo
fanno attraverso `explore`, l'unica leva che fonda insediamenti. Ma una politica
e' una catena ordinata, e per ogni cella vale la PRIMA regola che scatta: una
regola scritta in fondo puo' non agire mai. Contare le regole scritte, come fa
`analisi_stile_modelli.py`, dice che cosa il modello ha deciso; contare le volte
che ciascuna ha agito dice che cosa la colonia ha subito. Sono due cose diverse,
e il registro `governor_policy_hits.json` permette di misurare la seconda.

**Perche' conta.** L'unico esempio di politica che il prompt mostra al
governatore contiene due regole: se il cibo per occupante scende sotto 2,
`sustenance` per 3; se gli occupanti superano 150, `explore` per 2. La seconda
incoraggia a fondare. I modelli non la contraddicono quando la riscrivono: la
copiano e a volte la rinforzano. La compressione deve quindi venire da dove la
regola si trova nella catena, non da che cosa dice.

Uso:
    python scripts/analisi_regole_scattate.py
"""
from __future__ import annotations

import argparse
import io
import json
import re
import sys
from collections import Counter
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]

MODELLI = [
    ("gpt-oss 20B", ["runs/campagna_scarsa_v3/llm_completo_amm",
                     "runs/campagna_scarsa_v3/llm_completo_amm_rep2",
                     "runs/campagna_scarsa_v3/llm_completo_amm_rep3"]),
    ("gpt-oss 120B", ["runs/modelli/gptoss120b/llm_completo_amm"]),
    ("Qwen3.8 27B", ["runs/modelli/qwen38_27b/llm_completo_amm"]),
    ("Qwen3.6 27B", ["runs/modelli/qwen36_27b/llm_completo_amm"]),
]

PESO = re.compile(r"explore x([0-9]*\.?[0-9]+)")
COND = re.compile(r"^se ([a-z_]+) ([<>]) ")


def scatti(bracci: list[str]) -> Counter:
    tot: Counter = Counter()
    for rel in bracci:
        for cartella in sorted((RADICE / rel).glob("*seed*")):
            f = cartella / "governor_policy_hits.json"
            if not f.exists():
                continue
            for testo, n in json.loads(f.read_text(encoding="utf-8")).items():
                tot[testo] += int(n)
    return tot


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()

    print("Che cosa scatta davvero, per modello. Mondo di riferimento, cinque semi.\n")
    print(f"{'modello':14s} {'scatti':>10s} {'scoraggia':>10s} {'incoraggia':>11s} "
          f"{'neutra':>8s} {'espansiva':>10s} {'su occupants':>13s}")
    print("-" * 82)
    for nome, bracci in MODELLI:
        c = scatti(bracci)
        if not c:
            print(f"{nome:14s}  (nessun registro)")
            continue
        tot = sum(c.values())
        giu = su = neutre = 0
        occ_tot = occ_espansive = 0
        for testo, n in c.items():
            m = PESO.search(testo)
            w = float(m.group(1)) if m else 1.0
            if w < 1:
                giu += n
            elif w > 1:
                su += n
            else:
                neutre += n
            mc = COND.match(testo)
            if mc and mc.group(1) == "occupants" and mc.group(2) == ">":
                occ_tot += n
                if w > 1:
                    occ_espansive += n
        print(f"{nome:14s} {tot:10d} {giu / tot * 100:9.1f}% {su / tot * 100:10.1f}% "
              f"{neutre / tot * 100:7.1f}% {su / tot * 100:9.1f}% "
              f"{occ_tot / tot * 100:12.2f}%")

    print("\nLa colonna che conta e' l'ultima: quante volte ha agito una regola")
    print("condizionata sull'affollamento, cioe' la forma della regola che il prompt")
    print("mostra come esempio e che incoraggia a fondare insediamenti.\n")

    print("Le cinque regole piu' attive, per modello:")
    for nome, bracci in MODELLI:
        c = scatti(bracci)
        if not c:
            continue
        tot = sum(c.values())
        print(f"\n  {nome} ({tot} scatti)")
        for testo, n in c.most_common(5):
            print(f"    {n / tot * 100:5.1f}%  {testo[:96]}")
    return 0


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    raise SystemExit(main())
