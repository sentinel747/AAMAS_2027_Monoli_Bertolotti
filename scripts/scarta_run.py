# -*- coding: utf-8 -*-
"""Mette da parte una run bocciata, perche' ripartendo venga RIFATTA e non saltata.

**Il buco che chiude.** La coda si ferma quando `qualita_run.py` boccia una run,
ma decide che cosa e' gia' fatto cercando il seme dentro `results.jsonl` --- e
una run non comparabile il suo risultato lo ha scritto lo stesso. Ripartendo, la
coda la salterebbe: cioe' la terrebbe. Il log dice «va rifatta, non tenuta», ma
finora non c'era niente che lo imponesse.

**Non cancella niente.** La cartella del seme viene spostata sotto `_scartate/`
con la data, e la riga di `results.jsonl` viene tolta dopo aver salvato il file
intero in `results.jsonl.prima-di-<data>`. Se si scopre che la run era buona, si
rimette indietro tutto con due `move`.

Uso:
    python scripts/scarta_run.py runs/base_pulita/variante_D 5
    python scripts/scarta_run.py runs/base_pulita/variante_D 5 --motivo "3 tornate perse"
    python scripts/scarta_run.py runs/base_pulita/variante_D 5 --prova   # dice e non fa
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import time
from pathlib import Path


def righe_registro(p: Path) -> list[str]:
    if not p.exists():
        return []
    return [r for r in p.read_text(encoding="utf-8", errors="replace").splitlines()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cartella", help="la cartella del braccio, es. runs/base_pulita/variante_D")
    ap.add_argument("seme", type=int)
    ap.add_argument("--motivo", default="")
    ap.add_argument("--prova", action="store_true", help="mostra e non tocca niente")
    args = ap.parse_args()

    base = Path(args.cartella)
    if not base.is_dir():
        print("non trovo", base)
        return 1

    quando = time.strftime("%Y%m%d-%H%M%S")
    destinazione = base / "_scartate" / f"seme{args.seme}-{quando}"

    # 1. le cartelle del seme: il runner le chiama `<qualcosa>_seed<N>`
    cartelle = sorted(p for p in base.glob(f"*seed{args.seme}") if p.is_dir())
    # 2. la riga del registro
    registro = base / "results.jsonl"
    righe = righe_registro(registro)
    tenute, tolte = [], []
    for r in righe:
        if not r.strip():
            continue
        try:
            fuori = json.loads(r)
        except json.JSONDecodeError:
            tenute.append(r)
            continue
        (tolte if int(fuori.get("seed", -1)) == args.seme else tenute).append(r)

    print(f"braccio   : {base}")
    print(f"seme      : {args.seme}")
    print(f"cartelle  : {[p.name for p in cartelle] or 'nessuna'}")
    print(f"righe di results.jsonl da togliere: {len(tolte)} (ne restano {len(tenute)})")
    print(f"destinazione: {destinazione}")
    if args.motivo:
        print(f"motivo    : {args.motivo}")

    if not cartelle and not tolte:
        print("\nniente da scartare: la run non risulta eseguita.")
        return 0
    if args.prova:
        print("\n--prova: non ho toccato niente.")
        return 0

    destinazione.mkdir(parents=True, exist_ok=True)
    for p in cartelle:
        shutil.move(str(p), str(destinazione / p.name))
        print("spostata", p.name)

    if tolte:
        copia = registro.with_suffix(f".jsonl.prima-di-{quando}")
        shutil.copy2(registro, copia)
        io.open(registro, "w", encoding="utf-8").write(
            "\n".join(tenute) + ("\n" if tenute else ""))
        io.open(destinazione / "results-scartati.jsonl", "w", encoding="utf-8").write(
            "\n".join(tolte) + "\n")
        print(f"registro ripulito (copia intera in {copia.name})")

    nota = destinazione / "PERCHE.txt"
    io.open(nota, "w", encoding="utf-8").write(
        f"Scartata il {quando}\nbraccio: {base}\nseme: {args.seme}\n"
        f"motivo: {args.motivo or 'non dichiarato'}\n\n"
        "Spostata qui perche' la coda, ripartendo, cerca il seme dentro\n"
        "results.jsonl per sapere che cosa e' gia' fatto: lasciandola al suo\n"
        "posto sarebbe stata saltata, cioe' tenuta.\n")
    print("\nfatto. Ripartendo, la coda rifara' questo seme.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
