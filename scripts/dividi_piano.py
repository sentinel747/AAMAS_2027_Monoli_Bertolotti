# -*- coding: utf-8 -*-
"""Divide un piano fra piu' lavoratori, per BRACCIO e mai per riga.

**Perche' per braccio.** Due processi che appendono allo stesso `results.jsonl`
lo corrompono, e la corruzione non da' errore: da' un file che si legge quasi
tutto. Dividendo per cartella, ogni lavoratore possiede cartelle intere e
nessuno scrive dove scrive l'altro --- e' la sola forma di parallelismo che non
introduce un modo nuovo di rompersi.

Le cartelle vanno a turno, in ordine di dimensione decrescente, a chi in quel
momento ha meno run: un accorgimento da niente che evita il caso in cui un
lavoratore finisce in dieci ore e l'altro in venti.

Uso:  python scripts/dividi_piano.py runs/piano.txt 2
      -> scrive runs/piano.txt.1 e runs/piano.txt.2
"""

from __future__ import annotations

import io
import sys
from collections import OrderedDict
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    piano = Path(sys.argv[1])
    quanti = int(sys.argv[2])
    righe = [r for r in piano.read_text(encoding="utf-8").splitlines() if r.strip()]

    per_cartella: "OrderedDict[str, list[str]]" = OrderedDict()
    for r in righe:
        pezzi = r.split("\t")
        if len(pezzi) < 4:
            continue
        per_cartella.setdefault(pezzi[1], []).append(r)

    ceste: list[list[str]] = [[] for _ in range(quanti)]
    for cartella, elenco in sorted(per_cartella.items(),
                                   key=lambda kv: -len(kv[1])):
        meno = min(range(quanti), key=lambda i: len(ceste[i]))
        ceste[meno].extend(elenco)

    for i, cesta in enumerate(ceste, start=1):
        fuori = piano.with_suffix(piano.suffix + f".{i}")
        # `newline=""` e non `write_text`: su Windows Python tradurrebbe
        # ogni "\n" in "\r\n", e il "\r" finirebbe dentro l'ultimo campo
        # letto dalla coda --- cioe' dentro gli argomenti della simulazione.
        with io.open(fuori, "w", encoding="utf-8", newline="") as f:
            f.write("\n".join(cesta) + "\n")
        cartelle = sorted({r.split("\t")[1] for r in cesta})
        print(f"{fuori}: {len(cesta)} run in {len(cartelle)} cartelle")
        for c in cartelle:
            print(f"    {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
