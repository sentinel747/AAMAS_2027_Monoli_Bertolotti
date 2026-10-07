# -*- coding: utf-8 -*-
"""Due esecuzioni della stessa baseline devono dare gli stessi numeri.

**Perche' esiste.** La baseline del mondo di riferimento viene rieseguita per
ottenere le istantanee che la campagna originale non scriveva. La riesecuzione
e' legittima solo se produce la STESSA colonia: la baseline non interroga alcun
modello e la simulazione e' deterministica fra processi distinti, quindi i vivi
e le celle devono coincidere seme per seme.

Se non coincidono, qualcosa e' cambiato nel modello fra allora e oggi, e in
archivio finirebbero due baseline diverse con lo stesso nome --- il difetto
peggiore possibile, perche' ogni confronto appaiato del lavoro ci si appoggia.
Qui la differenza si vede subito e in chiaro.

Uso:
    python scripts/confronta_baseline.py --nuova runs/.../ctrl_none \\
        --archivio runs/campagna_scarsa/ctrl_none
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

RADICE = Path(__file__).resolve().parents[1]


def _per_seme(cartella: Path) -> dict[int, tuple[int, int]]:
    """seme -> (vivi, celle), dal registro della cartella."""
    fuori: dict[int, tuple[int, int]] = {}
    percorso = cartella / "results.jsonl"
    if not percorso.exists():
        return fuori
    for riga in io.open(percorso, encoding="utf-8", errors="replace"):
        riga = riga.strip()
        if not riga:
            continue
        try:
            r = json.loads(riga)
        except json.JSONDecodeError:
            continue
        fuori[int(r["seed"])] = (
            int(r.get("population", 0) or 0),
            int((r.get("espansione") or {}).get("celle_totali", 0) or 0),
        )
    return fuori


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nuova", type=Path, required=True)
    ap.add_argument("--archivio", type=Path, required=True)
    args = ap.parse_args()

    nuova = _per_seme(RADICE / args.nuova if not args.nuova.is_absolute() else args.nuova)
    vecchia = _per_seme(RADICE / args.archivio if not args.archivio.is_absolute() else args.archivio)

    comuni = sorted(set(nuova) & set(vecchia))
    if not comuni:
        print("nessun seme in comune fra le due cartelle: non posso verificare niente.")
        return 1

    diverse = []
    for seme in comuni:
        if nuova[seme] != vecchia[seme]:
            diverse.append((seme, vecchia[seme], nuova[seme]))
        print(
            "seme %d  archivio %d vivi / %d celle   riesecuzione %d vivi / %d celle   %s"
            % (seme, *vecchia[seme], *nuova[seme], "IDENTICA" if nuova[seme] == vecchia[seme] else "DIVERSA")
        )

    if diverse:
        print()
        print("Le due baseline NON coincidono su %d semi su %d." % (len(diverse), len(comuni)))
        print("Non e' un difetto della riesecuzione: vuol dire che il modello e'")
        print("cambiato dalla campagna originale a oggi. I due insiemi di numeri non")
        print("sono confrontabili e non vanno mescolati in nessuna tabella.")
        return 1

    print()
    print("Tutte identiche su %d semi: la riesecuzione aggiunge le istantanee e" % len(comuni))
    print("non cambia una cifra. Il confronto appaiato del mondo di riferimento si")
    print("puo' calcolare con queste mappe.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
