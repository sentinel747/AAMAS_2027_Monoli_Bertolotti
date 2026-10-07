# -*- coding: utf-8 -*-
"""Quali effetti reggono su DUE modelli, e quali erano un caso.

**Perche' serve.** Una campagna sola dice se un effetto supera il rumore di
QUEL modello. Non dice se l'effetto e' una proprieta' dei governatori LLM o una
proprieta' di quel modello. La differenza non e' accademica: su queste due
campagne ci sono effetti dichiarati REGGE in una e di segno OPPOSTO nell'altra,
entrambi con cinque semi concordi su cinque. Uno dei due sarebbe finito in tesi
come risultato.

**La regola applicata qui.** Un effetto e' RIPRODOTTO solo se, in entrambe le
campagne, supera il rispettivo rumore, ha cinque semi concordi su cinque e ---
la condizione che fa il lavoro --- ha lo **stesso segno**. Tutto il resto e'
specifico del modello, e in tesi ci va dichiarato come tale.

Il rumore e' quello di ciascuna campagna: i due modelli non hanno la stessa
varianza e confrontarli con un metro solo sarebbe la stessa scorciatoia che
questo lavoro evita altrove.

Uso:  python scripts/replica_due_modelli.py [campagna A] [campagna B]
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from esiti_campagna import (COPPIE_MINIME, EFFETTI, GRANDEZZE, RIPETIZIONE,
                            SEMI_MINIMI, differenze)


def quadro(base: str) -> dict:
    """Per ogni (grandezza, effetto): (media, concordi, n, supera_il_rumore)."""
    fuori = {}
    for chiave in GRANDEZZE:
        rumore = [abs(d) for d in differenze(base, *RIPETIZIONE, chiave)]
        if len(rumore) < COPPIE_MINIME:
            continue
        tipico = sum(rumore) / len(rumore)
        if tipico <= 0:
            continue
        fuori[(chiave, '__rumore__')] = tipico
        for etichetta, sinistra, destra in EFFETTI:
            diffs = differenze(base, sinistra, destra, chiave)
            n = len(diffs)
            if n < SEMI_MINIMI:
                continue
            media = sum(diffs) / n
            concordi = max(sum(1 for d in diffs if d > 0),
                           sum(1 for d in diffs if d < 0))
            fuori[(chiave, etichetta)] = (media, concordi, n,
                                          abs(media) > tipico)
    return fuori


def main() -> int:
    a = sys.argv[1] if len(sys.argv) > 1 else 'runs/base_qwen'
    b = sys.argv[2] if len(sys.argv) > 2 else 'runs/base_pulita'
    qa, qb = quadro(a), quadro(b)

    print('RIPRODUCIBILITA\' FRA DUE MODELLI')
    print(f'  A = {a}')
    print(f'  B = {b}')
    print()
    print('RIPRODOTTO = in ENTRAMBE supera il rumore, 5/5 semi concordi, stesso segno.')
    print()

    riprodotti, contraddetti, solo_uno = [], [], []

    for chiave in GRANDEZZE:
        if (chiave, '__rumore__') not in qa or (chiave, '__rumore__') not in qb:
            continue
        ra, rb = qa[(chiave, '__rumore__')], qb[(chiave, '__rumore__')]
        print(f'=== {chiave.upper()} ===   rumore: A {ra:.3g}   B {rb:.3g}')
        print(f'  {"effetto":<34}{"A":>9}{"":>6}{"B":>9}{"":>6}  esito')
        for etichetta, _s, _d in EFFETTI:
            va, vb = qa.get((chiave, etichetta)), qb.get((chiave, etichetta))
            if va is None or vb is None:
                continue
            ma, ca, na, sa = va
            mb, cb, nb, sb = vb
            forte_a = (ca == na and sa)
            forte_b = (cb == nb and sb)
            concorde = (ma > 0) == (mb > 0)
            if forte_a and forte_b and concorde:
                esito = 'RIPRODOTTO'
                riprodotti.append((chiave, etichetta, ma, mb))
            elif forte_a and forte_b:
                esito = '*** CONTRADDETTO: segni opposti ***'
                contraddetti.append((chiave, etichetta, ma, mb))
            elif forte_a or forte_b:
                quale = 'A' if forte_a else 'B'
                esito = f'solo in {quale}'
                solo_uno.append((chiave, etichetta, ma, mb, quale))
            else:
                esito = '-'
            print(f'  {etichetta:<34}{ma:>+9.3g}{f" {ca}/{na}":>6}'
                  f'{mb:>+9.3g}{f" {cb}/{nb}":>6}  {esito}')
        print()

    print('=' * 78)
    print(f'RIPRODOTTI su entrambi i modelli: {len(riprodotti)}')
    for c, e, ma, mb in riprodotti:
        print(f'   {c:<15}{e:<34}A {ma:+.4g}   B {mb:+.4g}')
    print()
    print(f'CONTRADDETTI (forti in entrambi, segno opposto): {len(contraddetti)}')
    for c, e, ma, mb in contraddetti:
        print(f'   {c:<15}{e:<34}A {ma:+.4g}   B {mb:+.4g}')
    print()
    print(f'FORTI IN UNA CAMPAGNA SOLA (non riproducibili): {len(solo_uno)}')
    for c, e, ma, mb, quale in solo_uno:
        altro = mb if quale == 'A' else ma
        mio = ma if quale == 'A' else mb
        print(f'   {c:<15}{e:<34}{quale} {mio:+.4g} forte, '
              f'l\'altra {altro:+.4g} no')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
