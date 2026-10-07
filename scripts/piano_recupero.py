# -*- coding: utf-8 -*-
"""Il piano delle sole run che mancano a una campagna, nell'ordine che conta.

**Perche' non basta rilanciare la coda.** Rilanciata, la coda salta le run gia'
fatte e rifa' le mancanti: funziona, ma le rifa' nell'ordine del piano. Quando
il fornitore e' instabile l'ordine decide che cosa si salva, perche' la campagna
puo' interrompersi di nuovo a meta'. Il braccio che misura il rumore va per
primo: senza quello nessun altro numero della campagna si puo' leggere, quindi
e' l'unico che non ci si puo' permettere di perdere due volte.

**Cio' che manca si deduce dal piano, non dalle cartelle.** La prima versione
elencava le cartelle su disco e per ciascuna guardava quali semi mancassero:
cosi' un braccio che non era mai partito --- cartella assente --- risultava
invisibile, e il recupero dichiarava «non manca nulla». Il braccio piu' esposto
a quell'errore era proprio quello del rumore, che sta in fondo al piano ed e'
quindi il primo a non avere nessuna run quando il fornitore cade a meta'
campagna. Il piano invece dice che cosa DEVE esistere, e il confronto con cio'
che esiste e' la definizione di cio' che manca.

**Le righe si copiano, non si riscrivono.** I piani portano gli argomenti esatti
con cui gli altri bracci sono stati eseguiti; riscriverli a mano rischierebbe
una differenza di configurazione dentro la stessa campagna, che e' il difetto
che questo progetto passa il tempo a evitare.

Uso:
    python scripts/piano_recupero.py                    # OSS-55
    python scripts/piano_recupero.py runs/base_qwen     # un'altra campagna

Esce con 0 se ha scritto un piano, con 1 se non manca nulla o se non e' riuscito
a costruirlo: chi lo chiama in automatico deve poter distinguere i due casi da
«ho scritto un piano vuoto», che e' il modo peggiore di fallire.
"""

from __future__ import annotations

import io
import json
import os
import sys

#: Prima il braccio che misura il rumore: senza, il resto non si legge.
PRIORITA = {'variante_D_rep2': 0}


def _normalizza(percorso: str) -> str:
    """Un solo modo di scrivere un percorso, cosi' le chiavi combaciano.

    Senza questo, un argomento con la barra finale produce `runs/x//braccio` e
    nessuna riga di piano viene riconosciuta: il recupero uscirebbe vuoto senza
    che nulla sembri rotto.
    """
    return percorso.replace(os.sep, '/').strip('/')


def righe_di_piano(piani: list[str]) -> list[tuple[str, str, str, str]]:
    """Le righe dei piani: (nome, cartella, seme, riga intera), senza doppioni.

    Una run puo' comparire due volte, se e' stata riaccodata a mano dopo un
    abbandono: si tiene la prima.
    """
    fuori, viste = [], set()
    for piano in piani:
        if not os.path.exists(piano):
            continue
        for riga in io.open(piano, encoding='utf-8', newline=''):
            pezzi = riga.rstrip('\r\n').split('\t')
            if len(pezzi) < 4:
                continue
            chiave = (_normalizza(pezzi[1]), pezzi[2])
            if chiave in viste:
                continue
            viste.add(chiave)
            fuori.append((pezzi[0], _normalizza(pezzi[1]), pezzi[2],
                          riga.rstrip('\r\n')))
    return fuori


def semi_presenti(cartella: str) -> set[str]:
    """I semi che hanno davvero prodotto un risultato in questa cartella."""
    percorso = os.path.join(cartella, 'results.jsonl')
    if not os.path.exists(percorso):
        # Cartella assente o senza registro: nessun seme fatto. NON significa
        # «braccio da ignorare» --- e' il caso del braccio mai partito.
        return set()
    presenti = set()
    for riga in io.open(percorso, encoding='utf-8'):
        riga = riga.strip()
        if riga:
            presenti.add(str(json.loads(riga)['seed']))
    return presenti


def main() -> int:
    base = _normalizza(sys.argv[1] if len(sys.argv) > 1 else 'runs/base_pulita')
    etichetta = os.path.basename(base)
    piani = [f'runs/piano_{etichetta}.txt']
    piani += [f'runs/piano_{etichetta}.txt.{n}' for n in range(1, 9)]
    fuori_file = f'runs/piano_recupero_{etichetta}.txt'

    tutte = righe_di_piano(piani)
    if not tutte:
        print(f'nessun piano trovato per {etichetta}: cercati {piani[0]} e i '
              'suoi pezzi. Senza il piano non so che cosa dovrebbe esistere.',
              file=sys.stderr)
        return 1

    fatti: dict[str, set[str]] = {}
    mancanti = []
    for nome, cartella, seme, riga in tutte:
        if cartella not in fatti:
            fatti[cartella] = semi_presenti(cartella)
        if seme not in fatti[cartella]:
            mancanti.append((PRIORITA.get(os.path.basename(cartella), 1),
                             seme, nome, riga))

    if not mancanti:
        print(f'{base}: non manca nulla ({len(tutte)} run previste, tutte fatte).')
        return 1

    mancanti.sort()
    io.open(fuori_file, 'w', encoding='utf-8', newline='').write(
        '\n'.join(r for _, _, _, r in mancanti) + '\n')

    print(f'{fuori_file}: {len(mancanti)} run da rifare su {len(tutte)} previste')
    for _, seme, nome, _riga in mancanti:
        print(f'   {nome}  seme {seme}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
