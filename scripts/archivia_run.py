# -*- coding: utf-8 -*-
"""Sposta in `runs/_scartate/` le cartelle che non servono piu', con la prova.

**Sposta, non cancella.** `runs/_scartate/` e' gia' la convenzione di questo
repository per il materiale fuori uso: nessuno script di analisi ci guarda, e
quindi una cartella li' dentro non puo' piu' finire per sbaglio in un conteggio.
Se poi si vuole liberare il disco, cancellare quella cartella sola e' un comando
solo, dato sapendo che cosa contiene.

Prima di ogni spostamento ricontrolla tre cose:
 - la cartella non e' fra le campagne DICHIARATE (`audit_registri.CAMPAGNE`),
   che sono l'insieme da cui escono i numeri della tesi;
 - la destinazione non esiste gia' (nessuna sovrascrittura);
 - il numero di file prima e dopo coincide.

Uso:  python archivia_run.py            (prova a vuoto, non tocca niente)
      python archivia_run.py --esegui
"""

import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, '.')
from scripts.audit_registri import CAMPAGNE  # noqa: E402

DICHIARATE = {c.rstrip('/') for elenco in CAMPAGNE.values() for c in elenco}

DESTINAZIONE = Path('runs/_scartate')

#: (cartella, sottocartella dell'archivio, perche')
PIANO = [
    ('runs/campagna_scarsa/modello_qwen3627b', '',
     'Il fornitore e\' caduto in corsa: 21 tornate su 40 al seme 4 e 33 su 39 al '
     'seme 5, e con il difetto corretto oggi ognuna abrogava la legge in vigore. '
     'Il quarto modello dei risultati NON viene da qui ma da runs/modelli/qwen36_27b, '
     'che ha zero chiamate fallite e zero politiche vuote.'),
    ('runs/campagna_scarsa/modello_gemma431b', '',
     'Otto tornate perse su cinque semi e nessun documento che la citi: la '
     'famiglia gemma non entra nei risultati.'),
    ('runs/campagna_2026-09-02_ISRU_ZERO_SCARTATA', '',
     'Scartata gia\' nel nome: ISRU a zero, la colonia non poteva lavorare il '
     'regolito. Nessuna citazione.'),
]

#: Le campagne di agosto: precedono la conservazione della massa (25 agosto) e
#: l'economia del materiale (31 agosto), quindi i loro numeri non sono
#: confrontabili con niente di quello che usiamo. Nessuna e' citata per nome.
AGOSTO = [
    'runs/campagna_espansione_a', 'runs/campagna_espansione_b',
    'runs/campagna_espansione_c', 'runs/campagna_espansione_d',
    'runs/campagna_espansione_e', 'runs/campagna_espansione_f',
    'runs/campagna_espansione_g', 'runs/campagna_espansione_h',
    'runs/classificazione_a', 'runs/classificazione_b', 'runs/classificazione_c',
    'runs/espansione_llm_espansi', 'runs/esperimento1',
    'runs/prova_bloccante', 'runs/prova_llm', 'runs/prova_output',
    'runs/scenari_fragmented', 'runs/scenari_high_hazard', 'runs/scenari_ice_rich',
    'runs/scenari_mineral_rich', 'runs/scenari_scarce_resources',
    'runs/valida600_a', 'runs/valida600_b', 'runs/valida600_c',
    'runs/verifica_600', 'runs/verifica_sintesi',
]
for c in AGOSTO:
    PIANO.append((c, '2026-08_pre_conservazione',
                  'Agosto: anteriore alla conservazione della massa e '
                  'all\'economia del materiale, numeri non confrontabili.'))


def conta_file(p: Path) -> int:
    return sum(len(f) for _, _, f in os.walk(p))


def main() -> int:
    esegui = '--esegui' in sys.argv
    totale = 0
    for cartella, sotto, perche in PIANO:
        p = Path(cartella)
        if not p.exists():
            print('  SALTO (non esiste):', cartella)
            continue
        dentro = any(cartella == d or cartella.startswith(d + '/')
                     or d.startswith(cartella + '/') for d in DICHIARATE)
        if dentro:
            print('  RIFIUTO:', cartella, '-> e\' una campagna dichiarata')
            continue
        dest_dir = DESTINAZIONE / sotto if sotto else DESTINAZIONE
        dest = dest_dir / p.name
        if dest.exists():
            print('  RIFIUTO:', cartella, '-> la destinazione esiste gia\'')
            continue
        n = conta_file(p)
        totale += n
        print('  %-50s -> %-44s %5d file' % (cartella, dest.as_posix(), n))
        if esegui:
            dest_dir.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
            dopo = conta_file(dest)
            assert dopo == n, f'file persi spostando {cartella}: {n} -> {dopo}'
    print('\n%d file in %d cartelle%s' %
          (totale, len(PIANO), '' if esegui else ' (prova a vuoto: niente e\' stato spostato)'))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
