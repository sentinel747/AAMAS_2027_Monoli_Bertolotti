from __future__ import annotations

"""Lo stato cold per-agente che il ciclo decisionale legge, come colonne.

Task B del piano ``docs/superpowers/plans/2026-08-07-decision-loop-rust-port.md``.

Il ciclo decisionale legge quasi tutto dagli array: misurato su 20 passi e 6.000
decisioni, il **96%** degli accessi ad attributo sull'agente finisce su colonne
SoA o sul loro plumbing. Il restante 4% -- 6,07 letture per decisione -- va a
``AgentSideState``, che e' un oggetto Python e quindi non e' leggibile dal lato
nativo senza una richiamata verso l'interprete: esattamente il costo per query
che ha fatto fallire il Task 3.

I campi sono pero' pochissimi, e tutti scalari:

===============================  =====================  ========
campo                            letture per decisione  tipo
===============================  =====================  ========
``founder_kit_reserved``         2,97                   bool
``movement_distance_m_per_step`` 2,00                   float
``actions_taken``                0,99                   int
``local_x_m`` / ``local_y_m``    0,05                   float
===============================  =====================  ========

``scout_phase`` e ``settle_phase`` **non** sono qui: sono gia' colonne, codificate
nell'intero ``mission`` (vedi ``AgentView.scout_phase``), quindi il lato nativo le
legge senza mediazione.

**Perche' materializzare invece di rispecchiare.** L'alternativa era aggiungere
colonne permanenti mantenute a ogni punto di scrittura, come si e' fatto a suo
tempo per ``mission``. Sarebbe piu' veloce ma introdurrebbe un invariante da
mantenere -- colonna e campo devono restare coerenti -- e la deriva fra i due
sarebbe silenziosa. Materializzare una volta per passo costa ~0,1% del passo
(misurato) e **non puo' derivare**, perche' la sorgente resta unica. Se un giorno
quel costo contasse, la promozione a colonna resta possibile; il contrario no.
"""

import numpy as np

# I nomi sono quelli letti dal ciclo, misurati e non dedotti: se il ciclo
# cominciasse a leggerne un altro, il lato nativo leggerebbe un valore che non
# esiste. Il test `tests/core/test_cold_state.py` confronta questo elenco con
# cio' che la fase di decisione tocca davvero.
COLD_FIELDS = (
    "actions_taken",
    "founder_kit_reserved",
    "local_x_m",
    "local_y_m",
    "movement_distance_m_per_step",
)


def materialize(side: dict, ids, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Le colonne cold per le righe indicate, nell'ordine di ``rows``.

    ``ids`` e' la mappa riga -> identificativo di ``AgentArrays``; ``rows`` sono
    le righe che decidono in questo passo. L'ordine del risultato segue ``rows``,
    non l'iterazione del dizionario: il lato nativo indicizza per posizione.
    """
    entries = [side[ids[int(row)]] for row in rows]
    return {
        "actions_taken": np.fromiter(
            (int(entry.actions_taken) for entry in entries),
            dtype=np.int64,
            count=len(entries),
        ),
        "founder_kit_reserved": np.fromiter(
            (bool(entry.founder_kit_reserved) for entry in entries),
            dtype=np.bool_,
            count=len(entries),
        ),
        "local_x_m": np.fromiter(
            (float(entry.local_x_m) for entry in entries),
            dtype=np.float64,
            count=len(entries),
        ),
        "local_y_m": np.fromiter(
            (float(entry.local_y_m) for entry in entries),
            dtype=np.float64,
            count=len(entries),
        ),
        "movement_distance_m_per_step": np.fromiter(
            (float(entry.movement_distance_m_per_step) for entry in entries),
            dtype=np.float64,
            count=len(entries),
        ),
    }
