# -*- coding: utf-8 -*-
"""Chi amministra quale cella: i distretti della colonia.

**La regola, nelle parole con cui e' stata chiesta.** Un amministratore ogni
tre celle; la cella madre resta sotto il controllo diretto del governo; un
amministratore nasce non appena una cella resta orfana, e ammette altre due
sotto di se'.

**Perche' l'assegnazione e' permanente.** Un distretto che si ricomponesse a
ogni passo renderebbe l'amministratore una funzione dello stato invece che un
attore: le sue decisioni si riferirebbero a un territorio diverso da quello su
cui vengono applicate, e la serie storica delle sue scelte non sarebbe
leggibile. Qui una cella, una volta assegnata, resta al suo distretto per tutta
la run; i distretti crescono soltanto, e solo fino al proprio tetto.

**Perche' l'ordine e' deterministico.** L'assegnazione dipende dall'ordine in
cui le celle compaiono, quindi quell'ordine deve essere una funzione della
mappa e non della memoria: si scandisce per indice piatto crescente (riga, poi
colonna). Due run con lo stesso seme producono gli stessi distretti, che e' la
condizione perche' il confronto appaiato misuri il trattamento.
"""

from __future__ import annotations

import numpy as np

#: Quante celle puo' tenere un amministratore. Tre, come chiesto: la cella che
#: lo fa nascere piu' altre due.
CELLE_PER_DISTRETTO = 3

#: Il valore che marca "nessun distretto": la cella madre e le celle non ancora
#: assegnate, entrambe di competenza diretta del governo.
SENZA_DISTRETTO = -1


class Distretti:
    """L'assegnazione cella -> amministratore, che cresce e non si rimescola."""

    def __init__(self, celle_per_distretto: int = CELLE_PER_DISTRETTO) -> None:
        self.celle_per_distretto = max(1, int(celle_per_distretto))
        #: (y, x) -> identificativo di distretto
        self._assegnate: dict[tuple[int, int], int] = {}
        #: identificativo -> quante celle contiene
        self._capienza: dict[int, int] = {}
        #: la cella madre, esclusa per sempre
        self._madre: tuple[int, int] | None = None
        self._prossimo_id = 0

    # ------------------------------------------------------------------ stato

    @property
    def madre(self) -> tuple[int, int] | None:
        return self._madre

    def numero_distretti(self) -> int:
        return len(self._capienza)

    def celle_di(self, distretto: int) -> list[tuple[int, int]]:
        return sorted(c for c, d in self._assegnate.items() if d == distretto)

    def mappa(self) -> dict[int, list[tuple[int, int]]]:
        fuori: dict[int, list[tuple[int, int]]] = {}
        for cella, distretto in sorted(self._assegnate.items()):
            fuori.setdefault(distretto, []).append(cella)
        return fuori

    # ----------------------------------------------------------- aggiornamento

    def imposta_madre(self, madre: tuple[int, int] | None) -> None:
        """La cella madre si dichiara una volta e non cambia.

        Se arrivasse dopo che la stessa cella e' gia' stata assegnata a un
        distretto — succede quando il sito non e' noto al primo tick — la si
        toglie da quel distretto, che liberera' un posto.
        """
        if madre is None or self._madre is not None:
            return
        self._madre = (int(madre[0]), int(madre[1]))
        distretto = self._assegnate.pop(self._madre, None)
        if distretto is not None:
            self._capienza[distretto] -= 1
            if self._capienza[distretto] <= 0:
                del self._capienza[distretto]

    def aggiorna(self, occupate: np.ndarray) -> list[int]:
        """Assegna le celle occupate non ancora viste. Torna i distretti nuovi.

        `occupate` e' una maschera booleana sulla griglia intera.
        """
        nuovi: list[int] = []
        ys, xs = np.nonzero(np.asarray(occupate, dtype=np.bool_))
        for y, x in zip(ys.tolist(), xs.tolist()):
            cella = (int(y), int(x))
            if cella == self._madre or cella in self._assegnate:
                continue
            libero = next(
                (
                    d
                    for d in sorted(self._capienza)
                    if self._capienza[d] < self.celle_per_distretto
                ),
                None,
            )
            if libero is None:
                libero = self._prossimo_id
                self._prossimo_id += 1
                self._capienza[libero] = 0
                nuovi.append(libero)
            self._assegnate[cella] = libero
            self._capienza[libero] += 1
        return nuovi

    def griglia(self, shape: tuple[int, int]) -> np.ndarray:
        """La mappa degli identificativi, nella forma della griglia del mondo."""
        ids = np.full(shape, SENZA_DISTRETTO, dtype=np.int16)
        for (y, x), distretto in self._assegnate.items():
            ids[y, x] = distretto
        return ids

    # ------------------------------------------------------------- diagnostica

    def riassunto(self) -> dict:
        return {
            "cells_per_district": self.celle_per_distretto,
            "mother_cell": list(self._madre) if self._madre else None,
            "districts": {
                str(d): [list(c) for c in self.celle_di(d)]
                for d in sorted(self._capienza)
            },
        }
