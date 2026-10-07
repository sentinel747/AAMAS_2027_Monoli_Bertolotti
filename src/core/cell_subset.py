from __future__ import annotations

"""Pubblicazione del menu di cella solo dove c'e' un colono.

``compute_cell_masks``, ``compute_cell_proposals`` e ``compute_needs`` sono
scritte come NumPy su tutta la griglia: a ogni step producono array
``[H, W, N_ACTIONS]`` per l'intera mappa. Il loro risultato pero' viene letto in
un punto solo per agente -- ``decide_batch`` fa
``cell_masks[agents.y[rows], agents.x[rows]]`` e il percorso scalare fa
``proposals.priority[agent.y, agent.x]``. Nessuno consulta il menu di una cella
in cui non c'e' nessuno.

Sulla configurazione reale lo scarto non e' marginale: la mappa e' 180x360 =
64.800 celle e la colonia ne occupa fra 1 e 5, quindi il 99,99% del menu
pubblicato veniva buttato. Il sottostadio costava il 30,9% del passo
(``docs/benchmarks/README.md``).

Questo modulo tiene **una sola copia delle formule** -- quelle in
``cell_proposals`` e ``cell_action_mask`` restano invariate -- e ne restringe
soltanto l'estensione: ``CellSubset`` presenta le celle scelte come una griglia
``1 x K`` con gli stessi nomi di campo, cosi' le stesse funzioni girano su K
celle invece che su H*W.

**Perche' il risultato e' bit-identico.** Ogni operazione di quelle funzioni e'
per-cella: elementwise, oppure riduzioni lungo l'asse delle azioni o delle
strutture (``argsort(axis=2)``, ``sum(axis=-1)``, ``any(axis=-1)``). Nessuna
riduce lungo gli assi spaziali, quindi il valore calcolato per una cella non
dipende da quali altre celle stanno nell'array. L'unica eccezione e' la
frontiera di ``EXPLORE``, che legge i quattro vicini: e' estratta in
``cell_proposals.explore_frontier``, si calcola sulla griglia intera (poche
operazioni booleane su 64.800 byte) e si passa gia' pronta.
"""

import numpy as np

from src.core.cell_proposals import CellProposals


class CellIndexedArray:
    """Array compatto ``[K, ...]`` che si indicizza come se fosse ``[H, W, ...]``.

    Esiste per non toccare i consumatori: ``decide_batch`` e il percorso scalare
    leggono entrambi ``X[y, x]``, con y/x scalari o array, e continuano a farlo.
    Ogni altra forma di accesso e' rifiutata invece di essere reinterpretata: una
    lettura su tutta la griglia e' esattamente cio' che questa rappresentazione
    serve a evitare, e restituirle qualcosa di plausibile nasconderebbe l'errore.
    """

    __slots__ = ("_compact", "_keys", "_width")

    def __init__(self, compact: np.ndarray, keys: np.ndarray, width: int) -> None:
        self._compact = compact
        self._keys = keys
        self._width = int(width)

    def __getitem__(self, key):
        if not isinstance(key, tuple) or len(key) != 2:
            raise TypeError(
                "un menu di cella si indirizza come [y, x]; ricevuto "
                f"{key!r}. L'accesso a tutta la griglia e' proprio cio' che "
                "questa rappresentazione evita."
            )
        if self._keys.size == 0:
            raise KeyError("nessuna cella pubblicata: il sottoinsieme e' vuoto")
        y, x = key
        flat = (
            np.asarray(y, dtype=np.int64) * self._width
            + np.asarray(x, dtype=np.int64)
        )
        column = np.searchsorted(self._keys, flat)
        # `searchsorted` puo' restituire K per una chiave oltre l'ultima: si
        # riporta in campo e poi si verifica l'uguaglianza, cosi' una cella
        # assente diventa un errore esplicito e non la cella sbagliata.
        column = np.minimum(column, self._keys.size - 1)
        if not np.array_equal(self._keys[column], flat):
            raise KeyError(
                "nessun menu pubblicato per questa cella: il sottoinsieme e' "
                "costruito sulle celle degli agenti che decidono, quindi una "
                "lettura altrove significa che il consumatore ha cambiato quali "
                "celle legge"
            )
        return self._compact[column]

    def __repr__(self) -> str:  # pragma: no cover - diagnostica
        return (
            f"CellIndexedArray(celle={self._keys.size}, "
            f"forma_compatta={self._compact.shape})"
        )


class _GatheredCells:
    """Le celle scelte, con i nomi di campo di ``CellArrays``."""

    def __init__(self, source, rows: np.ndarray, cols: np.ndarray) -> None:
        grid = (int(source.H), int(source.W))
        for name, value in vars(source).items():
            if isinstance(value, np.ndarray) and value.ndim >= 2 and value.shape[:2] == grid:
                value = value[rows, cols][None, ...]
            # I campi non-griglia (dizionari dei cantieri, cache delle viste)
            # passano per riferimento: queste funzioni li leggono soltanto.
            setattr(self, name, value)
        # Scritti per ultimi di proposito: `H` e `W` sono campi anche della
        # sorgente, e assegnarli prima lascerebbe che il ciclo ripristini
        # l'estensione piena, facendo tornare ogni funzione ad allocare output
        # grandi quanto la mappa mentre ne legge K celle.
        self.H = 1
        self.W = int(rows.size)


class CellSubset:
    """Le celle su cui il passo ha davvero bisogno del menu."""

    __slots__ = ("cells", "keys", "rows", "cols", "width")

    def __init__(self, source, keys: np.ndarray) -> None:
        self.width = int(source.W)
        self.keys = np.asarray(keys, dtype=np.int64)
        self.rows = (self.keys // self.width).astype(np.intp)
        self.cols = (self.keys % self.width).astype(np.intp)
        self.cells = _GatheredCells(source, self.rows, self.cols)

    @classmethod
    def at_agent_cells(cls, source, agents, rows: np.ndarray) -> "CellSubset":
        """Le celle distinte occupate dagli agenti indicati.

        Si costruisce sugli agenti e non su ``occupancy > 0`` perche' e'
        esattamente l'insieme che verra' letto: cosi' una lettura fuori
        sottoinsieme non puo' esistere, e se esistesse sarebbe un ``KeyError``.
        ``np.unique`` ordina e deduplica, quindi il sottoinsieme non dipende
        dall'ordine delle righe.
        """
        width = int(source.W)
        ys = np.asarray(agents.y[rows], dtype=np.int64)
        xs = np.asarray(agents.x[rows], dtype=np.int64)
        return cls(source, np.unique(ys * width + xs))

    @classmethod
    def at_flagged_cells(cls, source, flags: np.ndarray) -> "CellSubset":
        """Le celle dove ``flags`` e' vero, in ordine di indice piatto.

        Serve a ``compute_needs``, il cui risultato e' azzerato fuori dalle celle
        insediate (``needs *= settled_mask(...)``): calcolarlo su tutta la mappa
        produce 64.800 righe per ottenerne **una** non nulla, misurato su
        configurazione reale.
        """
        ys, xs = np.nonzero(np.asarray(flags, dtype=np.bool_))
        return cls(source, (ys.astype(np.int64) * int(source.W) + xs).astype(np.int64))

    def scatter(self, compact: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
        """Rimette le righe calcolate al loro posto su una griglia di zeri.

        Il consumatore -- ``redistribute`` -- indicizza per coordinata su tutta
        la mappa, quindi la forma piena va ricostruita. Lo zero non e' un
        riempimento arbitrario: e' esattamente il valore che il percorso storico
        produce fuori dalle celle selezionate.
        """
        full = np.zeros(shape, dtype=compact.dtype)
        if self.keys.size:
            full[self.rows, self.cols] = compact[0]
        return full

    def gather(self, grid: np.ndarray) -> np.ndarray:
        """Riduce un array su griglia intera alla forma ``1 x K`` del sottoinsieme."""
        return grid[self.rows, self.cols][None, ...]

    def publish(self, proposals: CellProposals) -> CellProposals:
        """Riveste il risultato compatto perche' i consumatori non cambino."""
        return CellProposals(
            mask=self._wrap(proposals.mask),
            priority=self._wrap(proposals.priority),
            quota=self._wrap(proposals.quota),
        )

    def _wrap(self, compact: np.ndarray) -> CellIndexedArray:
        return CellIndexedArray(compact[0], self.keys, self.width)
