"""Il menu ristretto deve valere quanto quello su tutta la griglia.

Il guadagno del sottoinsieme si regge su una sola affermazione: le formule di
``cell_action_mask``/``cell_proposals``/``needs`` sono per-cella, quindi il
valore calcolato per una cella non dipende da quali altre celle stanno
nell'array. Questi test la verificano invece di darla per buona, e coprono
l'unica eccezione nota (la frontiera di EXPLORE, che legge i vicini).
"""

import numpy as np
import pytest

from src.agents import pillars
from src.agents.action_space import ActionType
from src.core import constants as C
from src.core.arrays import CellArrays
from src.core.cell_action_mask import compute_cell_masks
from src.core.cell_proposals import (
    TASK_ACTIONS,
    compute_cell_proposals,
    explore_frontier,
)
from src.core.cell_subset import CellIndexedArray, CellSubset
from src.core.needs import RedistributionConfig, compute_needs
from src.world.structures import StructureType
from src.world.world_generator import WorldGenerator

OCCUPIED = ((7, 8), (3, 21), (17, 2))


def _cells_with_colony():
    """Griglia con tre celle abitate, com'e' una colonia che si e' espansa."""
    world = WorldGenerator(seed=71).generate(32, 20)
    cells = CellArrays.from_world(world)
    for index, (y, x) in enumerate(OCCUPIED):
        cells.occupancy[y, x] = 4 + index
        cells.struct_count[y, x, C.S[StructureType.SHELTER]] = 1
        cells.struct_integrity[y, x, C.S[StructureType.SHELTER]] = 1.0
        cells.water_ice[y, x] = 20.0
        cells.cell_res[y, x, C.R["minerals"]] = 20.0
        cells.cell_res[y, x, C.R["construction_material"]] = 20.0
        cells.vegetation[y, x] = 1.0
        cells.proto_soil[y, x] = 1.0
    # Esplorato a macchie: senza contrasto la frontiera di EXPLORE sarebbe
    # costante e il test non distinguerebbe una gather corretta da una sbagliata.
    cells.explored[:] = True
    cells.explored[0:6, :] = False
    cells.explored[:, 25:] = False
    return cells


class _AgentsAt:
    """Il minimo che ``CellSubset.at_agent_cells`` legge davvero."""

    def __init__(self, coords):
        self.y = np.array([y for y, _ in coords], dtype=np.int64)
        self.x = np.array([x for _, x in coords], dtype=np.int64)


def _full_grid(cells, top_k=5):
    needs = compute_needs(cells, RedistributionConfig())
    return compute_cell_proposals(
        cells, compute_cell_masks(cells), needs, top_k=top_k
    )


def _subset_of(cells, agents, rows, top_k=5):
    subset = CellSubset.at_agent_cells(cells, agents, rows)
    needs = compute_needs(subset.cells, RedistributionConfig())
    compact = compute_cell_proposals(
        subset.cells,
        compute_cell_masks(subset.cells),
        needs,
        top_k=top_k,
        frontier=subset.gather(explore_frontier(cells)),
    )
    return subset.publish(compact)


def test_subset_menu_is_bit_identical_to_the_whole_grid_menu():
    cells = _cells_with_colony()
    # Piu' agenti per cella e in ordine sparso: il sottoinsieme non deve
    # dipendere ne' dai duplicati ne' dall'ordine delle righe.
    coords = [OCCUPIED[2], OCCUPIED[0], OCCUPIED[2], OCCUPIED[1], OCCUPIED[0]]
    agents = _AgentsAt(coords)
    rows = np.arange(len(coords))

    full = _full_grid(cells)
    restricted = _subset_of(cells, agents, rows)

    for y, x in OCCUPIED:
        assert np.array_equal(restricted.mask[y, x], full.mask[y, x])
        # `array_equal` e non `allclose`: la tesi dichiara parita' bit-exact,
        # e una tolleranza qui la trasformerebbe in parita' approssimata.
        assert np.array_equal(restricted.priority[y, x], full.priority[y, x])
        assert np.array_equal(restricted.quota[y, x], full.quota[y, x])


def test_subset_menu_matches_the_whole_grid_under_vectorized_access():
    """``decide_batch`` indicizza con array, non con scalari."""
    cells = _cells_with_colony()
    coords = list(OCCUPIED) * 3
    agents = _AgentsAt(coords)
    rows = np.arange(len(coords))

    full = _full_grid(cells)
    restricted = _subset_of(cells, agents, rows)

    ys, xs = agents.y, agents.x
    assert np.array_equal(restricted.mask[ys, xs], full.mask[ys, xs])
    assert np.array_equal(restricted.priority[ys, xs], full.priority[ys, xs])
    assert np.array_equal(restricted.quota[ys, xs], full.quota[ys, xs])


def test_explore_priority_survives_the_restriction():
    """La frontiera e' l'unica formula che legge i vicini: va sorvegliata a parte.

    Se ``explore_frontier`` fosse calcolata sulla striscia 1xK invece che sulla
    mappa, i vicini sarebbero quelli sbagliati e la priorita' di EXPLORE
    cambierebbe silenziosamente: il resto del menu resterebbe corretto e nessun
    altro test se ne accorgerebbe.
    """
    cells = _cells_with_colony()
    agents = _AgentsAt(OCCUPIED)
    rows = np.arange(len(OCCUPIED))
    explore = pillars.ACTION_INDEX[ActionType.EXPLORE]

    # Menu non troncato: con top_k=5 le necessita' di acqua/cibo/materiali
    # scacciano EXPLORE dal menu e la sua priorita' viene azzerata, quindi il
    # confronto passerebbe leggendo zero da entrambe le parti.
    full = _full_grid(cells, top_k=len(TASK_ACTIONS))
    restricted = _subset_of(cells, agents, rows, top_k=len(TASK_ACTIONS))

    values = [float(full.priority[y, x, explore]) for y, x in OCCUPIED]
    assert len(set(values)) > 1, "fixture inerte: la frontiera non discrimina"
    for (y, x), expected in zip(OCCUPIED, values):
        # In due passi e non `[y, x, explore]`: il menu pubblicato accetta solo
        # l'indirizzo di cella, ed e' proprio quel rifiuto a garantire che
        # nessuno rilegga per sbaglio tutta la griglia.
        assert float(restricted.priority[y, x][explore]) == expected


def test_frontier_argument_defaults_to_the_extracted_formula():
    """Estrarre la frontiera non deve aver cambiato il percorso storico."""
    cells = _cells_with_colony()
    masks = compute_cell_masks(cells)
    needs = compute_needs(cells, RedistributionConfig())

    implicit = compute_cell_proposals(cells, masks, needs, top_k=5)
    explicit = compute_cell_proposals(
        cells, masks, needs, top_k=5, frontier=explore_frontier(cells)
    )
    assert np.array_equal(implicit.priority, explicit.priority)


def test_reading_a_cell_outside_the_subset_is_an_error():
    """Meglio un'eccezione che la riga di un'altra cella."""
    cells = _cells_with_colony()
    agents = _AgentsAt(OCCUPIED[:1])
    restricted = _subset_of(cells, agents, np.arange(1))

    y, x = OCCUPIED[1]
    with pytest.raises(KeyError):
        restricted.mask[y, x]


def test_whole_grid_access_is_refused_instead_of_reinterpreted():
    cells = _cells_with_colony()
    agents = _AgentsAt(OCCUPIED)
    restricted = _subset_of(cells, agents, np.arange(len(OCCUPIED)))

    with pytest.raises(TypeError):
        restricted.mask[:, :, 0]
    with pytest.raises(TypeError):
        restricted.priority[0]


def test_empty_subset_publishes_nothing_without_crashing():
    """Uno step senza agenti vivi non e' un errore, e' una colonia vuota."""
    cells = _cells_with_colony()
    agents = _AgentsAt([])
    restricted = _subset_of(cells, agents, np.arange(0))

    with pytest.raises(KeyError):
        restricted.mask[OCCUPIED[0][0], OCCUPIED[0][1]]


def test_subset_reports_one_row_per_distinct_cell():
    cells = _cells_with_colony()
    coords = list(OCCUPIED) + list(OCCUPIED)
    subset = CellSubset.at_agent_cells(cells, _AgentsAt(coords), np.arange(len(coords)))

    assert subset.cells.H == 1
    assert subset.cells.W == len(OCCUPIED)
    assert subset.cells.occupancy.shape == (1, len(OCCUPIED))
    # `keys` ordinato: il sottoinsieme e' lo stesso comunque arrivino gli agenti.
    assert list(subset.keys) == sorted(y * cells.W + x for y, x in OCCUPIED)


def test_indexed_array_round_trips_every_published_cell():
    keys = np.array([5, 9, 40], dtype=np.int64)
    compact = np.arange(9, dtype=np.float64).reshape(3, 3)
    wrapped = CellIndexedArray(compact, keys, width=10)

    for column, key in enumerate(keys):
        y, x = divmod(int(key), 10)
        assert np.array_equal(wrapped[y, x], compact[column])


def test_flagged_subset_and_scatter_reproduce_the_full_grid_needs():
    """La restrizione di ``compute_needs`` deve dare la matrice identica.

    Non "equivalente a meno di zeri": identica. Il consumatore
    (``redistribute``) indicizza per coordinata su tutta la mappa, quindi ogni
    cella non selezionata deve valere esattamente cio' che il percorso storico
    ci scriveva -- e ci scriveva zero, perche' ``compute_needs`` chiude con
    ``needs *= settled_mask(...)``.
    """
    import numpy as np

    from src.core import constants as C
    from src.core.arrays import CellArrays
    from src.core.cell_subset import CellSubset
    from src.core.needs import RedistributionConfig, compute_needs, settled_mask
    from src.world.structures import StructureType

    cells = CellArrays(6, 5)
    # Tre celle insediate con contenuti diversi, sparse: se lo scatter sbagliasse
    # l'ordine delle righe il test lo vedrebbe, cosa che una sola cella non fa.
    for (y, x), structure in (
        ((0, 0), StructureType.HABITAT),
        ((3, 4), StructureType.GREENHOUSE),
        ((5, 1), StructureType.SOLAR_ARRAY),
    ):
        cells.struct_count[y, x, C.S[structure]] = 2
        cells.struct_integrity[y, x, C.S[structure]] = 0.4
        cells.occupancy[y, x] = 7
        cells.cell_res[y, x, C.R["water"]] = 1.5
    cfg = RedistributionConfig()

    reference = compute_needs(cells, cfg)
    subset = CellSubset.at_flagged_cells(cells, settled_mask(cells))
    restricted = subset.scatter(
        compute_needs(subset.cells, cfg), (cells.H, cells.W, C.NN)
    )

    assert restricted.shape == reference.shape
    assert restricted.dtype == reference.dtype
    # Confronto bit-exact e non `allclose`: e' il vincolo del progetto, ed e'
    # lecito qui perche' ogni operazione di `compute_needs` e' elementwise per
    # cella e la sua unica riduzione sta sull'asse delle strutture.
    assert np.array_equal(restricted, reference)
    assert np.count_nonzero(np.any(reference != 0.0, axis=2)) == 3


def test_an_empty_flag_set_scatters_to_all_zeros():
    """Una mappa senza celle insediate: nessuna riga da rimettere a posto."""
    import numpy as np

    from src.core import constants as C
    from src.core.arrays import CellArrays
    from src.core.cell_subset import CellSubset

    cells = CellArrays(4, 3)
    subset = CellSubset.at_flagged_cells(cells, np.zeros((4, 3), dtype=np.bool_))
    scattered = subset.scatter(
        np.zeros((1, 0, C.NN), dtype=np.float64), (4, 3, C.NN)
    )
    assert scattered.shape == (4, 3, C.NN)
    assert not scattered.any()
