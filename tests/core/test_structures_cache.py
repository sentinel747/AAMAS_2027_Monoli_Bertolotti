"""Coerenza della cache di `CellView.structures` (perf, src/core).

`CellView.structures` è cachata su `CellArrays._struct_cache` (una lista di
`StructureView` per (y, x)) per non ri-materializzare gli wrapper a ogni accesso
— era il collo di bottiglia del profilo. La cache è invalidata SOLO dove gli
array cambiano fuori dal setter di `StructureView.integrity`:
  - `WorldView.add_structure` (struct_count++) → pop della cella;
  - il wear bulk in `kernel_biology.update_cells` → clear totale.
Il setter di integrità NON invalida di proposito (tiene coerente il proprio
`_integrity_cache`). Questi test esercitano gli scenari che il confronto v1/v2 a
seed fisso può non toccare (BUILD+MAINTAIN sulla stessa cella, wear dopo una
lettura), a prescindere dal motore a oggetti.
"""
from __future__ import annotations

import numpy as np

from src.core import constants as C
from src.core.arrays import CellArrays
from src.core.kernel_biology import update_cells
from src.core.views import WorldView
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType


def _world(width: int = 3, height: int = 3) -> GridWorld:
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


def _worldview() -> WorldView:
    w = _world()
    return WorldView(CellArrays.from_world(w), _agents_none(), dict(w.planetary_state), {})


def _agents_none():
    # WorldView needs an AgentArrays; an empty one is enough for structure tests.
    from src.core.arrays import AgentArrays

    return AgentArrays.from_agents({})


def test_add_structure_invalidates_count_same_cell():
    """BUILD due volte sulla stessa cella nello stesso step: la seconda lettura
    di cell.structures deve vedere il conteggio aggiornato (invalidazione su
    struct_count++), altrimenti un builder co-locato vedrebbe un conteggio
    stantio — è esattamente il percorso della contesa greenhouse-coverage."""
    wv = _worldview()
    cell = wv.get_cell(1, 1)
    assert len(cell.structures) == 0  # fast path: le celle vuote non sono cachate

    wv.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    # get_cell crea una CellView fresca; la cache vive su CellArrays.
    assert len(wv.get_cell(1, 1).structures) == 1
    # anche riusando la STESSA CellView la cache è stata invalidata dal pop.
    assert len(cell.structures) == 1

    wv.add_structure(Structure(type=StructureType.SOLAR_ARRAY, x=1, y=1, integrity=1.0))
    types = {s.type for s in wv.get_cell(1, 1).structures}
    assert types == {StructureType.GREENHOUSE, StructureType.SOLAR_ARRAY}


def test_add_structure_does_not_leak_to_other_cells():
    """L'invalidazione è per-cella: costruire su (1,1) non deve svuotare la
    cache di un'altra cella con strutture proprie."""
    wv = _worldview()
    wv.add_structure(Structure(type=StructureType.GREENHOUSE, x=0, y=0, integrity=1.0))
    assert len(wv.get_cell(0, 0).structures) == 1
    wv.add_structure(Structure(type=StructureType.HABITAT, x=1, y=1, integrity=1.0))
    # (0,0) resta con la sua struttura.
    assert len(wv.get_cell(0, 0).structures) == 1
    assert len(wv.get_cell(1, 1).structures) == 1


def test_bulk_wear_invalidates_cached_integrity():
    """Leggere cell.structures (materializza/cacha), poi il wear bulk in
    update_cells muta struct_integrity in blocco: la lettura successiva deve
    riflettere l'integrità ridotta, non il valore cachato pre-wear."""
    w = _world()
    ca = CellArrays.from_world(w)
    wv = WorldView(ca, _agents_none(), dict(w.planetary_state), {})
    wv.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))

    before = wv.get_cell(1, 1).structures[0].integrity
    assert before == 1.0  # materializza e cacha a integrità piena

    ps = dict(w.planetary_state)
    ps.setdefault("mean_temperature_c", -63.0)
    ca.occupancy[:] = 0
    # dt grande + dust/radiazione producono usura non nulla.
    ca.dust[1, 1] = 0.5
    update_cells(ca, _agents_none(), ps, 3650.0, cell_degradation=True, full_grid=True)

    gi = C.S[StructureType.GREENHOUSE]
    after = wv.get_cell(1, 1).structures[0].integrity
    # la cache è stata svuotata dal wear: la nuova lettura riflette l'array.
    mean_from_array = float(ca.struct_integrity[1, 1, gi]) / int(ca.struct_count[1, 1, gi])
    assert after == mean_from_array
    assert after < before  # c'è stata usura


def test_maintain_pattern_reads_updated_integrity_within_step():
    """Pattern MAINTAIN di action_space: itera cell.structures, setta integrity
    via setter, poi RILEGGE cell.structures. Con la lista cachata condivisa la
    rilettura deve dare il valore aggiornato (coerenza del setter, nessuna
    doppia applicazione)."""
    w = _world()
    ca = CellArrays.from_world(w)
    wv = WorldView(ca, _agents_none(), dict(w.planetary_state), {})
    wv.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=0.5))
    wv.add_structure(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=0.5))

    cell = wv.get_cell(1, 1)
    for s in cell.structures:  # come MAINTAIN: setter su ogni view
        s.integrity = min(1.0, s.integrity + 0.4)
    # rilettura da una CellView fresca (come fa il messaggio di MAINTAIN).
    reread = [s.integrity for s in wv.get_cell(1, 1).structures]
    assert all(np.isclose(v, 0.9) for v in reread)
    # e l'array è coerente: media per tipo = 0.9 su 2 strutture.
    gi = C.S[StructureType.GREENHOUSE]
    assert np.isclose(float(ca.struct_integrity[1, 1, gi]) / 2.0, 0.9)


def test_structure_cache_reports_hits_misses_and_invalidations():
    wv = _worldview()
    wv.add_structure(
        Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0)
    )
    cell = wv.get_cell(1, 1)
    cell.structures
    cell.structures
    before = wv._cells.cache_metrics()
    assert before["structure_view_misses"] == 1.0
    assert before["structure_view_hits"] == 1.0
    assert before["structure_cache_hit_rate"] == 0.5

    wv.add_structure(
        Structure(type=StructureType.SOLAR_ARRAY, x=1, y=1, integrity=1.0)
    )
    after = wv._cells.cache_metrics()
    assert after["structure_invalidated_entries"] == 1.0


def test_empty_cells_use_fast_path_without_polluting_structure_cache():
    wv = _worldview()

    for y in range(wv.height):
        for x in range(wv.width):
            assert wv.get_cell(x, y).structures == []

    metrics = wv._cells.cache_metrics()
    assert metrics["structure_empty_fast_paths"] == 9.0
    assert metrics["structure_view_misses"] == 0.0
    assert metrics["structure_cached_cell_count"] == 0.0


def test_structure_type_counts_match_object_and_vector_worlds():
    world = _world()
    world.add_structure(
        Structure(type=StructureType.GREENHOUSE, x=0, y=0, integrity=1.0)
    )
    world.add_structure(
        Structure(type=StructureType.GREENHOUSE, x=2, y=2, integrity=1.0)
    )
    world.add_structure(
        Structure(type=StructureType.OXYGEN_PLANT, x=2, y=2, integrity=1.0)
    )
    view = WorldView(
        CellArrays.from_world(world),
        _agents_none(),
        dict(world.planetary_state),
        {},
    )

    assert world.structure_type_counts() == {
        "greenhouse": 2,
        "oxygen_plant": 1,
    }
    vector_counts = view.structure_type_counts()
    assert vector_counts["greenhouse"] == 2
    assert vector_counts["oxygen_plant"] == 1
    assert sum(vector_counts.values()) == 3
