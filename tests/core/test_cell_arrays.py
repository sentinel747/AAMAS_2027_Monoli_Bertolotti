import numpy as np
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.terrain import TerrainType
from src.world.structures import Structure, StructureType
from src.core import constants as C
from src.core.arrays import CellArrays


def _world():
    # GridWorld has no width/height/seed convenience constructor (it is a plain
    # dataclass requiring a pre-built cells grid); build the minimal writable
    # grid the same way src/world/world_generator.py does.
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(4)] for y in range(3)]
    w = GridWorld(width=4, height=3, cells=cells)
    cell = w.get_cell(1, 1)
    cell.structures.append(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0))
    cell.structures.append(Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=0.6))
    cell.structures.append(Structure(type=StructureType.SOLAR_ARRAY, x=1, y=1, integrity=0.3))
    cell.resources.water = 7.0
    cell.water_ice = 2.5
    return w


def test_from_world_clusters_structures_by_type():
    ca = CellArrays.from_world(_world())
    g = C.S[StructureType.GREENHOUSE]
    s = C.S[StructureType.SOLAR_ARRAY]
    assert ca.struct_count[1, 1, g] == 2
    assert np.isclose(ca.struct_integrity[1, 1, g], 1.6)
    assert ca.struct_count[1, 1, s] == 1
    assert np.isclose(ca.cell_res[1, 1, C.R["water"]], 7.0)
    assert np.isclose(ca.water_ice[1, 1], 2.5)


def test_struct_fx_matches_object_sum():
    w = _world()
    cell = w.get_cell(1, 1)
    expected_food = sum(s.local_effect.get("food", 0.0) for s in cell.structures)
    expected_energy = sum(s.local_effect.get("energy", 0.0) for s in cell.structures)
    ca = CellArrays.from_world(w)
    # 2 serre a integrita' media 0.8 (>0.4: eff=0.8) => food = 1.0*0.8*2
    assert np.isclose(ca.struct_fx[1, 1, C.E["food"]], expected_food, atol=1e-6)
    # solar a 0.3 (<=0.4: eff=0.15) => energy = 1.5*0.15
    assert np.isclose(ca.struct_fx[1, 1, C.E["energy"]], expected_energy, atol=1e-6)


def test_empty_cells_have_zero_fx():
    ca = CellArrays.from_world(_world())
    assert ca.struct_fx[0, 0].sum() == 0.0
