import numpy as np
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType
from src.core import constants as C
from src.core.arrays import CellArrays
from src.core.needs import RedistributionConfig, compute_needs, settled_mask


def _grid(width: int, height: int) -> GridWorld:
    # GridWorld has no width/height/seed convenience constructor (plain dataclass
    # requiring a pre-built cells grid) - build it the same way the other
    # tests/core/*.py modules do (test_cell_arrays.py, test_kernel_biology_equivalence.py).
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


def _cells_with_outpost():
    w = _grid(5, 5)
    w.get_cell(2, 2).structures.append(Structure(type=StructureType.SHELTER, x=2, y=2))
    ca = CellArrays.from_world(w)
    ca.occupancy[2, 2] = 3
    return ca


def test_settled_mask_requires_structure_or_site():
    ca = _cells_with_outpost()
    m = settled_mask(ca)
    assert m[2, 2] and m.sum() == 1
    ca.site_progress[0, 4, C.S[StructureType.SHELTER]] = 10.0
    assert settled_mask(ca)[0, 4]


def test_water_need_scales_with_occupants_and_stock():
    ca = _cells_with_outpost()
    cfg = RedistributionConfig()
    needs = compute_needs(ca, cfg)
    # 3 occupanti * 4.0 target * 2 = 24, magazzino 0 -> need 24
    assert np.isclose(needs[2, 2, C.ND["water"]], 24.0)
    ca.cell_res[2, 2, C.R["water"]] = 24.0
    assert compute_needs(ca, cfg)[2, 2, C.ND["water"]] == 0.0


def test_trio_needs_flag_missing_structures():
    ca = _cells_with_outpost()
    needs = compute_needs(ca, RedistributionConfig())
    assert needs[2, 2, C.ND["build_solar"]] == 1.0
    assert needs[2, 2, C.ND["build_oxygen_plant"]] == 1.0
    assert needs[2, 2, C.ND["build_greenhouse"]] == 1.0
    assert needs[0, 0].sum() == 0.0  # non insediata: zero ovunque


def test_build_need_persists_until_population_coverage_or_active_site():
    ca = _cells_with_outpost()
    cfg = RedistributionConfig()
    ca.occupancy[2, 2] = 20  # solar target ceil(20/7)=3; O2 target ceil(20/10)=2
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_solar"]] == 3.0
    ca.struct_count[2, 2, C.S[StructureType.SOLAR_ARRAY]] = 1
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_solar"]] == 2.0
    ca.struct_count[2, 2, C.S[StructureType.SOLAR_ARRAY]] = 3
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_solar"]] == 0.0
    ca.struct_count[2, 2, C.S[StructureType.SOLAR_ARRAY]] = 1
    ca.site_progress[2, 2, C.S[StructureType.SOLAR_ARRAY]] = 30.0
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_solar"]] == 0.0  # cantiere attivo: non richiedere di nuovo

    ca.site_progress[2, 2, C.S[StructureType.SOLAR_ARRAY]] = -1.0
    ca.struct_count[2, 2, C.S[StructureType.OXYGEN_PLANT]] = 1
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_oxygen_plant"]] == 1.0
    ca.struct_count[2, 2, C.S[StructureType.OXYGEN_PLANT]] = 2
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_oxygen_plant"]] == 0.0


def test_greenhouse_need_scales_with_greenhouse_per_capita():
    ca = _cells_with_outpost()
    cfg = RedistributionConfig(greenhouse_per_capita=0.25)
    # 3 occupanti * 0.25 = 0.75 -> ceil = 1 serra attesa; 0 presenti -> bisogno.
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_greenhouse"]] == 1.0
    ca.struct_count[2, 2, C.S[StructureType.GREENHOUSE]] = 1
    assert compute_needs(ca, cfg)[2, 2, C.ND["build_greenhouse"]] == 0.0


def test_maintenance_need_is_the_integrity_a_repair_would_restore():
    """Il bisogno E' cio' che `integrita_da_ripristinare` calcola per l'esecutore.

    Con integrita' 0,3 una manutenzione porta la struttura a 0,7: recupera 0,4,
    non 0,3. La vecchia formula misurava la distanza da una soglia (0,6) che
    l'esecutore non conosce, e le due grandezze non coincidevano mai.
    """
    ca = _cells_with_outpost()
    si = C.S[StructureType.SHELTER]
    ca.struct_integrity[2, 2, si] = 0.3
    needs = compute_needs(ca, RedistributionConfig())
    assert np.isclose(needs[2, 2, C.ND["maintenance"]], 0.4, atol=1e-6)

    # Sopra 0,6 di integrita' il recupero e' cio' che manca a 1,0, non 0,4.
    ca.struct_integrity[2, 2, si] = 0.85
    needs = compute_needs(ca, RedistributionConfig())
    assert np.isclose(needs[2, 2, C.ND["maintenance"]], 0.15, atol=1e-6)


def test_maintenance_need_wakes_up_before_the_old_threshold():
    """Il difetto che ha estinto sei run su sette il 26-27 agosto.

    La produzione di una struttura scala con la sua integrita', quindi una cella
    perde l'autosufficienza molto prima di 0,6 — misurato a 0,82. Con la vecchia
    formula il bisogno restava ESATTAMENTE zero fin li', e nessuna manutenzione
    veniva mai proposta: la colonia moriva di sete con il magazzino pieno di
    cibo. Qui si blocca la proprieta' che conta, cioe' che l'usura si veda
    appena esiste.
    """
    ca = _cells_with_outpost()
    si = C.S[StructureType.SHELTER]
    for integrita in (0.99, 0.9, 0.82, 0.7):
        ca.struct_integrity[2, 2, si] = integrita
        needs = compute_needs(ca, RedistributionConfig())
        assert needs[2, 2, C.ND["maintenance"]] > 0.0, integrita

    # A impianti nuovi non c'e' nulla da fare, e deve restare cosi'.
    ca.struct_integrity[2, 2, si] = 1.0
    needs = compute_needs(ca, RedistributionConfig())
    assert needs[2, 2, C.ND["maintenance"]] == 0.0


def test_habitat_pressure_uses_shelter_habitat_infirmary_weights():
    ca = _cells_with_outpost()
    ca.occupancy[2, 2] = 5  # shelter_slots = 1*1.0 = 1.0 -> pressione = 5-1 = 4
    needs = compute_needs(ca, RedistributionConfig())
    assert np.isclose(needs[2, 2, C.ND["habitat_pressure"]], 4.0)


def test_needs_matrix_not_imported_by_any_agents_module():
    """Keep one narrow seam even though kernel now feeds the need array to
    cell proposals: decision modules consume data, not this implementation."""
    import pathlib
    import re

    src_dir = pathlib.Path(__file__).resolve().parents[2] / "src"
    allowed = {
        (src_dir / "core" / "kernel.py").resolve(),
        (src_dir / "core" / "redistribution.py").resolve(),
    }
    offenders = []
    for path in src_dir.rglob("*.py"):
        if path.name == "needs.py" or path.resolve() in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"^\s*(from|import)\s+src\.core\.needs\b", text, re.MULTILINE):
            offenders.append(str(path))
    assert offenders == [], (
        f"only src/core/kernel.py and src/core/redistribution.py may import "
        f"src.core.needs: {offenders}"
    )
