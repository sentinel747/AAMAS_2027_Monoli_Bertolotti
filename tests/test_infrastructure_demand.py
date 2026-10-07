"""Capacita' vitale guidata dalla domanda (audit run 120726_testlogic5_1000steps).

Nella run da 1000 step serre (7) e impianti O2 (5) sono rimasti allo stock
iniziale per 19 anni con la popolazione raddoppiata: la serra costava ice=1 che
nessun agente possedeva (ghiaccio aggregato 0.04 su 207 coloni) e le proposte
scattavano solo su bisogni personali, mai sulla copertura di cella. Ora la
serra costa water=1 (raggiungibile via pozzo colonia) e, a griglia di
sopravvivenza pronta, serre e impianti O2 vengono proposti finche' i rapporti
di copertura di build_policy non sono soddisfatti.
"""

from types import SimpleNamespace

from src.agents.population import spawn_initial_agents
from src.world.structures import BUILD_COSTS, Structure, StructureType
from src.world.world_generator import WorldGenerator


def _mature_underserved_colony(seed=12, colonists=20):
    """Cella con griglia di sopravvivenza pronta ma copertura sotto fabbisogno."""
    world = WorldGenerator(seed=seed).generate(8, 8)
    world.step = 50
    agents = spawn_initial_agents(1, world, seed=seed)
    agent = next(iter(agents.values()))
    x, y = agent.x, agent.y
    cell = world.get_cell(x, y)
    cell.agents_present.extend(f"ghost_{i:02d}" for i in range(colonists - 1))
    world.add_structure(Structure(StructureType.GREENHOUSE, x, y))      # servono ceil(20/7)=3
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))    # servono ceil(20/10)=2
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))    # O2 saturo
    for _ in range(10):
        world.add_structure(Structure(StructureType.HABITAT, x, y))     # shelter saturo (10*2 >= 20)
    agent.local_x_m = 30_000.0  # lontano dalle strutture esistenti (blocco 250 m)
    agent.local_y_m = 30_000.0
    return world, agent, cell


def test_greenhouse_cost_uses_water_not_ice():
    cost = BUILD_COSTS[StructureType.GREENHOUSE].to_dict()
    assert cost.get("ice", 0) == 0
    assert cost.get("water", 0) >= 1


def test_planned_infrastructure_scales_greenhouses_with_population():
    world, agent, cell = _mature_underserved_colony()
    agent.inventory.water = 5.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 10.0
    agent.inventory.construction_material = 3.0
    agent.inventory.minerals = 1.0
    agent.oxygen_level = 1.0

    request = agent._planned_infrastructure(
        {s.type for s in cell.structures},
        SimpleNamespace(local_danger=0.0, construction_sites={}),
        world,
    )

    assert request is not None
    assert request.action.value == "build_greenhouse"


def test_greenhouse_not_proposed_when_water_reserve_is_survival_buffer():
    world, agent, cell = _mature_underserved_colony()
    agent.inventory.water = 1.0  # sotto il buffer: l'ultima acqua non va in malta
    agent.inventory.ice = 0.0
    agent.inventory.food = 10.0
    agent.inventory.construction_material = 3.0
    agent.inventory.minerals = 1.0  # oxygen plant (4 mat, 3 min) non accessibile

    request = agent._planned_infrastructure(
        {s.type for s in cell.structures},
        SimpleNamespace(local_danger=0.0, construction_sites={}),
        world,
    )

    assert request is None or request.action.value != "build_greenhouse"


def test_needs_resources_follows_coverage_not_existence():
    world, agent, cell = _mature_underserved_colony()
    # Tutto costruito almeno una volta, ma la serra e' sotto copertura e
    # l'agente non puo' permettersela: deve contare come fabbisogno risorse.
    for _ in range(7):
        world.add_structure(Structure(StructureType.SOLAR_ARRAY, agent.x, agent.y))
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y))
    agent.inventory.water = 5.0
    agent.inventory.construction_material = 0.0
    agent.inventory.minerals = 0.0

    existing = {s.type for s in cell.structures}
    assert agent._needs_resources(existing, cell) is True


# --- Lo stallo della colonia: capienza esattamente pari agli occupanti -------
#
# Misurato il 2026-08-30 su cento coloni fondatori: la cella madre finiva con
# centootto occupanti e capienza centootto, le due regole si chiudevano l'una
# sull'altra — costruire richiedeva `capienza < occupanti`, nascere richiedeva
# `capienza > occupanti` — e la colonia restava ferma per sempre con il 45% dei
# coloni senza alcuna azione ammissibile. I test che seguono bloccano la
# proprieta' che ne e' uscita: **una cella a copertura esatta deve ancora poter
# costruire**, e la regola che lo decide deve essere una sola.


def test_una_cella_a_copertura_esatta_puo_ancora_costruire():
    from src.agents.build_policy import structure_saturated

    world = WorldGenerator(seed=77).generate(8, 8)
    x, y = 4, 4
    cell = world.get_cell(x, y)
    cell.agents_present = [SimpleNamespace(agent_id=f"a{i}") for i in range(70)]
    for _ in range(10):
        world.add_structure(Structure(StructureType.HABITAT, x, y))
    for _ in range(50):
        world.add_structure(Structure(StructureType.SHELTER, x, y))
    for _ in range(10):
        world.add_structure(Structure(StructureType.GREENHOUSE, x, y))

    # 50 rifugi + 10 habitat x 2 = 70 posti per 70 occupanti: copertura esatta.
    assert not structure_saturated(StructureType.SHELTER, cell)
    # 10 serre coprono 70 coloni al rapporto di 7: anche qui copertura esatta.
    assert not structure_saturated(StructureType.GREENHOUSE, cell)


def test_la_saturazione_e_una_regola_sola_fra_i_due_motori():
    """La maschera ad array e la saturazione a oggetti non possono divergere."""
    import numpy as np

    from src.agents import pillars
    from src.agents.action_space import ActionType, BUILD_ACTIONS
    from src.agents.build_policy import structure_saturated
    from src.core.arrays import CellArrays
    from src.core.cell_action_mask import compute_cell_masks

    world = WorldGenerator(seed=78).generate(8, 8)
    x, y = 4, 4
    cell = world.get_cell(x, y)
    # Terreno edificabile per costruzione (2026-09-01): questa prova confronta
    # la SATURAZIONE fra i due motori, e il divieto geografico di costruire su
    # montagna e canyon spegnerebbe la maschera per una ragione che non c'entra.
    from src.world.terrain import TerrainType as _T
    cell.terrain = _T.REGOLITH_PLAIN
    for tipo, quante in (
        (StructureType.HABITAT, 6),
        (StructureType.SHELTER, 18),
        (StructureType.GREENHOUSE, 5),
        (StructureType.SOLAR_ARRAY, 5),
        (StructureType.OXYGEN_PLANT, 3),
    ):
        for _ in range(quante):
            world.add_structure(Structure(tipo, x, y))
    cell.agents_present = [SimpleNamespace(agent_id=f"a{i}") for i in range(30)]

    cells = CellArrays.from_world(world)
    cells.occupancy[y, x] = 30
    masks = compute_cell_masks(cells)

    for action, structure_type in BUILD_ACTIONS.items():
        ammessa_array = bool(masks[y, x, pillars.ACTION_INDEX[action]])
        satura_oggetti = structure_saturated(structure_type, cell)
        assert ammessa_array == (not satura_oggetti), (
            f"{action.value}: maschera={ammessa_array} saturazione={satura_oggetti}"
        )


def test_il_cancello_fisiologico_usa_la_stessa_somma_degli_alloggi():
    """`physio_gate` riscrive la somma in interi: deve coincidere sempre."""
    import numpy as np

    from src.core import constants as C
    from src.core.physio_gate import _support_capacity_positive
    from src.world.occupancy import housing_slots_from_structures

    for rifugi in range(0, 5):
        for habitat in range(0, 5):
            for infermerie in range(0, 5):
                counts = np.zeros(len(C.STRUCTURES), dtype=np.int64)
                counts[C.S[StructureType.SHELTER]] = rifugi
                counts[C.S[StructureType.HABITAT]] = habitat
                counts[C.S[StructureType.INFIRMARY]] = infermerie
                counts[C.S[StructureType.GREENHOUSE]] = 3
                counts[C.S[StructureType.SOLAR_ARRAY]] = 3
                counts[C.S[StructureType.OXYGEN_PLANT]] = 3
                alloggi = int(
                    housing_slots_from_structures(rifugi, habitat, infermerie)
                )
                atteso = min(alloggi, 3 * 7, 3 * 7, 3 * 10) > 0
                assert bool(_support_capacity_positive(counts)) == atteso
