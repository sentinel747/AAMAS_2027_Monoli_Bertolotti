"""Logistica idrica di frontiera (audit run 140726_testlogic2_1000steps).

Nella run da 1000 step post-espansione 18 dei 19 decessi erano disidratazioni
in celle-sobborgo con soli shelter/heater: strutture che contano come "life
support" per la regola del riposo ma NON hanno effetto water. L'agente
riposava al rifugio asciutto finche' l'orologio della sete (7 step) non
superava la traversata di rientro (~12 mosse). Regole corrette: riposo per
fatica solo dove c'e' acqua o con borraccia capiente; al pozzo si fa il pieno;
a secco e sotto scorta si torna verso strutture CON acqua.
"""

from types import SimpleNamespace

from src.agents.action_space import ActionType
from src.agents.build_policy import copertura_completa, structure_saturated
from src.agents.population import spawn_initial_agents
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _observation():
    return SimpleNamespace(nearby_agents=[], nearby_agents_details=[], local_danger=0.0, local_habitability=0.0, agent_count=1, construction_sites={})


def _dry_suburb_world(seed=31):
    world = WorldGenerator(seed=seed).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=seed)
    agent = next(iter(agents.values()))
    # Sobborgo asciutto: solo shelter (nessun effetto water).
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y))
    # La colonia madre con acqua sta a distanza.
    hx, hy = (agent.x + 2) % world.width, agent.y
    world.add_structure(Structure(StructureType.GREENHOUSE, hx, hy))
    return world, agent


def test_dry_shelter_cell_is_not_a_rest_trap():
    world, agent = _dry_suburb_world()
    agent.inventory.water = 2.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 5.0
    agent.fatigue = 0.55  # prima: riposo vicino al "life support" (asciutto)

    request = agent._movement_or_observe(world, _observation())

    assert request.action != ActionType.REST
    assert request.action in {ActionType.MOVE, ActionType.EXPLORE}
    assert "returning" in (request.message or "")


def test_wet_cell_low_canteen_tops_up_instead_of_resting():
    world = WorldGenerator(seed=32).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=32)
    agent = next(iter(agents.values()))
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
    agent.inventory.water = 2.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 5.0
    agent.fatigue = 0.0

    request = agent._movement_or_observe(world, _observation())

    assert request.action == ActionType.REFILL_WATER


def test_rest_for_fatigue_still_allowed_where_water_flows():
    world = WorldGenerator(seed=33).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=33)
    agent = next(iter(agents.values()))
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
    agent.inventory.water = 6.0
    agent.inventory.food = 5.0
    agent.fatigue = 0.55

    request = agent._movement_or_observe(world, _observation())

    assert request.action == ActionType.REST


def test_solar_arrays_follow_population_coverage_ratio():
    """1 pannello ogni 7 coloni, come le serre, piu' il margine di crescita.

    Nella run da 1000 passi la formula speciale per-cella saturava localmente
    mentre il power margin dell'intera colonia scendeva a 0,57 (28 pannelli
    contro ~49 necessari). Dal 2026-08-30 il fabbisogno da coprire e' la
    popolazione **piu' il margine** (`MARGINE_CAPIENZA_COSTRUITA`), perche' una
    cella che copre esattamente i presenti non puo' piu' crescere: qui venti
    coloni ne fanno coprire ventitre, cioe' quattro pannelli invece di tre.
    La maturita' della cella, che e' un'altra domanda, resta a copertura esatta
    e la misura `copertura_completa`.
    """
    world = WorldGenerator(seed=34).generate(8, 8)
    cell = world.get_cell(4, 4)
    cell.agents_present.extend(f"g{i}" for i in range(20))
    for _ in range(2):
        world.add_structure(Structure(StructureType.SOLAR_ARRAY, 4, 4))
    # ceil((20 + 1 di riserva) / 7) = 3: due non bastano.
    assert structure_saturated(StructureType.SOLAR_ARRAY, cell) is False

    world.add_structure(Structure(StructureType.SOLAR_ARRAY, 4, 4))
    assert structure_saturated(StructureType.SOLAR_ARRAY, cell) is True
    # La maturita' della cella e' un'altra domanda e non porta riserva.
    assert copertura_completa(StructureType.SOLAR_ARRAY, cell) is True
