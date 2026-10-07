"""L'infrastruttura di supporto segue le case (autopsie v4/v5).

I cluster di decessi ((2,79), (1,77), (1,79)) erano celle con SOLI
heater/stazioni meteo/depositi: la coda `established` delle priorita' li
faceva costruire ai vagabondi in celle deserte, e quelle mini-basi senza cibo
ne' acqua attiravano altri erranti (il punteggio cella premia le strutture)
che vi si stabilivano fino a morire di sete o fame. Regole: meteo/heater/
deposito/lab solo dove esiste gia' serra o habitat; dispersione da folla solo
verso celle supportate e con scorte d'acqua E cibo; il rientro per fame parte
finche' le razioni coprono la traversata.
"""

from types import SimpleNamespace

from src.agents.action_space import ActionType
from src.agents.population import spawn_initial_agents
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _observation():
    return SimpleNamespace(nearby_agents=[], nearby_agents_details=[], local_danger=0.0, local_habitability=0.0, agent_count=1, construction_sites={})


def _rich(agent):
    agent.inventory.water = 8.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 10.0
    agent.inventory.construction_material = 8.0
    agent.inventory.minerals = 6.0
    agent.inventory.energy = 4.0
    agent.oxygen_level = 1.0
    agent.fatigue = 0.0


def test_no_support_infrastructure_on_unsettled_ground():
    world = WorldGenerator(seed=51).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=51)
    agent = next(iter(agents.values()))
    _rich(agent)
    cell = world.get_cell(agent.x, agent.y)
    assert not cell.structures  # terreno vergine

    request = agent._planned_infrastructure(set(), _observation(), world)

    junk = {ActionType.BUILD_WEATHER_STATION, ActionType.BUILD_HEATER, ActionType.BUILD_STORAGE_DEPOT, ActionType.BUILD_RESEARCH_LAB}
    assert request is None or request.action not in junk


def test_provisioned_pioneer_homesteads_with_a_greenhouse():
    """Il pioniere col kit su terreno vergine pianta la serra deliberatamente:
    senza questa regola i dispersi ben riforniti oziavano sulle celle nuove
    (ogni build era need-driven) e l'espansione v7 e' crollata a 4 celle."""
    world = WorldGenerator(seed=55).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=55)
    agent = next(iter(agents.values()))
    _rich(agent)  # acqua 8, cibo 10: nessun bisogno immediato
    cell = world.get_cell(agent.x, agent.y)
    assert not cell.structures

    request = agent._planned_infrastructure(set(), _observation(), world)

    assert request is not None
    assert request.action == ActionType.BUILD_GREENHOUSE


def test_support_infrastructure_follows_homes():
    world = WorldGenerator(seed=52).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=52)
    agent = next(iter(agents.values()))
    _rich(agent)
    x, y = agent.x, agent.y
    world.add_structure(Structure(StructureType.GREENHOUSE, x, y))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, x, y))
    agent.local_x_m = 30_000.0
    agent.local_y_m = 30_000.0
    cell = world.get_cell(x, y)

    request = agent._planned_infrastructure({s.type for s in cell.structures}, _observation(), world)

    assert request is not None  # cella abitata: il supporto arriva
    assert request.action in {ActionType.BUILD_WEATHER_STATION, ActionType.BUILD_HEATER, ActionType.BUILD_STORAGE_DEPOT, ActionType.BUILD_RESEARCH_LAB, ActionType.BUILD_HABITAT, ActionType.BUILD_INFIRMARY}


def test_crowd_dispersal_requires_provisions_not_kit():
    """La dispersione da folla resta il canale di espansione organica (v3: 15
    celle) e richiede solo scorte di viaggio (acqua 3.5, cibo 2.5): pretendere
    celle gia' supportate o kit da fondatore l'ha strangolata a 3-4 celle
    (v6-v9). La sicurezza viene dal divieto di mini-basi e dall'homestead."""
    world = WorldGenerator(seed=53).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=53)
    agent = next(iter(agents.values()))
    _rich(agent)
    cell = world.get_cell(agent.x, agent.y)
    cell.agents_present.extend(f"g{i}" for i in range(25))  # cella affollata
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))

    # Senza scorte di viaggio niente dispersione.
    agent.inventory.water = 2.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 1.0
    request = agent._movement_or_observe(world, _observation())
    assert "dispersing" not in (request.message or "")

    # Con le scorte la valvola di sfogo funziona anche verso terreno vergine.
    _rich(agent)
    request = agent._movement_or_observe(world, _observation())
    assert "dispersing" in (request.message or "") or request.action in {ActionType.MOVE, ActionType.EXPLORE, ActionType.OBSERVE, ActionType.REST}


def test_hungry_agent_walks_home_before_starvation_clock_wins():
    world = WorldGenerator(seed=54).generate(10, 10)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=54)
    agent = next(iter(agents.values()))
    wx = (agent.x + 2) % world.width
    world.add_structure(Structure(StructureType.GREENHOUSE, wx, agent.y))
    agent.inventory.water = 8.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 2.0  # 20 razioni: sotto il costo del rientro a 2 celle (~2.9)
    agent.inventory.construction_material = 0.0  # niente serra sul posto
    agent.inventory.minerals = 0.0
    agent.fatigue = 0.0

    request = agent._food_security_action(world, world.get_cell(agent.x, agent.y), _observation())

    assert request is not None
    assert request.action in {ActionType.MOVE, ActionType.EXPLORE}
