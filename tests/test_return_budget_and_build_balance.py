"""Soglie di rientro e portafoglio costruzioni bilanciato (validazioni v4-v10).

1. Chi vaga fuori dal supporto idrico rientra a soglie FISSE (4.0 acqua /
   2.5 cibo): il budget scalato sulla distanza (v5) rimbalzava i pionieri sul
   confine delle celle vergini prima che potessero fondare, e con il divieto
   di mini-basi nel deserto gli sbandati profondi che voleva salvare non
   esistono piu'.
2. Con serre e pannelli allo stesso rapporto (1/7), la serra — prima nella
   lista priorita' — ombreggiava il solar nello slot di costruzione (18 build
   solar in v4 contro 32 in v3): i tipi a domanda vengono proposti in ordine
   di deficit proporzionale.
"""

from types import SimpleNamespace

from src.agents.action_space import ActionType
from src.agents.population import spawn_initial_agents
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _observation():
    return SimpleNamespace(nearby_agents=[], nearby_agents_details=[], local_danger=0.0, local_habitability=0.0, agent_count=1, construction_sites={})


def test_low_canteen_away_from_wells_walks_home():
    world = WorldGenerator(seed=41).generate(12, 12)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=41)
    agent = next(iter(agents.values()))
    wx = (agent.x + 2) % world.width
    world.add_structure(Structure(StructureType.GREENHOUSE, wx, agent.y))
    agent.inventory.water = 3.0  # sotto il pavimento fisso di 4.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 10.0
    agent.fatigue = 0.0

    request = agent._movement_or_observe(world, _observation())

    assert request.action in {ActionType.MOVE, ActionType.EXPLORE}
    assert "returning" in (request.message or "")


def test_provisioned_agent_far_from_well_keeps_working():
    world = WorldGenerator(seed=42).generate(12, 12)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=42)
    agent = next(iter(agents.values()))
    wx = (agent.x + 2) % world.width
    world.add_structure(Structure(StructureType.GREENHOUSE, wx, agent.y))
    agent.inventory.water = 8.0  # ben oltre il budget di ritorno (2 celle = ~3.6)
    agent.inventory.ice = 0.0
    agent.inventory.food = 10.0
    agent.fatigue = 0.0

    request = agent._movement_or_observe(world, _observation())

    assert "returning" not in (request.message or "")


def test_demand_priorities_follow_proportional_deficit():
    world = WorldGenerator(seed=43).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=43)
    agent = next(iter(agents.values()))
    x, y = agent.x, agent.y
    cell = world.get_cell(x, y)
    cell.agents_present.extend(f"g{i}" for i in range(20))
    # Serre gia' a copertura (3/3), solar scoperto (1/3), ossigeno coperto (2/2).
    for _ in range(3):
        world.add_structure(Structure(StructureType.GREENHOUSE, x, y))
    for _ in range(2):
        world.add_structure(Structure(StructureType.OXYGEN_PLANT, x, y))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, x, y))
    agent.local_x_m = 30_000.0
    agent.local_y_m = 30_000.0
    agent.inventory.water = 8.0
    agent.inventory.food = 10.0
    agent.inventory.construction_material = 6.0
    agent.inventory.minerals = 4.0
    agent.oxygen_level = 1.0

    request = agent._planned_infrastructure({s.type for s in cell.structures}, _observation(), world)

    assert request is not None
    assert request.action == ActionType.BUILD_SOLAR_ARRAY


def test_dry_shelter_recovery_rest_requires_water_or_reserve():
    world = WorldGenerator(seed=44).generate(8, 8)
    agents = spawn_initial_agents(1, world, seed=44)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y))
    agent.fatigue = 0.60
    agent.inventory.water = 1.0
    agent.inventory.ice = 0.0

    assert agent._should_recover(cell, _observation()) is False  # rifugio asciutto, borraccia vuota

    agent.inventory.water = 5.0
    assert agent._should_recover(cell, _observation()) is True  # riposo sicuro: scorta capiente
