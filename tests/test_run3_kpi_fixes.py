"""Fix dell'audit KPI della run 140726_testlogic3_1000steps (1000 step, 200 coloni).

Tre difetti osservati nei log della run:

1) RETE SOCIALE A STELLA — `nearby_agents` e' ordinata e tutti i rami
   COMMUNICATE usavano l'indice [0]: il 98.9% dei 34.6k messaggi andava ad
   agent_000 (grado 408, densita' 0.01, peso mediano degli archi 1). In piu'
   la clausola `local_habitability > 0.025` nel need_signal e' sempre vera
   nelle celle colonia, quindi communicate girava al tetto di 1-su-4 della
   finestra recente (23.9% di tutte le azioni).
2) AVAMPOSTI SENZA ENERGIA/OSSIGENO — il trio a domanda (serra/O2/solar)
   girava solo con la griglia di sopravvivenza completa (che richiede un
   impianto O2): 0 pannelli e 0 impianti fuori dalla cella madre, power
   margin 1.09 -> 0.66.
3) MORTI DI FAME SULLA VIA DEL GHIACCIO — 11 coloni morti insieme a (1,77)
   con sazieta' 0 e idratazione 1.0: le marce verso il ghiaccio non
   guardavano il cibo trasportato e la bevuta forzata del streak-breaker a
   idratazione piena tassava di ~33% la marcia di rientro.
"""

from types import SimpleNamespace

from src.agents.action_space import ActionType
from src.agents.population import spawn_initial_agents
from src.agents.vitals import death_cause
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _observation(**overrides):
    base = dict(
        nearby_agents=[],
        nearby_agents_details=[],
        local_danger=0.0,
        local_habitability=0.0,
        agent_count=1,
        construction_sites={},
    )
    base.update(overrides)
    return SimpleNamespace(**base)


def _lone_agent(seed=41):
    world = WorldGenerator(seed=seed).generate(8, 8)
    world.step = 60
    agents = spawn_initial_agents(1, world, seed=seed)
    agent = next(iter(agents.values()))
    return world, agent


# ---------------------------------------------------------------- difetto 1


def test_communication_targets_rotate_across_neighbors():
    _, agent = _lone_agent(seed=42)
    observation = _observation(nearby_agents=["agent_101", "agent_102", "agent_103"])

    targets = set()
    for actions_taken in range(6):
        agent.actions_taken = actions_taken
        targets.add(agent._communication_target(observation))

    # Prima del fix il destinatario era sempre nearby_agents[0].
    assert len(targets) == 3


def test_colony_wellbeing_alone_does_not_force_chat():
    _, agent = _lone_agent(seed=43)
    agent.actions_taken = 1  # cadenza (ogni 6) non scattata
    agent.cooperation = 0.9
    agent.health = 1.0
    agent.oxygen_level = 1.0
    agent.hydration = 1.0
    agent.stress_index = 0.0
    agent.inventory.food = 5.0
    agent.recent_actions = ["move", "move", "move"]
    observation = _observation(nearby_agents=["agent_101"], local_habitability=0.05)

    # Prima del fix local_habitability > 0.025 rendeva need_signal sempre vero.
    assert agent._should_coordinate(observation) is False


def test_chronic_cell_danger_is_not_a_personal_need_signal():
    """Il rischio ambientale cronico della cella affollata non e' una notizia
    per chi ci vive: in validazione v1 teneva communicate al 23.8% (tetto
    1-su-4) invece della cadenza 1-su-6."""
    _, agent = _lone_agent(seed=49)
    agent.actions_taken = 1
    agent.cooperation = 0.9
    agent.health = 1.0
    agent.oxygen_level = 1.0
    agent.hydration = 1.0
    agent.stress_index = 0.0
    agent.inventory.food = 5.0
    agent.recent_actions = ["move", "move", "move"]
    observation = _observation(nearby_agents=["agent_101"], local_danger=1.5)

    assert agent._should_coordinate(observation) is False


# ---------------------------------------------------------------- difetto 2


def test_settled_outpost_without_o2_plant_still_builds_solar():
    world, agent = _lone_agent(seed=44)
    cell = world.get_cell(agent.x, agent.y)
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))
    cell.agents_present.extend(f"crowd_{i}" for i in range(7))  # 8 coloni presenti

    agent.inventory.water = 5.0
    agent.inventory.ice = 0.0
    agent.inventory.food = 5.0
    agent.inventory.construction_material = 2.0
    agent.inventory.minerals = 2.0
    agent.inventory.energy = 2.0
    agent.oxygen_level = 1.0

    request = agent._planned_infrastructure(
        {s.type for s in cell.structures}, _observation(agent_count=8), world
    )

    # Prima del fix la cella senza impianto O2 non entrava nel trio a domanda:
    # la coda "established" proponeva la stazione meteo al posto del solare.
    assert request is not None
    assert request.action == ActionType.BUILD_SOLAR_ARRAY


# ---------------------------------------------------------------- difetto 3


def _food_poor_wanderer(seed):
    world, agent = _lone_agent(seed=seed)
    cell = world.get_cell(agent.x, agent.y)
    cell.water_ice = 0.0
    cell.resources.ice = 0.0
    # La colonia con acqua e cibo sta a due celle di distanza.
    hx, hy = (agent.x + 2) % world.width, agent.y
    world.add_structure(Structure(StructureType.GREENHOUSE, hx, hy))
    agent.inventory.ice = 0.0
    agent.inventory.construction_material = 0.5
    agent.inventory.minerals = 0.5
    agent.steps_without_water = 0
    return world, agent, cell


def test_forecast_ice_march_blocked_when_rations_short():
    world, agent, cell = _food_poor_wanderer(seed=45)
    agent.inventory.water = 3.0  # sotto la soglia 3.2 che fa partire la marcia
    agent.inventory.food = 2.0   # dispensa vuota: niente pellegrinaggio del ghiaccio

    request = agent._water_forecast_action(world, cell, _observation(agent_count=6))

    assert request is not None
    assert request.action == ActionType.MOVE
    assert "returning to colony support" in (request.message or "")


def test_security_ice_march_blocked_when_rations_short():
    world, agent, cell = _food_poor_wanderer(seed=46)
    agent.inventory.water = 1.5  # emergenza idrica vera
    agent.inventory.food = 2.0
    agent.autonomy_preference = 0.9
    agent.hydration = 0.80

    request = agent._water_security_action(world, cell, _observation())

    assert request is not None
    assert request.action == ActionType.MOVE
    assert "returning to colony support" in (request.message or "")


def test_streak_breaker_does_not_sip_at_full_hydration():
    world, agent, cell = _food_poor_wanderer(seed=47)
    agent.inventory.water = 2.0
    agent.inventory.food = 5.0
    agent.hydration = 1.0
    agent.recent_actions = ["move", "move", "move"]

    request = agent._water_security_action(world, cell, _observation())

    # Prima del fix: DRINK_WATER forzato ogni 4 passi anche a idratazione piena.
    assert request is None or request.action != ActionType.DRINK_WATER


def test_streak_breaker_still_drinks_when_hydration_drains():
    world, agent, cell = _food_poor_wanderer(seed=48)
    agent.inventory.water = 2.0
    agent.inventory.food = 5.0
    agent.hydration = 0.80
    agent.recent_actions = ["move", "move", "move"]

    request = agent._water_security_action(world, cell, _observation())

    assert request is not None
    assert request.action == ActionType.DRINK_WATER


# ------------------------------------------------------- autopsie leggibili


def test_death_cause_classification():
    starved = SimpleNamespace(hydration=1.0, satiety=0.0, oxygen_level=0.99, steps_without_water=0, steps_without_food=14)
    parched = SimpleNamespace(hydration=0.0, satiety=0.5, oxygen_level=0.99, steps_without_water=7, steps_without_food=0)
    choked = SimpleNamespace(hydration=0.8, satiety=0.8, oxygen_level=0.0, steps_without_water=0, steps_without_food=0)
    crushed = SimpleNamespace(hydration=0.8, satiety=0.8, oxygen_level=0.9, steps_without_water=0, steps_without_food=0)

    assert death_cause(starved) == "starvation"
    assert death_cause(parched) == "dehydration"
    assert death_cause(choked) == "hypoxia"
    assert death_cause(crushed) == "health_collapse"
