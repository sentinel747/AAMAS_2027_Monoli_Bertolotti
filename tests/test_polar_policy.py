from src.agents.action_space import ActionType
from src.agents.action_space import ActionRequest, validate_action
from src.agents.population import spawn_initial_agents
from src.world.perception import observe
from src.world.structures import StructureType
from src.world.terrain import traversal_risk_for_cell
from src.world.world_generator import WorldGenerator


def test_generated_polar_caps_are_environmentally_costly():
    world = WorldGenerator(seed=0).generate(50, 50)
    cap = world.get_cell(25, 0)
    mid = world.get_cell(25, 25)
    cap_band = [world.get_cell(x, y) for y in (0, 1, 48, 49) for x in range(world.width)]
    mid_band = [world.get_cell(x, y) for y in range(20, 30) for x in range(world.width)]

    assert cap.polar_severity == 1.0
    assert cap.local_temperature_modifier < mid.local_temperature_modifier - 15.0
    assert traversal_risk_for_cell(cap) > traversal_risk_for_cell(mid)
    assert sum(c.habitability_score for c in cap_band) / len(cap_band) < sum(c.habitability_score for c in mid_band) / len(mid_band)


def test_world_has_sparse_nonpolar_buried_ice_without_matching_polar_supply():
    world = WorldGenerator(seed=0).generate(50, 50)

    # **Il confronto e' PER CELLA, non sul totale (2026-09-01).** Dal fondo
    # idrico sotterraneo ogni cella ha dell'acqua, quindi il totale extrapolare
    # non e' piu' una frazione trascurabile di quello polare — ed e' corretto
    # cosi': il regolito marziano trattiene acqua ovunque, e Curiosity ne ha
    # misurata circa il 2% in peso all'equatore. Cio' che il modello deve
    # continuare a dire e' che la CALOTTA e' la grande riserva: una sua cella
    # vale molte volte una cella qualunque.
    polari = sorted(
        cell.water_ice + cell.resources.ice
        for row in world.cells for cell in row if cell.polar_severity > 0.0
    )
    extrapolari = sorted(
        cell.water_ice + cell.resources.ice
        for row in world.cells for cell in row if cell.polar_severity == 0.0
    )

    assert polari and extrapolari
    assert all(v > 0.0 for v in extrapolari), "ogni cella ha del ghiaccio sotto"
    mediana_polare = polari[len(polari) // 2]
    mediana_extra = extrapolari[len(extrapolari) // 2]
    assert mediana_polare > mediana_extra * 3.0, (
        f"calotta {mediana_polare:.2f} contro fascia libera {mediana_extra:.2f}"
    )


def test_rule_based_agent_does_not_open_with_advanced_construction():
    world = WorldGenerator(seed=0).generate(50, 50)
    agents = spawn_initial_agents(1, world, seed=0)
    agent = next(iter(agents.values()))
    world.step = 1
    observation = observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents)

    request = agent.decide(observation, world)

    assert request.action not in {
        ActionType.BUILD_SOLAR_ARRAY,
        ActionType.BUILD_RESEARCH_LAB,
        ActionType.BUILD_GREENHOUSE,
        ActionType.BUILD_HABITAT,
        ActionType.BUILD_WEATHER_STATION,
        ActionType.BUILD_STORAGE_DEPOT,
    }


def test_rule_based_water_search_avoids_hard_polar_cap_until_emergency():
    world = WorldGenerator(seed=0).generate(50, 50)
    agents = spawn_initial_agents(1, world, seed=0)
    agent = next(iter(agents.values()))
    world.step = 12
    agent.x = 25
    agent.y = 5
    agent.inventory.water = 0.0
    agent.inventory.ice = 1.0
    world.place_agent(agent.agent_id, agent.x, agent.y)
    observation = observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents)

    request = agent.decide(observation, world)

    if request.action == ActionType.MOVE:
        target = world.get_cell(request.target["x"], request.target["y"])
        assert target.polar_severity <= 0.45


def test_planned_building_is_blocked_on_unprepared_polar_cells():
    world = WorldGenerator(seed=0).generate(50, 50)
    agents = spawn_initial_agents(1, world, seed=0)
    agent = next(iter(agents.values()))
    world.remove_agent(agent.agent_id, agent.x, agent.y)
    agent.x = 25
    agent.y = 0
    agent.role = "engineer"
    world.place_agent(agent.agent_id, agent.x, agent.y)
    world.step = 10
    observation = observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents)

    request = agent.decide(observation, world)

    assert request.action not in {ActionType.BUILD_SOLAR_ARRAY, ActionType.BUILD_HEATER}
    assert StructureType.SOLAR_ARRAY not in {s.type for s in world.get_cell(agent.x, agent.y).structures}


def test_build_validation_allows_density_as_area_scaled_feedback():
    world = WorldGenerator(seed=0).generate(12, 12)
    agents = spawn_initial_agents(1, world, seed=0)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.construction_sites["shelter"] = 50.0
    cell.construction_sites["oxygen_plant"] = 20.0

    agent.inventory.construction_material = 10
    agent.inventory.minerals = 10
    result = validate_action(agent, world, ActionRequest(agent.agent_id, ActionType.BUILD_SOLAR_ARRAY))

    assert result.accepted


def test_move_validation_allows_dense_target_cell_as_risk_feedback():
    world = WorldGenerator(seed=0).generate(12, 12)
    agents = spawn_initial_agents(16, world, seed=0)
    agent = agents["agent_000"]
    tx, ty = agent.x + 1, agent.y
    for other in list(agents.values())[1:]:
        world.remove_agent(other.agent_id, other.x, other.y)
        other.x = tx
        other.y = ty
        world.place_agent(other.agent_id, tx, ty)

    result = validate_action(agent, world, ActionRequest(agent.agent_id, ActionType.MOVE, {"x": tx, "y": ty}))

    assert result.accepted
