import math

from src.agents.action_space import (
    ICE_COLLECTION_YIELD_PER_ACTION,
    ActionRequest,
    ActionType,
    execute_action,
)
from src.agents.action_space import valid_actions_for
from src.agents.population import spawn_initial_agents
from src.world.resources import ResourceBundle
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def test_agent_can_move_and_collect():
    world = WorldGenerator(seed=2).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=2)
    agent = next(iter(agents.values()))
    result = execute_action(agent, agents, world, ActionRequest(agent.agent_id, ActionType.MOVE, {"x": agent.x + 1, "y": agent.y}))
    assert result.accepted
    cell = world.get_cell(agent.x, agent.y)
    cell.resources.minerals = 1
    result = execute_action(agent, agents, world, ActionRequest(agent.agent_id, ActionType.COLLECT_MINERALS))
    assert result.accepted


def test_collect_ice_is_atomic_and_never_opens_a_construction_site():
    world = WorldGenerator(seed=202).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=202)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.water_ice = 2.0
    before_inventory = agent.inventory.ice

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.COLLECT_ICE),
    )

    assert result.accepted
    assert math.isclose(
        agent.inventory.ice - before_inventory,
        ICE_COLLECTION_YIELD_PER_ACTION,
    )
    assert math.isclose(cell.water_ice, 2.0 - ICE_COLLECTION_YIELD_PER_ACTION)
    assert cell.construction_sites == {}


def test_collect_ice_recovers_and_clears_legacy_progress():
    world = WorldGenerator(seed=203).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=203)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.water_ice = 5.0
    cell.construction_sites["ice_extraction"] = 75.0
    before_inventory = agent.inventory.ice

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.COLLECT_ICE),
    )

    assert result.accepted
    assert math.isclose(agent.inventory.ice - before_inventory, 1.6)
    assert result.data["legacy_progress_recovered"] == 75.0
    assert "ice_extraction" not in cell.construction_sites


def test_explore_marks_the_reached_cell_as_explored():
    world = WorldGenerator(seed=21).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=21)
    agent = next(iter(agents.values()))
    target_x = (agent.x + 1) % world.width

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.EXPLORE, {"x": target_x, "y": agent.y}),
    )

    assert result.accepted
    assert world.get_cell(agent.x, agent.y).explored


def test_move_and_explore_accept_wrapped_longitude_neighbor():
    world = WorldGenerator(seed=22).generate(12, 8)
    agents = spawn_initial_agents(1, world, seed=22, start_x=0, start_y=4)
    agent = next(iter(agents.values()))
    agent.local_x_m = 0.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(
            agent.agent_id,
            ActionType.EXPLORE,
            {"x": world.width - 1, "y": agent.y},
        ),
    )

    assert result.accepted
    assert agent.x == world.width - 1
    assert world.get_cell(agent.x, agent.y).explored


def test_operational_range_drives_perception_and_movement_together():
    def configured_agent(range_m: float):
        world = WorldGenerator(seed=12).generate(10, 10)
        agents = spawn_initial_agents(
            1,
            world,
            seed=12,
            operational_range_m=range_m,
        )
        agent = next(iter(agents.values()))
        return agent

    short = configured_agent(100.0)
    full_cell = configured_agent(59_000.0)
    assert short.perception_radius_m == short.movement_distance_m_per_step == 100.0
    assert full_cell.perception_radius_m == full_cell.movement_distance_m_per_step == 59_000.0


def test_spawn_clamps_operational_range_to_public_range():
    low_world = WorldGenerator(seed=13).generate(10, 10)
    low_agent = next(
        iter(
            spawn_initial_agents(
                1, low_world, seed=13, operational_range_m=1.0
            ).values()
        )
    )
    high_world = WorldGenerator(seed=14).generate(10, 10)
    high_agent = next(
        iter(
            spawn_initial_agents(
                1, high_world, seed=14, operational_range_m=120_000.0
            ).values()
        )
    )

    assert low_agent.movement_distance_m_per_step == 100.0
    assert low_agent.perception_radius_m == 100.0
    assert high_agent.movement_distance_m_per_step == 59_000.0
    assert high_agent.perception_radius_m == 59_000.0


def test_raw_surface_biomass_is_not_immediately_edible_food():
    world = WorldGenerator(seed=3).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=3)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    cell.vegetation_biomass = 1.0
    cell.proto_soil_development = 0.0

    assert ActionType.FORAGE.value not in valid_actions_for(agent, world)


def test_build_actions_are_hidden_only_for_same_local_structure_position():
    world = WorldGenerator(seed=4).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=4)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    agent.local_x_m = 1_000.0
    agent.local_y_m = 1_000.0
    cell.agent_positions_m[agent.agent_id] = {"x": agent.local_x_m, "y": agent.local_y_m}

    # Enough colonists that one distant shelter does not yet cover the need
    # (shelter capacity is population-gated by build_policy.structure_saturated).
    cell.agents_present.extend(f"ghost_{i}" for i in range(6))
    cell.structures.append(Structure(StructureType.SHELTER, agent.x, agent.y, local_x_m=8_000.0, local_y_m=8_000.0))
    assert ActionType.BUILD_SHELTER.value in valid_actions_for(agent, world)

    cell.structures.append(Structure(StructureType.SHELTER, agent.x, agent.y, local_x_m=1_050.0, local_y_m=1_050.0))
    assert ActionType.BUILD_SHELTER.value not in valid_actions_for(agent, world)


def test_nearby_agents_resume_same_local_construction_site_without_paying_again():
    world = WorldGenerator(seed=5).generate(10, 10)
    agents = spawn_initial_agents(2, world, seed=5)
    agent_a, agent_b = list(agents.values())
    agent_b.x = agent_a.x
    agent_b.y = agent_a.y
    world.place_agent(agent_b.agent_id, agent_b.x, agent_b.y)
    cell = world.get_cell(agent_a.x, agent_a.y)
    agent_a.local_x_m = 1_000.0
    agent_a.local_y_m = 1_000.0
    agent_b.local_x_m = 1_120.0
    agent_b.local_y_m = 1_040.0
    cell.agent_positions_m[agent_a.agent_id] = {"x": agent_a.local_x_m, "y": agent_a.local_y_m}
    cell.agent_positions_m[agent_b.agent_id] = {"x": agent_b.local_x_m, "y": agent_b.local_y_m}
    agent_a.inventory = ResourceBundle(construction_material=3, minerals=1, water=1)
    agent_b.inventory = ResourceBundle()

    first = execute_action(agent_a, agents, world, ActionRequest(agent_a.agent_id, ActionType.BUILD_GREENHOUSE))
    second = execute_action(agent_b, agents, world, ActionRequest(agent_b.agent_id, ActionType.BUILD_GREENHOUSE))

    assert first.accepted
    assert second.accepted
    assert cell.construction_sites == {}
    assert [s.type for s in cell.structures] == [StructureType.GREENHOUSE]


def test_different_structure_types_receive_distinct_nearby_sites():
    world = WorldGenerator(seed=51).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=51)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    agent.local_x_m = 10_000.0
    agent.local_y_m = 10_000.0
    cell.agent_positions_m[agent.agent_id] = {
        "x": agent.local_x_m,
        "y": agent.local_y_m,
    }
    agent.inventory = ResourceBundle(
        construction_material=20,
        minerals=20,
        water=5,
        energy=5,
    )

    greenhouse = ActionRequest(agent.agent_id, ActionType.BUILD_GREENHOUSE)
    solar = ActionRequest(agent.agent_id, ActionType.BUILD_SOLAR_ARRAY)
    assert execute_action(agent, agents, world, greenhouse).accepted
    assert execute_action(agent, agents, world, greenhouse).accepted
    assert execute_action(agent, agents, world, solar).accepted
    assert execute_action(agent, agents, world, solar).accepted

    assert len(cell.structures) == 2
    first, second = cell.structures
    distance = ((first.local_x_m - second.local_x_m) ** 2 + (first.local_y_m - second.local_y_m) ** 2) ** 0.5
    assert distance > 750.0
    assert {first.type, second.type} == {
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
    }


def test_physiological_recovery_restores_body_and_enabled_psychology():
    world = WorldGenerator(seed=15).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=15)
    agent = next(iter(agents.values()))
    world.metadata["individual_survival_priority_enabled"] = False
    world.metadata["psychosocial_enabled"] = True
    agent.health = 0.2
    agent.satiety = 0.3
    agent.hydration = 0.1
    agent.oxygen_level = 0.4
    agent.fatigue = 0.9
    agent.stress_index = 0.8
    agent.morale = 0.2
    agent.steps_without_water = 3
    agent.steps_without_food = 8
    # Dal 2026-08-25 il ripristino costa una razione di cibo, acqua e ossigeno:
    # il magazzino della cella deve averle, altrimenti non avviene.
    cella = world.get_cell(agent.x, agent.y)
    cella.resources.food = 1.0
    cella.resources.water = 1.0
    cella.resources.oxygen = 1.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.PHYSIOLOGICAL_RECOVERY),
    )

    assert result.accepted
    assert agent.health == agent.satiety == agent.hydration == 1.0
    assert agent.oxygen_level == 1.0
    assert agent.fatigue == agent.stress_index == 0.0
    assert agent.morale == 1.0
    assert agent.steps_without_water == agent.steps_without_food == 0


def test_il_ripristino_fisiologico_senza_scorte_non_avviene():
    """Era una guarigione completa dal nulla, e annullava le razioni.

    Nella modalita' in cui e' disponibile (`individual_survival_priority_enabled`
    spento) l'azione portava tutti i vitali a 1,0 e azzerava i contatori di
    privazione senza costo ne' massa: qualunque scarsita' era aggirabile.
    """
    world = WorldGenerator(seed=15).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=15)
    agent = next(iter(agents.values()))
    world.metadata["individual_survival_priority_enabled"] = False
    agent.health = 0.2
    agent.satiety = 0.3
    agent.inventory.food = 0.0
    agent.inventory.water = 0.0
    agent.inventory.oxygen = 0.0
    cella = world.get_cell(agent.x, agent.y)
    cella.resources.food = 0.0
    cella.resources.water = 0.0
    cella.resources.oxygen = 0.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.PHYSIOLOGICAL_RECOVERY),
    )

    assert not result.accepted
    assert agent.health == 0.2
    assert agent.satiety == 0.3


def test_move_uses_real_continuous_target_and_respects_metric_range():
    world = WorldGenerator(seed=202).generate(360, 180)
    agents = spawn_initial_agents(
        1,
        world,
        seed=202,
        start_x=180,
        start_y=90,
        operational_range_m=1_000.0,
    )
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    agent.local_x_m = float(cell.geometry["width_m"]) * 0.5
    agent.local_y_m = float(cell.geometry["height_m"]) * 0.5
    request = ActionRequest(
        agent.agent_id,
        ActionType.MOVE,
        {"x": agent.x, "y": agent.y, "local_x_m": 0.0, "local_y_m": 0.0},
    )

    result = execute_action(agent, agents, world, request)

    assert result.accepted
    assert 999.999 <= result.data["distance_m"] <= 1_000.001
    assert math.hypot(
        agent.local_x_m - float(cell.geometry["width_m"]) * 0.5,
        agent.local_y_m - float(cell.geometry["height_m"]) * 0.5,
    ) <= 1_000.001
    assert request.target["local_x_m"] == 0.0
    assert request.target["local_y_m"] == 0.0


def test_generated_move_destinations_are_seeded_continuous_and_not_cell_centres():
    def destination(agent_id: str):
        world = WorldGenerator(seed=203).generate(360, 180)
        agents = spawn_initial_agents(1, world, seed=203, start_x=180, start_y=90)
        agent = next(iter(agents.values()))
        agent.agent_id = agent_id
        agents = {agent_id: agent}
        world.step = 17
        request = ActionRequest(agent_id, ActionType.MOVE, {"x": agent.x, "y": agent.y})
        result = execute_action(agent, agents, world, request)
        return request.target, result.data

    target_a1, result_a1 = destination("agent_a")
    target_a2, result_a2 = destination("agent_a")
    target_b, _result_b = destination("agent_b")

    assert target_a1 == target_a2
    assert result_a1 == result_a2
    assert target_a1 != target_b
    assert target_a1["local_x_m"] != 29_500.0
    assert target_a1["local_y_m"] != 29_500.0
    assert 0.0 <= result_a1["local_x_m"] <= 59_000.0
    assert 0.0 <= result_a1["local_y_m"] <= 59_000.0


def test_physiological_recovery_leaves_legacy_psychology_untouched_when_off():
    world = WorldGenerator(seed=16).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=16)
    agent = next(iter(agents.values()))
    world.metadata["individual_survival_priority_enabled"] = False
    world.metadata["psychosocial_enabled"] = False
    agent.stress_index = 0.8
    agent.morale = 0.2

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.PHYSIOLOGICAL_RECOVERY),
    )

    assert result.accepted
    assert agent.stress_index == 0.8
    assert agent.morale == 0.2


def test_physiological_recovery_is_hidden_and_rejected_in_individual_mode():
    world = WorldGenerator(seed=17).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=17)
    agent = next(iter(agents.values()))

    assert ActionType.PHYSIOLOGICAL_RECOVERY.value not in valid_actions_for(
        agent, world
    )
    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(agent.agent_id, ActionType.PHYSIOLOGICAL_RECOVERY),
    )
    assert not result.accepted
