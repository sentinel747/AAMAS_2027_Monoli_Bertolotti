"""Integrated regressions for the 260726 frontier-starvation failure."""

import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionRequest, ActionType, execute_action
from src.agents.population import spawn_initial_agents
from src.agents.preference_agent import decide_preferences
from src.agents.vitals import tick_agent_vitals
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _place_agent(world, agent, x: int, y: int) -> None:
    world.get_cell(agent.x, agent.y).agents_present.remove(agent.agent_id)
    agent.x, agent.y = x, y
    cell = world.get_cell(x, y)
    agent.local_x_m = float(cell.geometry["width_m"]) * 0.5
    agent.local_y_m = float(cell.geometry["height_m"]) * 0.5
    cell.agents_present.append(agent.agent_id)


def test_execute_move_crosses_359_to_zero_with_59km_range():
    world = WorldGenerator(seed=101).generate(360, 180)
    agents = spawn_initial_agents(1, world, seed=101)
    agent = next(iter(agents.values()))
    _place_agent(world, agent, 359, 80)
    agent.movement_distance_m_per_step = 59_000.0

    result = execute_action(
        agent,
        agents,
        world,
        ActionRequest(
            agent.agent_id,
            ActionType.MOVE,
            {"x": 0, "y": 80},
            "crossing antimeridian",
        ),
    )

    assert result.accepted
    assert (agent.x, agent.y) == (0, 80)


def test_food_budget_return_survives_wrapped_trip_to_greenhouse():
    world = WorldGenerator(seed=102).generate(360, 180)
    agents = spawn_initial_agents(1, world, seed=102)
    agent = next(iter(agents.values()))
    _place_agent(world, agent, 348, 80)
    world.add_structure(Structure(StructureType.GREENHOUSE, 0, 80))
    agent.movement_distance_m_per_step = 59_000.0
    agent.inventory.food = 2.5
    agent.inventory.water = 8.0
    agent.inventory.ice = 0.0
    agent.satiety = 1.0
    agent.hydration = 1.0
    agent.health = 1.0
    mask = np.ones(pillars.N_ACTIONS, dtype=np.bool_)

    visited_x = []
    for step in range(1, 27):
        world.step = step
        request = decide_preferences(
            agent, world, mask, 0.99, "softmax", {}, step
        )
        assert request.action == ActionType.MOVE
        result = execute_action(agent, agents, world, request)
        assert result.accepted
        visited_x.append(agent.x)
        tick_agent_vitals(agent, world.get_cell(agent.x, agent.y), 7.0)
        assert agent.health > 0.0
        if agent.x == 0:
            break

    assert all(
        current in {previous, (previous + 1) % world.width}
        for previous, current in zip([348, *visited_x[:-1]], visited_x)
    )
    assert agent.x == 0
    assert agent.inventory.food >= 0.1 - 1.0e-9
