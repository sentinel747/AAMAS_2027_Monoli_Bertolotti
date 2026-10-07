from src.agents.action_space import ActionRequest, ActionType, execute_action, validate_action
from src.agents.population import spawn_initial_agents
from src.world.resources import ResourceBundle
from src.world.world_generator import WorldGenerator


def test_invalid_far_move_rejected():
    world = WorldGenerator(seed=3).generate(10, 10)
    agents = spawn_initial_agents(1, world, seed=3)
    agent = next(iter(agents.values()))
    result = validate_action(agent, world, ActionRequest(agent.agent_id, ActionType.MOVE, {"x": 9, "y": 9}))
    assert not result.accepted


def test_share_resource_requires_proximity():
    world = WorldGenerator(seed=4).generate(20, 20)
    agents = spawn_initial_agents(2, world, seed=4)
    first, second = list(agents.values())
    world.move_agent(second.agent_id, second.x, second.y, 19, 19)
    second.x, second.y = 19, 19
    request = ActionRequest(first.agent_id, ActionType.SHARE_RESOURCE, {"agent_id": second.agent_id, "resources": {"minerals": 1}})

    result = validate_action(first, world, request)

    assert not result.accepted


def test_nearby_resource_share_is_executable_action():
    world = WorldGenerator(seed=5).generate(10, 10)
    agents = spawn_initial_agents(2, world, seed=5)
    first, second = list(agents.values())
    second.x, second.y = first.x, first.y
    world.place_agent(second.agent_id, second.x, second.y)
    first.inventory = ResourceBundle(minerals=2)
    request = ActionRequest(first.agent_id, ActionType.SHARE_RESOURCE, {"agent_id": second.agent_id, "resources": {"minerals": 1}})

    result = execute_action(first, agents, world, request)

    assert result.accepted
    assert second.inventory.minerals >= 1
