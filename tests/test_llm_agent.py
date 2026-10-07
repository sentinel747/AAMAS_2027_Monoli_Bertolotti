from src.agents.llm_agent import LLMAgent
from src.llm.provider import LLMResponse
from src.world.perception import observe
from src.world.world_generator import WorldGenerator


class CountingProvider:
    provider_id = "test"
    model = "test-model"

    def __init__(self):
        self.calls = 0

    def complete_json(self, prompt, schema_hint=None):
        self.calls += 1
        return LLMResponse(
            text='{"thought_summary":"observe once","public_message":"","chosen_action":"observe","target":{"type":"self"}}',
            provider="test",
            api_call_attempted=True,
        )


def test_llm_agent_calls_provider_once_per_world_step():
    world = WorldGenerator(seed=1).generate(8, 8)
    provider = CountingProvider()
    agent = LLMAgent("agent_001", "Test Agent", "colonist", 4, 4, provider=provider)
    agents = {agent.agent_id: agent}
    world.place_agent(agent.agent_id, agent.x, agent.y)
    world.step = 3
    observation = observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents)

    agent.decide(observation, world)
    agent.decide(observation, world)
    world.step = 4
    agent.decide(observation, world)

    assert provider.calls == 2
    assert any(event["type"] == "llm_step_call_skipped" for event in world.events)
