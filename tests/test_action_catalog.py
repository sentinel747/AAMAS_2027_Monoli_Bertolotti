from src.agents.action_space import ActionType, valid_actions_for
from src.agents.decision_schema import VALID_ACTIONS
from src.agents.llm_agent import LLMAgent
from src.agents.rule_based_agent import RuleBasedAgent
from src.world.resources import ResourceBundle
from src.world.world_generator import WorldGenerator


def test_llm_schema_uses_same_action_catalog_as_simulator():
    assert VALID_ACTIONS == {action.value for action in ActionType}


def test_rule_based_and_llm_agents_receive_same_available_actions_when_state_matches():
    world = WorldGenerator(seed=21).generate(8, 8)
    rule_agent = RuleBasedAgent("rule", "Rule", "colonist", 4, 4)
    llm_agent = LLMAgent("llm", "LLM", "colonist", 4, 4)
    inventory = ResourceBundle(
        ice=5,
        minerals=5,
        energy=5,
        oxygen=5,
        food=5,
        construction_material=8,
        med_kits=2,
    )
    rule_agent.inventory = inventory
    llm_agent.inventory = ResourceBundle.from_dict(inventory.to_dict())
    rule_agent.health = llm_agent.health = 1.0

    assert set(valid_actions_for(rule_agent, world)) == set(valid_actions_for(llm_agent, world))
