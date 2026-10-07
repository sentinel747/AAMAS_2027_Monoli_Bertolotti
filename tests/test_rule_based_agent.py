from src.agents.action_space import ActionType
from src.agents.rule_based_agent import RuleBasedAgent
from src.world.perception import observe
from src.world.resources import ResourceBundle
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _observe(agent, world, agents=None):
    return observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents or {agent.agent_id: agent})


def test_rule_based_agent_prioritizes_medical_survival():
    world = WorldGenerator(seed=10).generate(8, 8)
    agent = RuleBasedAgent("a", "Survivor", "colonist", 4, 4)
    agent.health = 0.40
    world.place_agent(agent.agent_id, agent.x, agent.y)

    request = agent.decide(_observe(agent, world), world)

    assert request.action == ActionType.USE_MED_KIT


def test_rule_based_agent_builds_greenhouse_for_colony_food():
    world = WorldGenerator(seed=11).generate(8, 8)
    agent = RuleBasedAgent("b", "Bio", "biologist", 4, 4)
    agent.inventory = ResourceBundle(construction_material=5, minerals=4, ice=2, water=4, food=2, oxygen=2, energy=2, med_kits=1)
    world.place_agent(agent.agent_id, agent.x, agent.y)

    request = agent.decide(_observe(agent, world), world)

    assert request.action == ActionType.BUILD_GREENHOUSE


def test_rule_based_agent_can_build_extra_water_support_in_same_large_cell():
    world = WorldGenerator(seed=17).generate(8, 8)
    world.step = 10
    agent = RuleBasedAgent("planner", "Water Planner", "colonist", 4, 4)
    agent.local_x_m = 1_000.0
    agent.local_y_m = 1_000.0
    agent.inventory = ResourceBundle(construction_material=5, minerals=3, oxygen=1, water=5, food=4, med_kits=1)
    agents = {agent.agent_id: agent}
    world.place_agent(agent.agent_id, agent.x, agent.y)
    cell = world.get_cell(agent.x, agent.y)
    cell.water_ice = 0.0
    cell.resources.ice = 0.0
    cell.agent_positions_m[agent.agent_id] = {"x": agent.local_x_m, "y": agent.local_y_m}
    for index in range(9):
        other = RuleBasedAgent(f"other_{index}", f"Other {index}", "colonist", agent.x, agent.y)
        agents[other.agent_id] = other
        world.place_agent(other.agent_id, other.x, other.y)
    world.add_structure(Structure(StructureType.HABITAT, agent.x, agent.y, local_x_m=8_000.0, local_y_m=8_000.0))
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y, local_x_m=8_500.0, local_y_m=8_500.0))

    request = agent.decide(_observe(agent, world, agents), world)

    # Col costo della serra in acqua (non piu' ghiaccio mai posseduto) il piano
    # idrico sceglie direttamente la serra: e' il supporto idrico piu' mirato.
    # NB: la formula colony-wide della previsione e' deliberatamente
    # pessimista a scala — la variante pro capite e' stata provata e RITIRATA
    # nell'audit run5 (serre 68->32, ISRU 0.53->0.33 senza sbloccare le
    # spedizioni): e' il motore edilizio di fatto delle serre.
    assert request.action == ActionType.BUILD_GREENHOUSE
    assert "water" in request.message


def test_rule_based_engineer_respects_local_structure_limit_when_core_life_support_exists():
    world = WorldGenerator(seed=12).generate(8, 8)
    agent = RuleBasedAgent("c", "Builder", "colonist", 4, 4)
    agent.inventory = ResourceBundle(construction_material=8, minerals=6, oxygen=2, energy=3, ice=2, food=2, med_kits=1)
    world.place_agent(agent.agent_id, agent.x, agent.y)
    world.add_structure(Structure(StructureType.SHELTER, agent.x, agent.y))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, agent.x, agent.y))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, agent.x, agent.y))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, agent.x, agent.y))
    world.add_structure(Structure(StructureType.GREENHOUSE, agent.x, agent.y))

    request = agent.decide(_observe(agent, world), world)

    assert request.action != ActionType.BUILD_HABITAT


def test_rule_based_agent_shares_resources_with_critical_neighbor():
    world = WorldGenerator(seed=13).generate(8, 8)
    helper = RuleBasedAgent("helper", "Helper", "colonist", 4, 4)
    neighbor = RuleBasedAgent("neighbor", "Neighbor", "colonist", 4, 5)
    neighbor.health = 0.30
    helper.inventory.med_kits = 2
    agents = {helper.agent_id: helper, neighbor.agent_id: neighbor}
    world.place_agent(helper.agent_id, helper.x, helper.y)
    world.place_agent(neighbor.agent_id, neighbor.x, neighbor.y)

    request = helper.decide(_observe(helper, world, agents), world)

    assert request.action == ActionType.SHARE_RESOURCE
    assert request.target["agent_id"] == neighbor.agent_id
    assert request.target["resources"]["med_kits"] == 1.0


def test_rule_based_agent_moves_toward_polar_ice_before_water_runs_out():
    world = WorldGenerator(seed=15).generate(50, 50)
    agent = RuleBasedAgent("water", "Water Planner", "explorer", 25, 25)
    agent.inventory.water = 1.0
    agent.inventory.ice = 0.0
    world.place_agent(agent.agent_id, agent.x, agent.y)
    # **La cella di partenza va prosciugata (2026-09-01).** Da quando ogni cella
    # ha un fondo idrico sotterraneo, un colono assetato beve o scava dove si
    # trova e non ha alcun motivo di viaggiare: la prova misurerebbe il fondo
    # idrico invece del rientro d'emergenza che dichiara di misurare.
    _qui = world.get_cell(agent.x, agent.y)
    _qui.water_ice = 0.0
    _qui.resources.ice = 0.0
    _qui.liquid_water = 0.0

    request = agent.decide(_observe(agent, world), world)

    assert request.action == ActionType.MOVE
    assert abs(request.target["y"] - 0) < abs(agent.y - 0) or abs(request.target["y"] - (world.height - 1)) < abs(agent.y - (world.height - 1))
    assert "ice" in request.message


def test_rule_based_agent_breaks_dehydration_streak_before_moving():
    world = WorldGenerator(seed=16).generate(8, 8)
    agent = RuleBasedAgent("drink", "Water Discipline", "colonist", 4, 4)
    agent.inventory.water = 0.0
    agent.inventory.ice = 1.0
    agent.steps_without_water = 3
    agent.recent_actions.extend([ActionType.MOVE.value, ActionType.MOVE.value, ActionType.MOVE.value])
    world.place_agent(agent.agent_id, agent.x, agent.y)

    request = agent.decide(_observe(agent, world), world)

    assert request.action == ActionType.DRINK_WATER


def test_rule_based_agent_communicates_with_contextual_phrase_when_near_neighbor():
    world = WorldGenerator(seed=14).generate(8, 8)
    speaker = RuleBasedAgent("speaker", "Speaker", "colonist", 4, 4)
    neighbor = RuleBasedAgent("neighbor", "Neighbor", "colonist", 4, 5)
    world.place_agent(speaker.agent_id, speaker.x, speaker.y)
    world.place_agent(neighbor.agent_id, neighbor.x, neighbor.y)
    agents = {speaker.agent_id: speaker, neighbor.agent_id: neighbor}
    world.add_structure(Structure(StructureType.SHELTER, speaker.x, speaker.y))
    world.add_structure(Structure(StructureType.SOLAR_ARRAY, speaker.x, speaker.y))
    world.add_structure(Structure(StructureType.OXYGEN_PLANT, speaker.x, speaker.y))
    # +pozzo (2026-09-01): senza acqua la cella non ha capienza vitale
    world.add_structure(Structure(StructureType.WATER_EXTRACTOR, speaker.x, speaker.y))
    world.add_structure(Structure(StructureType.GREENHOUSE, speaker.x, speaker.y))
    world.add_structure(Structure(StructureType.HABITAT, speaker.x, speaker.y))
    speaker.inventory = ResourceBundle(food=4, minerals=0, construction_material=0, ice=0, med_kits=1)

    request = speaker.decide(_observe(speaker, world, agents), world)

    assert request.action == ActionType.COMMUNICATE
    assert request.message
