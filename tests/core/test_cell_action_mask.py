from src.agents import pillars
from src.agents.action_space import ActionType, valid_actions_for
from src.agents.rule_based_agent import RuleBasedAgent
from src.core.arrays import CellArrays
from src.core.cell_action_mask import compute_cell_masks
from src.world.resources import ResourceBundle
from src.world.world_generator import WorldGenerator


_CELL_GATED = (
    ActionType.COLLECT_ICE,
    ActionType.COLLECT_MINERALS,
    ActionType.COLLECT_MATERIALS,
    ActionType.FORAGE,
)


def test_cell_mask_matches_valid_actions_for_cell_conditions():
    world = WorldGenerator(seed=3).generate(32, 20)
    cells = CellArrays.from_world(world)
    masks = compute_cell_masks(cells)
    rich = RuleBasedAgent(
        agent_id="probe",
        name="P",
        role="colonist",
        x=0,
        y=0,
        inventory=ResourceBundle(
            energy=99,
            oxygen=99,
            water=99,
            food=99,
            construction_material=99,
            minerals=99,
            med_kits=99,
        ),
    )
    for y in range(0, world.height, 3):
        for x in range(0, world.width, 3):
            rich.x, rich.y = x, y
            valid = set(valid_actions_for(rich, world))
            for action in _CELL_GATED:
                actual = masks[y, x, pillars.ACTION_INDEX[action]]
                assert bool(actual) == (action.value in valid), f"({x},{y}) {action}"

    for action in (
        ActionType.DRINK_WATER,
        ActionType.EAT_FOOD,
        ActionType.REST,
        ActionType.USE_MED_KIT,
        ActionType.COMMUNICATE,
        ActionType.SHARE_RESOURCE,
        ActionType.MOVE,
        ActionType.EXPLORE,
        ActionType.OBSERVE,
        ActionType.DO_NOTHING,
    ):
        assert masks[:, :, pillars.ACTION_INDEX[action]].all()


def test_build_mask_blocks_saturated_and_allows_open_sites():
    from src.core import constants as C
    from src.world.structures import StructureType

    world = WorldGenerator(seed=3).generate(32, 20)
    cells = CellArrays.from_world(world)
    y, x = 5, 5
    cells.occupancy[y, x] = 1
    cells.struct_count[y, x, C.S[StructureType.GREENHOUSE]] = 1
    cells.struct_integrity[y, x, C.S[StructureType.GREENHOUSE]] = 1.0
    cells.site_progress[y, x, C.S[StructureType.SOLAR_ARRAY]] = 0.5
    masks = compute_cell_masks(cells)
    assert not masks[y, x, pillars.ACTION_INDEX[ActionType.BUILD_GREENHOUSE]]
    assert masks[y, x, pillars.ACTION_INDEX[ActionType.BUILD_SOLAR_ARRAY]]
