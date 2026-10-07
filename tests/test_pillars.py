from src.agents.action_space import ActionType
from src.agents import pillars


def test_every_action_has_exactly_one_pillar_or_is_fallback():
    seen = {}
    for pillar, actions in pillars.PILLAR_ACTIONS.items():
        for action in actions:
            assert action not in seen, f"{action} in due pilastri: {seen[action]} e {pillar}"
            seen[action] = pillar
    assert pillars.FALLBACK_ACTION not in seen
    covered = set(seen) | {pillars.FALLBACK_ACTION} | set(pillars.AUTOMATIC_CELL_SERVICES)
    assert covered == set(ActionType)


def test_pillar_table_matches_spec():
    P = pillars
    assert P.N_PILLARS == 6
    assert set(P.PILLAR_ACTIONS[P.P_SUSTENANCE]) == {
        ActionType.DRINK_WATER,
        ActionType.REFILL_WATER,
        ActionType.EAT_FOOD,
        ActionType.FORAGE,
        ActionType.COLLECT_ICE,
    }
    assert set(P.PILLAR_ACTIONS[P.P_RESOURCES]) == {
        ActionType.COLLECT_MINERALS, ActionType.COLLECT_MATERIALS}
    assert ActionType.MAINTAIN_STRUCTURE in P.PILLAR_ACTIONS[P.P_BUILD]
    assert len([a for a in P.PILLAR_ACTIONS[P.P_BUILD] if a.value.startswith("build_")]) == 11
    assert set(P.PILLAR_ACTIONS[P.P_LIFE]) == {
        ActionType.PHYSIOLOGICAL_RECOVERY,
        ActionType.REST,
        ActionType.USE_MED_KIT,
    }
    assert P.PILLAR_ACTIONS[P.P_SOCIAL] == ()
    assert P.AUTOMATIC_CELL_SERVICES == {
        ActionType.COMMUNICATE, ActionType.SHARE_RESOURCE}
    assert set(P.PILLAR_ACTIONS[P.P_EXPLORE]) == {
        ActionType.MOVE, ActionType.EXPLORE, ActionType.OBSERVE}


def test_action_index_is_enum_order():
    assert pillars.ACTION_ORDER == tuple(ActionType)
    assert pillars.ACTION_INDEX[ActionType.OBSERVE] == 0
    assert pillars.N_ACTIONS == len(ActionType)
