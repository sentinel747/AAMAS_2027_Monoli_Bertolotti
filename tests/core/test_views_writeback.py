"""Write-back tests for the read-write facades (src/core/views.py).

Task 7: `AgentView`/`CellView` stop being read-only snapshots and become
proxies backed directly by `AgentArrays`/`CellArrays`, so that
`src/agents/action_space.py` (`execute_action`), which mutates state in place
(`cell.water_ice -= ...`, `agent.inventory.ice += ...`, ...), runs completely
unmodified against array-backed state. The integration tests at the bottom
are the ones that prove this: they run the SAME actions through the real
object engine (`BaseAgent`/`GridWorld`) and through the array/view engine
(`AgentArrays`/`CellArrays`/`AgentView`/`CellView`/`WorldView`) from
identical starting state and assert the two end up equal.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.agents.action_space import ActionRequest, ActionType, execute_action
from src.agents.base_agent import BaseAgent
from src.agents.rule_based_agent import RuleBasedAgent
from src.core import constants as C
from src.core.arrays import (
    AgentArrays,
    CellArrays,
    MISSION_SCOUT_BACK,
    MISSION_SCOUT_OUT,
    MISSION_SETTLE_OUT,
)
from src.core.views import AgentSideState, AgentView, CellView, WorldView
from src.simulation.extreme_events import ExtremeEventEngine
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.mars_geometry import cell_geometry
from src.world.resources import ResourceBundle
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType


# ---------------------------------------------------------------------------
# Step 1: write-back unit tests (proxies in isolation)
# ---------------------------------------------------------------------------


def _agent_view(**agent_kwargs) -> tuple[BaseAgent, AgentArrays, AgentView]:
    agent = BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1, **agent_kwargs)
    aa = AgentArrays.from_agents({"a0": agent})
    side = AgentSideState.from_agent(agent)
    return agent, aa, AgentView(aa, aa.index["a0"], side)


def _cell_view() -> tuple[CellArrays, CellView]:
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(3)] for y in range(3)]
    world = GridWorld(width=3, height=3, cells=cells)
    ca = CellArrays.from_world(world)
    return ca, CellView(ca, 1, 1, world.planetary_state)


def test_inventory_proxy_writes_through_to_arrays():
    _, aa, av = _agent_view()
    row = av.row
    av.inventory.ice += 2.5
    assert np.isclose(aa.inv[row, C.R["ice"]], 2.5)
    av.inventory.water = 1.0
    assert np.isclose(av.inventory.water, 1.0)
    assert np.isclose(aa.inv[row, C.R["water"]], 1.0)

    expected_keys = set(ResourceBundle().to_dict().keys())
    assert set(av.inventory.to_dict().keys()) == expected_keys


def test_cell_resources_proxy_writes_through():
    ca, cv = _cell_view()
    cv.resources.minerals = 5.0
    cv.resources.minerals -= 1.0
    assert np.isclose(ca.cell_res[1, 1, C.R["minerals"]], 4.0)
    assert np.isclose(cv.resources.minerals, 4.0)


def test_can_afford_and_remove_match_resourcebundle_semantics():
    ca, cv = _cell_view()
    cv.resources.minerals = 2.0
    cv.resources.construction_material = 1.0

    bundle = ResourceBundle(minerals=2.0, construction_material=1.0)
    cost_ok = ResourceBundle(minerals=2.0, construction_material=1.0)
    cost_too_much = ResourceBundle(minerals=2.0, construction_material=1.01)

    assert cv.resources.can_afford(cost_ok) is True
    assert bundle.can_afford(cost_ok) is True
    assert cv.resources.can_afford(cost_too_much) is False
    assert bundle.can_afford(cost_too_much) is False

    # remove() must be all-or-nothing on both sides, and leave state unchanged on failure.
    assert cv.resources.remove(cost_too_much) is False
    assert bundle.remove(cost_too_much) is False
    assert np.isclose(cv.resources.minerals, bundle.minerals)
    assert np.isclose(cv.resources.construction_material, bundle.construction_material)

    assert cv.resources.remove(cost_ok) is True
    assert bundle.remove(cost_ok) is True
    assert np.isclose(cv.resources.minerals, bundle.minerals)
    assert np.isclose(cv.resources.construction_material, bundle.construction_material)
    assert np.isclose(cv.resources.minerals, 0.0)


def test_agent_scalar_setters_write_through():
    _, aa, av = _agent_view()
    row = av.row
    av.fatigue = 0.7
    assert np.isclose(aa.fatigue[row], 0.7)
    av.hydration -= 0.1
    assert np.isclose(aa.hydration[row], 1.0 - 0.1)
    av.x = 3
    assert aa.x[row] == 3
    av.health = 0.4
    av.satiety = 0.3
    av.oxygen_level = 0.6
    av.stress_index = 0.2
    av.morale = 0.9
    av.cooperation = 0.5
    av.protocol_compliance = 0.6
    av.autonomy_preference = 0.7
    av.steps_without_water = 4
    av.steps_without_food = 2
    assert np.isclose(aa.health[row], 0.4)
    assert np.isclose(aa.satiety[row], 0.3)
    assert np.isclose(aa.oxygen[row], 0.6)
    assert np.isclose(aa.stress[row], 0.2)
    assert np.isclose(aa.morale[row], 0.9)
    assert np.isclose(aa.cooperation[row], 0.5)
    assert np.isclose(aa.compliance[row], 0.6)
    assert np.isclose(aa.autonomy[row], 0.7)
    assert aa.steps_without_water[row] == 4
    assert aa.steps_without_food[row] == 2


def test_cell_scalar_setters_write_through():
    ca, cv = _cell_view()
    cv.water_ice -= 0.1
    cv.vegetation_biomass = 2.0
    cv.liquid_water = 1.2
    cv.pollution_risk = 0.05
    cv.habitability_score = 0.4
    cv.proto_soil_development = 0.3
    cv.organic_matter = 0.6
    cv.explored = True
    cv.radiation_level = 0.9
    cv.dust_level = 0.2
    cv.local_temperature_modifier = -5.0
    assert np.isclose(ca.water_ice[1, 1], -0.1)
    assert np.isclose(ca.vegetation[1, 1], 2.0)
    assert np.isclose(ca.liquid_water[1, 1], 1.2)
    assert np.isclose(ca.pollution[1, 1], 0.05)
    assert np.isclose(ca.habitability[1, 1], 0.4)
    assert np.isclose(ca.proto_soil[1, 1], 0.3)
    assert np.isclose(ca.organic[1, 1], 0.6)
    assert bool(ca.explored[1, 1]) is True
    assert np.isclose(ca.radiation[1, 1], 0.9)
    assert np.isclose(ca.dust[1, 1], 0.2)
    assert np.isclose(ca.temp_mod[1, 1], -5.0)


def test_cell_nutrients_proxy_write_through():
    ca, cv = _cell_view()
    cv.nutrients["N"] = 3.0
    cv.nutrients["P"] += 1.0
    assert np.isclose(ca.nut_n[1, 1], 3.0)
    assert np.isclose(ca.nut_p[1, 1], 1.0)
    assert dict(cv.nutrients) == {"N": 3.0, "P": 1.0, "C": 0.0}


def test_cell_construction_sites_dict_like_write_through():
    ca, cv = _cell_view()
    key = "greenhouse@2:3"
    assert key not in cv.construction_sites
    cv.construction_sites[key] = 0.0
    assert key in cv.construction_sites
    assert cv.construction_sites[key] == 0.0
    cv.construction_sites[key] = 25.0
    assert cv.construction_sites.get(key) == 25.0
    si = C.S[StructureType.GREENHOUSE]
    assert np.isclose(ca.site_progress[1, 1, si], 25.0)
    del cv.construction_sites[key]
    assert key not in cv.construction_sites
    assert np.isclose(ca.site_progress[1, 1, si], -1.0)


def test_cell_construction_sites_non_structural_key():
    # "ice_extraction" (used by COLLECT_ICE) has no "@" and is not a StructureType
    # value, so it does not map onto site_progress at all (see arrays.py
    # `_site_key_to_type`); the proxy must still round-trip it.
    ca, cv = _cell_view()
    assert "ice_extraction" not in cv.construction_sites
    cv.construction_sites["ice_extraction"] = 25.0
    assert cv.construction_sites["ice_extraction"] == 25.0
    del cv.construction_sites["ice_extraction"]
    assert "ice_extraction" not in cv.construction_sites


def test_agent_mission_state_defaults_to_none():
    _, _, av = _agent_view()
    assert av.scout_target is None
    assert av.scout_home is None
    assert av.scout_phase is None
    assert av.scout_steps == 0
    assert av.settle_target is None
    assert av.settle_phase is None
    assert av.settle_steps == 0
    assert av.scout_next_at == -1
    assert av.settle_next_at == -1


def test_agent_mission_state_roundtrips_scout():
    _, aa, av = _agent_view()
    av.scout_phase = "out"
    av.scout_steps = 0
    av.scout_target = (5, 6)
    av.scout_home = (1, 1)
    assert av.scout_phase == "out"
    assert av.scout_target == (5, 6)
    assert av.scout_home == (1, 1)
    assert int(aa.mission[av.row]) == MISSION_SCOUT_OUT

    av.scout_steps += 1
    assert av.scout_steps == 1

    av.scout_phase = "back"
    assert av.scout_phase == "back"
    assert int(aa.mission[av.row]) == MISSION_SCOUT_BACK

    av.scout_phase = None
    av.scout_target = None
    av.scout_home = None
    av.scout_steps = 0
    assert av.scout_phase is None
    assert av.scout_target is None
    assert av.scout_home is None


def test_agent_mission_state_roundtrips_settle():
    _, aa, av = _agent_view()
    av.settle_phase = "out"
    av.settle_target = (2, 2)
    assert av.settle_phase == "out"
    assert av.settle_target == (2, 2)
    assert int(aa.mission[av.row]) == MISSION_SETTLE_OUT
    # scout_target shares storage with settle_target but is only meaningful
    # while a scout mission is active; while settling, scout_phase reads None.
    assert av.scout_phase is None

    av.settle_phase = None
    av.settle_target = None
    assert av.settle_phase is None
    assert av.settle_target is None


def test_agent_cold_state_setters_write_through():
    agent, _, av = _agent_view()
    av.current_goal = "build a greenhouse"
    assert av.current_goal == "build a greenhouse"
    av.actions_taken = 5
    assert av.actions_taken == 5
    av.recent_actions.append("rest")
    assert av.recent_actions == ["rest"]
    av.local_x_m = 12.5
    av.local_y_m = 40.0
    assert np.isclose(av.local_x_m, 12.5)
    assert np.isclose(av.local_y_m, 40.0)
    assert av.memory is not None


def test_worldview_log_event_add_structure_and_ice_cache():
    ca, _ = _cell_view()
    aa = AgentArrays(4)
    wv = WorldView(ca, aa, {}, {})
    wv.metadata["_total_ice_cache"] = 10.0
    wv.adjust_total_ice_cache(-2.5)
    assert np.isclose(wv.metadata["_total_ice_cache"], 7.5)

    structure = Structure(type=StructureType.SHELTER, x=1, y=1, integrity=0.9)
    wv.add_structure(structure)
    si = C.S[StructureType.SHELTER]
    assert ca.struct_count[1, 1, si] == 1
    assert np.isclose(ca.struct_integrity[1, 1, si], 0.9)
    assert len(wv.events) == 1
    assert wv.events[0]["type"] == "structure_built"


# ---------------------------------------------------------------------------
# Step 3: execute_action() runs UNMODIFIED against the views and produces the
# same array state as the object engine produces in objects.
# ---------------------------------------------------------------------------


def _pair(cell_setup=None, agent_kwargs=None, structures=None):
    """Build matching (object engine, array/view engine) starting states."""
    cell_setup = cell_setup or (lambda c: None)
    agent_kwargs = agent_kwargs or {}
    structures = structures or []

    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(3)] for y in range(3)]
    cell = cells[1][1]
    cell_setup(cell)
    for s in structures:
        cell.structures.append(s)
    world = GridWorld(width=3, height=3, cells=cells)

    agent = BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1, **agent_kwargs)

    aa = AgentArrays.from_agents({"a0": agent})
    ca = CellArrays.from_world(world)
    side = AgentSideState.from_agent(agent)
    av = AgentView(aa, aa.index["a0"], side)
    wv = WorldView(
        ca,
        aa,
        dict(world.planetary_state),
        {},
        {"a0": side},
    )

    return agent, world, av, wv


def _assert_parity(agent, world, av, wv):
    cell = world.get_cell(1, 1)
    cv = wv.get_cell(1, 1)
    assert av.inventory.to_dict() == pytest.approx(agent.inventory.to_dict(), abs=1e-5)
    assert np.isclose(av.health, agent.health, atol=1e-5)
    assert np.isclose(av.satiety, agent.satiety, atol=1e-5)
    assert np.isclose(av.hydration, agent.hydration, atol=1e-5)
    assert np.isclose(av.fatigue, agent.fatigue, atol=1e-5)
    assert np.isclose(av.oxygen_level, agent.oxygen_level, atol=1e-5)
    assert cv.resources.to_dict() == pytest.approx(cell.resources.to_dict(), abs=1e-5)
    assert np.isclose(cv.water_ice, cell.water_ice, atol=1e-5)
    assert np.isclose(cv.liquid_water, cell.liquid_water, atol=1e-5)
    assert np.isclose(cv.vegetation_biomass, cell.vegetation_biomass, atol=1e-5)


def test_cell_view_exposes_live_agent_subcell_positions():
    _agent, _world, av, wv = _pair()
    av.local_x_m = 12_345.0
    av.local_y_m = 54_321.0

    cell = wv.get_cell(av.x, av.y)

    assert cell.agent_positions_m == {
        "a0": {"x": 12_345.0, "y": 54_321.0}
    }
    assert cell.to_public_dict()["agent_positions_m"] == cell.agent_positions_m


def test_execute_action_collect_ice_matches_object_engine():
    def setup(cell):
        cell.water_ice = 5.0

    agent, world, av, wv = _pair(cell_setup=setup)
    for _ in range(5):  # atomic weekly yield; five calls retain the old 2-unit throughput
        r1 = execute_action(agent, {}, world, ActionRequest("a0", ActionType.COLLECT_ICE))
        r2 = execute_action(av, {}, wv, ActionRequest("a0", ActionType.COLLECT_ICE))
        assert r1.accepted == r2.accepted
    assert av.inventory.ice > 0
    assert not world.get_cell(1, 1).construction_sites
    assert not wv.get_cell(1, 1).construction_sites
    _assert_parity(agent, world, av, wv)


def test_execute_action_collect_minerals_matches_object_engine():
    def setup(cell):
        cell.resources.minerals = 3.0

    agent, world, av, wv = _pair(cell_setup=setup)
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.COLLECT_MINERALS))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.COLLECT_MINERALS))
    assert av.inventory.minerals > 0
    _assert_parity(agent, world, av, wv)


def test_execute_action_collect_materials_matches_object_engine():
    def setup(cell):
        cell.resources.construction_material = 3.0

    agent, world, av, wv = _pair(cell_setup=setup)
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.COLLECT_MATERIALS))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.COLLECT_MATERIALS))
    assert av.inventory.construction_material > 0
    _assert_parity(agent, world, av, wv)


def test_execute_action_eat_food_matches_object_engine():
    agent, world, av, wv = _pair(agent_kwargs={"inventory": ResourceBundle(food=5.0)})
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.EAT_FOOD))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.EAT_FOOD))
    _assert_parity(agent, world, av, wv)


def test_execute_action_drink_water_structure_branch_matches_object_engine():
    structures = [Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0)]
    agent, world, av, wv = _pair(
        agent_kwargs={"inventory": ResourceBundle(water=0.0, ice=0.0)}, structures=structures
    )
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.DRINK_WATER))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.DRINK_WATER))
    _assert_parity(agent, world, av, wv)


def test_execute_action_drink_water_inventory_water_branch_matches_object_engine():
    agent, world, av, wv = _pair(agent_kwargs={"inventory": ResourceBundle(water=1.0, ice=0.0)})
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.DRINK_WATER))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.DRINK_WATER))
    _assert_parity(agent, world, av, wv)


def test_execute_action_drink_water_inventory_ice_branch_matches_object_engine():
    agent, world, av, wv = _pair(agent_kwargs={"inventory": ResourceBundle(water=0.0, ice=1.0)})
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.DRINK_WATER))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.DRINK_WATER))
    _assert_parity(agent, world, av, wv)


def test_execute_action_drink_water_liquid_water_branch_matches_object_engine():
    def setup(cell):
        cell.liquid_water = 1.0

    agent, world, av, wv = _pair(cell_setup=setup, agent_kwargs={"inventory": ResourceBundle(water=0.0, ice=0.0)})
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.DRINK_WATER))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.DRINK_WATER))
    _assert_parity(agent, world, av, wv)


def test_execute_action_drink_water_cell_ice_branch_matches_object_engine():
    def setup(cell):
        cell.water_ice = 1.0

    agent, world, av, wv = _pair(cell_setup=setup, agent_kwargs={"inventory": ResourceBundle(water=0.0, ice=0.0)})
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.DRINK_WATER))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.DRINK_WATER))
    _assert_parity(agent, world, av, wv)


def test_execute_action_refill_water_from_cell_ice_matches_object_engine():
    def setup(cell):
        cell.water_ice = 2.0

    agent, world, av, wv = _pair(
        cell_setup=setup,
        agent_kwargs={"inventory": ResourceBundle(water=0.5, ice=0.0)},
    )
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.REFILL_WATER))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.REFILL_WATER))
    assert agent.inventory.water == av.inventory.water == 0.5
    assert agent.inventory.ice == av.inventory.ice == 1.0
    _assert_parity(agent, world, av, wv)


def test_execute_action_forage_greenhouse_branch_matches_object_engine():
    structures = [Structure(type=StructureType.GREENHOUSE, x=1, y=1, integrity=1.0)]

    def setup(cell):
        # La serra deposita il raccolto nella giacenza della cella a ogni
        # passo; raccogliere e' prelevare da li'. Senza scorta l'azione non
        # ha nulla da dare, e il test misurerebbe la fame invece della parita'.
        cell.resources.food = 50.0

    agent, world, av, wv = _pair(structures=structures, cell_setup=setup)
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.FORAGE))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.FORAGE))
    assert av.inventory.food > 5.0
    _assert_parity(agent, world, av, wv)


def test_execute_action_forage_open_field_branch_matches_object_engine():
    def setup(cell):
        cell.vegetation_biomass = 2.0
        cell.proto_soil_development = 0.6

    agent, world, av, wv = _pair(cell_setup=setup)
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.FORAGE))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.FORAGE))
    _assert_parity(agent, world, av, wv)


def test_execute_action_rest_matches_object_engine():
    structures = [Structure(type=StructureType.SHELTER, x=1, y=1, integrity=1.0)]

    def setup(cell):
        cell.habitability_score = 0.5

    agent, world, av, wv = _pair(cell_setup=setup, agent_kwargs={"fatigue": 0.6}, structures=structures)
    execute_action(agent, {}, world, ActionRequest("a0", ActionType.REST))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.REST))
    _assert_parity(agent, world, av, wv)


def test_execute_action_build_creates_and_advances_construction_site_matches_object_engine():
    # BUILD_TIME[HABITAT] == 3, so 100.0/3 progress-per-step. The object
    # engine accumulates progress in plain float64; the array engine stores
    # it in a float32 `site_progress` cell, so the exact call at which
    # accumulated progress first crosses the >=100.0 completion threshold can
    # differ by one call between the two engines - both must still converge
    # to "exactly one finished HABITAT" within a few calls, which is what
    # this test checks (see task-7-report.md for this documented precision
    # gap). The first call, at least, must behave identically on both sides
    # (open the site, deduct the same cost).
    inv = ResourceBundle(construction_material=5.0, minerals=3.0, oxygen=1.0)
    agent, world, av, wv = _pair(agent_kwargs={"inventory": inv})
    cell = world.get_cell(1, 1)
    cv = wv.get_cell(1, 1)

    r1 = execute_action(agent, {}, world, ActionRequest("a0", ActionType.BUILD_HABITAT))
    r2 = execute_action(av, {}, wv, ActionRequest("a0", ActionType.BUILD_HABITAT))
    assert r1.accepted == r2.accepted
    _assert_parity(agent, world, av, wv)  # cost deducted identically on the opening call
    assert len(cell.structures) == 0 and len(cv.construction_sites) == 1

    for _ in range(5):
        if cell.structures:
            break
        execute_action(agent, {}, world, ActionRequest("a0", ActionType.BUILD_HABITAT))
    for _ in range(5):
        if not cv.construction_sites:
            break
        execute_action(av, {}, wv, ActionRequest("a0", ActionType.BUILD_HABITAT))

    assert len(cell.structures) == 1
    assert cell.structures[0].type == StructureType.HABITAT
    si = C.S[StructureType.HABITAT]
    assert int(wv._cells.struct_count[1, 1, si]) == 1
    assert np.isclose(float(wv._cells.struct_integrity[1, 1, si]), cell.structures[0].integrity, atol=1e-3)
    assert len(cv.construction_sites) == 0
    assert av.inventory.to_dict() == pytest.approx(agent.inventory.to_dict(), abs=1e-5)


# ---------------------------------------------------------------------------
# Review findings (task-7 review): four Important defects, each proven with a
# test that fails before the corresponding fix.
# ---------------------------------------------------------------------------


class _ScoutSettleView(AgentView):
    """Test-only shim: `_maybe_start_scout`/`_maybe_start_settlement` call
    `self._scout_interval()`/`self._settle_interval()` internally, and Python
    resolves that through `type(self)`, not through whatever object happened
    to be passed as `self` - so calling the real unbound methods with a plain
    `AgentView` as `self` fails on that internal call even though neither
    helper touches anything but `self.curiosity`/`self.agent_id` (both real
    `AgentView` properties). Borrowing the two interval methods here is
    test-side glue for Python's method lookup, not a change to
    rule_based_agent.py - the bodies executed are the untouched originals.
    """

    _scout_interval = RuleBasedAgent._scout_interval
    _settle_interval = RuleBasedAgent._settle_interval


def test_scout_and_settle_next_at_evolve_independently():
    # Important 1: scout_next_at and settle_next_at used to share one
    # AgentArrays column (mission_next_at). rule_based_agent.decide() runs
    # _settlement_action() before _scouting_action() every tick
    # (rule_based_agent.py:995,1154), so on a shared column the settlement
    # stagger (_settle_interval(), min 20) would silently clobber the
    # scouting stagger (_scout_interval(), min 8) computed just afterwards.
    # Both _maybe_start_* methods only touch self.agent_id/curiosity on their
    # very first call (next_at < 0), so they can be driven directly without a
    # real cell/world.
    agent = RuleBasedAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1, curiosity=0.5)
    aa = AgentArrays.from_agents({"a0": agent})
    side = AgentSideState.from_agent(agent)
    av = _ScoutSettleView(aa, aa.index["a0"], side)

    assert RuleBasedAgent._maybe_start_settlement(agent, None, None) is None
    assert RuleBasedAgent._maybe_start_settlement(av, None, None) is None
    assert RuleBasedAgent._maybe_start_scout(agent, None, None) is None
    assert RuleBasedAgent._maybe_start_scout(av, None, None) is None

    # Sanity: on the real object engine the two staggers are never equal
    # (SETTLE_INTERVAL_STEPS=60 vs SCOUT_INTERVAL_STEPS=24, both with a
    # curiosity-dependent multiplier and a distinct per-agent-id offset).
    assert agent.settle_next_at != agent.scout_next_at

    assert av.settle_next_at == agent.settle_next_at
    assert av.scout_next_at == agent.scout_next_at
    assert av.settle_next_at != av.scout_next_at


def test_structure_integrity_maintain_and_wear_match_object_engine_with_two_structures():
    # Important 2: StructureView.integrity used to serve the getter from a
    # fresh mean-of-array read every time. cell.structures builds every
    # StructureView of a type in ONE call, but a loop that then mutates each
    # one in turn (action_space.py MAINTAIN_STRUCTURE, extreme_events.py wear)
    # would have iteration 2 read the mean already updated by iteration 1 and
    # compound the delta. Two 0.5-integrity shelters repaired by +0.4: the
    # object engine gives mean 0.9 (0.5+0.4 each, independently); the buggy
    # array engine gave 1.0 (second read sees 0.9, +0.4 clamped to 1.0).
    structures = [
        Structure(type=StructureType.SHELTER, x=1, y=1, integrity=0.5),
        Structure(type=StructureType.SHELTER, x=1, y=1, integrity=0.5),
    ]
    inv = ResourceBundle(construction_material=5.0)
    agent, world, av, wv = _pair(agent_kwargs={"inventory": inv}, structures=structures)
    cell = world.get_cell(1, 1)
    cv = wv.get_cell(1, 1)

    execute_action(agent, {}, world, ActionRequest("a0", ActionType.MAINTAIN_STRUCTURE))
    execute_action(av, {}, wv, ActionRequest("a0", ActionType.MAINTAIN_STRUCTURE))

    obj_mean = sum(s.integrity for s in cell.structures) / len(cell.structures)
    arr_mean = cv.structures[0].integrity
    assert np.isclose(obj_mean, 0.9, atol=1e-5)  # matches the review's worked example
    assert np.isclose(arr_mean, obj_mean, atol=1e-5)

    # Now the real extreme_events wear pass (Global Dust Storm), on both
    # engines, called directly with a synthetic event to skip the random
    # detection scaffolding in ExtremeEventEngine.advance().
    event = {"type": "Global Dust Storm", "severity": 1.0}
    ExtremeEventEngine()._apply_event_impacts(world, {"a0": agent}, event, [{"x": 1, "y": 1}])
    ExtremeEventEngine()._apply_event_impacts(wv, {"a0": av}, event, [{"x": 1, "y": 1}])

    obj_mean_after = sum(s.integrity for s in cell.structures) / len(cell.structures)
    arr_mean_after = cv.structures[0].integrity
    assert np.isclose(arr_mean_after, obj_mean_after, atol=1e-5)


def test_add_structure_updates_habitability_matching_object_engine():
    # Important 3: WorldView.add_structure never touched cells.habitability,
    # but GridWorld.add_structure calls Cell.recompute_habitability, which
    # folds each structure's local_effect["habitability"] into the score
    # (cell.py:185-187). Establish a real (non-default) baseline habitability
    # on the object engine first, snapshot it into CellArrays, then build the
    # SAME structure through both engines and compare.
    #
    # cell_degradation is turned off for this test: recompute_habitability's
    # density-load block (cell.py:165-171) reads len(self.structures), which
    # changes the instant the structure is appended, so a FULL recompute
    # picks up a pollution_risk side effect from the structure count itself -
    # on a bare test Cell (area_m2 default 1.0) even a single structure
    # saturates the density_pollution cap. That coupling is orthogonal to
    # what this test checks (the habitability delta from local_effect) and
    # is exactly the "sacrifice accepted" full-recompute gap this fix's
    # "minimum acceptable" incremental approach does not attempt to close.
    from src.world.cell import set_cell_degradation

    set_cell_degradation(False)
    try:
        cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(3)] for y in range(3)]
        world = GridWorld(width=3, height=3, cells=cells)
        cell = world.get_cell(1, 1)
        cell.recompute_habitability(world.planetary_state)
        baseline = cell.habitability_score
        assert 0.0 < baseline < 1.0  # sanity: not sitting on the clamp boundary

        ca = CellArrays.from_world(world)
        aa = AgentArrays(4)
        wv = WorldView(ca, aa, dict(world.planetary_state), {})
        assert np.isclose(wv.get_cell(1, 1).habitability_score, baseline, atol=1e-5)

        world.add_structure(Structure(type=StructureType.SHELTER, x=1, y=1, integrity=1.0))
        wv.add_structure(Structure(type=StructureType.SHELTER, x=1, y=1, integrity=1.0))

        assert np.isclose(
            wv.get_cell(1, 1).habitability_score, world.get_cell(1, 1).habitability_score, atol=1e-5
        )
    finally:
        set_cell_degradation(True)


def test_add_structure_habitability_reflects_stale_array_state():
    # Task 8, Important 3 (controller decision: KEEP the full single-cell
    # `recompute_habitability` call in `WorldView.add_structure` - it mirrors
    # `GridWorld.add_structure` -> `Cell.recompute_habitability` exactly,
    # including the density-pollution floor side effect (grid.py:133-139);
    # fidelity to the object engine is the governing criterion, not
    # incrementalism).
    #
    # `test_add_structure_updates_habitability_matching_object_engine` (above)
    # builds a FRESH `CellArrays` right before calling `add_structure`, so
    # `occupancy`/`struct_count` are still exactly what `from_world()` just
    # captured - staleness never gets a chance to appear there. Here `ca` is
    # mutated BY HAND after construction - an agent-occupied cell and an
    # already-standing HEATER, written directly into the arrays rather than
    # through `add_structure`/`from_world` - simulating a `CellArrays` that
    # has drifted from a snapshot the way live agent movement and prior
    # structure builds would drift it during a real run (this drift is
    # exactly what wiring `update_cells` into the engines, a later task, is
    # meant to keep continuously synced - see the report). The object engine
    # is walked to the equivalent state by hand so both sides describe the
    # same world. If `WorldView.add_structure`'s recompute ever stopped
    # reading the LIVE `cells.occupancy`/`struct_count` arrays (e.g. cached
    # a value at construction time instead), this test would catch it; the
    # fresh-CellArrays test above could not.
    from src.world.cell import set_cell_degradation

    set_cell_degradation(True)
    try:
        cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(3)] for y in range(3)]
        world = GridWorld(width=3, height=3, cells=cells)
        cell = world.get_cell(1, 1)

        ca = CellArrays.from_world(world)  # snapshot: occupancy=0, struct_count=0
        aa = AgentArrays(4)
        wv = WorldView(ca, aa, dict(world.planetary_state), {})

        # Drift #1: two agents now occupy the cell; CellArrays never got told.
        cell.agents_present = ["a0", "a1"]
        ca.occupancy[1, 1] = 2

        # Drift #2: a HEATER already standing there, added to the object
        # engine through the real API but mirrored into `ca` by direct array
        # writes (not through add_structure/from_world) - the kind of
        # out-of-band mutation that leaves a snapshot stale.
        world.add_structure(Structure(type=StructureType.HEATER, x=1, y=1, integrity=1.0))
        heater_si = C.S[StructureType.HEATER]
        ca.struct_count[1, 1, heater_si] += 1
        ca.struct_integrity[1, 1, heater_si] += 1.0
        ca.recompute_struct_fx(1, 1)

        world.add_structure(Structure(type=StructureType.SHELTER, x=1, y=1, integrity=1.0))
        wv.add_structure(Structure(type=StructureType.SHELTER, x=1, y=1, integrity=1.0))

        # Sanity: the density-pollution floor really engaged from this drift
        # (area_m2 defaults to 1.0 on a bare test Cell, so any nonzero
        # occupancy/struct_count saturates the 0.08 cap) and matches.
        assert np.isclose(ca.pollution[1, 1], 0.08, atol=1e-6)
        assert np.isclose(ca.pollution[1, 1], cell.pollution_risk, atol=1e-6)
        assert np.isclose(
            wv.get_cell(1, 1).habitability_score, world.get_cell(1, 1).habitability_score, atol=1e-5
        )
    finally:
        set_cell_degradation(True)


def test_move_onto_hazardous_terrain_does_not_raise_and_matches_object_engine():
    # Important 4: action_space.py's apply_environmental_entry reads
    # agent.name unconditionally once traversal_risk >= 0.75 (to log a
    # hazard_environment event). AgentView had no `name` property, so any
    # MOVE onto high-hazard terrain raised AttributeError through the views.
    #
    # MOVE's actual cell-to-cell resolution runs on real-world geometry
    # (center_lat_deg/center_lon_deg/width_m/height_m -
    # action_space.py:182-256). Task 12 made `CellView.geometry` compute the
    # SAME real geometry `WorldGenerator` populates in production
    # (`src.world.mars_geometry.cell_geometry(x, y, W, H)`, a pure function
    # of position/grid size - src/core/views.py's own docstring) instead of
    # only ever carrying `area_m2` - so a hand-built test `Cell` with its
    # `geometry` dict left at the dataclass default (`{}`) would now diverge
    # from the vectorized side, which always computes it. Setting the same
    # real geometry on the object-side cells here restores the equivalence
    # this test actually cares about (hazard entry logging through the
    # views), rather than relying on "both sides leave geometry empty" being
    # incidentally true.
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(3)] for y in range(3)]
    cells[1][1].terrain = TerrainType.LAVA_TUBE  # traversal_risk 0.93 + radiation extra >= 0.75
    for yy, row in enumerate(cells):
        for xx, cell in enumerate(row):
            geom = cell_geometry(xx, yy, 3, 3)
            cell.geometry = {
                "center_lat_deg": geom.center_lat_deg,
                "center_lon_deg": geom.center_lon_deg,
                "width_m": geom.width_m,
                "height_m": geom.height_m,
                "area_m2": geom.area_m2,
                "area_km2": geom.area_km2,
            }
    world = GridWorld(width=3, height=3, cells=cells)

    agent = BaseAgent(agent_id="a0", name="scout-alpha", role="colonist", x=1, y=1, faction="mars-one")
    aa = AgentArrays.from_agents({"a0": agent})
    ca = CellArrays.from_world(world)
    side = AgentSideState.from_agent(agent)
    av = AgentView(aa, aa.index["a0"], side)
    wv = WorldView(ca, aa, dict(world.planetary_state), {})

    assert av.name == "scout-alpha"
    request = ActionRequest("a0", ActionType.MOVE, target={"x": 1, "y": 1})

    r1 = execute_action(agent, {}, world, request)
    r2 = execute_action(av, {}, wv, request)  # must not raise AttributeError: 'AgentView' object has no attribute 'name'

    assert r1.accepted == r2.accepted
    assert (agent.x, agent.y) == (av.x, av.y)
    assert np.isclose(agent.health, av.health, atol=1e-5)
    assert np.isclose(agent.fatigue, av.fatigue, atol=1e-5)
    assert any(e["type"] == "hazard_environment" for e in wv.events)


# ---------------------------------------------------------------------------
# Minor findings from the same review.
# ---------------------------------------------------------------------------


def test_scout_home_setter_skips_write_on_none():
    # Setting scout_home = None used to write (0, 0) into home_x/home_y,
    # permanently destroying the agent's spawn home even though the getter
    # already reads None while at rest (gated on `mission`).
    _, aa, av = _agent_view()
    row = av.row
    assert (int(aa.home_x[row]), int(aa.home_y[row])) == (1, 1)  # spawned at (1, 1)

    av.scout_phase = "out"
    av.scout_home = None  # must be a no-op on the underlying array
    assert (int(aa.home_x[row]), int(aa.home_y[row])) == (1, 1)

    av.scout_phase = None
    assert av.scout_home is None  # at rest: getter still reads None


def test_worldview_day_step_read_live_from_metadata():
    # day/step used to be copied once at construction time, so events logged
    # later through the same WorldView carried a stale step/day forever.
    ca, _ = _cell_view()
    aa = AgentArrays(4)
    metadata = {"day": 1, "step": 10}
    wv = WorldView(ca, aa, {}, metadata)
    assert wv.day == 1
    assert wv.step == 10

    metadata["day"] = 5
    metadata["step"] = 42
    assert wv.day == 5
    assert wv.step == 42

    wv.log_event("test_event", "test")
    assert wv.events[-1]["day"] == 5
    assert wv.events[-1]["step"] == 42


def test_leggere_construction_sites_non_crea_stato():
    """Un percorso di sola osservazione non deve mutare cio' che osserva.

    `ConstructionSitesProxy._extra` era un `setdefault`, quindi bastava
    *interrogare* `construction_sites` perche' comparisse una voce vuota nel
    registro `_site_extra`. Il salvataggio degli artifact lo fa per ogni cella,
    per comporre `cell_infrastructure.json`: misurato il 2026-08-28,
    l'impronta dello stato differiva fra una run salvata e la stessa run non
    salvata, sul solo campo `cells.site_extra`. Il comportamento della
    simulazione non cambiava — le voci sono dizionari vuoti — ma l'oracolo che
    il progetto usa per ogni verifica ne risultava invalidato.
    """
    from src.core.state_digest import digest_cells

    cells = CellArrays(4, 4)
    prima = digest_cells(cells)

    vista = CellView(cells, 1, 2, {})
    # tutte le forme di LETTURA del proxy, nessuna delle quali deve creare nulla
    assert "greenhouse@0:0" not in vista.construction_sites
    assert vista.construction_sites.get("greenhouse@0:0") is None
    assert len(vista.construction_sites) == 0
    assert list(vista.construction_sites) == []
    with pytest.raises(KeyError):
        vista.construction_sites["assente"]

    assert digest_cells(cells) == prima, "leggere ha mutato lo stato"

    # la SCRITTURA invece deve creare, altrimenti il proxy non servirebbe
    vista.construction_sites["cantiere_libero"] = 0.5
    assert digest_cells(cells) != prima
    assert vista.construction_sites["cantiere_libero"] == 0.5
    del vista.construction_sites["cantiere_libero"]
