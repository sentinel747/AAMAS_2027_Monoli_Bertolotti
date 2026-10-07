import numpy as np
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.terrain import TerrainType
from src.world.structures import Structure, StructureType
from src.agents.base_agent import BaseAgent
from src.core.arrays import AgentArrays, CellArrays
from src.core.views import AgentView, CellView, WorldView, AgentSideState


def _world_with_structures():
    # GridWorld has no width/height/seed convenience constructor (it is a plain
    # dataclass requiring a pre-built cells grid); build the minimal writable
    # grid the same way tests/core/test_cell_arrays.py and
    # src/world/world_generator.py do.
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(4)] for y in range(3)]
    w = GridWorld(width=4, height=3, cells=cells)
    cell = w.get_cell(2, 1)
    cell.structures.append(Structure(type=StructureType.SHELTER, x=2, y=1, integrity=0.9))
    cell.structures.append(Structure(type=StructureType.SHELTER, x=2, y=1, integrity=0.9))
    return w


def test_cellview_public_dict_has_same_keys_as_object():
    w = _world_with_structures()
    expected = w.get_cell(2, 1).to_public_dict()
    ca = CellArrays.from_world(w)
    got = CellView(ca, 2, 1, w.planetary_state).to_public_dict()
    assert set(got.keys()) == set(expected.keys())
    assert len(got["structures"]) == 2                      # ri-espansione per tipo
    assert got["structures"][0]["type"] == "shelter"
    assert np.isclose(got["structures"][0]["integrity"], 0.9)
    for key in ("terrain", "x", "y", "water_ice", "habitability"):
        assert got[key] == expected[key] or np.isclose(got[key], expected[key])


def test_agentview_dict_has_same_keys_as_object():
    agent = BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=2)
    expected = agent.to_dict()
    aa = AgentArrays.from_agents({"a0": agent})
    side = {"a0": AgentSideState.from_agent(agent)}
    got = AgentView(aa, 0, side["a0"]).to_dict()
    assert set(got.keys()) == set(expected.keys())
    assert got["agent_id"] == "a0" and got["x"] == 1
    assert set(got["inventory"].keys()) == set(expected["inventory"].keys())


def test_worldview_get_agents_in_range():
    w = _world_with_structures()
    agents = {"a0": BaseAgent(agent_id="a0", name="a0", role="colonist", x=1, y=1),
              "a1": BaseAgent(agent_id="a1", name="a1", role="colonist", x=3, y=1)}
    aa = AgentArrays.from_agents(agents)
    wv = WorldView(CellArrays.from_world(w), aa, w.planetary_state, {})
    assert set(wv.get_agents_in_range(1, 1, 1)) == {"a0"}
    assert set(wv.get_agents_in_range(2, 1, 1)) == {"a0", "a1"}
