import numpy as np
from src.agents.base_agent import BaseAgent
from src.core import constants as C
from src.core.arrays import AgentArrays


def _mk(agent_id, x, y, **kw):
    return BaseAgent(agent_id=agent_id, name=agent_id, role="colonist", x=x, y=y, **kw)


def test_from_agents_copies_vitals_and_inventory():
    agents = {"a0": _mk("a0", 2, 3), "a1": _mk("a1", 5, 1)}
    agents["a1"].hydration = 0.4
    agents["a1"].inventory.food = 9.0
    aa = AgentArrays.from_agents(agents)
    assert aa.n == 2
    i1 = aa.index["a1"]
    assert aa.x[i1] == 5 and aa.y[i1] == 1
    assert np.isclose(aa.hydration[i1], 0.4)
    assert np.isclose(aa.inv[i1, C.R["food"]], 9.0)
    assert aa.alive[: aa.n].all()


def test_capacity_margin_leaves_room_and_spawn_appends():
    agents = {f"a{i}": _mk(f"a{i}", 0, 0) for i in range(4)}
    aa = AgentArrays.from_agents(agents, capacity_margin=2.0)
    assert aa.x.shape[0] >= 8
    row = aa.spawn("baby0", 1, 1, np.zeros(C.NR, dtype=np.float32))
    assert aa.ids[row] == "baby0" and aa.alive[row]
    assert aa.n == 5


def test_dead_rows_keep_index_stable():
    agents = {"a0": _mk("a0", 0, 0), "a1": _mk("a1", 1, 1)}
    aa = AgentArrays.from_agents(agents)
    aa.alive[aa.index["a0"]] = False
    assert list(aa.alive_rows()) == [aa.index["a1"]]
    assert aa.index["a0"] == 0  # mai ricompattato


def test_grow_preserves_sentinels_and_data():
    # Start with 8 agents, capacity_margin=1.0 → capacity=max(8, 8)=8
    agents = {f"a{i}": _mk(f"a{i}", i, i) for i in range(8)}
    aa = AgentArrays.from_agents(agents, capacity_margin=1.0)
    assert aa.x.shape[0] == 8
    # Record a0's state before growth
    i0 = aa.index["a0"]
    old_x0 = aa.x[i0]
    # Spawn 2 more agents: n=8, spawn brings n=9, triggers _grow() → capacity=16
    inv_row = np.zeros(C.NR, dtype=np.float32)
    r9 = aa.spawn("baby1", 100, 101, inv_row)
    assert r9 == 8  # 9th row
    r10 = aa.spawn("baby2", 200, 201, inv_row)
    assert r10 == 9  # 10th row
    # Verify capacity grew
    assert aa.x.shape[0] == 16
    # Verify pre-existing data survived
    assert aa.x[i0] == old_x0
    # Verify new row has sentinel defaults
    assert aa.target_x[r9] == -1
    assert aa.target_y[r9] == -1
    assert aa.mission_next_at[r9] == -1
    assert aa.last_comm_step[r9] == -10**6
    # Verify spawn still set initial vitals correctly
    assert np.isclose(aa.health[r9], 1.0)
    assert np.isclose(aa.morale[r9], 0.82)
