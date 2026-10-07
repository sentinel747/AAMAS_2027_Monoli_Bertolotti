"""Task 11: SnapshotBuffer append/flush round-trip.

Builds a minimal CoreState the same way tests/core/test_kernel_step.py does
(no shared tests/core/helpers.py module exists yet - see task-11-brief.md's
fallback instruction), drives a handful of real src.core.kernel.step calls,
and asserts the buffer's .npz round-trip on columns that actually MOVE across
the recorded steps (not a column frozen at its initial value - see this
task's own "recurring lesson" note)."""

from __future__ import annotations

import random

import numpy as np

from src.agents.rule_based_agent import RuleBasedAgent
from src.core.arrays import AgentArrays, CellArrays
from src.core.kernel import CoreState, step
from src.core.snapshot import AGENT_COLUMNS, SnapshotBuffer
from src.core.views import AgentSideState
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType

DOOMED_ID = "a2"

BASE_CONFIG = {
    "headless": {"fast_observation": True, "rule_based_observation": True},
    "model": {"psychosocial_enabled": True},
    "social": {"earth_mars_delay_minutes": 12.0},
    "world": {"cell_degradation": True},
    "seed": 321,
}


def _grid(width: int, height: int) -> GridWorld:
    cells = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(width)] for y in range(height)]
    return GridWorld(width=width, height=height, cells=cells)


def _agent_specs():
    return [
        {"agent_id": "a0", "name": "a0", "role": "colonist", "x": 2, "y": 2,
         "perception_radius": 2, "survival_priority": 0.80},
        {"agent_id": "a1", "name": "a1", "role": "colonist", "x": 1, "y": 1,
         "perception_radius": 1, "survival_priority": 0.85},
        {"agent_id": "a2", "name": "a2", "role": "colonist", "x": 4, "y": 3,
         "perception_radius": 3, "survival_priority": 0.60},
    ]


def make_state(doomed_ids: frozenset[str] = frozenset()) -> CoreState:
    w = _grid(6, 5)
    home = w.get_cell(2, 2)
    for st in (StructureType.HABITAT, StructureType.GREENHOUSE, StructureType.SOLAR_ARRAY, StructureType.OXYGEN_PLANT):
        w.add_structure(Structure(type=st, x=2, y=2, integrity=1.0))
    home.water_ice, home.liquid_water = 4.0, 0.6
    home.nutrients.update({"N": 5.0, "P": 1.0, "C": 3.0})
    w.get_cell(1, 1).water_ice = 1.5
    w.get_cell(4, 3).radiation_level = 0.5
    w.planetary_state.update({
        "mean_temperature_c": -55.0,
        "pressure_pa": 650.0,
        "liquid_water_stability": 0.2,
    })

    agents_src: dict[str, RuleBasedAgent] = {}
    for spec in _agent_specs():
        a = RuleBasedAgent(**spec)
        if spec["agent_id"] in doomed_ids:
            # Same recipe as tests/core/test_kernel_step.py's
            # _make_lethal_agents: hydration=0 + no water/ice reserves +
            # a bare cell (4,3 has no water anywhere nearby) guarantees a
            # dehydration death on the very first tick_vitals call.
            a.hydration = 0.0
            a.inventory.water = 0.0
            a.inventory.ice = 0.0
        agents_src[spec["agent_id"]] = a

    aa = AgentArrays.from_agents(agents_src)
    ca = CellArrays.from_world(w)
    side = {aid: AgentSideState.from_agent(a) for aid, a in agents_src.items()}
    ps = dict(w.planetary_state)
    return CoreState(agents=aa, cells=ca, side=side, planetary_state=ps, metadata={})


def run_steps(state: CoreState, n: int, seed: int = 0, dt_days: float = 365.0):
    rng = np.random.default_rng(seed)
    random.seed(seed)
    outcomes = []
    for i in range(1, n + 1):
        outcomes.append(step(state, i, dt_days, BASE_CONFIG, rng))
    return outcomes


def test_snapshot_appends_and_flushes_npz(tmp_path):
    state = make_state()
    buf = SnapshotBuffer(capacity_steps=8)
    outcomes = run_steps(state, 5, seed=0)
    for i, out in enumerate(outcomes):
        buf.append(i, i * 10.0, state.agents, state.cells, out)

    path = buf.flush_npz(tmp_path / "run.npz")
    data = np.load(path)

    assert data["steps"].shape == (5,)
    assert np.array_equal(data["steps"], np.arange(5))
    assert np.array_equal(data["day"], np.arange(5) * 10.0)
    assert data["agent_health"].shape[0] == 5
    assert data["agent_health"].shape[1] == state.agents.n
    for name in AGENT_COLUMNS:
        assert f"agent_{name}" in data


def test_snapshot_capacity_grows_past_initial_hint(tmp_path):
    # capacity_steps=2 forces at least one internal _grow_steps doubling
    # across 5 appends - the interface (capacity_steps as a sizing hint, not
    # a hard cap) is part of what this test locks in.
    state = make_state()
    buf = SnapshotBuffer(capacity_steps=2)
    outcomes = run_steps(state, 5, seed=1)
    for i, out in enumerate(outcomes):
        buf.append(i, float(i), state.agents, state.cells, out)

    path = buf.flush_npz(tmp_path / "grown.npz")
    data = np.load(path)
    assert data["steps"].shape == (5,)
    assert data["agent_x"].shape == (5, state.agents.n)


def test_snapshot_agent_health_column_round_trips_and_actually_moves(tmp_path):
    # "Beware the recurring lesson": pick a column that actually changes
    # across the recorded steps, and assert the buffered value matches the
    # live AgentArrays value observed at append time (not just any value).
    #
    # append() must be interleaved WITH each step() call, not run after the
    # fact over an already-finished trajectory: AgentArrays is mutated
    # in-place by step(), so calling append() in a second pass over
    # `run_steps`'s returned outcomes (after every step already ran) would
    # copy the SAME final-state health into every recorded row - a bug in the
    # test, not in SnapshotBuffer, but exactly the kind of "asserted quantity
    # never moves" trap this task's brief warns about.
    state = make_state()
    buf = SnapshotBuffer(capacity_steps=8)
    rng = np.random.default_rng(2)
    random.seed(2)

    recorded_health = []
    for i in range(1, 7):
        out = step(state, i, 365.0, BASE_CONFIG, rng)
        buf.append(i, float(i), state.agents, state.cells, out)
        recorded_health.append(state.agents.health[: state.agents.n].copy())

    path = buf.flush_npz(tmp_path / "health.npz")
    data = np.load(path)

    for i, expected in enumerate(recorded_health):
        assert np.allclose(data["agent_health"][i], expected, atol=1e-6)

    # Sanity: health actually moved somewhere across the run - a column
    # frozen at the same value every recorded step would make the round-trip
    # check above vacuous.
    distinct_steps = {tuple(np.round(row, 6)) for row in data["agent_health"]}
    assert len(distinct_steps) > 1


def test_snapshot_flush_with_zero_appends_still_emits_agent_columns(tmp_path):
    # Minor gap (task-11 review): _agent_cols is only populated inside
    # append(), so a buffer flushed without a single append() call used to
    # write an .npz with NO agent_* keys at all. A shell wiring this buffer
    # in later would expect agent_x/agent_health/etc. to always be present
    # keys (empty is fine, missing is not) - pin that flush_npz on an
    # untouched buffer still emits every AGENT_COLUMNS key as an empty
    # (shape (0, 0)) array.
    buf = SnapshotBuffer(capacity_steps=4)
    path = buf.flush_npz(tmp_path / "empty.npz")
    data = np.load(path)

    assert data["steps"].shape == (0,)
    for name, dtype in AGENT_COLUMNS.items():
        key = f"agent_{name}"
        assert key in data
        assert data[key].shape == (0, 0)
        assert data[key].dtype == dtype


def test_snapshot_capacity_steps_zero_and_negative_clamp_to_one(tmp_path):
    # Minor gap (task-11 review): capacity_steps is silently clamped to 1 at
    # construction (`max(1, int(capacity_steps))`) - pin that both 0 and a
    # negative value are accepted (no raise) and produce a usable buffer
    # with an internal capacity of 1, rather than a zero/negative-length
    # array that would break the first append()'s growth check.
    for capacity_steps in (0, -5):
        buf = SnapshotBuffer(capacity_steps=capacity_steps)
        assert buf.steps.shape == (1,)
        assert buf.day.shape == (1,)
        assert buf.cell_resource_total.shape[0] == 1

        # A single append must not immediately require growth (capacity 1
        # is exactly enough for the first recorded step).
        state = make_state()
        out = step(state, 1, 365.0, BASE_CONFIG, np.random.default_rng(0))
        buf.append(0, 0.0, state.agents, state.cells, out)
        assert buf.steps.shape == (1,)

        path = buf.flush_npz(tmp_path / f"clamp_{capacity_steps}.npz")
        data = np.load(path)
        assert data["steps"].shape == (1,)


def test_snapshot_death_and_knowledge_scalars_are_light_summaries(tmp_path):
    # a2 starts doomed: guaranteed dehydration death on step 1, giving a
    # non-zero death_count somewhere in the recorded run - otherwise
    # death_count would stay at 0 for every step and the column would prove
    # nothing (see make_state's doomed_ids docstring note).
    state = make_state(doomed_ids=frozenset({DOOMED_ID}))

    buf = SnapshotBuffer(capacity_steps=8)
    outcomes = run_steps(state, 3, seed=3)
    for i, out in enumerate(outcomes):
        buf.append(i, float(i), state.agents, state.cells, out)

    path = buf.flush_npz(tmp_path / "deaths.npz")
    data = np.load(path)

    assert data["death_count"].sum() == sum(len(o.deaths) for o in outcomes)
    assert data["death_count"].sum() >= 1
    assert np.allclose(data["knowledge_gain"], [float(o.knowledge_gain) for o in outcomes])
