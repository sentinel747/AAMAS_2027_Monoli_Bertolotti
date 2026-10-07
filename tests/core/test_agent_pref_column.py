import numpy as np

from src.agents import pillars
from src.agents.rule_based_agent import RuleBasedAgent
from src.core.arrays import AgentArrays


def _make_agent(idx: int, prefs=None, skills=None) -> RuleBasedAgent:
    agent = RuleBasedAgent(
        agent_id=f"agent_{idx:03d}", name=f"A{idx}", role="colonist", x=1, y=1
    )
    if prefs is not None:
        agent.pillar_preferences = tuple(prefs)
    if skills is not None:
        agent.pillar_skills = tuple(skills)
    return agent


def test_from_agents_copies_preferences_and_defaults_to_uniform():
    explicit = pillars.sample_pillar_preferences(0, 0)
    agents = {
        "agent_000": _make_agent(0, explicit),
        "agent_001": _make_agent(1, None),
    }
    arrays = AgentArrays.from_agents(agents)
    assert arrays.pref.shape[1] == pillars.N_PILLARS
    assert arrays.pref.dtype == np.float64
    assert np.array_equal(arrays.pref[0], explicit)
    assert np.allclose(
        arrays.pref[1], np.full(pillars.N_PILLARS, 1.0 / pillars.N_PILLARS)
    )


def test_from_agents_copies_skills_and_defaults_to_one():
    skills = np.linspace(0.7, 1.3, pillars.N_PILLARS)
    arrays = AgentArrays.from_agents(
        {"agent_000": _make_agent(0, skills=skills), "agent_001": _make_agent(1)}
    )
    assert np.array_equal(arrays.skill[0], skills)
    assert np.array_equal(arrays.skill[1], np.ones(pillars.N_PILLARS))


def test_grow_preserves_pref_rows_and_initializes_new_rows():
    agents = {
        "agent_000": _make_agent(0, pillars.sample_pillar_preferences(0, 0))
    }
    arrays = AgentArrays.from_agents(agents, capacity_margin=1.0)
    before = arrays.pref[0].copy()
    for i in range(1, 40):
        arrays.spawn(f"agent_{i:03d}", 1, 1, np.zeros(arrays.inv.shape[1]))
    assert np.array_equal(arrays.pref[0], before)
    assert np.allclose(arrays.pref[39], 1.0 / pillars.N_PILLARS)


def test_commit_spawned_agent_copies_preferences():
    from src.agents.population import spawn_initial_agents
    from src.core.shell_common import (
        build_core_state,
        commit_spawned_agent,
        rebuild_agent_views,
    )
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=0).generate(48, 30)
    object_agents = spawn_initial_agents(2, world, seed=0)
    core, _world_view = build_core_state(world, object_agents)
    agents = rebuild_agent_views(core)
    born = _make_agent(
        2,
        pillars.sample_pillar_preferences(0, 2),
        np.linspace(0.8, 1.2, pillars.N_PILLARS),
    )
    commit_spawned_agent(core, agents, born)
    row = core.agents.index["agent_002"]
    assert np.array_equal(core.agents.pref[row], np.asarray(born.pillar_preferences))
    assert np.shares_memory(agents["agent_002"].pillar_preferences, core.agents.pref)
    assert np.array_equal(agents["agent_002"].pillar_preferences, core.agents.pref[row])
    assert np.array_equal(core.agents.skill[row], np.asarray(born.pillar_skills))
    assert np.shares_memory(agents["agent_002"].pillar_skills, core.agents.skill)
