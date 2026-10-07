from collections import Counter

import numpy as np

from src.agents import pillars
from src.agents.population import spawn_initial_agents
from src.agents.role_profiles import (
    DEFAULT_ROLE_DISTRIBUTION,
    ROLE_PROFILES,
    assign_initial_roles,
    sample_role,
    sample_role_preferences,
    sample_role_skills,
)
from src.world.world_generator import WorldGenerator


def test_initial_role_percentages_produce_exact_counts():
    roles = assign_initial_roles(100, DEFAULT_ROLE_DISTRIBUTION, seed=81)
    assert Counter(roles) == {
        "biologist": 20,
        "technician": 20,
        "engineer": 20,
        "medic": 15,
        "coordinator": 10,
        "explorer": 15,
    }


def test_role_assignment_preferences_and_skills_are_deterministic():
    first = assign_initial_roles(37, DEFAULT_ROLE_DISTRIBUTION, seed=82)
    second = assign_initial_roles(37, DEFAULT_ROLE_DISTRIBUTION, seed=82)
    assert first == second
    role = first[4]
    assert np.array_equal(
        sample_role_preferences(82, 4, role, 0.25),
        sample_role_preferences(82, 4, role, 0.25),
    )
    assert np.array_equal(
        sample_role_skills(82, 4, role, 0.25),
        sample_role_skills(82, 4, role, 0.25),
    )


def test_spawned_groups_have_normalized_individual_profiles():
    world = WorldGenerator(seed=83).generate(32, 20)
    agents = spawn_initial_agents(
        50,
        world,
        seed=83,
        role_distribution=DEFAULT_ROLE_DISTRIBUTION,
        role_preference_randomness=0.20,
    )
    assert set(agent.role for agent in agents.values()) == set(DEFAULT_ROLE_DISTRIBUTION)
    for agent in agents.values():
        preferences = np.asarray(agent.pillar_preferences)
        skills = np.asarray(agent.pillar_skills)
        assert preferences.shape == (pillars.N_PILLARS,)
        assert np.isclose(preferences.sum(), 1.0)
        assert skills.shape == (pillars.N_PILLARS,)
        assert np.all((skills >= 0.5) & (skills <= 1.6))


def test_role_profiles_emphasize_the_expected_pillars():
    assert np.argmax(ROLE_PROFILES["engineer"].preferences) == pillars.P_BUILD
    assert np.argmax(ROLE_PROFILES["medic"].preferences) == pillars.P_LIFE
    assert np.argmax(ROLE_PROFILES["explorer"].preferences) == pillars.P_EXPLORE
    assert ROLE_PROFILES["biologist"].skills[pillars.P_SUSTENANCE] > 1.0
    assert ROLE_PROFILES["coordinator"].skills[pillars.P_SOCIAL] > 1.0


def test_newborn_role_stream_is_independent_and_repeatable():
    role = sample_role(84, 123, DEFAULT_ROLE_DISTRIBUTION)
    assert role == sample_role(84, 123, DEFAULT_ROLE_DISTRIBUTION)
    assert role in DEFAULT_ROLE_DISTRIBUTION


def test_newborn_receives_configured_role_preferences_and_skills(monkeypatch):
    from src.agents import population
    from src.world.structures import Structure, StructureType

    world = WorldGenerator(seed=85).generate(32, 20)
    agents = spawn_initial_agents(1, world, seed=85)
    founder = next(iter(agents.values()))
    for structure_type in (
        StructureType.HABITAT,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, founder.x, founder.y))
    # Il corredo del neonato lo paga la CELLA (2026-08-30): senza scorte in
    # magazzino la colonia non potrebbe mantenerlo e la nascita sarebbe vietata.
    _mag = world.get_cell(founder.x, founder.y).resources
    _mag.food, _mag.water, _mag.oxygen = 50.0, 40.0, 20.0
    monkeypatch.setattr(population, "_colony_growth_probability", lambda *_: 1.0)
    config = {
        "agents": {
            "role_distribution": {"engineer": 100},
            "role_preference_randomness": 0.10,
        },
        "population": {
            "enabled": True,
            "max_agents": 2,
            "daily_spawn_probability": 1.0,
            "min_habitability_for_growth": -1.0,
        },
    }
    born = population.maybe_spawn_agent(agents, world, config, day=1, seed=85)
    assert born is not None
    assert born.role == "engineer"
    assert np.array_equal(
        np.asarray(born.pillar_preferences),
        sample_role_preferences(85, 1, "engineer", 0.10),
    )
    assert np.array_equal(
        np.asarray(born.pillar_skills),
        sample_role_skills(85, 1, "engineer", 0.10),
    )


def test_runner_passes_role_config_into_founders():
    from src.simulation.agent_coupled_runner import AgentCoupledRunner

    config = {
        "seed": 86,
        "world": {"width": 32, "height": 20},
        "agents": {
            "count": 12,
            "llm_count": 0,
            "role_distribution": {"medic": 100},
            "decision_mode": "preferences",
        },
        "population": {"enabled": False},
        "environmental_layer": {"enabled": False},
    }
    runner = AgentCoupledRunner(config)
    assert {agent.role for agent in runner.agents.values()} == {"medic"}
    assert np.all(runner.core.agents.skill[:12, pillars.P_LIFE] > 1.0)
