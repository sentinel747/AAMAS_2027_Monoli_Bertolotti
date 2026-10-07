import numpy as np

from src.agents import pillars


def test_preferences_are_normalized_and_deterministic():
    p1 = pillars.sample_pillar_preferences(seed=7, agent_index=3)
    p2 = pillars.sample_pillar_preferences(seed=7, agent_index=3)
    assert p1.shape == (pillars.N_PILLARS,)
    assert p1.dtype == np.float64
    assert abs(float(p1.sum()) - 1.0) < 1e-12
    assert np.array_equal(p1, p2)
    assert (p1 > 0).all()


def test_preferences_differ_by_agent_and_seed():
    a = pillars.sample_pillar_preferences(seed=7, agent_index=0)
    b = pillars.sample_pillar_preferences(seed=7, agent_index=1)
    c = pillars.sample_pillar_preferences(seed=8, agent_index=0)
    assert not np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_spawned_agents_get_preferences_and_traits_are_unchanged():
    from src.world.world_generator import WorldGenerator
    from src.agents.population import spawn_initial_agents

    world = WorldGenerator(seed=0).generate(48, 30)
    agents = spawn_initial_agents(4, world, seed=0)
    golden_cooperation = [
        0.625231410711,
        0.663296521782,
        0.834358005438,
        0.686896435781,
    ]
    golden_curiosity = [
        0.457711154814,
        0.521755432781,
        0.536482177783,
        0.577549294932,
    ]
    assert [round(a.cooperation, 12) for a in agents.values()] == golden_cooperation
    assert [round(a.curiosity, 12) for a in agents.values()] == golden_curiosity
    for idx, agent in enumerate(agents.values()):
        expected = pillars.sample_pillar_preferences(seed=0, agent_index=idx)
        assert agent.pillar_preferences is not None
        assert np.array_equal(np.asarray(agent.pillar_preferences), expected)


def test_born_agent_gets_deterministic_preferences(monkeypatch):
    from src.world.world_generator import WorldGenerator
    from src.world.structures import Structure, StructureType
    from src.agents import population

    world = WorldGenerator(seed=0).generate(48, 30)
    agents = population.spawn_initial_agents(2, world, seed=0)
    x, y = world.width // 2, world.height // 2
    for structure_type in (
        StructureType.HABITAT,
        StructureType.HABITAT,
        StructureType.GREENHOUSE,
        StructureType.SOLAR_ARRAY,
        StructureType.OXYGEN_PLANT,
        # +pozzo (2026-09-01): l'acqua e' un requisito di capienza
        StructureType.WATER_EXTRACTOR,
    ):
        world.add_structure(Structure(structure_type, x, y))
        # Il corredo del neonato lo paga la CELLA (2026-08-30): senza scorte in
        # magazzino la colonia non potrebbe mantenerlo e la nascita sarebbe vietata.
        _mag = world.get_cell(x, y).resources
        _mag.food, _mag.water, _mag.oxygen = 50.0, 40.0, 20.0
    monkeypatch.setattr(population, "_colony_growth_probability", lambda *_: 1.0)
    config = {
        "population": {
            "enabled": True,
            "max_agents": 10,
            "daily_spawn_probability": 1.0,
            "min_habitability_for_growth": -1.0,
        },
        "simulation": {"days_per_step": 7},
    }
    born = None
    for day in range(200):
        born = population.maybe_spawn_agent(agents, world, config, day, seed=0)
        if born is not None:
            break
    assert born is not None, "nessuna nascita in 200 giorni con probabilita' 1.0"
    expected = pillars.sample_pillar_preferences(seed=0, agent_index=2)
    assert np.array_equal(np.asarray(born.pillar_preferences), expected)


def test_spawn_and_birth_use_configured_operational_range(monkeypatch):
    from src.agents import population
    from src.world.structures import Structure, StructureType
    from src.world.world_generator import WorldGenerator

    world = WorldGenerator(seed=4).generate(32, 180)
    agents = population.spawn_initial_agents(
        1, world, seed=4, operational_range_m=120_000.0
    )
    founder = next(iter(agents.values()))
    assert founder.perception_radius_m == 59_000.0
    assert founder.movement_distance_m_per_step == 59_000.0
    assert founder.perception_radius == 1

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
        "agents": {"operational_range_m": 120_000.0},
        "population": {
            "enabled": True,
            "max_agents": 2,
            "daily_spawn_probability": 1.0,
            "min_habitability_for_growth": -1.0,
        },
    }
    born = population.maybe_spawn_agent(agents, world, config, day=1, seed=4)
    assert born is not None
    assert born.perception_radius_m == 59_000.0
    assert born.movement_distance_m_per_step == 59_000.0
    assert born.perception_radius == 1
