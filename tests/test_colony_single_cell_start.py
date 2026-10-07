from src.agents.population import spawn_initial_agents
from src.world.colony_site import colony_spawn_cells
from src.world.initial_support import seed_initial_colony_support
from src.world.world_generator import WorldGenerator


def test_initial_agents_share_the_colony_cell():
    world = WorldGenerator(seed=0).generate(20, 20)
    agents = spawn_initial_agents(50, world, seed=0, start_x=5, start_y=7)
    assert {(agent.x, agent.y) for agent in agents.values()} == {(5, 7)}


def test_initial_agents_spread_only_above_per_cell_threshold():
    world = WorldGenerator(seed=0).generate(20, 20)
    agents = spawn_initial_agents(25, world, seed=0, start_x=10, start_y=10, max_agents_per_cell=10)
    counts: dict[tuple[int, int], int] = {}
    for agent in agents.values():
        counts[(agent.x, agent.y)] = counts.get((agent.x, agent.y), 0) + 1
    assert len(counts) == 3
    assert (10, 10) in counts
    assert all(max(abs(x - 10), abs(y - 10)) <= 1 for x, y in counts)
    assert max(counts.values()) - min(counts.values()) <= 1


def test_aggregate_sample_uses_real_population_footprint():
    world = WorldGenerator(seed=0).generate(20, 20)
    agents = spawn_initial_agents(
        5, world, seed=0, start_x=10, start_y=10, total_population=30, max_agents_per_cell=10
    )
    cells = {(agent.x, agent.y) for agent in agents.values()}
    assert len(cells) == 3


def test_spawn_cells_stay_in_bounds_at_map_corner():
    world = WorldGenerator(seed=0).generate(20, 20)
    cells = colony_spawn_cells(world, 0, 0, 90, max_per_cell=10)
    assert len(cells) == 9
    assert len(set(cells)) == 9
    assert all(0 <= x < world.width and 0 <= y < world.height for x, y in cells)
    assert cells[0] == (0, 0)


def test_structures_share_single_cell_with_default_party():
    world = WorldGenerator(seed=0).generate(20, 20)
    config = {
        "agents": {"count": 50},
        "colony": {"start_x": 4, "start_y": 6, "initial_structures": {"habitat": 3, "greenhouse": 2}},
    }
    structures = seed_initial_colony_support(world, config, seed=0)
    assert {(structure.x, structure.y) for structure in structures} == {(4, 6)}


def test_structures_spread_with_large_landing_party():
    world = WorldGenerator(seed=0).generate(20, 20)
    config = {
        "agents": {"count": 2500},
        "colony": {"start_x": 10, "start_y": 10, "initial_structures": {"habitat": 6}},
    }
    structures = seed_initial_colony_support(world, config, seed=0)
    cells = {(structure.x, structure.y) for structure in structures}
    assert len(cells) == 3
    assert (10, 10) in cells
    assert all(max(abs(x - 10), abs(y - 10)) <= 1 for x, y in cells)


def test_agents_and_structures_share_footprint_above_threshold():
    world = WorldGenerator(seed=0).generate(20, 20)
    config = {
        "agents": {"count": 2500},
        "colony": {"start_x": 10, "start_y": 10, "initial_structures": {"habitat": 6, "greenhouse": 3}},
    }
    agents = spawn_initial_agents(2500, world, seed=0, start_x=10, start_y=10)
    structures = seed_initial_colony_support(world, config, seed=0)
    agent_cells = {(agent.x, agent.y) for agent in agents.values()}
    structure_cells = {(structure.x, structure.y) for structure in structures}
    assert structure_cells == agent_cells
