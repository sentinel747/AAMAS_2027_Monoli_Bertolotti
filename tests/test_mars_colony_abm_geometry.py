from src.agents.population import spawn_initial_agents
from src.simulation.colony_dynamics import compute_colony_dynamics_metrics
from src.social_network.network import SocialNetwork
from src.world.perception import observe
from src.world.mars_geometry import MARS_SURFACE_AREA_KM2
from src.world.initial_support import seed_initial_colony_support
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def test_grid_cells_expose_real_mars_scale_and_internal_positions():
    world = WorldGenerator(seed=23).generate(360, 180)
    assert world.metadata["planet"] == "Mars"
    assert world.metadata["grid_width_cells"] == 360
    assert world.metadata["grid_height_cells"] == 180
    assert abs(world.metadata["mars_surface_area_km2"] - MARS_SURFACE_AREA_KM2) < 1e-6
    assert world.metadata["mars_surface_gravity_m_s2"] == 3.71
    assert world.metadata["mars_mean_surface_pressure_mbar"] == 6.35
    assert world.metadata["mars_mean_temperature_c"] == -55.0

    equator = world.get_cell(180, 90)
    assert 40_000 < equator.geometry["width_m"] < 70_000
    assert 50_000 < equator.geometry["height_m"] < 70_000

    agents = spawn_initial_agents(3, world, seed=23)
    assert {agent.role for agent in agents.values()} == {"colonist"}
    for agent in agents.values():
        cell = world.get_cell(agent.x, agent.y)
        assert 0.0 <= agent.local_x_m <= cell.geometry["width_m"]
        assert 0.0 <= agent.local_y_m <= cell.geometry["height_m"]
        assert agent.perception_radius_m == 59_000.0
        assert agent.movement_distance_m_per_step == 59_000.0
        assert agent.agent_id in cell.agent_positions_m


def test_agent_perception_uses_metric_radius_and_partial_cell_visibility():
    world = WorldGenerator(seed=26).generate(360, 180)
    agents = spawn_initial_agents(1, world, seed=26, start_x=180, start_y=90)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    agent.local_x_m = cell.geometry["width_m"] * 0.5
    agent.local_y_m = cell.geometry["height_m"] * 0.5

    agent.perception_radius = 10
    agent.perception_radius_m = 1_000.0
    narrow = observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents)
    assert narrow.radius_m == 1_000.0
    assert {(item["x"], item["y"]) for item in narrow.visible_cells} == {(agent.x, agent.y)}
    current = narrow.visible_cells[0]
    assert current["visibility_status"] == "partial"
    assert 0.0 < current["visible_fraction"] < 1.0

    agent.local_x_m = cell.geometry["width_m"] - 100.0
    agent.local_y_m = cell.geometry["height_m"] * 0.5
    agent.perception_radius_m = 5_000.0
    edge = observe(world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents)
    visible_by_position = {(item["x"], item["y"]): item for item in edge.visible_cells}
    east = visible_by_position[(agent.x + 1, agent.y)]
    assert edge.radius_m == 5_000.0
    assert east["visibility_status"] == "partial"
    assert 0.0 < east["visible_fraction"] < 1.0
    assert east["visible_area_m2"] < world.get_cell(agent.x + 1, agent.y).geometry["area_m2"]


def test_metric_perception_crosses_longitude_seam():
    world = WorldGenerator(seed=27).generate(360, 180)
    agents = spawn_initial_agents(1, world, seed=27, start_x=0, start_y=90)
    agent = next(iter(agents.values()))
    cell = world.get_cell(agent.x, agent.y)
    agent.local_x_m = 100.0
    agent.local_y_m = cell.geometry["height_m"] * 0.5
    agent.perception_radius_m = 5_000.0

    observation = observe(
        world, agent.agent_id, agent.x, agent.y, agent.perception_radius, agents
    )
    visible = {(item["x"], item["y"]) for item in observation.visible_cells}

    assert (world.width - 1, agent.y) in visible


def test_colony_metrics_include_area_food_material_and_tool_feedback():
    world = WorldGenerator(seed=24).generate(24, 12)
    agents = spawn_initial_agents(4, world, seed=24)
    x, y = world.width // 2, world.height // 2
    world.add_structure(Structure(StructureType.GREENHOUSE, x, y, local_x_m=120.0, local_y_m=140.0))
    world.add_structure(Structure(StructureType.RESEARCH_LAB, x, y, local_x_m=180.0, local_y_m=200.0))
    for agent in agents.values():
        agent.inventory.food = 6.0
        agent.inventory.water = 4.0
        agent.inventory.construction_material = 5.0
        agent.inventory.tools = 2.0

    # **Le scorte pubblicate sono sacche PIU' magazzino (2026-08-30).** Il
    # generatore lascia sul terreno una giacenza propria, quindi il valore atteso
    # si scrive esplicitamente invece di assumere che la cella sia vuota: era
    # proprio quell'assunzione a nascondere il difetto, perche' il test passava
    # sommando le sole sacche.
    magazzino = world.get_cell(x, y).resources

    metrics = compute_colony_dynamics_metrics(agents, world, {}, SocialNetwork(set(agents)))

    assert metrics["colony_area_m2"] > 0
    assert metrics["food_stock"] == 24.0 + magazzino.food
    assert metrics["material_stock"] == 20.0 + magazzino.construction_material
    assert metrics["tool_stock"] == 8.0 + magazzino.tools
    assert magazzino.construction_material > 0.0
    assert metrics["tool_capacity_index"] > 0
    assert metrics["colony_prosperity_index"] > 0


def test_initial_colony_support_seeds_configured_structures_with_local_positions():
    world = WorldGenerator(seed=25).generate(24, 12)

    created = seed_initial_colony_support(
        world,
        {
            "colony": {
                "initial_structures": {
                    "habitat": 2,
                    "greenhouse": 1,
                    "solar_array": 1,
                    "oxygen_plant": 1,
                    "storage_depot": 1,
                    "weather_station": 1,
                }
            }
        },
        seed=25,
    )

    assert [structure.type for structure in created].count(StructureType.HABITAT) == 2
    assert [structure.type for structure in created].count(StructureType.GREENHOUSE) == 1
    assert [structure.type for structure in created].count(StructureType.SOLAR_ARRAY) == 1
    assert [structure.type for structure in created].count(StructureType.OXYGEN_PLANT) == 1
    assert [structure.type for structure in created].count(StructureType.STORAGE_DEPOT) == 1
    assert [structure.type for structure in created].count(StructureType.WEATHER_STATION) == 1
    cell = world.get_cell(world.width // 2, world.height // 2)
    assert len(cell.structures) == 7
    for structure in cell.structures:
        assert structure.owner == "initial_support"
        assert 0.0 <= structure.local_x_m <= cell.geometry["width_m"]
        assert 0.0 <= structure.local_y_m <= cell.geometry["height_m"]
