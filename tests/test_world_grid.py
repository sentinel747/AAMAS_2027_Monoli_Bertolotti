from src.world.world_generator import WorldGenerator
from src.world.terrain import TerrainType


def test_world_generation_50x50():
    world = WorldGenerator(seed=1).generate(50, 50)
    assert world.width == 50
    assert world.height == 50
    assert len(world.snapshot()["cells"]) == 2500


def test_world_generation_keeps_ice_terrain_near_poles():
    world = WorldGenerator(seed=1).generate(50, 50)
    ice_like = {"ice_deposit", "frozen_basin"}

    north_rows = world.cells[:5]
    south_rows = world.cells[-5:]
    middle_rows = world.cells[10:40]

    assert all(cell.terrain.value in ice_like for row in north_rows for cell in row)
    assert all(cell.terrain.value in ice_like for row in south_rows for cell in row)
    assert all(cell.terrain.value not in ice_like for row in middle_rows for cell in row)


def test_world_generation_is_reproducible_for_seed_and_map_profile():
    first = _world_signature(WorldGenerator(seed=42, map_profile="mineral_rich").generate(30, 30))
    second = _world_signature(WorldGenerator(seed=42, map_profile="mineral_rich").generate(30, 30))
    other_seed = _world_signature(WorldGenerator(seed=43, map_profile="mineral_rich").generate(30, 30))

    assert first == second
    assert first != other_seed


def test_balanced_profile_has_no_non_polar_ice_terrain():
    world = WorldGenerator(seed=7, map_profile="balanced").generate(50, 50)
    ice_like = {TerrainType.ICE_DEPOSIT, TerrainType.FROZEN_BASIN}

    for y, row in enumerate(world.cells):
        if WorldGenerator(seed=0)._polar_fraction(y, world.height) <= 0:
            assert all(cell.terrain not in ice_like for cell in row)


def test_new_mars_world_has_no_surface_water_or_edible_biomass():
    world = WorldGenerator(seed=8, map_profile="balanced").generate(50, 50)

    assert all(cell.liquid_water == 0 for row in world.cells for cell in row)
    assert all(cell.vegetation_biomass == 0 for row in world.cells for cell in row)


def test_longitude_neighbors_and_agent_range_wrap_at_map_edge():
    world = WorldGenerator(seed=9).generate(12, 8)
    y = 4
    wrapped = {(cell.x, cell.y) for cell in world.neighbors(0, y, 1)}
    assert (world.width - 1, y) in wrapped
    assert (1, y) in wrapped

    world.place_agent("west", world.width - 1, y)
    assert "west" in world.get_agents_in_range(0, y, 1)


def test_map_profiles_create_different_resource_and_hazard_distributions():
    scarce = WorldGenerator(seed=5, map_profile="scarce_resources").generate(50, 50)
    mineral = WorldGenerator(seed=5, map_profile="mineral_rich").generate(50, 50)
    hazard = WorldGenerator(seed=5, map_profile="high_hazard").generate(50, 50)
    balanced = WorldGenerator(seed=5, map_profile="balanced").generate(50, 50)
    ice_rich = WorldGenerator(seed=5, map_profile="ice_rich").generate(50, 50)

    assert _total_minerals(mineral) > _total_minerals(scarce)
    assert _hazard_count(hazard) > _hazard_count(balanced)
    assert _ice_terrain_count(ice_rich) > _ice_terrain_count(balanced)


def test_random_map_profile_is_seed_reproducible_but_records_request():
    first = WorldGenerator(seed=99, map_profile="random").generate(20, 20)
    second = WorldGenerator(seed=99, map_profile="random").generate(20, 20)

    assert first.metadata["requested_map_profile"] == "random"
    assert first.metadata["map_profile"] == second.metadata["map_profile"]
    assert _world_signature(first) == _world_signature(second)


def _world_signature(world):
    return [
        (cell.terrain.value, round(cell.water_ice, 3), round(cell.resources.minerals, 3), round(cell.radiation_level, 3), round(cell.dust_level, 3))
        for row in world.cells
        for cell in row
    ]


def _total_minerals(world) -> float:
    return sum(cell.resources.minerals for row in world.cells for cell in row)


def _hazard_count(world) -> int:
    hazards = {TerrainType.CRATER, TerrainType.CANYON, TerrainType.DUST_FIELD, TerrainType.LAVA_TUBE}
    return sum(1 for row in world.cells for cell in row if cell.terrain in hazards)


def _ice_terrain_count(world) -> int:
    ice_like = {TerrainType.ICE_DEPOSIT, TerrainType.FROZEN_BASIN}
    return sum(1 for row in world.cells for cell in row if cell.terrain in ice_like)
