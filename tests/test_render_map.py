from fastapi.testclient import TestClient

from src.api.main import app
from src.world.structures import Structure, StructureType
from src.world.terrain import TerrainType
from src.world.world_generator import WorldGenerator


def _make_world():
    return WorldGenerator(seed=3).generate(40, 20)


def test_render_map_payload_shapes_and_static_toggle():
    world = _make_world()
    payload = world.render_map_payload(include_static=True)

    assert payload["world_width"] == 40
    assert payload["world_height"] == 20
    assert payload["day"] == int(world.day)

    total = 40 * 20
    dynamic = payload["dynamic"]
    for key in ("habitability", "vegetation", "water_ice", "dust", "radiation", "explored"):
        assert len(dynamic[key]) == total
    assert all(value in (0, 1) for value in dynamic["explored"])

    static = payload["static"]
    assert static["terrain_types"] == [t.value for t in TerrainType]
    assert len(static["terrain"]) == total
    assert len(static["elevation"]) == total
    assert len(static["polar_severity"]) == total
    assert all(0 <= code < len(static["terrain_types"]) for code in static["terrain"])

    assert "static" not in world.render_map_payload(include_static=False)


def test_render_map_tracks_explored_and_structures():
    world = _make_world()
    x, y = 7, 5
    index = y * world.width + x

    assert world.render_map_payload()["dynamic"]["explored"][index] == 0
    world.mark_explored(x, y)
    assert world.render_map_payload()["dynamic"]["explored"][index] == 1

    world.add_structure(Structure(StructureType.GREENHOUSE, x, y, local_x_m=120.0, local_y_m=140.0))
    entries = world.render_map_payload()["structures"]
    assert {"x": x, "y": y, "type": "greenhouse", "local_x_m": 120.0, "local_y_m": 140.0} in entries


def test_render_map_route_smoke():
    client = TestClient(app)

    data = client.get("/api/world/render_map?include_static=1").json()
    total = data["world_width"] * data["world_height"]
    assert len(data["dynamic"]["habitability"]) == total
    assert len(data["static"]["terrain"]) == total
    assert data["static"]["terrain_types"] == [t.value for t in TerrainType]

    without_static = client.get("/api/world/render_map").json()
    assert "static" not in without_static
    assert len(without_static["dynamic"]["habitability"]) == total
