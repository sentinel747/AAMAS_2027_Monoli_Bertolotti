"""Coordinator follow-up on Task 12: the live GUI map/chunk API routes
(src/api/routes_world.py) call `world.compact_chunk`/`world.render_map_payload`
on whatever `controller.read_world()` returns - a `WorldView` once a scenario
is loaded (src/api/state_store.py). Both were missing entirely from
`WorldView` (`AttributeError`), caught by
`tests/test_api_routes.py::test_world_chunk_supports_colony_focus_bounds` and
`tests/test_render_map.py::test_render_map_route_smoke`.

This pins the payload EQUIVALENCE (not just "no longer raises"): for the same
world, `WorldView.compact_chunk`/`.render_map_payload` must produce the same
shape and values `GridWorld`'s own versions do (src/world/grid.py:178-253),
including the sparse per-structure sub-cell coordinates retained alongside
the dense count/integrity arrays.
"""

from __future__ import annotations

from src.core.arrays import AgentArrays, CellArrays
from src.core.views import WorldView
from src.world.structures import Structure, StructureType
from src.world.world_generator import WorldGenerator


def _build_world():
    world = WorldGenerator(seed=11, map_profile="balanced").generate(9, 7)
    world.add_structure(Structure(StructureType.GREENHOUSE, 3, 2, local_x_m=120.0, local_y_m=140.0))
    world.add_structure(Structure(StructureType.HABITAT, 3, 2))
    world.mark_explored(1, 1)
    return world


def _build_view(world):
    ca = CellArrays.from_world(world)
    aa = AgentArrays.from_agents({})
    return WorldView(ca, aa, dict(world.planetary_state), {"day": world.day, "step": world.step})


def test_compact_chunk_matches_object_engine():
    world = _build_world()
    view = _build_view(world)

    for args in ({"x": 0, "y": 0, "size": 4}, {"x": -20, "y": -10, "size": 5}, {"x": 5, "y": 4, "zoom": 2, "size": 100}):
        obj_chunk = world.compact_chunk(**args)
        vec_chunk = view.compact_chunk(**args)
        assert obj_chunk["x"] == vec_chunk["x"]
        assert obj_chunk["y"] == vec_chunk["y"]
        assert obj_chunk["zoom"] == vec_chunk["zoom"]
        assert obj_chunk["width"] == vec_chunk["width"]
        assert obj_chunk["height"] == vec_chunk["height"]
        assert obj_chunk["world_width"] == vec_chunk["world_width"]
        assert obj_chunk["world_height"] == vec_chunk["world_height"]
        assert len(obj_chunk["cells"]) == len(vec_chunk["cells"])
        for obj_cell, vec_cell in zip(obj_chunk["cells"], vec_chunk["cells"]):
            assert obj_cell["x"] == vec_cell["x"] and obj_cell["y"] == vec_cell["y"]
            assert obj_cell["terrain"] == vec_cell["terrain"]
            assert obj_cell["colony_site_score"] == vec_cell["colony_site_score"] or abs(obj_cell["colony_site_score"] - vec_cell["colony_site_score"]) < 1e-6


def test_render_map_payload_matches_object_engine():
    world = _build_world()
    view = _build_view(world)

    for include_static in (False, True):
        obj_payload = world.render_map_payload(include_static=include_static)
        vec_payload = view.render_map_payload(include_static=include_static)

        assert obj_payload["world_width"] == vec_payload["world_width"]
        assert obj_payload["world_height"] == vec_payload["world_height"]
        assert obj_payload["day"] == vec_payload["day"]
        assert ("static" in obj_payload) == ("static" in vec_payload)
        if include_static:
            assert obj_payload["static"]["terrain_types"] == vec_payload["static"]["terrain_types"]
            assert obj_payload["static"]["terrain"] == vec_payload["static"]["terrain"]
            assert obj_payload["static"]["elevation"] == vec_payload["static"]["elevation"]
            assert obj_payload["static"]["polar_severity"] == vec_payload["static"]["polar_severity"]

        obj_dyn, vec_dyn = obj_payload["dynamic"], vec_payload["dynamic"]
        for key in ("habitability", "vegetation", "water_ice", "dust", "radiation", "explored"):
            assert len(obj_dyn[key]) == len(vec_dyn[key]) == world.width * world.height
            # CellArrays stores these as float32 vs. the object model's
            # float64 - values agree well within one rounding unit (2/3
            # decimals here), but an exact list== can land on opposite sides
            # of a rounding boundary (e.g. 10.705 rounds to 10.70 in float64
            # but 10.71 in float32) without either side being wrong.
            for index, (obj_value, vec_value) in enumerate(zip(obj_dyn[key], vec_dyn[key])):
                assert abs(obj_value - vec_value) <= 0.011, f"{key}[{index}]: obj={obj_value} vec={vec_value}"

        obj_structs = sorted(
            (s["x"], s["y"], s["type"], s["local_x_m"], s["local_y_m"])
            for s in obj_payload["structures"]
        )
        vec_structs = sorted(
            (s["x"], s["y"], s["type"], s["local_x_m"], s["local_y_m"])
            for s in vec_payload["structures"]
        )
        assert obj_structs == vec_structs
        assert len(obj_structs) == 2  # sanity: the fixture's two structures are actually present


def test_world_view_add_structure_preserves_real_subcell_position_and_owner():
    world = _build_world()
    view = _build_view(world)
    structure = Structure(
        StructureType.SOLAR_ARRAY,
        4,
        3,
        local_x_m=12_345.0,
        local_y_m=45_678.0,
        owner="agent_007",
    )

    view.add_structure(structure)

    stored = [s for s in view.get_cell(4, 3).structures if s.type == StructureType.SOLAR_ARRAY]
    assert len(stored) == 1
    assert (stored[0].local_x_m, stored[0].local_y_m, stored[0].owner) == (
        12_345.0,
        45_678.0,
        "agent_007",
    )
    assert {
        "x": 4,
        "y": 3,
        "type": "solar_array",
        "local_x_m": 12_345.0,
        "local_y_m": 45_678.0,
    } in view.render_map_payload()["structures"]


def test_render_map_payload_tracks_explored_live():
    world = _build_world()
    view = _build_view(world)
    x, y = 4, 3
    index = y * world.width + x
    assert view.render_map_payload()["dynamic"]["explored"][index] == 0
    view.mark_explored(x, y)
    assert view.render_map_payload()["dynamic"]["explored"][index] == 1


def test_world_view_neighbors_match_longitude_wrapping():
    world = _build_world()
    view = _build_view(world)
    y = 3

    assert {(cell.x, cell.y) for cell in world.neighbors(0, y, 1)} == {
        (cell.x, cell.y) for cell in view.neighbors(0, y, 1)
    }
    assert (world.width - 1, y) in {
        (cell.x, cell.y) for cell in view.neighbors(0, y, 1)
    }


def test_world_view_exposes_same_infrastructure_occupancy_capacity():
    world = _build_world()
    x, y = 3, 2
    for _ in range(12):
        world.add_structure(Structure(StructureType.SHELTER, x, y))
    world.get_cell(x, y).agents_present.extend(f"a{index}" for index in range(15))
    view = _build_view(world)

    object_cell = world.get_cell(x, y)
    vector_cell = view.get_cell(x, y)
    assert object_cell.occupancy_capacity() == vector_cell.occupancy_capacity() == 14.0
    assert object_cell.overcrowding_excess() == vector_cell.overcrowding_excess() == 1.0
    assert object_cell.to_public_dict()["occupancy_capacity"] == 14.0
    assert vector_cell.to_public_dict()["overcrowding_excess"] == 1.0
