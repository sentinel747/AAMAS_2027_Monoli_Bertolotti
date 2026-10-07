"""Task 12: `colony_site_score` computed through the vectorized facade must
equal the object engine's for the same world - the deferred Task 4 fields
(`CellView.geometry`'s `center_lat_deg`/`area_km2`, `CellView.is_spawn`) this
wires. Before this fix, `colony_site_score` (src/world/colony_site.py:15-32)
was measurably divergent through a view: `_lat()` (:113-114) always read
`center_lat_deg` as the getattr default 0.0 (weight 0.16 in the score), the
area term always read `area_km2` as 0.0 (weight 0.06), and
`traversal_risk_for_cell`'s spawn short-circuit (src/world/terrain.py:44-47)
never fired through a view (weight 0.18 in the score, via `safety_score`).
"""

from __future__ import annotations

from src.core.arrays import AgentArrays, CellArrays
from src.core.views import CellView
from src.world.colony_site import colony_site_score
from src.world.initial_support import seed_initial_colony_support
from src.world.world_generator import WorldGenerator


CONFIG = {
    "seed": 7,
    "colony": {"start_x": 5, "start_y": 4, "initial_structures": {"habitat": 1, "greenhouse": 1}},
    "agents": {"count": 4},
}


def _build_world():
    world = WorldGenerator(seed=7, map_profile="balanced").generate(12, 10)
    seed_initial_colony_support(world, CONFIG, seed=7)
    return world


def test_colony_site_score_matches_object_engine_for_every_cell():
    world = _build_world()
    ca = CellArrays.from_world(world)
    aa = AgentArrays.from_agents({})

    mismatches = []
    saw_spawn_cell = False
    saw_nonzero_latitude = False
    for y in range(world.height):
        for x in range(world.width):
            obj_cell = world.get_cell(x, y)
            vec_cell = CellView(ca, x, y, dict(world.planetary_state), aa)

            if obj_cell.is_spawn:
                saw_spawn_cell = True
                assert vec_cell.is_spawn, f"({x},{y}): object is_spawn=True but view reports False"
            if abs(obj_cell.geometry.get("center_lat_deg", 0.0)) > 1.0:
                saw_nonzero_latitude = True

            obj_score = colony_site_score(obj_cell)
            vec_score = colony_site_score(vec_cell)
            if abs(obj_score - vec_score) > 1e-6:
                mismatches.append((x, y, obj_score, vec_score))

    assert saw_spawn_cell, "sanity: the seeded colony must mark at least one cell is_spawn"
    assert saw_nonzero_latitude, "sanity: a 10-row grid must have cells away from the equator"
    assert not mismatches, f"colony_site_score diverged on {len(mismatches)} cell(s): {mismatches[:5]}"


def test_geometry_matches_object_engine_exactly():
    world = _build_world()
    ca = CellArrays.from_world(world)

    for y in range(world.height):
        for x in range(world.width):
            obj_geom = world.get_cell(x, y).geometry
            vec_geom = CellView(ca, x, y, dict(world.planetary_state)).geometry
            for key in ("center_lat_deg", "center_lon_deg", "width_m", "height_m", "area_km2"):
                assert abs(obj_geom[key] - vec_geom[key]) < 1e-6, f"({x},{y}).{key}: obj={obj_geom[key]} vec={vec_geom[key]}"
            # area_m2 is stored as a float32 column (CellArrays.area_m2, ~1e11
            # magnitude here) - a relative tolerance, not an absolute one,
            # matches float32's ~7 significant digits.
            obj_area = obj_geom["area_m2"]
            assert abs(obj_area - vec_geom["area_m2"]) <= abs(obj_area) * 1e-4, f"({x},{y}).area_m2"
