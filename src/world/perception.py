from __future__ import annotations

from dataclasses import dataclass
import math
from types import SimpleNamespace

from .grid import GridWorld
from .terrain import terrain_hazard_tier, traversal_risk_for_cell
from .mars_geometry import MARS_MEAN_RADIUS_KM
from src.world.structures import StructureType


def _world_has_structure(world: GridWorld, structure_type: StructureType) -> bool:
    cells = getattr(world, "_cells", None)
    if cells is not None and hasattr(cells, "struct_count"):
        from src.core import constants as C

        return bool((cells.struct_count[:, :, C.S[structure_type]] > 0).any())
    positions = getattr(world, "_structure_positions", ())
    if positions:
        return any(
            structure.type == structure_type
            for x, y in positions
            for structure in world.get_cell(x, y).structures
        )
    return any(
        structure.type == structure_type
        for row in world.cells
        for cell in row
        for structure in cell.structures
    )


# (dx, dy, compass) — y increases downward on the grid (south). North = smaller y.
_ADJACENT_DELTAS: tuple[tuple[int, int, str], ...] = (
    (0, -1, "north"),
    (1, -1, "northeast"),
    (1, 0, "east"),
    (1, 1, "southeast"),
    (0, 1, "south"),
    (-1, 1, "southwest"),
    (-1, 0, "west"),
    (-1, -1, "northwest"),
)


@dataclass
class LocalObservation:
    agent_id: str
    x: int
    y: int
    radius: int
    radius_m: float
    visible_cells: list[dict]
    adjacent_tiles: list[dict]
    current_cell_digest: dict
    movement_advisories: list[str]
    nearby_agents: list[str]
    nearby_agents_details: list[dict]
    local_danger: float
    local_habitability: float
    construction_sites: dict[str, float]
    structure_count: int
    agent_count: int
    density_warnings: list[str]
    active_event: dict | None
    upcoming_event: dict | None
    observation_mode: str = "full"


def _summarize_neighbor(
    world: GridWorld,
    x: int,
    y: int,
    direction: str,
    agent_role: str = "unknown",
    visibility: dict | None = None,
) -> dict:
    cell = world.get_cell(x, y)
    visibility = visibility or {
        "min_distance_m": None,
        "visible_fraction": 0.0,
        "visible_area_m2": 0.0,
        "fully_visible": False,
        "visibility_status": "not_visible",
    }
    if visibility["visibility_status"] == "not_visible":
        return {
            "x": x,
            "y": y,
            "direction": direction,
            **visibility,
            "brief": f"{direction}: not visible within metric perception radius",
        }
    d = cell.to_public_dict()
    d["direction"] = direction
    note = terrain_hazard_tier(d["traversal_risk"])
    s_count = len(cell.structures)
    a_count = len(cell.agents_present)
    d.update(visibility)
    qualifier = "fully visible" if visibility["fully_visible"] else f"{visibility['visible_fraction']:.2%} visible"
    d["brief"] = f"{direction}: {cell.terrain.value} ({qualifier}, Risk: {note}, Structures: {s_count}, Agents: {a_count})"
    d["structure_count"] = s_count
    d["agent_count"] = a_count
    return d


def observe(world: GridWorld, agent_id: str, x: int, y: int, radius: int, agents_dict: dict | None = None,
            solo_visibili: bool = False) -> LocalObservation:
    """Osservazione metrica completa.

    `solo_visibili` (2026-09-24): fa lo stesso giro sulle celle visibili --
    stessa visibilita', stesse celle segnate come esplorate, stesso ordine --
    ma ogni cella e' una scheda minima (x, y e i campi di visibilita') e il
    resto dell'osservazione non si calcola. Serve al percorso a preferenze del
    kernel, che delle osservazioni legge solo `visible_cells` (per il conteggio
    nel registro) e `observation_mode`. Misura: a ~2000 coloni la scheda
    completa della cella madre, ricostruita da ogni cella che la vede, valeva
    ~14% del passo.
    """
    agent_role = "unknown"
    agent = None
    if agents_dict and agent_id in agents_dict:
        agent = agents_dict[agent_id]
        agent_role = getattr(agent, "role", "unknown")
    here_for_radius = world.get_cell(x, y)
    fallback_radius_m = max(float(here_for_radius.geometry.get("width_m", 1.0)), float(here_for_radius.geometry.get("height_m", 1.0))) * max(0, radius)
    radius_m = float(getattr(agent, "perception_radius_m", fallback_radius_m) if agent is not None else fallback_radius_m)
    local_x_m = float(getattr(agent, "local_x_m", here_for_radius.geometry.get("width_m", 1.0) * 0.5) if agent is not None else here_for_radius.geometry.get("width_m", 1.0) * 0.5)
    local_y_m = float(getattr(agent, "local_y_m", here_for_radius.geometry.get("height_m", 1.0) * 0.5) if agent is not None else here_for_radius.geometry.get("height_m", 1.0) * 0.5)

    cells: list[dict] = []
    danger = 0.0
    habitability = 0.0
    visibility_weight = 0.0

    min_side_m = max(1.0, min(float(here_for_radius.geometry.get("width_m", 1.0)), float(here_for_radius.geometry.get("height_m", 1.0))))
    radius_cells = max(0, int(math.ceil(radius_m / min_side_m)) + 1)
    y_min = max(0, y - radius_cells)
    y_max = min(world.height, y + radius_cells + 1)
    if radius_cells * 2 + 1 >= world.width:
        visible_x = range(world.width)
    else:
        visible_x = tuple((raw_x % world.width) for raw_x in range(x - radius_cells, x + radius_cells + 1))

    for yy in range(y_min, y_max):
        for xx in visible_x:
            cell = world.get_cell(xx, yy)
            visibility = _cell_visibility_metrics_m(world, x, y, local_x_m, local_y_m, radius_m, cell)
            if visibility["visibility_status"] == "not_visible":
                continue
            world.mark_explored(xx, yy)
            if solo_visibili:
                d = {"x": xx, "y": yy}
                d.update(visibility)
                cells.append(d)
                continue
            d = cell.to_public_dict()
            d.update(visibility)
            if not visibility["fully_visible"]:
                d["observation_note"] = (
                    "Partial metric visibility only: the agent sees the listed visible_area_m2 slice, "
                    "not the full physical cell."
                )
            cells.append(d)
            weight = max(0.0, float(visibility["visible_fraction"]))
            danger += (cell.radiation_level + cell.dust_level + cell.pollution_risk + traversal_risk_for_cell(cell) * 0.5) * weight
            habitability += cell.habitability_score * weight
            visibility_weight += weight

    if solo_visibili:
        return SimpleNamespace(visible_cells=cells, observation_mode="full")

    nearby_agent_ids = [aid for aid in world.get_agents_in_range(x, y, 1) if aid != agent_id]
    nearby_details = []
    if agents_dict:
        for aid in nearby_agent_ids:
            if a := agents_dict.get(aid):
                nearby_details.append({
                    "agent_id": a.agent_id,
                    "name": a.name,
                    "role": getattr(a, "role", "unknown"),
                    "health": getattr(a, "health", 1.0),
                    "hydration": getattr(a, "hydration", 1.0),
                    "satiety": getattr(a, "satiety", 1.0),
                })

    adjacent: list[dict] = []
    advisories: list[str] = []
    for dx, dy, compass in _ADJACENT_DELTAS:
        nx, ny = (x + dx) % world.width, y + dy
        if not 0 <= ny < world.height:
            continue
        ncell = world.get_cell(nx, ny)
        visibility = _cell_visibility_metrics_m(world, x, y, local_x_m, local_y_m, radius_m, ncell)
        adjacent.append(_summarize_neighbor(world, nx, ny, compass, agent_role, visibility))
        if visibility["visibility_status"] == "not_visible":
            continue
        tr = traversal_risk_for_cell(ncell)
        if tr >= 0.82:
            advisories.append(
                f"{compass.upper()}: {ncell.terrain.value} extreme environmental risk (~{tr:.2f}); "
                f"cave lava-tube terrain and fractures are dangerous without protection."
            )
        elif tr >= 0.55:
            advisories.append(
                f"{compass.upper()}: {ncell.terrain.value} elevated risk (~{tr:.2f}); "
                "expect dust, unstable ground, radiation, or steep terrain depending on geology."
            )

    here = world.get_cell(x, y)
    current_digest = here.to_public_dict()
    current_digest.update(_cell_visibility_metrics_m(world, x, y, local_x_m, local_y_m, radius_m, here))
    current_digest["position_notes"] = (
        "You are standing on this tile, but only the visible_area_m2 portion is visually perceived. "
        "visible_cells are filtered by metric perception radius; adjacent_tiles lists one-step movement "
        "directions and marks unseen neighbors as not_visible."
    )
    
    warnings = []
    if here.overcrowding_excess() > 0.0:
        warnings.append(
            f"WARNING: local capacity exceeded ({len(here.agents_present)} agents for "
            f"{here.occupancy_capacity():.1f} supported places)."
        )
    if here.pollution_risk > 0.04:
        warnings.append(f"WARNING: local pollution risk is {here.pollution_risk:.3f}.")

    # Detection logic for upcoming events
    # We only show upcoming_event if ANY agent has built a WEATHER_STATION
    has_weather_station = _world_has_structure(world, StructureType.WEATHER_STATION)
    
    # Injected by runner via world.metadata or similar if needed, but for now we look at the runner state indirectly
    # We'll assume the runner puts this info in world.metadata for observation purposes
    active_ev = world.metadata.get("active_event")
    upcoming_ev = world.metadata.get("upcoming_event") if has_weather_station else None

    count = max(visibility_weight, 1.0)
    return LocalObservation(
        agent_id=agent_id,
        x=x,
        y=y,
        radius=radius,
        radius_m=radius_m,
        visible_cells=cells,
        adjacent_tiles=sorted(adjacent, key=lambda row: row["direction"]),
        current_cell_digest=current_digest,
        movement_advisories=sorted(set(advisories)),
        nearby_agents=sorted(nearby_agent_ids),
        nearby_agents_details=nearby_details,
        local_danger=danger / count,
        local_habitability=habitability / count,
        construction_sites=current_digest.get("construction_sites", {}),
        structure_count=len(here.structures),
        agent_count=len(here.agents_present),
        density_warnings=warnings,
        active_event=active_ev,
        upcoming_event=upcoming_ev,
        observation_mode="full",
    )


def observe_fast(world: GridWorld, agent_id: str, x: int, y: int, radius: int, agents_dict: dict | None = None) -> LocalObservation:
    """Cheap local observation for headless/bulk runs.

    It keeps the current cell, immediate movement options and nearby-agent
    summaries, but skips metric visibility over the full radius. This is the
    path to use when agent count, not visual fidelity, is the bottleneck.
    """
    agent = agents_dict.get(agent_id) if agents_dict else None
    here = world.get_cell(x, y)
    world.mark_explored(x, y)
    radius_m = float(getattr(agent, "perception_radius_m", 0.0) or 0.0)
    current_digest = here.to_public_dict()
    current_digest.update(
        {
            "min_distance_m": 0.0,
            "max_corner_distance_m": 0.0,
            "visible_fraction": 1.0,
            "visible_area_m2": float(here.geometry.get("area_m2", 0.0)),
            "fully_visible": True,
            "visibility_status": "full",
        }
    )
    current_digest["position_notes"] = "Fast headless observation: current cell plus adjacent movement options."

    nearby_agent_ids = [aid for aid in world.get_agents_in_range(x, y, 1) if aid != agent_id]
    nearby_details = []
    if agents_dict:
        for aid in nearby_agent_ids:
            if a := agents_dict.get(aid):
                nearby_details.append(
                    {
                        "agent_id": a.agent_id,
                        "name": a.name,
                        "role": getattr(a, "role", "unknown"),
                        "health": getattr(a, "health", 1.0),
                        "hydration": getattr(a, "hydration", 1.0),
                        "satiety": getattr(a, "satiety", 1.0),
                    }
                )

    local_danger = here.radiation_level + here.dust_level + here.pollution_risk + traversal_risk_for_cell(here) * 0.5
    warnings = []
    if here.overcrowding_excess() > 0.0:
        warnings.append(
            f"WARNING: local capacity exceeded ({len(here.agents_present)} agents for "
            f"{here.occupancy_capacity():.1f} supported places)."
        )
    if here.pollution_risk > 0.04:
        warnings.append(f"WARNING: local pollution risk is {here.pollution_risk:.3f}.")

    return LocalObservation(
        agent_id=agent_id,
        x=x,
        y=y,
        radius=radius,
        radius_m=radius_m,
        visible_cells=[current_digest],
        adjacent_tiles=[],
        current_cell_digest=current_digest,
        movement_advisories=[],
        nearby_agents=sorted(nearby_agent_ids),
        nearby_agents_details=nearby_details,
        local_danger=local_danger,
        local_habitability=here.habitability_score,
        construction_sites=current_digest.get("construction_sites", {}),
        structure_count=len(here.structures),
        agent_count=len(here.agents_present),
        density_warnings=warnings,
        active_event=world.metadata.get("active_event"),
        upcoming_event=world.metadata.get("upcoming_event"),
        observation_mode="fast",
    )


def observe_rule_based(world: GridWorld, agent_id: str, x: int, y: int, radius: int, agents_dict: dict | None = None) -> LocalObservation:
    """Observation path for deterministic rule-based agents.

    Rule-based decisions only use the current cell, nearby agents, construction
    sites and immediate movement context. This keeps those fields exact without
    building the full metric-visibility payload needed by LLM prompts.
    """
    agent = agents_dict.get(agent_id) if agents_dict else None
    agent_role = getattr(agent, "role", "unknown") if agent is not None else "unknown"
    here = world.get_cell(x, y)
    radius_m = float(getattr(agent, "perception_radius_m", 0.0) or 0.0)
    area_m2 = float(here.geometry.get("area_m2", 0.0))
    current_visibility = {
        "min_distance_m": 0.0,
        "max_corner_distance_m": 0.0,
        "visible_fraction": 1.0,
        "visible_area_m2": area_m2,
        "fully_visible": True,
        "visibility_status": "full",
    }
    current_digest = here.to_public_dict()
    current_digest.update(current_visibility)
    current_digest["position_notes"] = (
        "Rule-based observation: exact current cell state plus immediate neighbors; "
        "full metric visible-cell payload is reserved for LLM perception."
    )
    world.mark_explored(x, y)

    nearby_agent_ids = [aid for aid in world.get_agents_in_range(x, y, 1) if aid != agent_id]
    nearby_details = []
    if agents_dict:
        for aid in nearby_agent_ids:
            if a := agents_dict.get(aid):
                nearby_details.append(
                    {
                        "agent_id": a.agent_id,
                        "name": a.name,
                        "role": getattr(a, "role", "unknown"),
                        "health": getattr(a, "health", 1.0),
                        "hydration": getattr(a, "hydration", 1.0),
                        "satiety": getattr(a, "satiety", 1.0),
                    }
                )

    adjacent: list[dict] = []
    advisories: list[str] = []
    for dx, dy, compass in _ADJACENT_DELTAS:
        nx, ny = (x + dx) % world.width, y + dy
        if not 0 <= ny < world.height:
            continue
        ncell = world.get_cell(nx, ny)
        neighbor_visibility = {
            "min_distance_m": 0.0,
            "max_corner_distance_m": 0.0,
            "visible_fraction": 1.0,
            "visible_area_m2": float(ncell.geometry.get("area_m2", 0.0)),
            "fully_visible": True,
            "visibility_status": "full",
        }
        adjacent.append(_summarize_neighbor(world, nx, ny, compass, agent_role, neighbor_visibility))
        tr = traversal_risk_for_cell(ncell)
        if tr >= 0.82:
            advisories.append(f"{compass.upper()}: {ncell.terrain.value} extreme environmental risk (~{tr:.2f}).")
        elif tr >= 0.55:
            advisories.append(f"{compass.upper()}: {ncell.terrain.value} elevated risk (~{tr:.2f}).")

    local_danger = here.radiation_level + here.dust_level + here.pollution_risk + traversal_risk_for_cell(here) * 0.5
    warnings = []
    if here.overcrowding_excess() > 0.0:
        warnings.append(
            f"WARNING: local capacity exceeded ({len(here.agents_present)} agents for "
            f"{here.occupancy_capacity():.1f} supported places)."
        )
    if here.pollution_risk > 0.04:
        warnings.append(f"WARNING: local pollution risk is {here.pollution_risk:.3f}.")

    return LocalObservation(
        agent_id=agent_id,
        x=x,
        y=y,
        radius=radius,
        radius_m=radius_m,
        visible_cells=[current_digest],
        adjacent_tiles=sorted(adjacent, key=lambda row: row["direction"]),
        current_cell_digest=current_digest,
        movement_advisories=sorted(set(advisories)),
        nearby_agents=sorted(nearby_agent_ids),
        nearby_agents_details=nearby_details,
        local_danger=local_danger,
        local_habitability=here.habitability_score,
        construction_sites=current_digest.get("construction_sites", {}),
        structure_count=len(here.structures),
        agent_count=len(here.agents_present),
        density_warnings=warnings,
        active_event=world.metadata.get("active_event"),
        upcoming_event=world.metadata.get("upcoming_event"),
        observation_mode="rule_based",
    )


def _cell_visibility_metrics_m(
    world: GridWorld,
    agent_x: int,
    agent_y: int,
    local_x_m: float,
    local_y_m: float,
    radius_m: float,
    cell,
) -> dict:
    min_distance, farthest_distance, visible_fraction = _cell_distance_and_fraction_m(
        world,
        agent_x,
        agent_y,
        local_x_m,
        local_y_m,
        radius_m,
        cell,
    )
    area_m2 = float(cell.geometry.get("area_m2", 0.0))
    visible_area_m2 = area_m2 * visible_fraction
    if visible_fraction <= 0.0:
        status = "not_visible"
    elif farthest_distance <= radius_m:
        status = "full"
    else:
        status = "partial"
    return {
        "min_distance_m": min_distance,
        "max_corner_distance_m": farthest_distance,
        "visible_fraction": visible_fraction,
        "visible_area_m2": visible_area_m2,
        "fully_visible": status == "full",
        "visibility_status": status,
    }


def _cell_distance_and_fraction_m(
    world: GridWorld,
    agent_x: int,
    agent_y: int,
    local_x_m: float,
    local_y_m: float,
    radius_m: float,
    cell,
) -> tuple[float, float, float]:
    """Approximate tangent-plane visibility from an agent's sub-cell point to a cell rectangle."""
    origin = world.get_cell(agent_x, agent_y)
    origin_geom = origin.geometry
    cell_geom = cell.geometry
    origin_lat = math.radians(float(origin_geom.get("center_lat_deg", 0.0)))
    origin_lon = math.radians(float(origin_geom.get("center_lon_deg", 0.0)))
    cell_lat = math.radians(float(cell_geom.get("center_lat_deg", 0.0)))
    cell_lon = math.radians(float(cell_geom.get("center_lon_deg", 0.0)))
    lon_delta = (cell_lon - origin_lon + math.pi) % (2.0 * math.pi) - math.pi
    mean_lat = (origin_lat + cell_lat) * 0.5
    planet_radius_m = MARS_MEAN_RADIUS_KM * 1000.0
    cell_center_x = lon_delta * math.cos(mean_lat) * planet_radius_m
    cell_center_y = -(cell_lat - origin_lat) * planet_radius_m
    agent_x_from_origin = local_x_m - float(origin_geom.get("width_m", 1.0)) * 0.5
    agent_y_from_origin = local_y_m - float(origin_geom.get("height_m", 1.0)) * 0.5
    width_m = float(cell_geom.get("width_m", 1.0))
    height_m = float(cell_geom.get("height_m", 1.0))
    half_width = width_m * 0.5
    half_height = height_m * 0.5
    relative_center_x = cell_center_x - agent_x_from_origin
    relative_center_y = cell_center_y - agent_y_from_origin
    dx = abs(relative_center_x) - half_width
    dy = abs(relative_center_y) - half_height
    min_distance = math.hypot(max(0.0, dx), max(0.0, dy))
    corners = (
        (relative_center_x - half_width, relative_center_y - half_height),
        (relative_center_x + half_width, relative_center_y - half_height),
        (relative_center_x - half_width, relative_center_y + half_height),
        (relative_center_x + half_width, relative_center_y + half_height),
    )
    farthest_distance = max(math.hypot(cx, cy) for cx, cy in corners)
    if min_distance > radius_m:
        return min_distance, farthest_distance, 0.0
    if farthest_distance <= radius_m:
        return min_distance, farthest_distance, 1.0

    samples_per_axis = 13
    visible_samples = 0
    total_samples = samples_per_axis * samples_per_axis
    for ix in range(samples_per_axis):
        px = relative_center_x - half_width + width_m * (ix + 0.5) / samples_per_axis
        for iy in range(samples_per_axis):
            py = relative_center_y - half_height + height_m * (iy + 0.5) / samples_per_axis
            if math.hypot(px, py) <= radius_m:
                visible_samples += 1
    return min_distance, farthest_distance, visible_samples / total_samples
