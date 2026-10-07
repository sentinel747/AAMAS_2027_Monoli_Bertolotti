from __future__ import annotations

import math
import random

from .colony_site import colony_spawn_cells, record_colony_site_metadata
from .grid import GridWorld
from .structures import Structure, StructureType


INITIAL_STRUCTURE_KEYS: dict[str, StructureType] = {
    "shelter": StructureType.SHELTER,
    "habitat": StructureType.HABITAT,
    "greenhouse": StructureType.GREENHOUSE,
    "solar_array": StructureType.SOLAR_ARRAY,
    "oxygen_plant": StructureType.OXYGEN_PLANT,
    # **Senza questa riga la dotazione non arriva a terra (2026-09-01).** La
    # configurazione poteva chiedere i pozzi, `dotazione_iniziale` li
    # derivava e il pannello li mostrava, ma qui la chiave non esisteva e i
    # pozzi venivano scartati in silenzio: la colonia madre nasceva con
    # capienza vitale ZERO, quindi senza figli e senza spedizioni.
    "water_extractor": StructureType.WATER_EXTRACTOR,
    "storage_depot": StructureType.STORAGE_DEPOT,
    "weather_station": StructureType.WEATHER_STATION,
}


def seed_initial_colony_support(world: GridWorld, config: dict, seed: int = 0) -> list[Structure]:
    colony_cfg = config.get("colony", {}) if isinstance(config.get("colony"), dict) else {}
    agents_cfg = config.get("agents", {}) if isinstance(config.get("agents"), dict) else {}
    center_x = min(world.width - 1, max(0, int(colony_cfg.get("start_x", world.width // 2))))
    center_y = min(world.height - 1, max(0, int(colony_cfg.get("start_y", world.height // 2))))
    record_colony_site_metadata(world, center_x, center_y)

    initial_structures = colony_cfg.get("initial_structures")
    if not isinstance(initial_structures, dict):
        initial_structures = agents_cfg.get("initial_structures") if isinstance(agents_cfg.get("initial_structures"), dict) else {}
    if not initial_structures:
        return []

    rng = random.Random(seed + 91001)
    created: list[Structure] = []

    flat: list[StructureType] = []
    for key, structure_type in INITIAL_STRUCTURE_KEYS.items():
        count = max(0, int(initial_structures.get(key, 0) or 0))
        flat.extend([structure_type] * count)

    # Same footprint as spawn_initial_agents: structures follow the settlers
    # onto extra cells only when the landing party outgrows a single cell.
    agent_count = max(1, int(agents_cfg.get("count", 1) or 1))
    spawn_cells = colony_spawn_cells(world, center_x, center_y, agent_count)

    total = max(1, len(flat))
    for index, structure_type in enumerate(flat):
        cell_x, cell_y = spawn_cells[index % len(spawn_cells)]
        cell = world.get_cell(cell_x, cell_y)
        cell.is_spawn = True
        geom = cell.geometry
        width_m = max(1.0, float(geom.get("width_m", 1.0)))
        height_m = max(1.0, float(geom.get("height_m", 1.0)))
        angle = (2.0 * math.pi * index / total) + rng.uniform(-0.10, 0.10)
        radius = 0.18 + 0.28 * ((index % 5) / 4.0)
        local_x = width_m * (0.5 + math.cos(angle) * radius)
        local_y = height_m * (0.5 + math.sin(angle) * radius)
        structure = Structure(
            structure_type,
            cell_x,
            cell_y,
            local_x_m=max(0.0, min(width_m, local_x)),
            local_y_m=max(0.0, min(height_m, local_y)),
            owner="initial_support",
        )
        world.add_structure(structure)
        created.append(structure)

    if created:
        world.log_event(
            "initial_colony_support_seeded",
            f"seeded {len(created)} initial colony support structures",
            structures=[structure.to_dict() for structure in created],
        )
    return created
