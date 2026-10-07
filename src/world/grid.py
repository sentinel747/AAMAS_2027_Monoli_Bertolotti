from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .cell import Cell
from .structures import Structure
from .terrain import TerrainType


import numpy as np

@dataclass
class GridWorld:
    width: int
    height: int
    cells: list[list[Cell]]
    day: int = 0
    step: int = 0
    events: list[dict] = field(default_factory=list)
    planetary_state: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    
    # NumPy caches for performance
    _habitability_cache: np.ndarray = field(init=False)
    _vegetation_cache: np.ndarray = field(init=False)
    _radiation_cache: np.ndarray = field(init=False)
    _temperature_cache: np.ndarray = field(init=False)
    
    # Spatial indexing for agents
    _agent_positions: dict[str, tuple[int, int]] = field(default_factory=dict)
    _structure_positions: set[tuple[int, int]] = field(default_factory=set)

    def __post_init__(self):
        self._habitability_cache = np.zeros((self.height, self.width), dtype=np.float32)
        self._vegetation_cache = np.zeros((self.height, self.width), dtype=np.float32)
        self._radiation_cache = np.zeros((self.height, self.width), dtype=np.float32)
        self._temperature_cache = np.zeros((self.height, self.width), dtype=np.float32)
        self._structure_positions = {
            (x, y)
            for y, row in enumerate(self.cells)
            for x, cell in enumerate(row)
            if cell.structures
        }
        self.metadata.setdefault(
            "_total_ice_cache",
            float(sum(cell.water_ice + cell.resources.ice for row in self.cells for cell in row)),
        )
        self.metadata.setdefault(
            "_explored_count",
            sum(1 for row in self.cells for cell in row if cell.explored),
        )
        self.sync_caches_from_cells()

    def sync_caches_from_cells(self):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                self._habitability_cache[y, x] = cell.habitability_score
                self._vegetation_cache[y, x] = cell.vegetation_biomass
                self._radiation_cache[y, x] = cell.radiation_level
                self._temperature_cache[y, x] = cell.local_temperature_modifier

    def sync_cells_from_caches(self):
        for y in range(self.height):
            for x in range(self.width):
                cell = self.cells[y][x]
                cell.habitability_score = float(self._habitability_cache[y, x])
                cell.vegetation_biomass = float(self._vegetation_cache[y, x])
                cell.radiation_level = float(self._radiation_cache[y, x])
                cell.local_temperature_modifier = float(self._temperature_cache[y, x])

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get_cell(self, x: int, y: int) -> Cell:
        if not self.in_bounds(x, y):
            raise IndexError(f"cell out of bounds: {x},{y}")
        return self.cells[y][x]

    def neighbors(self, x: int, y: int, radius: int = 1) -> list[Cell]:
        cells: list[Cell] = []
        seen: set[tuple[int, int]] = set()
        for yy in range(max(0, y - radius), min(self.height, y + radius + 1)):
            for raw_x in range(x - radius, x + radius + 1):
                # Longitude is periodic on Mars: the west neighbour of x=0
                # is x=width-1. Latitude remains bounded at the poles.
                xx = raw_x % self.width
                key = (xx, yy)
                if key == (x, y) or key in seen:
                    continue
                seen.add(key)
                cells.append(self.get_cell(xx, yy))
        return cells

    def place_agent(self, agent_id: str, x: int, y: int, local_x_m: float | None = None, local_y_m: float | None = None) -> None:
        cell = self.get_cell(x, y)
        if agent_id not in cell.agents_present:
            cell.agents_present.append(agent_id)
        if local_x_m is not None and local_y_m is not None:
            cell.agent_positions_m[agent_id] = {"x": float(local_x_m), "y": float(local_y_m)}
        self._agent_positions[agent_id] = (x, y)

    def remove_agent(self, agent_id: str, x: int, y: int) -> None:
        cell = self.get_cell(x, y)
        if agent_id in cell.agents_present:
            cell.agents_present.remove(agent_id)
        cell.agent_positions_m.pop(agent_id, None)
        if agent_id in self._agent_positions:
            del self._agent_positions[agent_id]

    def get_agents_in_range(self, x: int, y: int, radius: int) -> list[str]:
        """Find agents within a bounding box efficiently."""
        nearby = []
        y_min, y_max = y - radius, y + radius
        for aid, (ax, ay) in self._agent_positions.items():
            raw_dx = abs(ax - x)
            wrapped_dx = min(raw_dx, self.width - raw_dx)
            if wrapped_dx <= radius and y_min <= ay <= y_max:
                nearby.append(aid)
        return nearby

    def move_agent(
        self,
        agent_id: str,
        old_x: int,
        old_y: int,
        new_x: int,
        new_y: int,
        local_x_m: float | None = None,
        local_y_m: float | None = None,
    ) -> bool:
        if not self.in_bounds(new_x, new_y):
            return False
        self.remove_agent(agent_id, old_x, old_y)
        self.place_agent(agent_id, new_x, new_y, local_x_m, local_y_m)
        self.log_event("agent_moved", f"{agent_id} moved to ({new_x},{new_y})", agent_id=agent_id, x=new_x, y=new_y)
        return True

    def add_structure(self, structure: Structure) -> None:
        cell = self.get_cell(structure.x, structure.y)
        cell.structures.append(structure)
        self._structure_positions.add((structure.x, structure.y))

        cell.recompute_habitability(self.planetary_state)
        self._habitability_cache[structure.y, structure.x] = cell.habitability_score
        self._vegetation_cache[structure.y, structure.x] = cell.vegetation_biomass
        self.log_event("structure_built", f"{structure.type.value} built at ({structure.x},{structure.y})", structure=structure.to_dict())

    def structure_type_counts(self) -> dict[str, int]:
        """Return totals by type using the maintained structure-cell index."""
        counts: dict[str, int] = {}
        for x, y in self._structure_positions:
            for structure in self.get_cell(x, y).structures:
                key = structure.type.value
                counts[key] = counts.get(key, 0) + 1
        return counts

    def mark_explored(self, x: int, y: int) -> None:
        cell = self.get_cell(x, y)
        if not cell.explored:
            cell.explored = True
            self.metadata["_explored_count"] = int(self.metadata.get("_explored_count", 0)) + 1

    def adjust_total_ice_cache(self, delta: float) -> None:
        self.metadata["_total_ice_cache"] = max(0.0, float(self.metadata.get("_total_ice_cache", 0.0)) + float(delta))

    def log_event(self, event_type: str, message: str, **data) -> None:
        event = {
            "event_id": f"{len(self.events) + 1:08d}",
            "wall_time": datetime.now().isoformat(timespec="seconds"),
            # Both engines write world.step at the top of each step; stamping it
            # here keeps every event addressable in time without forcing each
            # call site to thread the step through.
            "step": data.pop("step", getattr(self, "step", None)),
            "day": self.day,
            "type": event_type,
            "message": message,
            "data": data,
        }
        self.events.append(event)
        if event_type.startswith("runtime_") or event_type in {"agent_died", "step_completed"}:
            print(f"[event {event['event_id']}] day={self.day} type={event_type} message={message}", flush=True)

    def snapshot(self) -> dict:
        return {
            "width": self.width,
            "height": self.height,
            "day": self.day,
            "metadata": dict(self.metadata),
            "cells": [cell.to_public_dict() for row in self.cells for cell in row],
            "events": self.events[-200:],
        }

    def compact_chunk(self, x: int = 0, y: int = 0, zoom: int = 1, size: int = 50) -> dict:
        size = max(1, int(size))
        start_x = max(0, min(self.width - 1, int(x)))
        start_y = max(0, min(self.height - 1, int(y)))
        max_x = min(self.width, start_x + size)
        max_y = min(self.height, start_y + size)
        cells = [self.get_cell(xx, yy).to_public_dict() for yy in range(start_y, max_y) for xx in range(start_x, max_x)]
        return {
            "x": start_x,
            "y": start_y,
            "zoom": zoom,
            "width": max_x - start_x,
            "height": max_y - start_y,
            "world_width": self.width,
            "world_height": self.height,
            "cells": cells,
        }

    def render_map_payload(self, include_static: bool = False) -> dict:
        """Compact full-planet snapshot for GUI polling (flat row-major arrays, index = y * width + x)."""
        terrain_types = list(TerrainType)
        terrain_index = {terrain: index for index, terrain in enumerate(terrain_types)}
        terrain: list[int] = []
        elevation: list[float] = []
        polar_severity: list[float] = []
        habitability: list[float] = []
        vegetation: list[float] = []
        water_ice: list[float] = []
        dust: list[float] = []
        radiation: list[float] = []
        explored: list[int] = []
        structures: list[dict] = []
        for row in self.cells:
            for cell in row:
                if include_static:
                    terrain.append(terrain_index[cell.terrain])
                    elevation.append(round(cell.elevation, 2))
                    polar_severity.append(round(cell.polar_severity, 2))
                habitability.append(round(cell.habitability_score, 3))
                vegetation.append(round(cell.vegetation_biomass, 3))
                water_ice.append(round(cell.water_ice, 2))
                dust.append(round(cell.dust_level, 2))
                radiation.append(round(cell.radiation_level, 2))
                explored.append(int(cell.explored))
                for structure in cell.structures:
                    structures.append(
                        {
                            "x": cell.x,
                            "y": cell.y,
                            "type": structure.type.value,
                            "local_x_m": round(float(structure.local_x_m or 0.0), 1),
                            "local_y_m": round(float(structure.local_y_m or 0.0), 1),
                        }
                    )
        payload: dict = {
            "world_width": self.width,
            "world_height": self.height,
            "day": int(self.day),
        }
        if include_static:
            payload["static"] = {
                "terrain": terrain,
                "terrain_types": [t.value for t in terrain_types],
                "elevation": elevation,
                "polar_severity": polar_severity,
            }
        payload["dynamic"] = {
            "habitability": habitability,
            "vegetation": vegetation,
            "water_ice": water_ice,
            "dust": dust,
            "radiation": radiation,
            "explored": explored,
        }
        payload["structures"] = structures
        return payload

    def recompute_habitability(self) -> None:
        # First ensure caches are up to date from cells (where biology/structures modified them)
        self.sync_caches_from_cells()
        
        # In a real production system, the whole habitability logic from Cell.py 
        # should be moved here into vectorized NumPy operations.
        # For now, we still call the cell method but speed up the iteration 
        # or use the caches for global metrics.
        for row in self.cells:
            for cell in row:
                cell.recompute_habitability(self.planetary_state)
        
        # Update caches after recomputation
        self.sync_caches_from_cells()

    def metrics(self, include_planetary: bool = True) -> dict[str, float]:
        # Using vectorized caches for global metrics
        total_cells = max(1, self.width * self.height)
        metrics = {
            "average_habitability": float(np.mean(self._habitability_cache)),
            "exploration_coverage": float(self.metadata.get("_explored_count", 0)) / total_cells,
            "total_ice": float(self.metadata.get("_total_ice_cache", 0.0)),
            "vegetation": float(np.sum(self._vegetation_cache)),
        }
        
        if not self._structure_positions:
            self._structure_positions = {
                (x, y)
                for y, row in enumerate(self.cells)
                for x, cell in enumerate(row)
                if cell.structures
            }
        structures_built = 0
        greenhouses = 0
        oxygen_plants = 0
        for x, y in self._structure_positions:
            if not self.in_bounds(x, y):
                continue
            for structure in self.cells[y][x].structures:
                structures_built += 1
                type_value = structure.type.value
                if type_value == "greenhouse":
                    greenhouses += 1
                elif type_value == "oxygen_plant":
                    oxygen_plants += 1
        metrics["structures_built"] = float(structures_built)
        metrics["greenhouses"] = float(greenhouses)
        metrics["oxygen_plants"] = float(oxygen_plants)

        conoscenza_colonia = float(
            sum(cell.resources.knowledge for row in self.cells for cell in row)
        )

        if include_planetary:
            metrics.update(self.planetary_state)

        # **Dopo il merge, e non prima.** `planetary_state` porta
        # `scientific_knowledge: 0.0` fra i propri default anche quando il layer
        # e' disattivato, quindi scrivendo prima il valore della colonia veniva
        # sovrascritto da quello zero — che e' esattamente il difetto che questa
        # riga corregge. Il layer planetario, quando produce davvero un valore,
        # continua a vincere: li' la grandezza ha una dinamica propria.
        if not float(metrics.get("scientific_knowledge", 0.0) or 0.0):
            metrics["scientific_knowledge"] = conoscenza_colonia
            # I bonus vanno ricalcolati con essa: `get_active_bonuses` gira solo
            # dentro il layer planetario, quindi senza questa riga la colonia
            # accumulava conoscenza e non sbloccava mai nulla.
            from src.simulation.planetary_coupling import active_bonuses

            for chiave, valore in active_bonuses(conoscenza_colonia).items():
                if not isinstance(valore, list):
                    metrics[chiave] = float(valore)
        return metrics
