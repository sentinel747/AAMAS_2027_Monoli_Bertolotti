from __future__ import annotations

import numpy as np


def update_biology_cells(world, planetary_metrics: dict, dt: float, full_grid: bool = False) -> dict[str, float]:
    """Advance biology without paying full-grid cost when empty cells cannot change.

    Empty baseline Mars cells with no agents, structures, construction sites,
    biomass, soil development, organic matter, liquid water or pollution have no
    biological state to advance. They can be skipped without dropping run
    artifacts; active and already-mutated cells are still updated exactly.
    """
    updated = 0
    melt_active = (
        float(planetary_metrics.get("liquid_water_stability", 0.0)) > 0.0
        and float(planetary_metrics.get("mean_temperature_c", -63.0)) > -10.0
    )
    if full_grid:
        iterator = ((x, y, world.get_cell(x, y)) for y in range(world.height) for x in range(world.width))
    else:
        iterator = (
            (x, y, cell)
            for y, row in enumerate(world.cells)
            for x, cell in enumerate(row)
            if _biology_cell_can_change(cell, melt_active)
        )

    ice_delta = 0.0
    for x, y, cell in iterator:
        ice_before = cell.water_ice + cell.resources.ice
        cell.update_biology(planetary_metrics, dt)
        ice_delta += (cell.water_ice + cell.resources.ice) - ice_before
        world._habitability_cache[y, x] = cell.habitability_score
        world._vegetation_cache[y, x] = cell.vegetation_biomass
        world._radiation_cache[y, x] = cell.radiation_level
        world._temperature_cache[y, x] = cell.local_temperature_modifier
        updated += 1

    if ice_delta:
        # Keep the world total-ice metric coherent with melt/refreeze exchanges.
        world.adjust_total_ice_cache(ice_delta)

    return {
        "vegetation": float(np.sum(world._vegetation_cache)),
        "updated_cells": float(updated),
    }


def _biology_cell_can_change(cell, melt_active: bool = False) -> bool:
    if cell.agents_present or cell.structures or cell.construction_sites:
        return True
    if cell.vegetation_biomass > 0.0 or cell.proto_soil_development > 0.0 or cell.organic_matter > 0.0:
        return True
    if cell.liquid_water > 0.0 or cell.pollution_risk > 0.0:
        return True
    if melt_active and cell.water_ice > 0.0:
        return True
    nutrients = getattr(cell, "nutrients", {}) or {}
    return any(float(value or 0.0) > 0.0 for value in nutrients.values())
