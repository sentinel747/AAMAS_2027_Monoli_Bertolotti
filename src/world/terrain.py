from __future__ import annotations

from enum import Enum


class TerrainType(str, Enum):
    REGOLITH_PLAIN = "regolith_plain"
    CRATER = "crater"
    MOUNTAIN = "mountain"
    CANYON = "canyon"
    ICE_DEPOSIT = "ice_deposit"
    MINERAL_RICH = "mineral_rich_area"
    DUST_FIELD = "dust_field"
    LAVA_TUBE = "lava_tube"
    FROZEN_BASIN = "frozen_basin"


TERRAIN_HABITABILITY = {
    TerrainType.REGOLITH_PLAIN: 0.08,
    TerrainType.CRATER: 0.04,
    TerrainType.MOUNTAIN: 0.03,
    TerrainType.CANYON: 0.06,
    TerrainType.ICE_DEPOSIT: 0.10,
    TerrainType.MINERAL_RICH: 0.07,
    TerrainType.DUST_FIELD: 0.02,
    TerrainType.LAVA_TUBE: 0.18,
    TerrainType.FROZEN_BASIN: 0.09,
}

# Traversal / environmental risk (0 = mild, 1 = extreme). Used for perception and injury on entry.
TERRAIN_TRAVERSAL_RISK: dict[TerrainType, float] = {
    TerrainType.REGOLITH_PLAIN: 0.10,
    TerrainType.CRATER: 0.28,
    TerrainType.MOUNTAIN: 0.22,
    TerrainType.CANYON: 0.38,
    TerrainType.ICE_DEPOSIT: 0.14,
    TerrainType.MINERAL_RICH: 0.18,
    TerrainType.DUST_FIELD: 0.30,
    TerrainType.LAVA_TUBE: 0.93,
    TerrainType.FROZEN_BASIN: 0.32,
}


#: **Dove non si costruisce (2026-09-01).**
#:
#: Marte non e' una piana uniforme: ospita l'Olympus Mons, il rilievo piu' alto
#: del sistema solare, e la Valles Marineris. Un avamposto non si pianta su una
#: parete rocciosa ne' dentro un canyon, e finora il modello lo permetteva
#: ovunque — reso ancora piu' evidente dal fondo regolitico, che ha dato a ogni
#: cella di che costruire.
#:
#: Non e' una soglia scelta su un'altimetria: sono due dei nove terreni che il
#: generatore gia' produce, e dipendono dal seme della mappa come tutto il resto.
#: Sono l'8,7% del pianeta (misurato sul seme 9: montagna 6,67%, canyon 1,99%).
#:
#: **Si vieta il CANTIERE, non il passaggio ne' l'estrazione.** Un colono puo'
#: attraversare una montagna, raccogliervi minerali e scavarvi ghiaccio: cio'
#: che non puo' fare e' erigervi una struttura. La geografia torna cosi' a
#: vincolare la forma dell'insediamento senza togliere nulla a cio' che il
#: pianeta offre.
TERRENI_NON_EDIFICABILI = frozenset(
    {TerrainType.MOUNTAIN, TerrainType.CANYON}
)


def edificabile(cell) -> bool:
    """Si puo' erigere una struttura su questa cella?"""
    return getattr(cell, "terrain", None) not in TERRENI_NON_EDIFICABILI


def traversal_risk_for_cell(cell) -> float:
    """Combined risk when standing or stepping into this cell."""
    if getattr(cell, "is_spawn", False):
        return 0.0
    base = TERRAIN_TRAVERSAL_RISK.get(cell.terrain, 0.2)
    structure_heat = cell.structure_heat() if hasattr(cell, "structure_heat") else 0.0
    cold_modifier = max(0.0, -(float(getattr(cell, "local_temperature_modifier", 0.0)) + structure_heat) - 10.0)
    extra = (
        0.12 * min(1.0, cell.radiation_level)
        + 0.08 * min(1.0, cell.dust_level)
        + 0.15 * min(1.0, cell.pollution_risk)
        + 0.24 * min(1.0, float(getattr(cell, "polar_severity", 0.0)))
        + 0.012 * min(20.0, cold_modifier)
    )
    return max(0.0, min(1.0, base + extra))


def terrain_hazard_tier(risk: float) -> str:
    if risk >= 0.82:
        return "extreme"
    if risk >= 0.55:
        return "high"
    if risk >= 0.30:
        return "moderate"
    return "low"
