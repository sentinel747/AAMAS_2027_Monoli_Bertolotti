from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import math
import os


def _precompute_enabled() -> bool:
    """Toggle di sola misura per il braccio di controllo del Task 0b.

    Il precompute e' semanticamente identico al ricalcolo (stessa funzione pura,
    stesso risultato), quindi non esiste ragione simulativa per spegnerlo: e'
    acceso per default. Resta disattivabile solo perche' il piano di migrazione
    (revisione 2, punto R2.1) esige che lo speed-up attribuito a Rust sia
    misurato contro un Python *con* lo stesso precompute, e per farlo serve poter
    eseguire i due bracci sulla stessa macchina nella stessa sessione.
    """
    return os.environ.get("MARSABM_PRECOMPUTE", "1") != "0"


MARS_MEAN_RADIUS_KM = 3390.0
MARS_SURFACE_AREA_KM2 = 4.0 * math.pi * MARS_MEAN_RADIUS_KM**2
MARS_EQUATORIAL_CIRCUMFERENCE_KM = 2.0 * math.pi * MARS_MEAN_RADIUS_KM
MARS_SOL_HOURS = 24.6
MARS_YEAR_SOLS = 669.6
MARS_YEAR_EARTH_DAYS = 687.0
MARS_SURFACE_GRAVITY_M_S2 = 3.71
MARS_GRAVITY_EARTH_RATIO = 0.38
MARS_MEAN_SURFACE_PRESSURE_MBAR = 6.35
MARS_MEAN_SURFACE_PRESSURE_PA = MARS_MEAN_SURFACE_PRESSURE_MBAR * 100.0
MARS_MEAN_TEMPERATURE_C = -55.0
MARS_MIN_TEMPERATURE_C = -153.0
MARS_MAX_TEMPERATURE_C = 20.0
MARS_ATMOSPHERE_CO2_FRACTION = 0.95
MARS_ATMOSPHERE_N2_FRACTION = 0.03
MARS_ATMOSPHERE_AR_FRACTION = 0.016


@dataclass(frozen=True)
class CellGeometry:
    center_lat_deg: float
    center_lon_deg: float
    width_m: float
    height_m: float
    area_m2: float
    area_km2: float


def _compute_cell_geometry(x: int, y: int, width: int, height: int) -> CellGeometry:
    """Approximate an equal-angle Mars grid cell on a spherical reference Mars."""
    lon_w = -180.0 + 360.0 * x / max(1, width)
    lon_e = -180.0 + 360.0 * (x + 1) / max(1, width)
    lat_n = 90.0 - 180.0 * y / max(1, height)
    lat_s = 90.0 - 180.0 * (y + 1) / max(1, height)
    center_lat = (lat_n + lat_s) * 0.5
    center_lon = (lon_w + lon_e) * 0.5

    lat_n_rad = math.radians(lat_n)
    lat_s_rad = math.radians(lat_s)
    lon_span_rad = math.radians(lon_e - lon_w)
    area_km2 = (MARS_MEAN_RADIUS_KM**2) * lon_span_rad * abs(math.sin(lat_n_rad) - math.sin(lat_s_rad))
    height_km = math.pi * MARS_MEAN_RADIUS_KM / max(1, height)
    width_km = area_km2 / max(height_km, 1e-9)
    return CellGeometry(
        center_lat_deg=center_lat,
        center_lon_deg=center_lon,
        width_m=width_km * 1000.0,
        height_m=height_km * 1000.0,
        area_m2=area_km2 * 1_000_000.0,
        area_km2=area_km2,
    )


# `cell_geometry` e' una funzione pura di (x, y, width, height) e restituisce un
# dataclass FROZEN: memoizzarla non puo' cambiare il comportamento, perche' i
# chiamanti non possono mutare il valore restituito e condividerlo e' quindi
# indistinguibile dal riceverne una copia nuova.
#
# Il profilo del kernel la mostrava chiamata 895.919 volte in 100 step
# (`CellView.geometry`, src/core/views.py:602, la ricalcola a ogni lettura) pur
# essendo invariante per l'intera run: la cache ha al massimo width*height voci
# (864 per una griglia 36x24), quindi `maxsize=None` e' limitato dalla griglia,
# non dalla durata della run.
#
# La memoizzazione e' assegnata al nome pubblico invece di essere avvolta in una
# funzione di dispatch: un ulteriore livello di chiamata Python annullerebbe parte
# del guadagno proprio sul percorso caldo che si vuole accelerare.
cell_geometry = lru_cache(maxsize=None)(_compute_cell_geometry)
if not _precompute_enabled():
    cell_geometry = _compute_cell_geometry


def grid_metadata(width: int, height: int) -> dict[str, float | int | str]:
    cell_area_km2 = MARS_SURFACE_AREA_KM2 / max(1, width * height)
    return {
        "planet": "Mars",
        "mars_mean_radius_km": MARS_MEAN_RADIUS_KM,
        "mars_equatorial_circumference_km": MARS_EQUATORIAL_CIRCUMFERENCE_KM,
        "mars_surface_area_km2": MARS_SURFACE_AREA_KM2,
        "mars_sol_hours": MARS_SOL_HOURS,
        "mars_year_sols": MARS_YEAR_SOLS,
        "mars_year_earth_days": MARS_YEAR_EARTH_DAYS,
        "mars_surface_gravity_m_s2": MARS_SURFACE_GRAVITY_M_S2,
        "mars_gravity_earth_ratio": MARS_GRAVITY_EARTH_RATIO,
        "mars_mean_surface_pressure_mbar": MARS_MEAN_SURFACE_PRESSURE_MBAR,
        "mars_mean_surface_pressure_pa": MARS_MEAN_SURFACE_PRESSURE_PA,
        "mars_mean_temperature_c": MARS_MEAN_TEMPERATURE_C,
        "mars_min_temperature_c": MARS_MIN_TEMPERATURE_C,
        "mars_max_temperature_c": MARS_MAX_TEMPERATURE_C,
        "mars_atmosphere_co2_fraction": MARS_ATMOSPHERE_CO2_FRACTION,
        "mars_atmosphere_n2_fraction": MARS_ATMOSPHERE_N2_FRACTION,
        "mars_atmosphere_ar_fraction": MARS_ATMOSPHERE_AR_FRACTION,
        "grid_projection": "equal_angle_lat_lon",
        "grid_width_cells": width,
        "grid_height_cells": height,
        "mean_cell_area_km2": cell_area_km2,
        "mean_cell_side_km": math.sqrt(cell_area_km2),
    }
