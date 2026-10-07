from __future__ import annotations

"""Shared operational-capacity formulas for object and vector cell models."""

import numpy as np


BASE_CELL_OCCUPANCY_CAPACITY = 10.0


def occupancy_capacity_from_housing(shelters, habitats, infirmaries):
    """Return supported local occupants using existing housing units.

    The historical ten-person threshold remains the capacity of an improvised
    camp. Housing can raise it: shelter=1, habitat=2, infirmary=0.5, matching
    the infrastructure-coverage contract in build_policy/needs. Scalars and
    numpy arrays are both accepted.
    """
    capacity = np.maximum(
        BASE_CELL_OCCUPANCY_CAPACITY,
        housing_slots_from_structures(shelters, habitats, infirmaries),
    )
    return float(capacity) if capacity.ndim == 0 else capacity


def housing_slots_from_structures(shelters, habitats, infirmaries):
    """I posti letto costruiti: rifugio=1, habitat=2, infermeria=0,5.

    **Una copia sola (2026-08-30).** La stessa somma era scritta a mano in nove
    punti — capienza di cella, capienza di supporto vitale, due maschere di
    saturazione, il bisogno di alloggi, tre metriche e il cancello fisiologico.
    Coincidevano ancora, ma e' la famiglia di difetto che questo modello produce
    di piu': due formule per la stessa grandezza che prima o poi divergono.
    Senza pavimento: il minimo di dieci posti dell'accampamento improvvisato e'
    una proprieta' della cella, non degli alloggi, e sta in
    ``occupancy_capacity_from_housing``.
    """
    return (
        np.asarray(shelters, dtype=np.float64)
        + np.asarray(habitats, dtype=np.float64) * 2.0
        + np.asarray(infirmaries, dtype=np.float64) * 0.5
    )


def overcrowding_excess_from_housing(
    occupancy, shelters, habitats, infirmaries
):
    capacity = occupancy_capacity_from_housing(shelters, habitats, infirmaries)
    excess = np.maximum(0.0, np.asarray(occupancy, dtype=np.float64) - capacity)
    return float(excess) if excess.ndim == 0 else excess
