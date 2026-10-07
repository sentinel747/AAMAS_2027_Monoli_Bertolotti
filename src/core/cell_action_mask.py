from __future__ import annotations

"""Vectorized cell-physics filter for preference-policy actions.

The mask deliberately contains no per-agent needs or planning. It mirrors the
cell-only gates in ``action_space.valid_actions_for`` and is computed once per
step for reuse by every colonist in a macro cell.
"""

import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionType, BUILD_ACTIONS
from src.core.arrays import _T_INDEX
from src.world.terrain import TERRENI_NON_EDIFICABILI
from src.agents.build_policy import costruzione_satura
from src.core import constants as C
from src.world.occupancy import housing_slots_from_structures
from src.world.structures import StructureType

# Exact cultivated-biomass thresholds from action_space._has_cultivated_biomass.
_FORAGE_VEG_MIN = 0.25
_FORAGE_SOIL_MIN = 0.35


def compute_cell_masks(cells) -> np.ndarray:
    """Return ``[height, width, action]`` cell-level admissibility flags."""
    height, width = cells.H, cells.W
    masks = np.ones((height, width, pillars.N_ACTIONS), dtype=np.bool_)

    def set_action(action: ActionType, values: np.ndarray) -> None:
        masks[:, :, pillars.ACTION_INDEX[action]] = values

    # Enabled explicitly by compute_cell_proposals only for the cell-only
    # decision experiment; it is not an ordinary action-space capability.
    set_action(
        ActionType.PHYSIOLOGICAL_RECOVERY,
        np.zeros((height, width), dtype=np.bool_),
    )

    set_action(
        ActionType.COLLECT_ICE,
        (cells.water_ice > 0.0) | (cells.cell_res[:, :, C.R["ice"]] > 0.0),
    )
    set_action(
        ActionType.REFILL_WATER,
        (cells.liquid_water > 0.0)
        | (cells.water_ice > 0.0)
        | (cells.cell_res[:, :, C.R["ice"]] > 0.0)
        | (cells.struct_fx[:, :, C.E["water"]] > 0.0),
    )
    set_action(
        ActionType.COLLECT_MINERALS,
        cells.cell_res[:, :, C.R["minerals"]] > 0.0,
    )
    set_action(
        ActionType.COLLECT_MATERIALS,
        cells.cell_res[:, :, C.R["construction_material"]] > 0.0,
    )
    greenhouse_here = (
        cells.struct_count[:, :, C.S[StructureType.GREENHOUSE]] > 0
    )
    cultivated = (cells.vegetation > _FORAGE_VEG_MIN) & (
        cells.proto_soil >= _FORAGE_SOIL_MIN
    )
    set_action(ActionType.FORAGE, greenhouse_here | cultivated)

    edificabile_mask = np.ones((cells.H, cells.W), dtype=np.bool_)
    for _terreno in TERRENI_NON_EDIFICABILI:
        edificabile_mask &= cells.terrain != _T_INDEX[_terreno]
    colonists = np.maximum(1, cells.occupancy.astype(np.int64))
    alloggi = housing_slots_from_structures(
        cells.struct_count[:, :, C.S[StructureType.SHELTER]],
        cells.struct_count[:, :, C.S[StructureType.HABITAT]],
        cells.struct_count[:, :, C.S[StructureType.INFIRMARY]],
    )
    for action, structure_type in BUILD_ACTIONS.items():
        structure_index = C.S[structure_type]
        # Una regola sola per la saturazione: vedi `costruzione_satura`.
        saturated = costruzione_satura(
            structure_type,
            cells.struct_count[:, :, structure_index],
            colonists,
            alloggi,
        )
        site_open = cells.site_progress[:, :, structure_index] >= 0.0
        # La geografia vieta il cantiere dove il terreno non lo sostiene: vedi
        # `TERRENI_NON_EDIFICABILI`. Un cantiere gia' aperto non fa eccezione,
        # perche' non potrebbe esistere.
        set_action(action, (~saturated | site_open) & edificabile_mask)

    return masks
