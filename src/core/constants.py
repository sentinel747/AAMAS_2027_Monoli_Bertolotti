from __future__ import annotations

from enum import IntEnum

import numpy as np

from src.world.resources import ResourceBundle
from src.world.structures import BUILD_COSTS, BUILD_TIME, Structure, StructureType

RESOURCES: tuple[str, ...] = tuple(ResourceBundle.__dataclass_fields__)
R: dict[str, int] = {name: i for i, name in enumerate(RESOURCES)}
NR: int = len(RESOURCES)

STRUCTURES: tuple[StructureType, ...] = tuple(StructureType)
S: dict[StructureType, int] = {t: i for i, t in enumerate(STRUCTURES)}
NS: int = len(STRUCTURES)

EFFECTS: tuple[str, ...] = (
    "oxygen", "habitability", "food", "water", "healing_bonus",
    "temperature", "energy", "biomass", "knowledge", "storage",
    "population_support", "prediction_range",
)
E: dict[str, int] = {name: i for i, name in enumerate(EFFECTS)}
NE: int = len(EFFECTS)

# Necessità pubblicate dalle celle insediate (spec sez. 2.3).
NEEDS: tuple[str, ...] = (
    "water", "food", "oxygen", "materials",
    "build_solar", "build_oxygen_plant", "build_greenhouse",
    "maintenance", "habitat_pressure", "build_water",
)
ND: dict[str, int] = {name: i for i, name in enumerate(NEEDS)}
NN: int = len(NEEDS)


class ACT(IntEnum):
    OBSERVE = 0
    MOVE = 1
    COLLECT = 2      # param: indice risorsa (ice/minerals/materials/forage)
    BUILD = 3        # param: indice struttura
    MAINTAIN = 4     # param: indice struttura
    EAT = 5
    DRINK = 6
    REST = 7
    MED_KIT = 8
    COMMUNICATE = 9  # param: indice riga destinatario


ACT_TO_ACTIONTYPE: dict[int, str] = {
    int(ACT.OBSERVE): "observe", int(ACT.MOVE): "move", int(ACT.COLLECT): "collect",
    int(ACT.BUILD): "build", int(ACT.MAINTAIN): "maintain_structure",
    int(ACT.EAT): "eat_food", int(ACT.DRINK): "drink_water", int(ACT.REST): "rest",
    int(ACT.MED_KIT): "use_med_kit", int(ACT.COMMUNICATE): "communicate",
}

## Task 14 fix (equivalence audit, controller decision - exact equivalence):
# BUILD_COST_M/STRUCT_FX_M are float64, not float32 - matmuls against the
# (now also float64) AgentArrays/CellArrays columns stay float64 end to end
# instead of silently truncating back to float32 precision partway through.
BUILD_COST_M = np.zeros((NS, NR), dtype=np.float64)
for _st, _bundle in BUILD_COSTS.items():
    for _res, _v in _bundle.to_dict().items():
        BUILD_COST_M[S[_st], R[_res]] = _v

BUILD_TIME_V = np.array([BUILD_TIME[t] for t in STRUCTURES], dtype=np.int16)

STRUCT_FX_M = np.zeros((NS, NE), dtype=np.float64)
for _st in STRUCTURES:
    for _eff, _v in Structure(type=_st, x=0, y=0, integrity=1.0).local_effect.items():
        STRUCT_FX_M[S[_st], E[_eff]] = _v

# Agent suit-reserve refill parameters (step_effects.py:89-96, kernel_biology.py's
# `_serialized_reserve_draw`). Homed here (rather than in src/simulation/step_effects.py,
# which imports src.world.grid) so src/core/kernel_biology.py — and transitively
# src/core/views.py — do not have to reach into the simulation layer for three
# float literals. step_effects.py re-exports these names unchanged (pure
# re-export, zero behavior change) so existing `from src.simulation.step_effects
# import AGENT_ENERGY_RESERVE_CAP` call sites keep working.
AGENT_ENERGY_RESERVE_CAP = 3.0
AGENT_OXYGEN_RESERVE_CAP = 3.0
AGENT_REFILL_PER_STEP = 0.25
