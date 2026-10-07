from __future__ import annotations

"""Mapping between preference pillars and simulation actions.

This is the single source of truth shared by the readable per-agent policy and
the vectorized kernel policy described in the preference-decision design.
"""

from src.agents.action_space import ActionType
import numpy as np

N_PILLARS = 6
P_SUSTENANCE, P_RESOURCES, P_BUILD, P_LIFE, P_SOCIAL, P_EXPLORE = range(N_PILLARS)

PILLAR_ACTIONS: dict[int, tuple[ActionType, ...]] = {
    P_SUSTENANCE: (
        ActionType.DRINK_WATER,
        ActionType.REFILL_WATER,
        ActionType.EAT_FOOD,
        ActionType.FORAGE,
        ActionType.COLLECT_ICE,
    ),
    P_RESOURCES: (
        ActionType.COLLECT_MINERALS,
        ActionType.COLLECT_MATERIALS,
    ),
    P_BUILD: (
        ActionType.BUILD_SHELTER,
        ActionType.BUILD_SOLAR_ARRAY,
        ActionType.BUILD_OXYGEN_PLANT,
        ActionType.BUILD_GREENHOUSE,
        ActionType.BUILD_WATER_EXTRACTOR,
        ActionType.BUILD_HEATER,
        ActionType.BUILD_INFIRMARY,
        ActionType.BUILD_HABITAT,
        ActionType.BUILD_RESEARCH_LAB,
        ActionType.BUILD_STORAGE_DEPOT,
        ActionType.BUILD_WEATHER_STATION,
        ActionType.MAINTAIN_STRUCTURE,
    ),
    P_LIFE: (
        ActionType.PHYSIOLOGICAL_RECOVERY,
        ActionType.REST,
        ActionType.USE_MED_KIT,
    ),
    # Social coordination and resource sharing are cell services in the
    # preference policy: they never consume a seven-day agent action.
    P_SOCIAL: (),
    P_EXPLORE: (
        ActionType.MOVE,
        ActionType.EXPLORE,
        ActionType.OBSERVE,
    ),
}

AUTOMATIC_CELL_SERVICES = frozenset(
    {ActionType.COMMUNICATE, ActionType.SHARE_RESOURCE}
)

ACTION_TO_PILLAR: dict[ActionType, int] = {
    action: pillar for pillar, actions in PILLAR_ACTIONS.items() for action in actions
}

FALLBACK_ACTION = ActionType.DO_NOTHING

ACTION_ORDER: tuple[ActionType, ...] = tuple(ActionType)
ACTION_INDEX: dict[ActionType, int] = {
    action: index for index, action in enumerate(ACTION_ORDER)
}
N_ACTIONS = len(ACTION_ORDER)

# Dedicated streams keep preference sampling and decisions independent from
# the stdlib Random(seed) sequence used by the historical tree policy.
_STREAM_PREF = 101
_STREAM_DECISION = 202


def sample_pillar_preferences(seed: int, agent_index: int) -> np.ndarray:
    """Return deterministic Dirichlet preferences for one agent."""
    rng = np.random.default_rng(
        np.random.SeedSequence([int(seed), int(agent_index), _STREAM_PREF])
    )
    return rng.dirichlet(np.ones(N_PILLARS, dtype=np.float64)).astype(
        np.float64, copy=False
    )


def decision_rng(seed: int, step_index: int) -> np.random.Generator:
    """Return the deterministic random stream used for one decision step."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), int(step_index), _STREAM_DECISION])
    )
