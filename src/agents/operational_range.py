"""Single source of truth for agent perception and movement range."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


MIN_OPERATIONAL_RANGE_M = 100.0
MAX_OPERATIONAL_RANGE_M = 59_000.0
DEFAULT_OPERATIONAL_RANGE_M = 59_000.0


def clamp_operational_range_m(value: Any) -> float:
    """Clamp a user/config value to the supported 0.1-59 km interval."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        numeric = DEFAULT_OPERATIONAL_RANGE_M
    return min(MAX_OPERATIONAL_RANGE_M, max(MIN_OPERATIONAL_RANGE_M, numeric))


def operational_range_from_config(
    agents_config: Mapping[str, Any] | None,
    default: float = DEFAULT_OPERATIONAL_RANGE_M,
) -> float:
    """Read the canonical range, safely migrating legacy split configurations.

    New payloads use ``operational_range_m``. If an old payload contains only
    the separate vision/movement values, the smaller value is selected: this
    preserves the strictest historical bound and prevents an invisible move.
    """
    config = agents_config or {}
    if config.get("operational_range_m") is not None:
        return clamp_operational_range_m(config["operational_range_m"])

    legacy = [
        config[key]
        for key in ("vision_radius_m", "movement_distance_m_per_step")
        if config.get(key) is not None
    ]
    if legacy:
        return min(clamp_operational_range_m(value) for value in legacy)
    return clamp_operational_range_m(default)


def synchronize_operational_range_config(config: dict[str, Any]) -> float:
    """Normalize a scenario in place and return its canonical range."""
    agents = config.get("agents")
    if not isinstance(agents, dict):
        agents = {}
        config["agents"] = agents
    range_m = operational_range_from_config(agents)
    agents["operational_range_m"] = range_m
    agents["vision_radius_m"] = range_m
    agents["movement_distance_m_per_step"] = range_m
    return range_m
