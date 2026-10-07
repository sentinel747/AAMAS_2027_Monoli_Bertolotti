from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from src.world.world_generator import MAP_PROFILES


DEFAULT_STANDARD_SCENARIOS_PATH = Path("configs/scenarios/standard_v1.yaml")


def load_standard_scenarios(path: str | Path = DEFAULT_STANDARD_SCENARIOS_PATH) -> dict[str, Any]:
    source = Path(path)
    data = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
    scenarios = data.get("scenarios") or []
    if not isinstance(scenarios, list):
        raise ValueError("standard scenarios file must contain a scenarios list")
    ids: set[str] = set()
    for row in scenarios:
        scenario_id = row.get("id")
        if not scenario_id:
            raise ValueError("each standard scenario must define id")
        if scenario_id in ids:
            raise ValueError(f"duplicate standard scenario id: {scenario_id}")
        ids.add(scenario_id)
        if not isinstance(row.get("config"), dict):
            raise ValueError(f"standard scenario {scenario_id} must define config")
        world_cfg = row["config"].get("world") or {}
        map_profile = world_cfg.get("map_profile", "balanced")
        if map_profile not in MAP_PROFILES:
            raise ValueError(f"standard scenario {scenario_id} has unknown map_profile: {map_profile}")
        social_cfg = row["config"].get("social") or {}
        if float(social_cfg.get("earth_mars_delay_minutes", 0.0) or 0.0) < 0:
            raise ValueError(f"standard scenario {scenario_id} has negative earth_mars_delay_minutes")
        climate_cfg = row["config"].get("climate") or {}
        if climate_cfg and climate_cfg.get("scenario") not in {None, "climatology", "clim_minEUV", "clim_maxEUV", "cold", "warm", "strm", "dust_storm", "dusty", "clear"}:
            raise ValueError(f"standard scenario {scenario_id} has unknown climate scenario: {climate_cfg.get('scenario')}")
    return data


def list_standard_scenarios(path: str | Path = DEFAULT_STANDARD_SCENARIOS_PATH) -> list[dict[str, Any]]:
    data = load_standard_scenarios(path)
    version = data.get("version", "standard-scenarios-v1")
    rows = []
    for row in data.get("scenarios", []):
        rows.append(
            {
                "id": row["id"],
                "label": row.get("label", row["id"]),
                "description": row.get("description", ""),
                "version": version,
                "config": deepcopy(row["config"]),
                "evaluation": deepcopy(row.get("evaluation", {})),
            }
        )
    return rows


def get_standard_scenario_config(scenario_id: str, path: str | Path = DEFAULT_STANDARD_SCENARIOS_PATH) -> dict[str, Any]:
    for row in list_standard_scenarios(path):
        if row["id"] == scenario_id:
            return deepcopy(row["config"])
    raise KeyError(f"unknown standard scenario: {scenario_id}")
