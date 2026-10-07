from fastapi.testclient import TestClient

from src.api.main import app
from src.experiments.scenarios import get_standard_scenario_config, list_standard_scenarios


def test_standard_scenarios_are_versioned_and_repeatable():
    scenarios = list_standard_scenarios()

    assert 4 <= len(scenarios) <= 6
    ids = {scenario["id"] for scenario in scenarios}
    assert "standard_v1_balanced_colony" in ids
    assert all(scenario["version"] == "standard-scenarios-v1" for scenario in scenarios)
    assert all(scenario["config"]["seed"] is not None for scenario in scenarios)
    assert all(scenario["config"]["days"] >= 1000 for scenario in scenarios)
    assert all(scenario["config"]["world"]["map_profile"] for scenario in scenarios)
    assert all(scenario["config"]["social"]["earth_mars_delay_minutes"] >= 0 for scenario in scenarios)
    assert all(scenario["config"]["climate"]["enabled"] is True for scenario in scenarios)
    assert all("evaluation" in scenario for scenario in scenarios)

    config = get_standard_scenario_config("standard_v1_resource_scarcity")
    assert config["name"] == "standard_v1_resource_scarcity"
    assert config["world"]["width"] == 28
    assert config["world"]["map_profile"] == "scarce_resources"


def test_standard_scenarios_are_available_to_gui_api():
    client = TestClient(app)
    payload = client.get("/api/scenarios/standard").json()

    assert 4 <= len(payload["scenarios"]) <= 6
    assert payload["scenarios"][0]["config"]["agents"]["count"] > 0
    assert payload["scenarios"][0]["config"]["world"]["map_profile"] == "balanced"
