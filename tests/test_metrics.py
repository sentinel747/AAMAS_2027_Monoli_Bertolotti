from src.experiments.metrics import composite_agent_score, emergence_indicators, planetary_metrics


def test_metrics_are_computed():
    metrics = planetary_metrics({"average_habitability": 0.2, "vegetation": 10, "oxygen_plants": 1, "total_ice": 5, "total_minerals": 5})
    assert metrics["terraforming_progress_index"] > 0
    assert metrics["colony_environment_progress_index"] == metrics["terraforming_progress_index"]
    indicators = emergence_indicators({"structures_built": 3, "cooperation_index": 0.4, "network_density": 0.5, "greenhouses": 1})
    assert indicators["settlement_clustering"]
    assert indicators["trade_network_emergence"]  # 0.4 su scala 0-1
    assert not emergence_indicators({"cooperation_index": 0.05})["trade_network_emergence"]


def test_composite_agent_score_uses_common_components():
    score = composite_agent_score(
        {
            "population": 8,
            "initial_agent_count": 10,
            "average_agent_health": 0.75,
            "planetary_terraforming_progress_index": 0.2,
            "colony_prosperity_index": 0.35,
            "food_margin": 1.1,
            "material_margin": 0.8,
            "average_habitability": 0.12,
            "structures_built": 6,
            "social_stability_score": 0.7,
            "cooperation_index": 0.5,
            "communication_frequency": 8,
            "action_acceptance_rate": 0.9,
            "action_diversity": 5,
            "step": 20,
        }
    )
    assert 0 < score["composite_agent_score"] <= 1
    assert score["survival_component"] > score["colony_component"]
    assert score["terraforming_component"] == score["colony_component"]
    assert set(score) == {
        "survival_component",
        "colony_component",
        "terraforming_component",
        "cooperation_component",
        "efficiency_component",
        "composite_agent_score",
    }
