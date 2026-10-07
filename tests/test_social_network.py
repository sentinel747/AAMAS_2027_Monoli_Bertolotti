from src.social_network.metrics import compute_social_metrics
from src.social_network.network import SocialNetwork
from src.agents.action_space import ActionRequest, ActionType
from src.simulation.agent_coupled_runner import AgentCoupledRunner


def test_social_network_updates_trust():
    network = SocialNetwork()
    network.update("a", "b", "communicate")
    metrics = compute_social_metrics(network)
    assert metrics["average_trust"] > 0.5
    assert metrics["network_density"] == 1.0


def test_cooperation_index_is_bounded_per_relationship():
    network = SocialNetwork()
    for _ in range(50):
        network.update("a", "b", "share_resource")  # scambi illimitati sulla stessa coppia
    saturated = compute_social_metrics(network)
    assert saturated["cooperation_index"] == 1.0  # satura, non cresce all'infinito

    network.update("a", "c", "communicate")  # relazione attiva ma senza scambi
    mixed = compute_social_metrics(network)
    assert 0.0 < mixed["cooperation_index"] < 1.0  # media per relazione
    assert mixed["social_stability_score"] < 1.0  # non piu' inchiodata a 1.0


def test_runner_records_proximity_communication():
    runner = AgentCoupledRunner(
        {
            "seed": 2,
            "days": 1,
            "simulation": {"days_per_step": 3650, "max_days": 36500},
            "world": {"width": 8, "height": 8},
            "agents": {"count": 2, "llm_count": 0},
        }
    )
    agent_ids = list(runner.agents)
    runner._record_social_effect(agent_ids[0], ActionRequest(agent_ids[0], ActionType.COMMUNICATE), True, [agent_ids[1]])

    metrics = compute_social_metrics(runner.social)

    assert metrics["communication_frequency"] == 1.0
    assert metrics["cooperation_index"] > 0.0
