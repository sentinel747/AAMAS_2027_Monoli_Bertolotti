from __future__ import annotations

from .network import SocialNetwork


def compute_social_metrics(network: SocialNetwork) -> dict[str, float]:
    node_count = len(network.agents)
    edge_count = len(network.edges)
    possible = node_count * (node_count - 1) / 2 if node_count > 1 else 1
    avg_trust = sum(edge.trust for edge in network.edges.values()) / max(1, edge_count)
    communication = sum(edge.communication_frequency for edge in network.edges.values())
    conflict = sum(edge.conflict for edge in network.edges.values())
    # Bounded cooperation: alliance is already 0..1 per relationship and trade
    # saturates after ~5 exchanges, so the index is the average cooperativeness
    # of the existing relationships (0..1). The previous lifetime event sum
    # grew without bound (2501 after 1000 steps) and pinned
    # social_stability_score at 1.0 through its 0.02 weight.
    cooperation = sum(min(1.0, edge.alliance + edge.trade * 0.2) for edge in network.edges.values()) / max(1, edge_count)
    conflict_pressure = sum(min(1.0, edge.conflict * 0.34) for edge in network.edges.values()) / max(1, edge_count)
    return {
        "average_trust": avg_trust,
        "network_density": edge_count / possible,
        "communication_frequency": float(communication),
        "conflict_intensity": conflict,
        "cooperation_index": cooperation,
        "social_stability_score": max(0.0, min(1.0, avg_trust + cooperation * 0.15 - conflict_pressure * 0.30)),
    }
