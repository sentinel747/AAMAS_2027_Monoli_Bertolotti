from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SocialEdge:
    trust: float = 0.5
    communication_frequency: int = 0
    alliance: float = 0.0
    conflict: float = 0.0
    trade: float = 0.0
    shared_knowledge: float = 0.0


@dataclass
class SocialNetwork:
    agents: set[str] = field(default_factory=set)
    edges: dict[tuple[str, str], SocialEdge] = field(default_factory=dict)

    def add_agent(self, agent_id: str) -> None:
        self.agents.add(agent_id)

    def edge(self, a: str, b: str) -> SocialEdge:
        key = tuple(sorted((a, b)))
        self.edges.setdefault(key, SocialEdge())
        return self.edges[key]

    def update(self, a: str, b: str, event_type: str) -> None:
        self.add_agent(a)
        self.add_agent(b)
        edge = self.edge(a, b)
        if event_type == "communicate":
            edge.communication_frequency += 1
            edge.trust = min(1.0, edge.trust + 0.02)
        elif event_type == "share_resource":
            edge.trade += 1
            edge.trust = min(1.0, edge.trust + 0.06)
        elif event_type == "conflict":
            edge.conflict += 1
            edge.trust = max(0.0, edge.trust - 0.12)
        elif event_type == "cooperate":
            edge.alliance = min(1.0, edge.alliance + 0.05)
            edge.trust = min(1.0, edge.trust + 0.04)

    def export_graph(self) -> dict:
        return {
            "nodes": [{"id": agent} for agent in sorted(self.agents)],
            "edges": [
                {"source": a, "target": b, "trust": edge.trust, "comm": edge.communication_frequency, "alliance": edge.alliance, "conflict": edge.conflict, "trade": edge.trade}
                for (a, b), edge in sorted(self.edges.items())
            ],
        }

    def to_dict(self) -> dict:
        return self.export_graph()
