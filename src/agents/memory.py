from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class AgentMemory:
    explored_cells: dict[tuple[int, int], dict] = field(default_factory=dict)
    cell_memory_limit: int | None = None
    recent_events: list[str] = field(default_factory=list)
    long_term_summary: str = ""
    social_memory: dict[str, str] = field(default_factory=dict)
    discovered_resources: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    past_decisions: list[dict] = field(default_factory=list)
    conversations: list[dict] = field(default_factory=list)
    task_queue: list[dict] = field(default_factory=list)
    recent_failures: list[dict] = field(default_factory=list)
    # Transient warnings (SYSTEM ALERTs) keyed by kind: they mirror the
    # CURRENT situation and are refreshed every step — set while the condition
    # holds, removed the moment it clears or the agent moves on. They used to
    # be appended to recent_events on every move, so stale alerts from cells
    # left long ago lingered in memory (and in LLM prompts) for dozens of steps.
    active_alerts: dict[str, str] = field(default_factory=dict)

    def add_event(self, event: str) -> None:
        self.recent_events.append(event)
        self.recent_events = self.recent_events[-30:]

    def set_alert(self, key: str, text: str) -> None:
        self.active_alerts[key] = text

    def clear_alert(self, key: str) -> None:
        self.active_alerts.pop(key, None)

    def add_failure(self, action: str, target: dict | None, step: int | None = None) -> None:
        self.recent_failures.append({"action": action, "target": target, "step": step})
        self.recent_failures = self.recent_failures[-10:]

    def remember_cell(self, x: int, y: int, cell_data: dict | None = None) -> None:
        if cell_data is None:
            if (x, y) not in self.explored_cells:
                self.explored_cells[(x, y)] = {"x": x, "y": y}
        else:
            self.explored_cells[(x, y)] = cell_data

        if self.cell_memory_limit and len(self.explored_cells) > self.cell_memory_limit:
            oldest = next(iter(self.explored_cells.keys()))
            self.explored_cells.pop(oldest)

    def summarize(self) -> str:
        recent = "; ".join(self.recent_events[-5:])
        base = self.long_term_summary or (recent if recent else "No important memory yet.")
        alerts = " | ".join(self.active_alerts.values())
        return f"{alerts} | {base}" if alerts else base

    def retrieve_relevant(self) -> list[str]:
        return list(self.active_alerts.values()) + self.recent_events[-10:]
