from __future__ import annotations

from dataclasses import dataclass, field

from src.world.resources import ResourceBundle

from .memory import AgentMemory


@dataclass
class BaseAgent:
    agent_id: str
    name: str
    role: str
    x: int
    y: int
    perception_radius: int = 3
    perception_radius_m: float = 59_000.0
    # Existing executors still expose separate fields for artifact/API
    # compatibility, but scenarios derive both from operational_range_m.
    movement_distance_m_per_step: float = 59_000.0
    faction: str | None = None
    inventory: ResourceBundle = field(
        default_factory=lambda: ResourceBundle(energy=2, oxygen=2, water=4.0, food=5.0, construction_material=5, minerals=3, med_kits=2.0)
    )
    local_x_m: float = 0.0
    local_y_m: float = 0.0
    health: float = 1.0
    satiety: float = 1.0
    oxygen_level: float = 1.0
    hydration: float = 1.0
    fatigue: float = 0.0
    steps_without_water: int = 0
    steps_without_food: int = 0
    current_goal: str = "survive and improve local habitability"
    recent_actions: list[str] = field(default_factory=list)
    actions_taken: int = 0
    cooperation: float = 0.7
    risk_tolerance: float = 0.4
    curiosity: float = 0.6
    survival_priority: float = 0.8
    stress_index: float = 0.15
    morale: float = 0.82
    protocol_compliance: float = 0.85
    autonomy_preference: float = 0.5
    mode: str = "rule_based"
    llm_provider_id: str | None = None
    llm_model: str | None = None
    # Scouting expedition state: periodic exploration missions away from the
    # colony ("out" toward scout_target, then "back" toward scout_home).
    scout_target: tuple[int, int] | None = None
    scout_home: tuple[int, int] | None = None
    scout_phase: str | None = None
    scout_steps: int = 0
    scout_next_at: int = -1
    # Settlement expedition state: one-way trip toward a virgin cell to found
    # an outpost once the home cell has every coverage ratio satisfied.
    settle_target: tuple[int, int] | None = None
    settle_phase: str | None = None
    settle_steps: int = 0
    settle_next_at: int = -1
    # Temporary logistics exemption while this agent assembles or carries a
    # founder kit.  The ordinary redistribution targets are intentionally
    # smaller than the water/food/material reserves needed by an expedition.
    founder_kit_reserved: bool = False
    # Normalized preferences over the six decision pillars. Population owns
    # initialization; None remains supported for direct test/fixture agents.
    pillar_preferences: tuple[float, ...] | None = None
    # Role-shaped capability multipliers aligned with the six pillars.
    pillar_skills: tuple[float, ...] | None = None
    memory: AgentMemory = field(default_factory=AgentMemory)

    def __post_init__(self):
        self.memory.cell_memory_limit = 20

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "x": self.x,
            "y": self.y,
            "local_x_m": self.local_x_m,
            "local_y_m": self.local_y_m,
            "perception_radius": self.perception_radius,
            "perception_radius_m": self.perception_radius_m,
            "movement_distance_m_per_step": self.movement_distance_m_per_step,
            "faction": self.faction,
            "inventory": self.inventory.to_dict(),
            "health": self.health,
            "satiety": self.satiety,
            "oxygen_level": self.oxygen_level,
            "hydration": self.hydration,
            "fatigue": self.fatigue,
            "steps_without_water": self.steps_without_water,
            "steps_without_food": self.steps_without_food,
            "stress_index": self.stress_index,
            "morale": self.morale,
            "protocol_compliance": self.protocol_compliance,
            "autonomy_preference": self.autonomy_preference,
            "current_goal": self.current_goal,
            "recent_actions": list(self.recent_actions[-10:]),
            "mode": self.mode,
            "llm_provider_id": self.llm_provider_id,
            "llm_model": self.llm_model,
            "memory_summary": self.memory.summarize(),
            "pillar_preferences": list(self.pillar_preferences) if self.pillar_preferences is not None else None,
            "pillar_skills": list(self.pillar_skills) if self.pillar_skills is not None else None,
        }

    async def async_decide(self, observation, world):
        return self.decide(observation, world)
