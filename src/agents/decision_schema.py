from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .action_space import ActionType


VALID_ACTIONS = {action.value for action in ActionType}


@dataclass
class AgentDecision:
    thought_summary: str
    public_message: str
    chosen_action: str
    target: dict[str, Any] = field(default_factory=lambda: {"type": "none"})
    magnitude: float = 1.0
    resource_offer: dict[str, float] = field(default_factory=dict)
    resource_request: dict[str, float] = field(default_factory=dict)
    cooperation_target: str = ""


def validate_decision(data: dict) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return False, ["decision must be an object"]
    if data.get("chosen_action") not in VALID_ACTIONS:
        errors.append("chosen_action is not in the valid action set")
    if "thought_summary" not in data:
        errors.append("thought_summary is required")
    return not errors, errors
