from __future__ import annotations

import json
from typing import Any


def build_action_log_row(step: int, day: float, agent, request, result, observation=None) -> dict[str, Any]:
    inventory = getattr(agent, "inventory", None)
    visible_cells = getattr(observation, "visible_cells", []) or []
    nearby_agents = getattr(observation, "nearby_agents", []) or []
    return {
        "step": step,
        "day": day,
        "agent_id": agent.agent_id,
        "agent_name": getattr(agent, "name", ""),
        "role": getattr(agent, "role", "colonist"),
        "mode": getattr(agent, "mode", "rule_based"),
        "x": getattr(agent, "x", None),
        "y": getattr(agent, "y", None),
        "local_x_m": getattr(agent, "local_x_m", 0.0),
        "local_y_m": getattr(agent, "local_y_m", 0.0),
        "perception_radius_m": getattr(agent, "perception_radius_m", None),
        "movement_distance_m_per_step": getattr(
            agent, "movement_distance_m_per_step", None
        ),
        "action": request.action.value,
        "target": _json_safe(getattr(request, "target", None)),
        "magnitude": getattr(request, "magnitude", 1.0),
        "accepted": result.accepted,
        "message": result.message,
        "request_message": getattr(request, "message", None),
        "result_data": _json_safe(getattr(result, "data", {})),
        "observation_mode": getattr(observation, "observation_mode", "unknown") if observation is not None else "none",
        "health": getattr(agent, "health", None),
        "hydration": getattr(agent, "hydration", None),
        "satiety": getattr(agent, "satiety", None),
        "oxygen_level": getattr(agent, "oxygen_level", None),
        "fatigue": getattr(agent, "fatigue", None),
        "stress_index": getattr(agent, "stress_index", None),
        "morale": getattr(agent, "morale", None),
        "inventory": inventory.to_dict() if hasattr(inventory, "to_dict") else {},
        "visible_cells_count": len(visible_cells),
        "partial_visible_cells_count": sum(1 for cell in visible_cells if cell.get("visibility_status") == "partial"),
        "nearby_agents_count": len(nearby_agents),
        "nearby_agents": list(nearby_agents),
    }


def _json_safe(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(item) for item in value]
        return str(value)

