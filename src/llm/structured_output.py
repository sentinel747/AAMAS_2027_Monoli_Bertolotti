from __future__ import annotations

import json

from src.agents.decision_schema import validate_decision


def parse_decision_json(text: str) -> tuple[dict | None, list[str]]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, [f"invalid JSON: {exc}"]
    ok, errors = validate_decision(data)
    return (data if ok else None), errors
