from src.agents.decision_schema import validate_decision
from src.llm.structured_output import parse_decision_json


def test_decision_schema_accepts_valid_decision():
    data = {"thought_summary": "observe first", "chosen_action": "observe", "confidence": 0.7}
    ok, errors = validate_decision(data)
    assert ok, errors


def test_decision_schema_rejects_invalid_action():
    data, errors = parse_decision_json('{"thought_summary":"x","chosen_action":"fly","confidence":0.5}')
    assert data is None
    assert errors
