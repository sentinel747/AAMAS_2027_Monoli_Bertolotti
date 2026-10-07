from __future__ import annotations

"""Print one deterministic, human-readable preference-decision example.

This is deliberately a diagnostic, not a second decision implementation: it
calls the same score and sampling primitives used by the simulation.
"""

import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents import pillars
from src.agents.preference_agent import choose_pillar, score_pillars


PILLAR_NAMES = (
    "sostentamento",
    "risorse",
    "costruzione",
    "vita_salute",
    "sociale",
    "esplorazione",
)


def build_example() -> dict:
    """Return a fixed example whose numbers can be checked by hand."""
    preferences = np.asarray([0.12, 0.08, 0.26, 0.14, 0.10, 0.30])
    urgencies = np.asarray([0.70, 0.40, 0.60, 0.20, 0.30, 0.50])
    # The current cell has no nearby colonist, so its social action is absent.
    availability = np.asarray([True, True, True, True, False, True])
    # Maximum priority of the actions proposed by the current cell for each
    # pillar. Social coordination is automatic, hence zero as an action.
    cell_priority = np.asarray([0.90, 0.70, 1.00, 0.80, 0.00, 0.60])
    # Engineer capability profile with small individual variation.
    skills = np.asarray([0.90, 1.10, 1.35, 0.95, 0.90, 0.95])
    random_draw = 0.76
    scores = score_pillars(
        preferences, urgencies, availability, cell_priority, skills
    )
    chosen = choose_pillar(scores, "softmax", random_draw)
    return {
        "agent_role": "engineer",
        "preferences": dict(zip(PILLAR_NAMES, preferences.tolist())),
        "urgencies": dict(zip(PILLAR_NAMES, urgencies.tolist())),
        "available_from_current_cell": dict(
            zip(PILLAR_NAMES, availability.tolist())
        ),
        "cell_pillar_priority": dict(zip(PILLAR_NAMES, cell_priority.tolist())),
        "pillar_skills": dict(zip(PILLAR_NAMES, skills.tolist())),
        "normalized_probabilities": dict(zip(PILLAR_NAMES, scores.tolist())),
        "random_draw": random_draw,
        "chosen_pillar": PILLAR_NAMES[chosen],
        "example_bound_action": "rest",
        "automatic_cell_services": [
            "deposit_surplus",
            "withdraw_by_need",
            "coordinate_without_spending_agent_step",
        ],
        "cell_proposed_actions": [
            {"action": "collect_ice", "priority": 0.90, "quota": 3},
            {"action": "forage", "priority": 0.80, "quota": 2},
            {"action": "collect_materials", "priority": 0.70, "quota": 2},
            {"action": "build_solar_array", "priority": 1.00, "quota": 1},
            {"action": "build_greenhouse", "priority": 0.95, "quota": 1},
            {"action": "move", "priority": 0.25, "quota": 2},
            {"action": "observe", "priority": 0.20, "quota": "unlimited"},
        ],
        "personal_safety_actions": [
            "drink_water", "eat_food", "rest", "use_med_kit", "do_nothing"
        ],
    }


def main() -> None:
    print(json.dumps(build_example(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
