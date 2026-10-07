from __future__ import annotations

import numpy as np

from scripts.explain_preference_decision import build_example


def test_documented_preference_example_uses_real_scoring_primitives():
    example = build_example()
    probabilities = example["normalized_probabilities"]

    assert np.isclose(sum(probabilities.values()), 1.0)
    assert probabilities["costruzione"] > probabilities["sostentamento"]
    assert probabilities["esplorazione"] > probabilities["sostentamento"]
    assert probabilities["sociale"] == 0.0
    assert example["chosen_pillar"] == "vita_salute"
    assert example["example_bound_action"] == "rest"
    assert len(example["cell_proposed_actions"]) == 7
    assert all(
        proposal["action"] not in {"communicate", "share_resource"}
        for proposal in example["cell_proposed_actions"]
    )
