from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, parse_policy
from src.governors.policy_candidates import CandidatePolicyFactory


def picture() -> ColonyPicture:
    stats = {
        "food_per_occupant": 2.0,
        "water_per_occupant": 2.5,
        "oxygen_per_occupant": 3.0,
        "ice_per_occupant": 1.0,
        "material_per_occupant": 1.5,
        "minerals_per_occupant": 0.5,
        "power_coverage": 0.8,
        "structure_integrity": 0.7,
        "occupants": 20.0,
    }
    return ColonyPicture(
        step=1,
        population=20,
        n_cells=2,
        indicators={
            name: {"mean": value, "std": 0.1, "min": value - 0.1, "max": value + 0.1}
            for name, value in stats.items()
        },
    )


def test_candidates_are_deterministic_bounded_and_parser_valid():
    factory = CandidatePolicyFactory()
    bounds = Bounds(0.25, 4.0)
    first = factory.build(picture(), bounds)
    second = factory.build(picture(), bounds)
    assert first == second
    assert len(first.semantic_options()) <= 16
    assert first.semantic_options()[-1].id == "unknown"
    for candidate in first.candidates:
        policy, drops = parse_policy(candidate.raw_policy, bounds)
        assert policy == candidate.policy
        assert drops.dropped_rules == 0


def test_candidate_weights_obey_the_configured_bounds():
    result = CandidatePolicyFactory().build(picture(), Bounds(0.5, 1.5))
    assert result.candidates
    for candidate in result.candidates:
        assert all(
            0.5 <= weight <= 1.5
            for rule in candidate.policy.rules
            for weight in rule.weights.values()
        )


def test_missing_indicators_produce_only_safe_control_options():
    result = CandidatePolicyFactory().build(
        ColonyPicture(step=1, population=0), Bounds(0.25, 4.0)
    )
    assert not result.candidates
    assert [option.id for option in result.semantic_options()] == [
        "keep_previous", "no_intervention", "unknown"
    ]


# --- Profilo v2 (piano 2026-09-22, Task 4) -----------------------------------

_V2_INDICATORS = (
    "food_per_occupant", "water_per_occupant", "oxygen_per_occupant",
    "ice_per_occupant", "material_per_occupant", "minerals_per_occupant",
    "power_coverage", "structure_integrity", "occupants",
)


def _bounds() -> Bounds:
    return Bounds(0.25, 4.0)


def _uniform_picture(value: float) -> ColonyPicture:
    return ColonyPicture(
        step=1, population=8, n_cells=1,
        indicators={
            name: {"mean": value, "std": 0.0, "min": value, "max": value}
            for name in _V2_INDICATORS
        },
    )


def _picture_with_spread(mean: float, std: float) -> ColonyPicture:
    return ColonyPicture(
        step=1, population=20, n_cells=2,
        indicators={
            name: {"mean": mean, "std": std, "min": mean - std, "max": mean + std}
            for name in _V2_INDICATORS
        },
    )


def test_v2_candidates_fire_when_all_cells_are_equal():
    from src.governors.policy_candidates import CANDIDATE_PROFILE_VERSION_V2
    candidates = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2).build(
        _uniform_picture(value=1.5), _bounds()
    )
    food = candidates.by_id()["mean_food_per_occupant"]
    rule = food.policy.rules[0]
    # la soglia supera il valore comune: la regola copre tutta la colonia
    assert rule.condition.op == "<"
    assert rule.condition.value > 1.5
    assert "every settled cell" in food.description


def test_v2_keeps_the_mean_when_cells_differ():
    from src.governors.policy_candidates import CANDIDATE_PROFILE_VERSION_V2
    candidates = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2).build(
        _picture_with_spread(mean=2.0, std=0.5), _bounds()
    )
    rule = candidates.by_id()["mean_food_per_occupant"].policy.rules[0]
    assert rule.condition.value == 2.0
    assert "below the colony average" in candidates.by_id()["mean_food_per_occupant"].description


def test_v2_adds_the_reference_constitution_as_candidates():
    from src.governors.policy_candidates import CANDIDATE_PROFILE_VERSION_V2
    ids = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2).build(
        _uniform_picture(value=1.0), _bounds()
    ).by_id()
    assert "reference_food_per_occupant" in ids
    assert "reference_occupants" in ids


def test_v2_option_count_fits_the_local_gateway():
    from src.governors.policy_candidates import CANDIDATE_PROFILE_VERSION_V2
    options = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2).build(
        _uniform_picture(value=1.0), _bounds()
    ).semantic_options(include_deep_reasoning=True)
    assert len(options) <= 16


def test_v1_is_unchanged():
    from src.governors.policy_candidates import CANDIDATE_PROFILE_VERSION
    assert CANDIDATE_PROFILE_VERSION == "jev-semif-governor-v1"
    rule = CandidatePolicyFactory().build(_uniform_picture(value=1.5), _bounds()).by_id()[
        "mean_food_per_occupant"].policy.rules[0]
    assert rule.condition.value == 1.5


def test_v2_admin_profile_exists_and_builds():
    from src.governors.policy_candidates import ADMIN_CANDIDATE_PROFILE_VERSION_V2, PROFILE_VERSIONS
    assert PROFILE_VERSIONS["v2"][1] == ADMIN_CANDIDATE_PROFILE_VERSION_V2
    built = CandidatePolicyFactory(ADMIN_CANDIDATE_PROFILE_VERSION_V2).build(
        _uniform_picture(value=1.0), _bounds()
    )
    assert "reference_occupants" in built.by_id()
