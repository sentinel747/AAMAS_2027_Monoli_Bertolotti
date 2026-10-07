import asyncio

from src.governors.config import (
    build_governor,
    read_administrator_settings,
    read_settings,
)
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy
from src.governors.semif_arm import SemifGovernorProposer
from src.semantic_governance.client import FakeSemanticDecisionProvider


PICTURE = ColonyPicture(
    step=1,
    population=10,
    n_cells=1,
    indicators={
        "food_per_occupant": {"mean": 2.0, "std": 0.5, "min": 1.0, "max": 3.0},
    },
)
BOUNDS = Bounds(0.25, 4.0)


def propose(selected: str):
    provider = FakeSemanticDecisionProvider({"jev-semif-governor-v1": selected})
    proposer = SemifGovernorProposer(provider, run_id="test-run")
    return asyncio.run(proposer.propose(PICTURE, BOUNDS))


def test_semif_selects_a_prevalidated_candidate():
    proposal = propose("select_candidate:mean_food_per_occupant")
    assert proposal.policy is not None and len(proposal.policy.rules) == 1
    assert proposal.provider == "fake"
    assert proposal.candidate_profile == "jev-semif-governor-v1"
    assert proposal.semantic_decision["selected_option"].startswith("select_candidate:")


def test_unknown_and_unavailable_candidate_keep_the_previous_policy():
    unknown = propose("unknown")
    unavailable = propose("select_candidate:not-present")
    assert unknown.policy is None
    assert "fallback" in unknown.rationale.lower()
    assert unavailable.policy is None
    assert "fallback" in unavailable.rationale.lower()


def test_no_intervention_is_distinct_from_keep_previous():
    no_intervention = propose("no_intervention")
    keep = propose("keep_previous")
    assert no_intervention.policy == Policy((), "SemIf selected no_intervention")
    assert keep.policy is None


def test_governor_factory_accepts_injected_fake_without_network():
    provider = FakeSemanticDecisionProvider({
        "jev-semif-governor-v1": "select_candidate:mean_food_per_occupant"
    })
    governor = build_governor(
        {
            "days": 2,
            "governors": {
                "arm": "semif",
                "cadence_steps": 1,
                "wait_seconds": 1,
                "semantic": {"run_id": "factory-test"},
            },
        },
        seed=7,
        semantic_provider=provider,
    )
    try:
        policy = governor.advance(1, PICTURE)
        assert policy is not None and len(policy.rules) == 1
        assert len(provider.requests) == 1
    finally:
        governor.close()


def test_semif_can_be_inherited_by_active_administrators():
    config = {
        "governors": {
            "arm": "semif",
            "administrators": {
                "enabled": True,
                "follow_governor_arm": True,
            },
        }
    }
    settings = read_administrator_settings(config, read_settings(config))
    assert settings.enabled is True
    assert settings.follow_governor_arm is True
    assert settings.arm == "semif"


def test_a_disabled_administrator_section_does_not_block_semif():
    config = {
        "governors": {
            "arm": "semif",
            "administrators": {"enabled": False},
        }
    }
    settings = read_administrator_settings(config, read_settings(config))
    assert settings.enabled is False


def test_rest_mode_requires_an_explicit_endpoint():
    try:
        build_governor(
            {"days": 2, "governors": {"arm": "semif", "cadence_steps": 1,
                                       "wait_seconds": 1, "semantic": {"provider": "rest"}}},
            seed=0,
        )
    except ValueError as error:
        assert "endpoint" in str(error)
    else:
        raise AssertionError("REST mode must not select an endpoint implicitly")
