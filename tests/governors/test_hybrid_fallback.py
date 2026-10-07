import asyncio

from src.governors.arms import GovernorProposal
from src.governors.hybrid_arm import HybridGovernorProposer
from src.governors.config import build_governor
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy
from src.semantic_governance.client import FakeSemanticDecisionProvider


PICTURE = ColonyPicture(
    step=3,
    population=12,
    n_cells=2,
    indicators={"food_per_occupant": {"mean": 2.0, "std": 0.2, "min": 1.8, "max": 2.2}},
)
BOUNDS = Bounds(0.25, 4.0)


class _LLM:
    def __init__(self) -> None:
        self.calls = 0

    async def propose(self, picture, bounds):
        self.calls += 1
        return GovernorProposal(
            policy=Policy((), "llm result"),
            rationale="llm result",
            provider="offline-llm-stub",
            model="stub",
        )


def _run(selected: str):
    semantic = FakeSemanticDecisionProvider({"jev-semif-governor-v1": selected})
    llm = _LLM()
    proposer = HybridGovernorProposer(semantic, llm, run_id="hybrid-test")
    return asyncio.run(proposer.propose(PICTURE, BOUNDS)), semantic, llm


def test_deep_reasoning_is_the_only_explicit_llm_escalation():
    proposal, semantic, llm = _run("deep_reasoning")
    assert llm.calls == 1
    assert len(semantic.requests) == 1
    assert proposal.provider == "offline-llm-stub"
    assert proposal.semantic_decision["selected_option"] == "deep_reasoning"
    assert "explicit deep_reasoning" in proposal.rationale
    assert "deep_reasoning" in semantic.requests[0].option_ids
    assert len(semantic.requests[0].option_ids) <= 16


def test_unknown_does_not_silently_escalate():
    proposal, _, llm = _run("unknown")
    assert llm.calls == 0
    assert proposal.policy is None
    assert "fallback" in proposal.rationale.lower()


def test_bounded_candidate_does_not_call_the_llm():
    proposal, _, llm = _run("select_candidate:mean_food_per_occupant")
    assert llm.calls == 0
    assert proposal.policy is not None
    assert len(proposal.policy.rules) == 1


def test_hybrid_governor_factory_uses_only_injected_offline_proposers():
    semantic = FakeSemanticDecisionProvider({
        "jev-semif-governor-v1": "deep_reasoning"
    })
    llm = _LLM()
    governor = build_governor(
        {
            "days": 2,
            "governors": {
                "arm": "hybrid",
                "cadence_steps": 1,
                "wait_seconds": 1,
                "semantic": {"run_id": "hybrid-factory"},
            },
        },
        seed=1,
        semantic_provider=semantic,
        hybrid_llm_proposer=llm,
    )
    try:
        governor.advance(1, PICTURE)
        assert llm.calls == 1
        assert len(semantic.requests) == 1
    finally:
        governor.close()
