import time

from src.governors.admin_semif_arm import (
    AdminSemanticContext,
    AmministratoreIbrido,
    AmministratoreSemIf,
)
from src.governors.administration import Amministrazione
from src.governors.config import build_administration
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds
from src.semantic_governance.client import FakeSemanticDecisionProvider
from src.semantic_governance.replay import ReplaySemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecision


BOUNDS = Bounds(0.25, 4.0)
PICTURE = ColonyPicture(
    step=4,
    population=8,
    n_cells=2,
    indicators={"food_per_occupant": {"mean": 1.5, "std": 0.5, "min": 1.0, "max": 2.0}},
)


class _Districts:
    def numero_distretti(self):
        return 1


class _AdminLLM:
    def __init__(self) -> None:
        self.calls = 0

    async def propose_text_async(self, prompt: str) -> dict:
        self.calls += 1
        return {
            "raw": {
                "accept": False,
                "policy": [{"weights": {"build": 2.0}}],
                "rationale": "offline LLM stub",
            },
            "provider": "offline-llm-stub",
            "model": "stub",
        }


def _context(district: int = 0, prompt: str = "admin prompt") -> AdminSemanticContext:
    return AdminSemanticContext(
        district=district,
        picture=PICTURE,
        governor_policy=None,
        colony_indicators=PICTURE.indicators,
        prompt=prompt,
    )


def _decision(proposer, district: int = 0):
    administration = Amministrazione(lambda _: proposer, _Districts(), BOUNDS)
    return administration.tornata([
        (district, [(0, 0), (0, 1)], "admin prompt", 8, _context(district))
    ])[0]


def test_admin_semif_rewrite_still_passes_through_the_existing_parser():
    provider = FakeSemanticDecisionProvider({
        "jev-semif-admin-v1": "rewrite_with_candidate:mean_food_per_occupant"
    })
    decision = _decision(
        AmministratoreSemIf(provider, BOUNDS, run_id="admin-test", district=0)
    )
    assert decision.policy is not None and len(decision.policy.rules) == 1
    assert decision.semantic_decision["selected_option"].startswith("rewrite_with_candidate:")
    assert decision.candidate_profile == "jev-semif-admin-v1"
    assert decision.semantic_escalated is False
    assert len(provider.requests) == 1, "one call per district, not one per cell"
    assert decision.to_json()["candidate_profile"] == "jev-semif-admin-v1"


def test_admin_semif_unknown_accepts_governor_without_llm():
    provider = FakeSemanticDecisionProvider({"jev-semif-admin-v1": "unknown"})
    decision = _decision(
        AmministratoreSemIf(provider, BOUNDS, run_id="admin-test", district=0)
    )
    assert decision.policy is None
    assert decision.accettata is True
    assert "fallback" in decision.rationale.lower()


def test_admin_hybrid_escalates_only_on_deep_reasoning():
    provider = FakeSemanticDecisionProvider({"jev-semif-admin-v1": "deep_reasoning"})
    llm = _AdminLLM()
    decision = _decision(
        AmministratoreIbrido(
            provider, llm, BOUNDS, run_id="admin-hybrid", district=0
        )
    )
    assert llm.calls == 1
    assert decision.policy is not None
    assert decision.semantic_escalated is True
    assert decision.semantic_decision["selected_option"] == "deep_reasoning"
    assert decision.provider == "offline-llm-stub"
    assert "deep_reasoning" in provider.requests[0].option_ids
    assert len(provider.requests[0].option_ids) <= 16


def test_admin_hybrid_unknown_never_calls_llm():
    provider = FakeSemanticDecisionProvider({"jev-semif-admin-v1": "unknown"})
    llm = _AdminLLM()
    decision = _decision(
        AmministratoreIbrido(
            provider, llm, BOUNDS, run_id="admin-hybrid", district=0
        )
    )
    assert llm.calls == 0
    assert decision.policy is None


def test_blocking_semantic_providers_remain_concurrent():
    starts = []

    class _SlowFake(FakeSemanticDecisionProvider):
        def decide(self, request):
            starts.append(time.perf_counter())
            time.sleep(0.2)
            return super().decide(request)

    proposers = {
        district: AmministratoreSemIf(
            _SlowFake({"jev-semif-admin-v1": "accept_governor"}),
            BOUNDS,
            run_id="concurrency-test",
            district=district,
        )
        for district in range(3)
    }
    administration = Amministrazione(lambda district: proposers[district], _Districts(), BOUNDS)
    requests = [
        (district, [(district, 0)], "prompt", 1, _context(district))
        for district in range(3)
    ]
    before = time.perf_counter()
    decisions = administration.tornata(requests)
    duration = time.perf_counter() - before
    assert len(decisions) == 3
    assert duration < 0.4
    assert max(starts) - min(starts) < 0.1


def test_admin_factories_build_semif_and_hybrid_without_network():
    semantic = FakeSemanticDecisionProvider({"jev-semif-admin-v1": "accept_governor"})
    semif = build_administration(
        {
            "governors": {
                "arm": "semif",
                "semantic": {"run_id": "admin-factory"},
                "administrators": {"enabled": True},
            }
        },
        seed=2,
        semantic_provider_factory=lambda _district: semantic,
    )
    assert isinstance(semif._proponente(0), AmministratoreSemIf)

    llm = _AdminLLM()
    hybrid = build_administration(
        {
            "governors": {
                "arm": "hybrid",
                "semantic": {"run_id": "admin-hybrid-factory"},
                "administrators": {"enabled": True},
            }
        },
        seed=2,
        semantic_provider_factory=lambda _district: semantic,
        hybrid_llm_factory=lambda _district: llm,
    )
    assert isinstance(hybrid._proponente(0), AmministratoreIbrido)


def test_admin_semantic_decision_replays_with_exact_district_identity():
    fake = FakeSemanticDecisionProvider({
        "jev-semif-admin-v1": "rewrite_with_candidate:mean_food_per_occupant"
    })
    first = _decision(
        AmministratoreSemIf(fake, BOUNDS, run_id="admin-replay", district=0)
    )
    replay = ReplaySemanticDecisionProvider([
        SemanticDecision.from_dict(first.semantic_decision)
    ])
    second = _decision(
        AmministratoreSemIf(replay, BOUNDS, run_id="admin-replay", district=0)
    )
    assert second.policy == first.policy
    assert second.semantic_decision["route"] == "replay"
