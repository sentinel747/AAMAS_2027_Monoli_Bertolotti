"""Governatore e amministratori SemIf a due livelli, profilo v2h (2026-09-23).

Livello 1: quale area (pilastro), oppure keep/no-intervention/accept/abstain.
Livello 2: quale candidata dentro l'area. Con 14 opzioni in una sola domanda la
confidenza stava a 0,25-0,35 e la soglia 0,65 scartava tutto (live, farm): due
domande strette al posto di una larga, come per gli agenti.
"""

import asyncio

import pytest

from src.governors.admin_semif_arm import AdminSemanticContext, AmministratoreSemIf
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy
from src.governors.policy_candidates import (
    ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
    CANDIDATE_PROFILE_VERSION_V2,
    CANDIDATE_PROFILE_VERSION_V2H,
    PROFILE_VERSIONS,
    CandidatePolicyFactory,
    is_hierarchical,
)
from src.governors.semif_arm import SemifGovernorProposer
from src.semantic_governance.client import FakeSemanticDecisionProvider
from src.semantic_governance.replay import ReplaySemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecision

INDICATORS = (
    "food_per_occupant", "water_per_occupant", "oxygen_per_occupant",
    "ice_per_occupant", "material_per_occupant", "minerals_per_occupant",
    "power_coverage", "structure_integrity", "occupants",
)
PICTURE = ColonyPicture(
    step=3, population=12, n_cells=1,
    indicators={n: {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0} for n in INDICATORS},
)
BOUNDS = Bounds(0.25, 4.0)
GOV_L1 = f"{CANDIDATE_PROFILE_VERSION_V2H}:L1"
ADM_L1 = f"{ADMIN_CANDIDATE_PROFILE_VERSION_V2H}:L1"


def _gov_l2(pillar):
    return f"{CANDIDATE_PROFILE_VERSION_V2H}:L2:{pillar}"


# --- profilo -------------------------------------------------------------------

def test_v2h_is_selectable_and_marked_hierarchical():
    assert PROFILE_VERSIONS["v2h"] == (
        CANDIDATE_PROFILE_VERSION_V2H, ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
    )
    assert is_hierarchical(CANDIDATE_PROFILE_VERSION_V2H)
    assert is_hierarchical(ADMIN_CANDIDATE_PROFILE_VERSION_V2H)
    assert not is_hierarchical(CANDIDATE_PROFILE_VERSION_V2)


def test_v2h_candidates_are_exactly_the_v2_candidates():
    v2 = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2).build(PICTURE, BOUNDS)
    v2h = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H).build(PICTURE, BOUNDS)
    # Stesse candidate: stessi id e stesse regole. La motivazione testuale
    # porta il nome del profilo e quindi differisce.
    assert [(c.id, c.policy.rules) for c in v2.candidates] == [
        (c.id, c.policy.rules) for c in v2h.candidates
    ]


def test_candidates_are_grouped_by_the_pillar_they_weight():
    groups = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H).build(PICTURE, BOUNDS).by_pillar()
    assert list(groups) == ["sustenance", "resources", "build", "life", "explore"]
    assert {c.id for c in groups["build"]} == {
        "mean_material_per_occupant", "mean_power_coverage", "mean_structure_integrity",
    }
    assert "reference_food_per_occupant" in {c.id for c in groups["sustenance"]}
    total = sum(len(v) for v in groups.values())
    assert total == len(CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H).build(PICTURE, BOUNDS).candidates)


# --- governatore ------------------------------------------------------------------

def _propose(decisions, hybrid=None):
    provider = FakeSemanticDecisionProvider(decisions)
    proposer = SemifGovernorProposer(
        provider, run_id="h", candidate_factory=CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H),
        deep_reasoning_proposer=hybrid,
    )
    return asyncio.run(proposer.propose(PICTURE, BOUNDS)), provider


def test_level1_has_few_options_and_one_area_per_pillar():
    _, provider = _propose({GOV_L1: "keep_previous"})
    first = provider.requests[0]
    assert first.question_id == GOV_L1
    assert first.option_ids == (
        "keep_previous", "no_intervention", "area:sustenance", "area:resources",
        "area:build", "area:life", "area:explore", "unknown",
    )


def test_area_then_candidate_applies_that_candidate():
    proposal, provider = _propose({
        GOV_L1: "area:sustenance",
        _gov_l2("sustenance"): "select_candidate:reference_food_per_occupant",
    })
    assert [r.question_id for r in provider.requests] == [GOV_L1, _gov_l2("sustenance")]
    assert provider.requests[1].option_ids[-1] == "unknown"
    assert all(o.startswith("select_candidate:") for o in provider.requests[1].option_ids[:-1])
    assert proposal.policy is not None and len(proposal.policy.rules) == 1
    assert [d["question_id"] for d in proposal.semantic_decisions] == [GOV_L1, _gov_l2("sustenance")]
    assert proposal.semantic_decision["question_id"] == _gov_l2("sustenance")


def test_keep_previous_asks_one_question_and_keeps_the_policy():
    proposal, provider = _propose({GOV_L1: "keep_previous"})
    assert len(provider.requests) == 1
    assert proposal.policy is None
    assert len(proposal.semantic_decisions) == 1


def test_no_intervention_is_still_an_explicit_empty_policy():
    proposal, _ = _propose({GOV_L1: "no_intervention"})
    assert proposal.policy == Policy((), "SemIf selected no_intervention")


def test_uncertain_level2_keeps_the_previous_policy():
    proposal, provider = _propose({GOV_L1: "area:build", _gov_l2("build"): "unknown"})
    assert len(provider.requests) == 2
    assert proposal.policy is None
    assert "level 2" in proposal.rationale


def test_level1_unknown_keeps_the_previous_policy():
    proposal, provider = _propose({GOV_L1: "unknown"})
    assert len(provider.requests) == 1
    assert proposal.policy is None


def test_hybrid_level1_offers_and_honours_deep_reasoning():
    class Deep:
        async def propose(self, picture, bounds):
            from src.governors.arms import GovernorProposal
            return GovernorProposal(policy=Policy((), "deep"), rationale="deep")

    proposal, provider = _propose({GOV_L1: "deep_reasoning"}, hybrid=Deep())
    assert "deep_reasoning" in provider.requests[0].option_ids
    assert proposal.rationale.startswith("SemIf selected explicit deep_reasoning")
    assert len(proposal.semantic_decisions) == 1


def test_governor_decisions_replay_exactly():
    live, fake = _propose({
        GOV_L1: "area:build", _gov_l2("build"): "select_candidate:mean_power_coverage",
    })
    recorded = [SemanticDecision.from_dict(d) for d in live.semantic_decisions]
    replayer = SemifGovernorProposer(
        ReplaySemanticDecisionProvider(recorded), run_id="h",
        candidate_factory=CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H),
    )
    replayed = asyncio.run(replayer.propose(PICTURE, BOUNDS))
    assert replayed.policy == live.policy


def test_record_and_collect_keep_both_decisions(tmp_path):
    from src.governors.record import GovernorRecorder
    from src.semantic_governance.artifacts import collect_semantic_decisions

    proposal, _ = _propose({
        GOV_L1: "area:life", _gov_l2("life"): "select_candidate:mean_oxygen_per_occupant",
    })
    recorder = GovernorRecorder(tmp_path / "governor_decisions.jsonl")
    _record(recorder, proposal)
    collected = collect_semantic_decisions(tmp_path)
    assert [d["question_id"] for d in collected] == [GOV_L1, _gov_l2("life")]


def _record(recorder, proposal):
    recorder.record(1, 1, PICTURE, proposal, proposal.policy, False)
    recorder.close()


# --- amministratori ---------------------------------------------------------------

def _governor_law():
    from src.governors.arms import reference_policy
    return reference_policy(BOUNDS)


def _admin(decisions, governor_policy="law"):
    provider = FakeSemanticDecisionProvider(decisions)
    admin = AmministratoreSemIf(
        provider, BOUNDS, run_id="h", district=2,
        candidate_profile=ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
    )
    context = AdminSemanticContext(
        district=2, picture=PICTURE,
        governor_policy=_governor_law() if governor_policy == "law" else governor_policy,
        colony_indicators=PICTURE.indicators, prompt="p",
    )
    return asyncio.run(admin.propose_semantic_async(context)), provider


def _adm_l2(pillar):
    return f"{ADMIN_CANDIDATE_PROFILE_VERSION_V2H}:L2:{pillar}"


def test_admin_level1_options():
    _, provider = _admin({ADM_L1: "accept_governor"})
    assert provider.requests[0].option_ids == (
        "accept_governor", "abstain", "area:sustenance", "area:resources",
        "area:build", "area:life", "area:explore", "unknown",
    )
    assert provider.requests[0].actor == "administrator:2"


def test_admin_area_then_candidate_rewrites_the_district():
    out, provider = _admin({
        ADM_L1: "area:explore", _adm_l2("explore"): "rewrite_with_candidate:reference_occupants",
    })
    assert [r.question_id for r in provider.requests] == [ADM_L1, _adm_l2("explore")]
    assert out["raw"]["accept"] is False
    assert out["raw"]["policy"]
    assert [d["question_id"] for d in out["semantic_decisions"]] == [ADM_L1, _adm_l2("explore")]


def test_admin_uncertain_level2_accepts_the_governor():
    out, provider = _admin({ADM_L1: "area:resources", _adm_l2("resources"): "unknown"})
    assert len(provider.requests) == 2
    assert out["raw"]["accept"] is True


def test_admin_accept_asks_only_one_question():
    out, provider = _admin({ADM_L1: "abstain"})
    assert len(provider.requests) == 1
    assert out["raw"]["accept"] is True


def test_admin_record_carries_both_decisions():
    out, _ = _admin({
        ADM_L1: "area:explore", _adm_l2("explore"): "rewrite_with_candidate:reference_occupants",
    })
    assert len(out["semantic_decisions"]) == 2


@pytest.mark.parametrize("profile", ["v1", "v2"])
def test_flat_profiles_are_unchanged(profile):
    gov, _ = PROFILE_VERSIONS[profile]
    provider = FakeSemanticDecisionProvider({gov: "keep_previous"})
    proposer = SemifGovernorProposer(provider, run_id="f", candidate_factory=CandidatePolicyFactory(gov))
    proposal = asyncio.run(proposer.propose(PICTURE, BOUNDS))
    assert [r.question_id for r in provider.requests] == [gov]
    assert getattr(proposal, "semantic_decisions", None) in (None, [proposal.semantic_decision])


def test_admin_decision_json_carries_both_decisions():
    from src.governors.administration import DecisioneAmministratore

    first = {"question_id": "L1"}
    second = {"question_id": "L2"}
    decisione = DecisioneAmministratore(
        distretto=2, policy=None, accettata=True, semantic_decision=second,
        semantic_decisions=[first, second], candidate_profile=ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
    )
    assert decisione.to_json()["semantic_decisions"] == [first, second]


# --- regola di decisione v2h: doppio del caso e margine sulla seconda ------------

from src.governors.policy_candidates import is_decisive  # noqa: E402


def _decision(probabilities, route="filter", reason=""):
    selected = max(probabilities, key=probabilities.get)
    return SemanticDecision(
        run_id="r", step=1, actor="governor", question_id="q", input_hash="h",
        selected_option=selected, probabilities=probabilities, confidence=0.5,
        route=route, fallback_reason=reason,
    )


def _spread(top, second, n, top_id="area:life", second_id="area:build"):
    rest = (1.0 - top - second) / (n - 2)
    probs = {top_id: top, second_id: second}
    probs.update({f"o{i}": rest for i in range(n - 3)})
    probs["unknown"] = 1.0 - sum(probs.values())
    return probs


def test_clear_preference_is_decisive():
    assert is_decisive(_decision(_spread(0.84, 0.06, 8)))


def test_near_tie_is_not_decisive():
    assert not is_decisive(_decision(_spread(0.41, 0.40, 8)))


def test_weak_top_among_many_is_not_decisive():
    # 0,22 supera la seconda di 12 punti ma resta sotto il doppio del caso (2/8)
    assert not is_decisive(_decision(_spread(0.22, 0.10, 8)))


def test_two_options_need_a_majority_and_the_margin():
    assert is_decisive(_decision({"select_candidate:x": 0.92, "unknown": 0.08}))
    assert not is_decisive(_decision({"select_candidate:x": 0.52, "unknown": 0.48}))


def test_four_options_floor_is_one_half():
    probs = {"select_candidate:a": 0.47, "select_candidate:b": 0.29,
             "select_candidate:c": 0.14, "unknown": 0.10}
    assert not is_decisive(_decision(probs))


def test_unknown_on_top_is_never_decisive():
    probs = {"unknown": 0.7, "area:life": 0.2, "keep_previous": 0.1}
    assert not is_decisive(_decision(probs, route="fallback", reason="selected_fallback"))


def test_error_fallback_is_never_decisive():
    probs = {"area:life": 0.0, "unknown": 1.0}
    assert not is_decisive(_decision(probs, route="fallback", reason="timeout"))


def _graded(table):
    """Fake che restituisce, per question_id, una distribuzione data."""
    def factory(request):
        probs = dict(table[request.question_id])
        for option in request.option_ids:
            probs.setdefault(option, 0.0)
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items() if k in request.option_ids}
        selected = max(probs, key=probs.get)
        return SemanticDecision(
            run_id=request.run_id, step=request.step, actor=request.actor,
            question_id=request.question_id, input_hash=request.input_hash,
            selected_option=selected, probabilities=probs, confidence=0.3,
            route="fallback" if selected == "unknown" else "filter",
        )
    return FakeSemanticDecisionProvider(factory=factory)


def test_governor_tie_at_level1_keeps_the_policy():
    provider = _graded({GOV_L1: {"area:sustenance": 0.41, "area:build": 0.40, "unknown": 0.19}})
    proposer = SemifGovernorProposer(
        provider, run_id="h", candidate_factory=CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H),
    )
    proposal = asyncio.run(proposer.propose(PICTURE, BOUNDS))
    assert len(provider.requests) == 1
    assert proposal.policy is None
    assert "not decisive" in proposal.rationale


def test_governor_clear_choices_at_both_levels_act():
    provider = _graded({
        GOV_L1: {"area:life": 0.82, "area:sustenance": 0.08, "unknown": 0.10},
        _gov_l2("life"): {"select_candidate:mean_oxygen_per_occupant": 0.92, "unknown": 0.08},
    })
    proposer = SemifGovernorProposer(
        provider, run_id="h", candidate_factory=CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V2H),
    )
    proposal = asyncio.run(proposer.propose(PICTURE, BOUNDS))
    assert proposal.policy is not None


def test_admin_tie_at_level1_accepts_the_governor():
    provider = _graded({ADM_L1: {"area:sustenance": 0.36, "accept_governor": 0.37, "unknown": 0.27}})
    admin = AmministratoreSemIf(
        provider, BOUNDS, run_id="h", district=1,
        candidate_profile=ADMIN_CANDIDATE_PROFILE_VERSION_V2H,
    )
    context = AdminSemanticContext(district=1, picture=PICTURE, governor_policy=_governor_law(),
                                   colony_indicators=PICTURE.indicators, prompt="p")
    out = asyncio.run(admin.propose_semantic_async(context))
    assert out["raw"]["accept"] is True
    assert len(provider.requests) == 1


def test_v2h_providers_do_not_threshold_confidence():
    from src.governors.config import _semantic_for_profile

    assert _semantic_for_profile({"candidate_profile": "v2h", "min_confidence": 0.65})["min_confidence"] == 0.0
    assert _semantic_for_profile({"candidate_profile": "v2", "min_confidence": 0.65})["min_confidence"] == 0.65
    assert _semantic_for_profile({"min_confidence": 0.65})["min_confidence"] == 0.65


def test_cli_section_for_v2h_has_no_provider_threshold():
    from scripts.run_governor_experiment import _semantic_section

    assert _semantic_section("fake", run_id="x", candidate_profile="v2h")["min_confidence"] == 0.0
    assert _semantic_section("fake", run_id="x", candidate_profile="v2")["min_confidence"] == 0.65


# --- amministratori v2h: deliberano solo con una legge di governo in vigore -----
# Scelta dell'utente (2026-09-23): il governo decide se e su cosa legiferare, gli
# amministratori adattano (accettano o riscrivono) la legge del governo.

@pytest.mark.parametrize("law", [None, Policy((), "SemIf selected no_intervention")])
def test_admin_without_a_governor_law_does_not_deliberate(law):
    out, provider = _admin({ADM_L1: "area:explore"}, governor_policy=law)
    assert provider.requests == []
    assert out["raw"]["accept"] is True
    assert out["raw"]["rationale"].startswith("no governor law in force")


def test_admin_with_a_governor_law_deliberates():
    _, provider = _admin({ADM_L1: "accept_governor"})
    assert len(provider.requests) == 1


def test_v2h_administrators_without_governor_are_refused():
    from src.governors.config import build_administration

    config = {"governors": {"arm": "none", "administrators": {
        "enabled": True, "follow_governor_arm": False, "arm": "semif",
        "semantic": {"provider": "fake", "run_id": "x", "candidate_profile": "v2h"},
    }}}
    with pytest.raises(ValueError, match="v2h"):
        build_administration(config, seed=1,
                             semantic_provider_factory=lambda _d: FakeSemanticDecisionProvider())


# --- telemetria di governo: interventi, cambi di legge, conferme, attese -------

def test_governance_summary_counts_changes_and_holds(tmp_path):
    import json

    from src.semantic_governance.artifacts import governance_summary

    law_a = {"rules": [{"if": {"indicator": "food_per_occupant"}, "weights": {"0": 3.0}}]}
    law_b = {"rules": [{"if": {"indicator": "power_coverage"}, "weights": {"2": 2.0}}]}
    gov_rows = [
        {"tick": 0, "governor": {"rationale": "x"}, "policy": law_a},
        {"tick": 1, "governor": {"rationale": "x"}, "policy": law_a},
        {"tick": 2, "governor": {"rationale": "SemIf level 1 not decisive"}, "policy": None},
        {"tick": 3, "governor": {"rationale": "x"}, "policy": law_b},
    ]
    (tmp_path / "governor_decisions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in gov_rows), encoding="utf-8")
    adm_rows = [{"last_round": [
        {"accepted_government_policy": True, "rationale": "no governor law in force; skip"},
        {"accepted_government_policy": False, "rationale": "rewrite"},
        {"accepted_government_policy": True, "rationale": "SemIf selected accept_governor"},
    ]}]
    (tmp_path / "administrator_decisions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in adm_rows), encoding="utf-8")
    summary = governance_summary(tmp_path)
    assert summary["governor"] == {
        "rounds": 4, "law_changes": 2, "reaffirmations": 1, "holds": 1, "abrogations": 0,
    }
    assert summary["administrators"] == {
        "district_rounds": 3, "rewrites": 1, "accepts": 1, "no_governor_law": 1,
    }


def test_governance_summary_reads_the_proposal_not_the_law_in_force(tmp_path):
    # Pilota del 2026-09-23: il registro scrive la legge IN VIGORE anche quando
    # il governo lascia com'e' (proposal None); contarla come riaffermazione
    # dava 0 attese su 100 tornate, mentre le attese vere erano 35.
    import json

    from src.semantic_governance.artifacts import governance_summary

    law_a = {"rationale": "a", "rules": [{"if": {"indicator": "food"}, "weights": {"0": 3.0}}]}
    law_b = {"rationale": "b", "rules": [{"if": {"indicator": "power"}, "weights": {"2": 2.0}}]}
    empty = {"rationale": "no_intervention", "rules": []}
    rows = [
        {"tick": 0, "governor": {"proposal": law_a}, "policy": law_a},   # cambio
        {"tick": 1, "governor": {"proposal": None}, "policy": law_a},    # attesa
        {"tick": 2, "governor": {"proposal": law_a}, "policy": law_a},   # riaffermazione
        {"tick": 3, "governor": {"proposal": law_b}, "policy": law_b},   # cambio
        {"tick": 4, "governor": {"proposal": empty}, "policy": empty},   # abrogazione
        {"tick": 5, "governor": {"proposal": None}, "policy": empty},    # attesa
    ]
    (tmp_path / "governor_decisions.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert governance_summary(tmp_path)["governor"] == {
        "rounds": 6, "law_changes": 2, "reaffirmations": 1, "holds": 2, "abrogations": 1,
    }
