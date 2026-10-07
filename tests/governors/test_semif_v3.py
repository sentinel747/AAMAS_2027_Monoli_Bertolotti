"""Profilo v3 del governo System 1 (2026-09-23, scelte dell'utente).

Rispetto a v2h, che resta intatto per la campagna gia' eseguita:

- **vince l'opzione piu' probabile**, senza soglia minima ne' distacco: il
  modello puo' sempre aspettare o non intervenire, quindi una politica vince
  solo se batte anche quelle scelte;
- **aspettare e non intervenire sono opzioni esplicite a ogni livello**;
- **descrizioni oneste e corte**: ogni candidata dice la condizione e su quante
  celle scatta adesso (nessuna / una parte / tutte, cioe' una costante), con
  l'informazione in testa perche' Laya tronca ogni opzione a ~22 token;
- la domanda dice il criterio: intervenire solo se un indicatore mostra un
  problema che la regola risolve; i pesi sono relativi;
- `power_coverage` e' escluso: il quadro lo legge dopo il consumo e vale 0
  anche con la corrente coperta (ricostruzione di jev_s4, passo 101).
"""

import asyncio

from src.governors.admin_semif_arm import AdminSemanticContext, AmministratoreSemIf
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds
from src.governors.policy_candidates import (
    ADMIN_CANDIDATE_PROFILE_VERSION_V3,
    CANDIDATE_PROFILE_VERSION_V2H,
    CANDIDATE_PROFILE_VERSION_V3,
    PROFILE_VERSIONS,
    CandidatePolicyFactory,
    is_hierarchical,
    is_v3,
    plurality_choice,
)
from src.governors.semif_arm import SemifGovernorProposer
from src.semantic_governance.client import FakeSemanticDecisionProvider
from src.semantic_governance.schemas import SemanticDecision

INDICATORS = (
    "food_per_occupant", "water_per_occupant", "oxygen_per_occupant",
    "ice_per_occupant", "material_per_occupant", "minerals_per_occupant",
    "power_coverage", "structure_integrity", "occupants",
)
UNA_CELLA = ColonyPicture(
    step=101, population=300, n_cells=1,
    indicators={
        **{n: {"mean": 1.0, "std": 0.0, "min": 1.0, "max": 1.0} for n in INDICATORS},
        "food_per_occupant": {"mean": 3.873, "std": 0.0, "min": 3.873, "max": 3.873},
        "occupants": {"mean": 300.0, "std": 0.0, "min": 300.0, "max": 300.0},
        "power_coverage": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
    },
    population_stats={"health_mean": 1.0},
    structures={"shelter": 186},
    rule_hits=(("food_per_occupant < 2 -> sustenance", 0),),
    deaths={(1, 2): {"starvation": 2}, (3, 4): {"hypoxia": 1}},
)
CELLE_DIVERSE = ColonyPicture(
    step=501, population=900, n_cells=6,
    indicators={
        **{n: {"mean": 2.0, "std": 0.5, "min": 1.0, "max": 3.0} for n in INDICATORS},
        "food_per_occupant": {"mean": 2.5, "std": 0.8, "min": 1.5, "max": 4.0},
        "occupants": {"mean": 150.0, "std": 200.0, "min": 3.0, "max": 700.0},
    },
)
BOUNDS = Bounds(0.25, 4.0)
GOV_L1 = f"{CANDIDATE_PROFILE_VERSION_V3}:L1"
ADM_L1 = f"{ADMIN_CANDIDATE_PROFILE_VERSION_V3}:L1"


def _gov_l2(area):
    return f"{CANDIDATE_PROFILE_VERSION_V3}:L2:{area}"


def _adm_l2(area):
    return f"{ADMIN_CANDIDATE_PROFILE_VERSION_V3}:L2:{area}"


def _candidates(picture=UNA_CELLA, profile=CANDIDATE_PROFILE_VERSION_V3):
    return CandidatePolicyFactory(profile).build(picture, BOUNDS)


# --- profilo e regola ---------------------------------------------------------

def test_v3_is_selectable_hierarchical_and_leaves_v2h_alone():
    assert PROFILE_VERSIONS["v3"] == (
        CANDIDATE_PROFILE_VERSION_V3, ADMIN_CANDIDATE_PROFILE_VERSION_V3,
    )
    assert is_hierarchical(CANDIDATE_PROFILE_VERSION_V3)
    assert is_hierarchical(ADMIN_CANDIDATE_PROFILE_VERSION_V3)
    assert is_v3(CANDIDATE_PROFILE_VERSION_V3) and not is_v3(CANDIDATE_PROFILE_VERSION_V2H)
    v2h = _candidates(profile=CANDIDATE_PROFILE_VERSION_V2H)
    assert "mean_power_coverage" in {c.id for c in v2h.candidates}
    assert v2h.by_id()["mean_power_coverage"].description == (
        "Weight build by 3 in every settled cell (current value 0) for power_coverage"
    )


def _decision(probabilities, selected=None):
    selected = selected or next(iter(probabilities))
    return SemanticDecision(
        run_id="r", step=1, actor="governor", question_id="q", input_hash="h" * 64,
        selected_option=selected, probabilities=probabilities, confidence=0.1,
        route="filter", runtime="test",
    )


def test_plurality_takes_the_most_probable_option_with_no_threshold():
    # 0,26 contro 0,25: nessuna soglia e nessun distacco, vince la piu' alta.
    d = _decision({"area:build": 0.26, "unknown": 0.25, "area:life": 0.24,
                   "no_intervention": 0.25}, selected="unknown")
    assert plurality_choice(d) == "area:build"


def test_plurality_tie_goes_to_the_provider_choice():
    assert plurality_choice(_decision({"area:build": 0.4, "area:life": 0.4, "unknown": 0.2},
                                      selected="area:life")) == "area:life"
    assert plurality_choice(_decision({"area:build": 0.4, "unknown": 0.4, "area:life": 0.2},
                                      selected="area:build")) == "area:build"


def test_plurality_without_probabilities_uses_the_provider_choice():
    d = _decision({"area:build": 1.0}, selected="area:build")
    assert plurality_choice(d) == "area:build"


# --- candidate oneste -----------------------------------------------------------

def test_v3_drops_power_coverage():
    ids = {c.id for c in _candidates().candidates}
    assert not any("power_coverage" in i for i in ids)
    ids = {c.id for c in _candidates(CELLE_DIVERSE).candidates}
    assert not any("power_coverage" in i for i in ids)


def test_uniform_cells_are_declared_a_colony_wide_constant():
    c = _candidates().by_id()["mean_food_per_occupant"]
    assert c.description.startswith("sustenance x3 where food_per_occupant <")
    assert "fires now on all 1 cells (a constant)" in c.description


def test_reference_rule_that_does_not_fire_says_so():
    c = _candidates().by_id()["reference_food_per_occupant"]
    assert c.description == "sustenance x3 where food_per_occupant < 2 | fires now on none of 1 cells"
    c = _candidates().by_id()["reference_occupants"]
    assert c.description == "explore x2 where occupants > 150 | fires now on all 1 cells (a constant)"


def test_spread_cells_fire_on_part_of_the_colony():
    by_id = _candidates(CELLE_DIVERSE).by_id()
    assert by_id["mean_food_per_occupant"].description == (
        "sustenance x3 where food_per_occupant < 2.5 | fires now on some of 6 cells"
    )
    # 2 sta fra il minimo 1,5 e il massimo 4: scatta su una parte.
    assert by_id["reference_food_per_occupant"].description.endswith("some of 6 cells")
    # 150 sta fra 3 e 700.
    assert by_id["reference_occupants"].description.endswith("some of 6 cells")


def test_v3_descriptions_are_short_enough_for_laya():
    for picture in (UNA_CELLA, CELLE_DIVERSE):
        for c in _candidates(picture).candidates:
            assert len(c.description) <= 96, c.description


def test_v3_candidates_are_the_same_rules_as_v2h_minus_power():
    v2h = {c.id: c.policy.rules for c in _candidates(profile=CANDIDATE_PROFILE_VERSION_V2H).candidates}
    v3 = {c.id: c.policy.rules for c in _candidates().candidates}
    v2h.pop("mean_power_coverage")
    assert v3 == v2h


# --- governatore ----------------------------------------------------------------

def _propose(decisions=None, factory=None, picture=UNA_CELLA):
    provider = FakeSemanticDecisionProvider(decisions, factory=factory)
    proposer = SemifGovernorProposer(
        provider, run_id="v3", candidate_factory=CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V3),
    )
    return asyncio.run(proposer.propose(picture, BOUNDS)), provider


def test_level1_offers_wait_no_intervention_and_the_areas():
    _, provider = _propose({GOV_L1: "unknown"})
    first = provider.requests[0]
    assert first.option_ids == (
        "unknown", "no_intervention", "area:sustenance", "area:resources",
        "area:build", "area:life", "area:explore",
    )
    descr = {o.id: o.description for o in first.options}
    assert descr["unknown"].startswith("Wait")
    assert descr["no_intervention"].startswith("No intervention: no indicator shows a problem")
    assert "Intervene only if an indicator shows a problem the rule would fix" in first.question
    assert "relative" in first.question
    assert all(len(d) <= 96 for d in descr.values())


def test_v3_state_is_compact_and_honest():
    _, provider = _propose({GOV_L1: "unknown"})
    state = provider.requests[0].state
    assert "power_coverage" not in state["indicators"]
    assert "structures" not in state
    assert state["deaths_since_last_round"] == 3
    assert state["policy_in_force"] == [{"rule": "food_per_occupant < 2 -> sustenance", "cell_steps": 0}]
    # Una cella: un valore solo, non media/std/min/max identici.
    assert state["indicators"]["food_per_occupant"] == 3.873
    _, provider = _propose({GOV_L1: "unknown"}, picture=CELLE_DIVERSE)
    assert provider.requests[0].state["indicators"]["food_per_occupant"] == {
        "mean": 2.5, "min": 1.5, "max": 4.0,
    }


def test_wait_keeps_the_policy_with_one_question():
    proposal, provider = _propose({GOV_L1: "unknown"})
    assert len(provider.requests) == 1
    assert proposal.policy is None


def test_no_intervention_repeals():
    proposal, _ = _propose({GOV_L1: "no_intervention"})
    assert proposal.policy is not None and proposal.policy.rules == ()


def test_area_then_candidate_with_a_bare_plurality_is_adopted():
    def factory(request):
        if request.question_id == GOV_L1:
            probs = {o: 0.1 for o in request.option_ids}
            probs.update({"area:sustenance": 0.3, "unknown": 0.2})
            selected = "unknown"  # la scelta del provider non conta: conta la piu' alta
        else:
            # 0,34 contro 0,22: nessuna soglia, vince comunque la piu' alta.
            probs = {o: 0.22 for o in request.option_ids}
            probs["select_candidate:reference_food_per_occupant"] = 0.34
            selected = "select_candidate:reference_food_per_occupant"
        return SemanticDecision(
            run_id=request.run_id, step=request.step, actor=request.actor,
            question_id=request.question_id, input_hash=request.input_hash,
            selected_option=selected, probabilities=probs, confidence=0.05,
            route="filter", runtime="test",
        )

    proposal, provider = _propose(factory=factory)
    assert [r.question_id for r in provider.requests] == [GOV_L1, _gov_l2("sustenance")]
    second = provider.requests[1]
    assert second.option_ids[-1] == "none_of_the_above"
    assert {o.id: o.description for o in second.options}["none_of_the_above"].startswith("None of these")
    assert proposal.policy is not None
    rule = proposal.policy.rules[0]
    assert rule.condition.indicator == "food_per_occupant" and rule.condition.value == 2.0


def test_none_of_these_keeps_the_policy():
    proposal, provider = _propose({GOV_L1: "area:sustenance", _gov_l2("sustenance"): "none_of_the_above"})
    assert len(provider.requests) == 2
    assert proposal.policy is None
    assert "none of these" in proposal.rationale


# --- amministratori -------------------------------------------------------------

def _admin(decisions, governor_policy="law"):
    from src.governors.arms import reference_policy

    provider = FakeSemanticDecisionProvider(decisions)
    admin = AmministratoreSemIf(
        provider, BOUNDS, run_id="v3", district=2,
        candidate_profile=ADMIN_CANDIDATE_PROFILE_VERSION_V3,
    )
    context = AdminSemanticContext(
        district=2, picture=UNA_CELLA,
        governor_policy=reference_policy(BOUNDS) if governor_policy == "law" else governor_policy,
        colony_indicators=UNA_CELLA.indicators, prompt="p",
    )
    return asyncio.run(admin.propose_semantic_async(context)), provider


def test_admin_without_governor_law_does_not_deliberate():
    result, provider = _admin({}, governor_policy=None)
    assert provider.requests == []
    assert result["raw"]["accept"] is True


def test_admin_level1_offers_keep_undecided_and_the_areas():
    result, provider = _admin({ADM_L1: "accept_governor"})
    first = provider.requests[0]
    assert first.option_ids == (
        "accept_governor", "unknown", "area:sustenance", "area:resources",
        "area:build", "area:life", "area:explore",
    )
    assert "Rewrite only if an indicator in this district shows a problem" in first.question
    assert "power_coverage" not in first.state["indicators"]
    assert first.state["governor_policy"]
    assert result["raw"]["accept"] is True
    result, _ = _admin({ADM_L1: "unknown"})
    assert result["raw"]["accept"] is True


def test_admin_area_then_candidate_rewrites_and_none_accepts():
    result, provider = _admin({
        ADM_L1: "area:sustenance",
        _adm_l2("sustenance"): "rewrite_with_candidate:reference_food_per_occupant",
    })
    assert provider.requests[1].option_ids[-1] == "none_of_the_above"
    assert result["raw"]["accept"] is False
    result, _ = _admin({ADM_L1: "area:sustenance", _adm_l2("sustenance"): "none_of_the_above"})
    assert result["raw"]["accept"] is True
