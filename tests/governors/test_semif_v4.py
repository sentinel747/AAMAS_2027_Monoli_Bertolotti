"""Profilo v4 del governo System 1 (2026-09-24): v3 piu' il metro del «problema».

Il pilota v3 (seme 4) ha mostrato amministratori Jev che riscrivevano il 99%
delle volte con «life x3 dove l'ossigeno e' sotto la media», mentre in questo
mondo nessuno muore di ipossia (campagna v2h, `none`: 5.126 morti di fame,
4.875 di sete, zero di ipossia). A Jev mancava il metro di che cosa sia un
problema. v4 aggiunge:

- i morti dall'ultima tornata **per causa**, al governo (colonia) e agli
  amministratori (distretto);
- per l'amministratore, **dove scatta adesso la legge del governo** nel suo
  distretto (nessuna / una parte / tutte le celle);
- un criterio che lega il problema alle cause di morte.

Regola di decisione, opzioni e candidate restano quelle di v3.
"""

import asyncio

from src.governors.admin_semif_arm import AdminSemanticContext, AmministratoreSemIf
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds
from src.governors.policy_candidates import (
    ADMIN_CANDIDATE_PROFILE_VERSION_V3,
    ADMIN_CANDIDATE_PROFILE_VERSION_V4,
    CANDIDATE_PROFILE_VERSION_V3,
    CANDIDATE_PROFILE_VERSION_V4,
    PROFILE_VERSIONS,
    CandidatePolicyFactory,
    is_hierarchical,
    is_v3,
)
from src.governors.semif_arm import SemifGovernorProposer
from src.semantic_governance.client import FakeSemanticDecisionProvider

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
    },
    population_stats={"health_mean": 1.0},
    deaths={(1, 2): {"starvation": 2}, (3, 4): {"starvation": 1, "dehydration": 4}},
)
BOUNDS = Bounds(0.25, 4.0)


def _gov(profile, picture=UNA_CELLA):
    provider = FakeSemanticDecisionProvider({f"{profile}:L1": "unknown"})
    proposer = SemifGovernorProposer(
        provider, run_id="v4", candidate_factory=CandidatePolicyFactory(profile),
    )
    asyncio.run(proposer.propose(picture, BOUNDS))
    return provider.requests[0]


def _admin(profile, deaths=None, decisions=None):
    from src.governors.arms import reference_policy

    provider = FakeSemanticDecisionProvider(decisions or {f"{profile}:L1": "accept_governor"})
    admin = AmministratoreSemIf(provider, BOUNDS, run_id="v4", district=2, candidate_profile=profile)
    context = AdminSemanticContext(
        district=2, picture=UNA_CELLA, governor_policy=reference_policy(BOUNDS),
        colony_indicators=UNA_CELLA.indicators, prompt="p",
        **({"deaths": deaths} if deaths is not None else {}),
    )
    result = asyncio.run(admin.propose_semantic_async(context))
    return result, provider


def test_v4_is_selectable_and_uses_the_v3_flow():
    assert PROFILE_VERSIONS["v4"] == (
        CANDIDATE_PROFILE_VERSION_V4, ADMIN_CANDIDATE_PROFILE_VERSION_V4,
    )
    assert is_v3(CANDIDATE_PROFILE_VERSION_V4) and is_v3(ADMIN_CANDIDATE_PROFILE_VERSION_V4)
    assert is_hierarchical(CANDIDATE_PROFILE_VERSION_V4)


def test_v4_candidates_are_the_v3_candidates():
    v3 = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V3).build(UNA_CELLA, BOUNDS)
    v4 = CandidatePolicyFactory(CANDIDATE_PROFILE_VERSION_V4).build(UNA_CELLA, BOUNDS)
    assert [(c.id, c.description, c.policy.rules) for c in v3.candidates] == [
        (c.id, c.description, c.policy.rules) for c in v4.candidates
    ]


def test_governor_v4_sees_deaths_by_cause_and_the_problem_criterion():
    req = _gov(CANDIDATE_PROFILE_VERSION_V4)
    assert req.state["deaths_since_last_round"] == {"dehydration": 4, "starvation": 3}
    assert "starvation" in req.question and "dehydration" in req.question
    assert req.question.startswith("Which area, if any,")
    # stesse opzioni di v3
    assert req.option_ids == _gov(CANDIDATE_PROFILE_VERSION_V3).option_ids


def test_governor_v3_state_is_unchanged():
    req = _gov(CANDIDATE_PROFILE_VERSION_V3)
    assert req.state["deaths_since_last_round"] == 7
    assert "starvation" not in req.question


def test_admin_v4_sees_district_deaths_and_where_the_governor_law_fires():
    _, provider = _admin(ADMIN_CANDIDATE_PROFILE_VERSION_V4, deaths={"dehydration": 3})
    state = provider.requests[0].state
    assert state["deaths_since_last_round"] == {"dehydration": 3}
    reach = {r["rule"]: r["fires"] for r in state["governor_policy_reach"]}
    assert reach == {
        "food_per_occupant < 2 -> sustenance x3": "fires now on none of 1 cells",
        "occupants > 150 -> explore x2": "fires now on all 1 cells (a constant)",
    }
    assert "die" in provider.requests[0].question


def test_admin_v4_without_deaths_gets_an_empty_count():
    _, provider = _admin(ADMIN_CANDIDATE_PROFILE_VERSION_V4)
    assert provider.requests[0].state["deaths_since_last_round"] == {}


def test_admin_v3_state_is_unchanged():
    _, provider = _admin(ADMIN_CANDIDATE_PROFILE_VERSION_V3, deaths={"dehydration": 3})
    state = provider.requests[0].state
    assert "governor_policy_reach" not in state
    assert state["deaths_since_last_round"] == 7  # dal quadro, come in v3
