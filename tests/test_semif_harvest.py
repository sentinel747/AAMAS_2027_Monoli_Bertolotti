"""Il raccoglitore di domande SemIf non deve cambiare la simulazione."""

from scripts.semif_harvest_cases import Harvester
from src.semantic_governance.schemas import SemanticDecisionRequest, SemanticOption


def _req(actor, qid, options):
    return SemanticDecisionRequest(
        run_id="r", step=1, actor=actor, question_id=qid, state={"a": 1},
        question="Which?", options=tuple(SemanticOption(o, o) for o in options),
    )


def test_agents_get_unknown_so_factors_stay_neutral():
    h = Harvester()
    d = h.decide(_req("agent:x", "jev-semif-agent-v1:L1", ["sustenance", "unknown"]))
    assert d.selected_option == "unknown"
    assert h.rows[d.input_hash]["kind"] == "agent_l1"


def test_level1_rotates_areas_and_level2_is_unknown():
    h = Harvester()
    opts = ["keep_previous", "area:life", "area:build", "unknown"]
    first = h.decide(_req("governor", "p:L1", opts)).selected_option
    second = h.decide(_req("governor", "p:L1", opts + ["no_intervention"])).selected_option
    assert {first, second} == {"area:life", "area:build"}
    l2 = h.decide(_req("governor", "p:L2:life", ["select_candidate:x", "unknown"]))
    assert l2.selected_option == "unknown"
    assert {r["kind"] for r in h.rows.values()} == {"governor_l1", "governor_l2"}


def test_same_question_is_stored_once():
    h = Harvester()
    r = _req("administrator:1", "p:L1", ["accept_governor", "area:life", "unknown"])
    h.decide(r)
    h.decide(r)
    assert len(h.rows) == 1
