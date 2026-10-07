"""Il registro v2 e la riesecuzione: cio' che si scrive si rilegge identico.

Il registro e' l'unica prova rieseguibile di una run LLM: un giro
scrittura-rilettura che non restituisca la stessa policy renderebbe la
"riesecuzione" una run diversa con lo stesso nome. E un registro del vecchio
consiglio non deve degradare in silenzio: deve rifiutarsi con la ragione.
"""

import json

import pytest

from src.agents import pillars
from src.governors.arms import GovernorProposal
from src.governors.observation import ColonyPicture
from src.governors.policy import Condition, Policy, PolicyDrops, Rule
from src.governors.record import (
    GovernorRecorder,
    ReplayProposer,
    _policy_from_json,
    _policy_to_json,
    load_records,
)

PICTURE = ColonyPicture(step=3, population=10)

POLICY = Policy(
    (
        Rule(Condition("food_per_occupant", "<", 2.0), {pillars.P_SUSTENANCE: 3.0}),
        Rule(None, {pillars.P_BUILD: 1.5, pillars.P_EXPLORE: 0.5}),
    ),
    rationale="prova",
)


def test_policy_json_round_trip_is_identity():
    assert _policy_from_json(_policy_to_json(POLICY)) == POLICY
    assert _policy_to_json(None) is None
    assert _policy_from_json(None) is None


def test_pillars_travel_by_name_not_by_index():
    payload = _policy_to_json(POLICY)
    assert set(payload["rules"][0]["weights"]) == {"sustenance"}
    assert set(payload["rules"][1]["weights"]) == {"build", "explore"}


def _record_one(tmp_path, proposal, policy, missed=False):
    path = tmp_path / "governor_decisions.jsonl"
    recorder = GovernorRecorder(path)
    recorder.record(
        tick=0, application_step=3, picture=PICTURE,
        proposal=proposal, policy=policy, missed=missed,
    )
    recorder.close()
    return path


def test_the_recorder_writes_one_readable_row(tmp_path):
    proposal = GovernorProposal(
        policy=POLICY, rationale="prova", provider="finto", model="m",
        tokens_in=5, tokens_out=9, latency_s=0.5, drops=PolicyDrops(clamped_weights=1),
    )
    path = _record_one(tmp_path, proposal, POLICY)
    rows = load_records(path)
    assert len(rows) == 1
    row = rows[0]
    assert row["tick"] == 0 and row["application_step"] == 3
    assert row["picture_step"] == 3 and row["picture_digest"]
    assert row["governor"]["provider"] == "finto"
    assert row["governor"]["drops"]["clamped_weights"] == 1
    assert _policy_from_json(row["policy"]) == POLICY


def test_the_recorder_preserves_semantic_decision_without_secrets(tmp_path):
    proposal = GovernorProposal(
        policy=POLICY,
        provider="fake",
        model="bounded",
        semantic_decision={
            "schema_version": "1.0",
            "selected_option": "select_candidate:mean_food_per_occupant",
            "fallback_reason": "",
        },
        candidate_profile="jev-semif-governor-v1",
    )
    row = load_records(_record_one(tmp_path, proposal, POLICY))[0]
    assert row["governor"]["candidate_profile"] == "jev-semif-governor-v1"
    assert row["governor"]["semantic_decision"]["schema_version"] == "1.0"


def test_a_miss_writes_a_row_without_governor(tmp_path):
    path = _record_one(tmp_path, None, POLICY, missed=True)
    row = load_records(path)[0]
    assert row["missed"] is True
    assert row["governor"] is None
    assert _policy_from_json(row["policy"]) == POLICY, (
        "nel tick mancato resta in vigore la policy precedente, e va registrata"
    )


def test_last_summary_is_compact_and_correct(tmp_path):
    recorder = GovernorRecorder(tmp_path / "r.jsonl")
    assert recorder.last_summary() is None
    proposal = GovernorProposal(policy=POLICY, rationale="prova", tokens_out=9)
    recorder.record(tick=0, application_step=3, picture=PICTURE,
                    proposal=proposal, policy=POLICY, missed=False)
    summary = recorder.last_summary()
    recorder.close()
    assert summary["rules_proposed"] == 2
    assert summary["rules_in_force"] == 2
    assert summary["rationale"] == "prova"


def test_replay_returns_the_recorded_policies_in_order(tmp_path):
    path = tmp_path / "r.jsonl"
    recorder = GovernorRecorder(path)
    second = Policy((Rule(None, {pillars.P_LIFE: 2.0}),), rationale="seconda")
    recorder.record(tick=0, application_step=3, picture=PICTURE,
                    proposal=GovernorProposal(policy=POLICY), policy=POLICY, missed=False)
    recorder.record(tick=1, application_step=5, picture=PICTURE,
                    proposal=GovernorProposal(policy=second), policy=second, missed=False)
    recorder.close()

    import asyncio
    replay = ReplayProposer(path)
    first = asyncio.run(replay.propose(PICTURE, None))
    assert first.policy == POLICY and first.provider == "replay"
    assert asyncio.run(replay.propose(PICTURE, None)).policy == second
    exhausted = asyncio.run(replay.propose(PICTURE, None))
    assert exhausted.policy == Policy(), (
        "esaurito il registro si propone il nulla, non si inventa"
    )


def test_a_council_era_record_is_refused_with_the_reason(tmp_path):
    path = tmp_path / "old.jsonl"
    path.write_text(
        json.dumps({
            "tick": 0, "application_step": 3, "missed": False,
            "governors": [{"mandate": "life_support"}],
            "directive": {"cells": []},
        }) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="consiglio"):
        ReplayProposer(path)


def test_a_record_with_an_unknown_indicator_is_refused_at_load(tmp_path):
    """Un registro scritto a mano (o da un'altra versione) con un indicatore
    fuori vocabolario deve fallire QUI, con il nome della voce — non con un
    `KeyError` dentro `kernel.step` a meta' riesecuzione."""
    path = tmp_path / "r.jsonl"
    path.write_text(
        json.dumps({
            "tick": 0, "application_step": 3, "missed": False,
            "governor": None,
            "policy": {"rationale": "", "rules": [
                {"if": {"indicator": "felicita", "op": "<", "value": 1.0},
                 "weights": {"build": 2.0}},
            ]},
        }) + chr(10),
        encoding="utf-8",
    )
    replay = ReplayProposer(path)
    import asyncio
    with pytest.raises(ValueError, match="felicita"):
        asyncio.run(replay.propose(PICTURE, None))


def test_a_record_with_an_unknown_pillar_or_op_is_refused_at_load(tmp_path):
    import asyncio

    for payload, match in (
        ({"if": None, "weights": {"felicita": 2.0}}, "felicita"),
        ({"if": {"indicator": "occupants", "op": ">=", "value": 1.0},
          "weights": {"build": 2.0}}, ">="),
    ):
        path = tmp_path / f"r_{match.strip('>=')}.jsonl"
        path.write_text(
            json.dumps({
                "tick": 0, "application_step": 3, "missed": False,
                "governor": None,
                "policy": {"rationale": "", "rules": [payload]},
            }) + chr(10),
            encoding="utf-8",
        )
        replay = ReplayProposer(path)
        with pytest.raises(ValueError):
            asyncio.run(replay.propose(PICTURE, None))


def test_a_model_rationale_with_line_separators_stays_one_row(tmp_path):
    """U+2028 dentro la rationale non deve spezzare il registro."""
    tricky = Policy((), rationale="prima seconda")
    path = _record_one(tmp_path, GovernorProposal(policy=tricky, rationale=tricky.rationale), tricky)
    raw = path.read_text(encoding="utf-8")
    assert raw.count("\n") == 1
    assert load_records(path)[0]["policy"]["rationale"] == "prima seconda"
