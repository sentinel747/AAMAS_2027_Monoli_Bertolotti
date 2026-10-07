"""Il braccio LLM: nessun errore esce da `propose`, e il prompt dice tutto.

Il provider e' iniettato, quindi tutto si verifica con provider finti e zero
credito. Le classi di pericolo: una risposta rotta che uccide la simulazione,
un prompt che tace un pezzo del vocabolario (il modello non deve indovinare),
e la contabilita' che riporta zero per una run che ha chiamato.
"""

import asyncio
import json

from src.governors.llm_arm import LLMProposer, build_governor_prompt
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, INDICATORS, MAX_RULES, PILLAR_BY_NAME

BOUNDS = Bounds(0.25, 4.0)
PICTURE = ColonyPicture(
    step=41,
    population=100,
    metrics={"food_margin": 1.1},
    indicators={"food_per_occupant": {"mean": 1.5, "std": 0.5, "min": 1.0, "max": 2.0}},
    population_stats={"health_mean": 0.9},
    structures={"greenhouse": 4},
    n_cells=2,
)


class _Response:
    def __init__(self, text, tokens_in=10, tokens_out=20):
        self.text = text
        self.tokens_in = tokens_in
        self.tokens_out = tokens_out
        self.cost_usd = 0.0
        self.error = ""


class _Provider:
    provider_id = "finto"
    model = "finto-1"

    def __init__(self, payload):
        self._payload = payload

    async def async_complete_json(self, prompt):
        if isinstance(self._payload, Exception):
            raise self._payload
        return _Response(self._payload)


def _propose(payload):
    return asyncio.run(LLMProposer(_Provider(payload)).propose(PICTURE, BOUNDS))


def test_a_valid_response_becomes_a_policy():
    raw = {
        "policy": [
            {"if": {"indicator": "food_per_occupant", "op": "<", "value": 2.0},
             "weights": {"sustenance": 3.0}},
        ],
        "rationale": "poco cibo nelle celle basse",
    }
    proposal = _propose(json.dumps(raw))
    assert proposal.policy is not None and len(proposal.policy.rules) == 1
    assert proposal.rationale == "poco cibo nelle celle basse"
    assert proposal.provider == "finto"
    assert proposal.tokens_out == 20


def test_non_json_is_silence_not_an_exception():
    proposal = _propose("non sono JSON")
    assert proposal.policy is None
    assert "non e' JSON" in proposal.rationale


def test_out_of_schema_json_is_silence_with_a_reason():
    proposal = _propose(json.dumps({"policy": "niente lista"}))
    assert proposal.policy is None
    assert "fuori schema" in proposal.rationale


def test_a_network_error_is_silence_with_the_error_text():
    proposal = _propose(RuntimeError("rete giu'"))
    assert proposal.policy is None
    assert "rete giu'" in proposal.rationale


def test_the_prompt_carries_the_whole_vocabulary():
    prompt = build_governor_prompt(PICTURE, BOUNDS)
    for indicator in INDICATORS:
        assert indicator in prompt
    for pillar in PILLAR_BY_NAME:
        assert pillar in prompt
    assert str(MAX_RULES) in prompt
    assert "0.25" in prompt and "4" in prompt
    assert "PRIMA regola" in prompt, "la semantica first-match va detta"
    assert '"if": null' in prompt.replace("\\\"", "\""), (
        "la regola incondizionata va detta"
    )
    assert "preferenze" in prompt, (
        "il modello deve sapere che l'else sono le preferenze individuali"
    )


def test_the_prompt_shows_the_aggregates_not_cells():
    prompt = build_governor_prompt(PICTURE, BOUNDS)
    assert "food_margin" in prompt
    assert "health_mean" in prompt
    assert "greenhouse" in prompt
    assert '"y"' not in prompt and "occupants\": 300" not in prompt, (
        "niente elenco di celle: il governatore scrive condizioni, non coordinate"
    )


def test_degenerate_indicators_are_flagged_in_the_prompt():
    """Misurato: sulla cella equatoriale il ghiaccio e' zero ovunque, e in
    monocella ogni condizione e' tutto-o-niente. Il modello non deve scoprirlo
    scrivendo una regola morta — la lezione della quota 0."""
    flat = lambda v: {"mean": v, "std": 0.0, "min": v, "max": v}
    single = ColonyPicture(
        step=1, population=300,
        indicators={"ice_per_occupant": flat(0.0), "occupants": flat(300.0)},
        n_cells=1,
    )
    prompt = build_governor_prompt(single, BOUNDS)
    assert "ice_per_occupant e' a ZERO" in prompt
    assert "UNA sola cella occupata" in prompt
    assert "occupants vale" not in prompt, (
        "in monocella tutto e' piatto: elencare ogni indicatore sarebbe rumore"
    )

    multi = ColonyPicture(
        step=1, population=300,
        indicators={
            "ice_per_occupant": flat(0.0),
            "occupants": {"mean": 150.0, "std": 10.0, "min": 140.0, "max": 160.0},
            "food_per_occupant": flat(2.0),
        },
        n_cells=3,
    )
    prompt = build_governor_prompt(multi, BOUNDS)
    assert "food_per_occupant vale 2" in prompt, "piatto ma non zero: va detto"
    assert "UNA sola cella" not in prompt


def test_the_prompt_says_weights_are_relative_and_materials_are_not_produced():
    """Le due lacune emerse dalla prima campagna LLM a policy (2026-08-24).

    Il modello scriveva `resources x2.5, gli altri x1` credendo di lasciarli
    com'erano: i punteggi sono rinormalizzati, quindi alzare tre pilastri
    penalizza gli altri — explore per primo, e l'esplorazione e' crollata del
    65%. E copiava la regola sui materiali senza sapere che raccoglierli non
    li produce: +50% di collect_materials che nessuna costruzione chiedeva.
    """
    prompt = build_governor_prompt(PICTURE, BOUNDS)
    assert "RELATIVI" in prompt
    assert "explore" in prompt[prompt.find("RELATIVI"):], (
        "l'esempio del pilastro da proteggere deve essere l'esplorazione"
    )
    assert "non ne PRODUCE" in prompt


def test_a_zero_indicator_warning_names_the_shadow():
    flat = lambda v: {"mean": v, "std": 0.0, "min": v, "max": v}
    picture = ColonyPicture(step=1, population=10,
                            indicators={"ice_per_occupant": flat(0.0)}, n_cells=1)
    assert "OSCURA tutte le regole successive" in build_governor_prompt(picture, BOUNDS)


def test_a_healthy_picture_carries_no_warnings():
    varied = lambda: {"mean": 2.0, "std": 0.5, "min": 1.0, "max": 3.0}
    picture = ColonyPicture(
        step=1, population=300,
        indicators={name: varied() for name in (
            "food_per_occupant", "ice_per_occupant", "material_per_occupant",
            "minerals_per_occupant", "occupants",
        )},
        n_cells=4,
    )
    assert "Avvertenze" not in build_governor_prompt(picture, BOUNDS)


def test_the_call_lands_in_the_cost_tracker():
    class Tracker:
        def __init__(self):
            self.calls = 0

        def record(self, **kwargs):
            self.calls += 1
            self.last = kwargs

    tracker = Tracker()
    proposer = LLMProposer(_Provider(json.dumps({"policy": []})), cost_tracker=tracker)
    asyncio.run(proposer.propose(PICTURE, BOUNDS))
    assert tracker.calls == 1
    assert tracker.last["tokens_out"] == 20
