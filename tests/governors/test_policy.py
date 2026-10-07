"""La validazione della policy: cio' che entra nei limiti, cio' che si scarta.

Le classi di pericolo sono le stesse della vecchia direttiva, perche' il testo
arriva dalla stessa fonte — un modello: valori fuori scala (da riportare e
contare), non-numeri che `json.loads` accetta (`NaN`, `Infinity`, booleani),
strutture fuori schema (da rifiutare in blocco), e la differenza fra "scarta la
voce" e "scarta tutto", che decide se una risposta mezza buona governa a meta'
o non governa affatto.
"""

import math

import pytest

from src.agents import pillars
from src.governors.policy import (
    Bounds,
    Condition,
    INDICATORS,
    MAX_RULES,
    PILLAR_BY_NAME,
    parse_policy,
)

BOUNDS = Bounds(0.25, 4.0)


def _rule(indicator="food_per_occupant", op="<", value=2.0, weights=None):
    return {
        "if": {"indicator": indicator, "op": op, "value": value},
        "weights": weights or {"sustenance": 3.0},
    }


def test_a_well_formed_policy_round_trips_into_typed_rules():
    policy, drops = parse_policy(
        {"policy": [_rule()], "rationale": "fame"}, BOUNDS
    )
    assert policy is not None
    assert policy.rationale == "fame"
    assert policy.rules[0].condition == Condition("food_per_occupant", "<", 2.0)
    assert policy.rules[0].weights == {pillars.P_SUSTENANCE: 3.0}
    assert drops == type(drops)()


def test_the_unconditional_rule_is_condition_none():
    policy, _ = parse_policy(
        {"policy": [{"if": None, "weights": {"build": 2.0}}]}, BOUNDS
    )
    assert policy is not None and policy.rules[0].condition is None


def test_social_is_not_in_the_vocabulary_because_it_has_no_actions():
    """La lezione della quota 0: mai offrire una leva che non fa niente.

    `PILLAR_ACTIONS[P_SOCIAL]` e' vuota, quindi un peso su `social` non
    toccherebbe alcuna azione: se il nome entrasse nel vocabolario, il modello
    potrebbe crederci e il registro mostrerebbe una politica che non esiste.
    """
    assert "social" not in PILLAR_BY_NAME
    assert not pillars.PILLAR_ACTIONS[pillars.P_SOCIAL]
    policy, drops = parse_policy(
        {"policy": [_rule(weights={"social": 3.0})]}, BOUNDS
    )
    # **Nessuna regola superstite e' silenzio, non non-intervento.** Il modello
    # ha provato a legiferare in un vocabolario che non esiste: restituire una
    # catena vuota la farebbe ADOTTARE, cioe' abrogherebbe la legge in vigore
    # per conto di un modello che non e' stato capito. Torna `None`, la catena
    # precedente resta, e i contatori dicono che cosa e' caduto.
    assert policy is None
    assert drops.unknown_pillars == 1
    assert drops.dropped_rules == 1


def test_every_named_pillar_matches_the_constants():
    for name, index in PILLAR_BY_NAME.items():
        assert pillars.PILLAR_ACTIONS[index], f"{name} non ha azioni da pesare"


def test_out_of_scale_weights_are_clamped_and_counted():
    policy, drops = parse_policy(
        {"policy": [_rule(weights={"sustenance": 99.0, "build": 0.01})]}, BOUNDS
    )
    assert policy.rules[0].weights == {
        pillars.P_SUSTENANCE: 4.0,
        pillars.P_BUILD: 0.25,
    }
    assert drops.clamped_weights == 2


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), "molto", None, True])
def test_a_non_number_weight_is_dropped_not_clamped(bad):
    """`min(max(nan, ...))` propagherebbe il NaN dichiarando di averlo limitato,
    e `True` e' un numero per `float()` ma non esprime un peso."""
    policy, drops = parse_policy(
        {"policy": [_rule(weights={"sustenance": bad, "build": 2.0})]}, BOUNDS
    )
    assert policy.rules[0].weights == {pillars.P_BUILD: 2.0}
    assert drops.unknown_pillars == 1


def test_an_unknown_indicator_drops_the_rule_and_counts_it():
    policy, drops = parse_policy(
        {"policy": [_rule(indicator="felicita"), _rule()]}, BOUNDS
    )
    assert len(policy.rules) == 1
    assert drops.unknown_indicators == 1
    assert drops.dropped_rules == 1


# **Nota sui tre test qui sotto.** La proposta ha UNA regola sola e quella
# regola cade: dal 14 settembre il risultato e' `None`, cioe' silenzio, e la
# catena in vigore resta dov'e'. Prima era una catena vuota, che veniva
# ADOTTATA: il modello sbagliava una parola e la legge spariva. Cio' che questi
# test verificano davvero --- che la regola sia scartata e contata --- non
# cambia.
@pytest.mark.parametrize("op", ["<=", ">=", "==", "tra"])
def test_an_unknown_operator_drops_the_rule(op):
    policy, drops = parse_policy({"policy": [_rule(op=op)]}, BOUNDS)
    assert policy is None
    assert drops.dropped_rules == 1


def test_a_non_finite_threshold_drops_the_rule():
    policy, drops = parse_policy({"policy": [_rule(value=float("inf"))]}, BOUNDS)
    assert policy is None
    assert drops.dropped_rules == 1


def test_a_rule_whose_weights_all_die_is_dropped():
    policy, drops = parse_policy(
        {"policy": [_rule(weights={"felicita": 3.0})]}, BOUNDS
    )
    assert policy is None
    assert drops.dropped_rules == 1


def test_una_catena_vuota_dichiarata_resta_un_non_intervento():
    """La distinzione che il disegno promette, e che era andata persa.

    `"policy": []` e' una scelta: il modello dichiara di non intervenire e la
    catena precedente viene sostituita. Una risposta SENZA la chiave `policy`
    --- il payload di ripiego di un fornitore, la risposta a un'altra domanda ---
    non ha dichiarato niente, e deve lasciare in piedi la legge in vigore.
    """
    vuota, _ = parse_policy({"policy": []}, BOUNDS)
    assert vuota is not None and not vuota.rules

    ripiego, _ = parse_policy(
        {"thought_summary": "ripiego", "chosen_action": "observe"}, BOUNDS
    )
    assert ripiego is None


def test_rules_beyond_the_cap_are_truncated_and_counted():
    policy, drops = parse_policy(
        {"policy": [_rule(value=float(i)) for i in range(MAX_RULES + 3)]}, BOUNDS
    )
    assert len(policy.rules) == MAX_RULES
    assert drops.excess_rules == 3


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "policy",
        {"policy": "non una lista"},
        {"policy": ["non un dizionario"]},
        {"policy": [{"if": "non una mappa", "weights": {"build": 2.0}}]},
        {"policy": [{"if": None, "weights": "non una mappa"}]},
    ],
)
def test_out_of_schema_input_returns_none_not_an_empty_policy(raw):
    """`None` e "policy vuota" sono esiti diversi: il primo lascia in vigore la
    policy precedente, la seconda la sostituisce col non-intervento."""
    policy, _ = parse_policy(raw, BOUNDS)
    assert policy is None


def test_an_empty_policy_is_legitimate_and_distinct_from_none():
    policy, drops = parse_policy({"policy": [], "rationale": "va tutto bene"}, BOUNDS)
    assert policy is not None
    assert policy.rules == ()
    assert drops == type(drops)()


def test_every_indicator_has_a_meaning_for_the_prompt():
    for name, meaning in INDICATORS.items():
        assert meaning.strip(), f"{name} senza spiegazione: il modello dovrebbe indovinare"


def test_thresholds_survive_exactly_no_float_massaging():
    policy, _ = parse_policy({"policy": [_rule(value=2.5)]}, BOUNDS)
    assert math.isclose(policy.rules[0].condition.value, 2.5, rel_tol=0, abs_tol=0)


def test_a_misspelt_pillar_is_read_as_the_pillar_and_counted_as_alias():
    """«sustainance» 117 volte su 487 riscritture (2026-09-06): il pilastro della
    fame spariva in silenzio. Ora si applica e si conta a parte."""
    policy, drops = parse_policy(
        {"policy": [{"if": None, "weights": {"sustainance": 3.0, "Sustain": 2.0, "build": 1.5}}]}, BOUNDS
    )
    assert policy is not None
    # Due grafie dello stesso pilastro: vince l'ultima scritta, come per un
    # dizionario JSON con chiavi ripetute; nessuna delle due e' "sconosciuta".
    assert policy.rules[0].weights[pillars.P_SUSTENANCE] == 2.0
    assert policy.rules[0].weights[pillars.P_BUILD] == 1.5
    assert drops.aliased_pillars == 2 and drops.unknown_pillars == 0


def test_a_null_rule_is_dropped_and_counted_not_fatal():
    """Il modello chiude la lista con `null`: 33 riscritture su 34 fuori schema
    nella campagna del 2026-09-06 erano questo. La voce cade, il resto vale."""
    policy, drops = parse_policy(
        {"policy": [{"if": None, "weights": {"build": 2.0}}, None]}, BOUNDS
    )
    assert policy is not None and len(policy.rules) == 1
    assert drops.dropped_rules == 1


def test_a_string_rule_is_still_out_of_schema():
    policy, _ = parse_policy({"policy": [{"if": None, "weights": {"build": 2.0}}, "se cibo < 1"]}, BOUNDS)
    assert policy is None
