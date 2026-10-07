# -*- coding: utf-8 -*-
"""Il consuntivo delle regole: il governo vede l'effetto della propria legge.

**Perche' esiste questa informazione.** La catena e' un if/elif e per ogni cella
vale la prima regola che scatta. E' la semantica di una legge, ed e' cio' che
rende l'effetto su una cella attribuibile a una riga sola. Ma ha un modo
preciso di tradire chi la scrive: una regola larga messa per prima porta via
tutte le celle alle successive, che restano scritte e non applicate. Senza
consuntivo, chi riscrive la politica non ha alcun modo di accorgersene, e
continuera' a credere che una regola in ombra stia agendo.
"""

import pytest

from src.governors.apply import ELSE_KEY, consuntivo_per_regola
from src.governors.llm_arm import build_governor_prompt
from src.governors.observation import ColonyPicture, picture_digest
from src.governors.policy import INDICATORS, Bounds, Condition, Policy, Rule


def _policy() -> Policy:
    from src.agents import pillars

    return Policy(
        (
            Rule(Condition("power_coverage", "<", 1.0), {pillars.P_BUILD: 3.0}),
            Rule(Condition("minerals_per_occupant", "<", 1.0), {pillars.P_EXPLORE: 2.5}),
        ),
        "",
    )


def _quadro(hits=()) -> ColonyPicture:
    return ColonyPicture(
        step=100,
        population=120,
        n_cells=4,
        metrics={"colony_prosperity_index": 0.7},
        indicators={
            nome: {"mean": 1.0, "std": 0.3, "min": 0.4, "max": 1.6}
            for nome in INDICATORS
        },
        population_stats={"health_mean": 0.9},
        structures={"greenhouse": 12},
        rule_hits=hits,
    )


def test_il_consuntivo_e_la_differenza_dall_ultimo_tick():
    """Il cumulato direbbe anche cio' che hanno fatto le politiche precedenti."""
    policy = _policy()
    prima = {"se power_coverage < 1 -> build x3": 100, ELSE_KEY: 10}
    ora = {"se power_coverage < 1 -> build x3": 180, ELSE_KEY: 12}
    consuntivo = dict(consuntivo_per_regola(policy, ora, prima))
    assert consuntivo["se power_coverage < 1 -> build x3"] == 80
    assert consuntivo[ELSE_KEY] == 2


def test_una_regola_mai_scattata_compare_a_zero():
    """Il caso che il consuntivo esiste per rendere visibile."""
    policy = _policy()
    consuntivo = consuntivo_per_regola(
        policy, {"se power_coverage < 1 -> build x3": 500}, {}
    )
    assert consuntivo[1][1] == 0, "la regola in ombra deve comparire, non sparire"
    assert consuntivo[1][0].startswith("se minerals_per_occupant")


def test_l_ordine_e_quello_di_scrittura_non_quello_dei_conteggi():
    """E' l'ordine che decide chi affama chi: riordinarlo nasconderebbe l'ombra."""
    policy = _policy()
    consuntivo = consuntivo_per_regola(
        policy, {"se power_coverage < 1 -> build x3": 500}, {}
    )
    assert consuntivo[0][0].startswith("se power_coverage")
    assert consuntivo[1][0].startswith("se minerals_per_occupant")
    assert consuntivo[-1][0] == ELSE_KEY


def test_senza_policy_il_consuntivo_e_vuoto():
    assert consuntivo_per_regola(None, {"x": 1}, {}) == ()
    assert consuntivo_per_regola(Policy((), ""), {"x": 1}, {}) == ()


def test_il_prompt_dichiara_la_regola_in_ombra():
    bounds = Bounds(0.25, 4.0)
    hits = (
        ("se power_coverage < 1 -> build x3", 500),
        ("se minerals_per_occupant < 1 -> explore x2.5", 0),
        (ELSE_KEY, 0),
    )
    testo = build_governor_prompt(_quadro(hits), bounds, "completo")
    assert "IN OMBRA" in testo
    assert "500 celle-passo" in testo
    # la riga in ombra e' la seconda, non la prima
    posizione_ombra = testo.index("IN OMBRA")
    assert "minerals_per_occupant" in testo[posizione_ombra - 120 : posizione_ombra]


def test_al_primo_tick_non_c_e_consuntivo():
    testo = build_governor_prompt(_quadro(()), Bounds(0.25, 4.0), "completo")
    assert "politica precedente" not in testo


def test_il_consuntivo_arriva_anche_al_gradino_cieco_ma_senza_nomi():
    """Riguarda la propria legge, non il dominio: e' meccanica, non contesto."""
    hits = (("se water_per_occupant < 0.5 -> sustenance x3", 40), (ELSE_KEY, 3))
    testo = build_governor_prompt(_quadro(hits), Bounds(0.25, 4.0), "cieco")
    assert "40 celle-passo" in testo
    assert "water_per_occupant" not in testo
    assert "regola 1" in testo


@pytest.mark.parametrize("livello", ["completo", "senza_aiuti", "nomi_veri", "cieco"])
def test_il_consuntivo_compare_a_ogni_gradino(livello):
    hits = (("se power_coverage < 1 -> build x3", 7), (ELSE_KEY, 1))
    assert "7 celle-passo" in build_governor_prompt(_quadro(hits), Bounds(0.25, 4.0), livello)


def test_il_digest_distingue_due_consuntivi_diversi():
    """Entra nel prompt, quindi entra nel quadro: un replay deve vederlo."""
    a = picture_digest(_quadro((("r", 1), (ELSE_KEY, 0))))
    b = picture_digest(_quadro((("r", 2), (ELSE_KEY, 0))))
    assert a != b
    assert a == picture_digest(_quadro((("r", 1), (ELSE_KEY, 0))))
