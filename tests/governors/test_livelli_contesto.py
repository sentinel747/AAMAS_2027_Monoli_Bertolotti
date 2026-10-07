# -*- coding: utf-8 -*-
"""I quattro gradini di contesto del prompt del governatore.

Il modo in cui questo braccio puo' rompersi in silenzio e' preciso: il gradino
cieco risponde in etichette neutre, il kernel conosce solo i nomi veri, e senza
la traduzione inversa *ogni* regola verrebbe scartata come "indicatore
sconosciuto". La run finirebbe identica alla baseline con un registro pieno di
proposte, e il braccio misurerebbe la traduzione mancante invece della cecita'.
"""

import pytest

from src.governors.context import (
    ALIAS_INDICATORI,
    ALIAS_PILASTRI,
    LIVELLI,
    maschera_dizionario,
    normalizza,
    traduci_proposta,
)
from src.governors.llm_arm import build_governor_prompt
from src.governors.observation import ColonyPicture
from src.governors.policy import INDICATORS, PILLAR_BY_NAME, Bounds, parse_policy


@pytest.fixture
def quadro() -> ColonyPicture:
    return ColonyPicture(
        step=180,
        population=59,
        n_cells=4,
        metrics={"colony_prosperity_index": 0.71, "food_stock": 212.0},
        indicators={
            nome: {"mean": 1.0, "std": 0.5, "min": 0.1, "max": 2.0}
            for nome in INDICATORS
        },
        population_stats={"health": 0.91, "hydration": 0.77},
        structures={"greenhouse": 12, "shelter": 48},
    )


@pytest.fixture
def bounds() -> Bounds:
    return Bounds(weight_min=0.25, weight_max=4.0)


#: Parole che dichiarano il dominio. Nel prompt cieco non ne deve restare una.
PAROLE_DI_DOMINIO = (
    "marzian", "colon", "cella", "celle", "serr", "ghiaccio", "acqua",
    "ossigeno", "minerali", "sopravviv",
)


def test_il_prompt_cieco_non_nomina_il_dominio(quadro, bounds):
    testo = build_governor_prompt(quadro, bounds, "cieco").lower()
    trovate = [p for p in PAROLE_DI_DOMINIO if p in testo]
    assert not trovate, f"il gradino cieco lascia passare {trovate}"
    for nome in INDICATORS:
        assert nome not in testo
    for nome in PILLAR_BY_NAME:
        assert f" {nome}" not in testo


def test_il_prompt_cieco_conserva_i_numeri(quadro, bounds):
    """Cieco sui nomi, non sui numeri: togliere dati confonderebbe i due assi."""
    testo = build_governor_prompt(quadro, bounds, "cieco")
    for alias in ALIAS_INDICATORI.values():
        assert alias in testo
    for alias in ALIAS_PILASTRI.values():
        assert alias in testo
    # tanti aggregati quanti ne aveva il quadro, solo rinominati
    assert testo.count("metrica_") == len(quadro.metrics)
    assert testo.count("totale_") == len(quadro.structures)


def test_il_dominio_resta_ai_due_gradini_informati(quadro, bounds):
    for livello in ("completo", "senza_aiuti"):
        testo = build_governor_prompt(quadro, bounds, livello)
        assert "governatore della colonia marziana" in testo
        assert "sustenance" in testo


def test_gli_aiuti_cadono_sopra_il_gradino_completo(quadro, bounds):
    completo = build_governor_prompt(quadro, bounds, "completo")
    senza = build_governor_prompt(quadro, bounds, "senza_aiuti")
    assert "politica di riferimento" in completo
    assert "politica di riferimento" not in senza
    assert len(senza) < len(completo)


def test_la_meccanica_resta_a_ogni_gradino(quadro, bounds):
    """Che i pesi siano relativi non e' contesto: e' la regola del gioco."""
    for livello in LIVELLI:
        testo = build_governor_prompt(quadro, bounds, livello)
        assert "RELATIVI" in testo
        assert "PRIMA regola" in testo
        assert '"rationale"' in testo


def test_la_scala_e_monotona(quadro, bounds):
    lunghezze = [len(build_governor_prompt(quadro, bounds, l)) for l in LIVELLI]
    assert lunghezze == sorted(lunghezze, reverse=True), lunghezze


def test_la_risposta_cieca_torna_ai_nomi_veri(bounds):
    grezza = {
        "policy": [
            {
                "if": {"indicator": ALIAS_INDICATORI["water_per_occupant"], "op": "<", "value": 0.5},
                "weights": {ALIAS_PILASTRI["sustenance"]: 3.0, ALIAS_PILASTRI["build"]: 1.4},
            },
            {"if": None, "weights": {ALIAS_PILASTRI["explore"]: 1.8}},
        ],
        "rationale": "voce_5 ha minimo basso",
    }
    policy, drops = parse_policy(traduci_proposta(grezza, "cieco"), bounds)
    assert policy is not None
    assert len(policy.rules) == 2
    assert drops.unknown_indicators == 0
    assert drops.unknown_pillars == 0
    assert policy.rules[0].condition.indicator == "water_per_occupant"


def test_senza_traduzione_la_risposta_cieca_sarebbe_tutta_scartata(bounds):
    """La ragione per cui la traduzione esiste, resa esplicita."""
    grezza = {
        "policy": [
            {
                "if": {"indicator": ALIAS_INDICATORI["water_per_occupant"], "op": "<", "value": 0.5},
                "weights": {ALIAS_PILASTRI["sustenance"]: 3.0},
            }
        ],
        "rationale": "",
    }
    _, drops = parse_policy(grezza, bounds)
    assert drops.unknown_indicators >= 1


def test_gli_altri_gradini_non_traducono(bounds):
    grezza = {"policy": [{"if": None, "weights": {"explore": 1.8}}], "rationale": ""}
    for livello in ("completo", "senza_aiuti", "nomi_veri"):
        assert traduci_proposta(grezza, livello) is grezza


def test_le_tabelle_di_alias_coprono_il_vocabolario():
    assert set(ALIAS_INDICATORI) == set(INDICATORS)
    assert set(ALIAS_PILASTRI) == set(PILLAR_BY_NAME)
    assert len(set(ALIAS_INDICATORI.values())) == len(ALIAS_INDICATORI)
    assert len(set(ALIAS_PILASTRI.values())) == len(ALIAS_PILASTRI)


def test_il_mascheramento_e_stabile_fra_chiamate():
    valori = {"beta": 2.0, "alfa": 1.0, "gamma": 3.0}
    primo = maschera_dizionario(valori, "m", "cieco")
    secondo = maschera_dizionario(dict(reversed(list(valori.items()))), "m", "cieco")
    assert primo == secondo == {"m_1": 1.0, "m_2": 2.0, "m_3": 3.0}


@pytest.mark.parametrize("grezzo, atteso", [
    ("cieco", "cieco"), ("CIECO", "cieco"), (" nomi_veri ", "nomi_veri"),
    (None, "completo"), ("", "completo"), ("inesistente", "completo"),
])
def test_un_livello_sconosciuto_vale_completo(grezzo, atteso):
    assert normalizza(grezzo) == atteso
