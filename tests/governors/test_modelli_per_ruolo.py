# -*- coding: utf-8 -*-
"""Governatore e amministratori possono parlare a modelli diversi.

**Perche'.** Fino al 2026-09-06 gli amministratori ereditavano il modello del
governatore: una scelta di disegno ("un decentramento che parlasse a un
modello migliore misurerebbe il modello, non il decentramento"), giusta per il
confronto centralizzato/decentrato. Ma la diversita' di stili di governo fra
modelli e' essa stessa una domanda di ricerca, e per porla serve poter dare a
ogni distretto un modello pescato da una lista --- in modo riproducibile a
parita' di seme, cosi' due run si confrontano ancora appaiate.

La chiave e' `governors.administrators.assignments`, una lista come quella
degli agenti. La vecchia `assignment` singola resta e vale come lista di uno.
"""

from src.governors.config import build_administration, read_administrator_settings, read_settings

ALFA = {"provider": "fallback", "model": "alfa"}
BETA = {"provider": "fallback", "model": "beta"}


def _config(**amministratori) -> dict:
    return {
        "governors": {
            "arm": "llm",
            "cadence_steps": 5,
            "administrators": {"enabled": True, "cells_per_district": 3, **amministratori},
        }
    }


def test_una_lista_di_modelli_per_gli_amministratori_e_letta():
    cfg = _config(assignments=[ALFA, BETA])
    s = read_administrator_settings(cfg, read_settings(cfg))
    assert s.assignments == [ALFA, BETA]
    # Una lista propria vuol dire braccio linguistico proprio, non ereditato.
    assert s.arm == "llm"
    assert s.follow_governor_arm is False


def test_senza_lista_l_assegnazione_singola_vale_come_lista_di_uno():
    cfg = _config(follow_governor_arm=False, arm="llm", assignment=ALFA)
    s = read_administrator_settings(cfg, read_settings(cfg))
    assert s.assignments == [ALFA]


def test_ogni_distretto_pesca_dalla_lista_e_la_pesca_e_riproducibile():
    cfg = _config(assignments=[ALFA, BETA])
    prima = build_administration(cfg, seed=3)
    modelli = [prima._proponente(d)._model for d in range(24)]
    assert set(modelli) == {"alfa", "beta"}, "con ventiquattro distretti entrambi i modelli devono comparire"
    seconda = build_administration(cfg, seed=3)
    assert [seconda._proponente(d)._model for d in range(24)] == modelli
    assert prima.riassunto()["models"] == ["fallback:alfa", "fallback:beta"]


def test_con_un_modello_solo_tutti_i_distretti_lo_usano():
    cfg = _config(assignments=[BETA])
    amm = build_administration(cfg, seed=3)
    assert {amm._proponente(d)._model for d in range(10)} == {"beta"}


def test_l_etichetta_del_braccio_dice_che_i_modelli_sono_misti():
    from scripts.run_governor_experiment import _etichetta

    assert _etichetta("llm", "completo", True, "", admin_misto=True) == "llm:completo+amm:misto"
    assert _etichetta("llm", "completo", True, "") == "llm:completo+amm"
