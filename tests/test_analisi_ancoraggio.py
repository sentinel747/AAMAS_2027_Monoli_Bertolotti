# -*- coding: utf-8 -*-
"""La misura dell'ancoraggio deve distinguere tre amministratori finti.

1. L'ANCORATO riscrive se e solo se riscriveva ieri, qualunque cosa vedano le
   sue celle: ancoraggio grezzo e depurato = +1, evidenza = 0.
2. Il RAZIONALE riscrive se e solo se ha visto morti nella finestra: dentro
   ogni strato di evidenza la decisione di ieri non conta (depurato = 0),
   mentre il grezzo puo' essere positivo per il solo fatto che i morti si
   ripetono nelle stesse celle. E' la ragione per cui esiste lo strato.
3. Il governatore che ripete la stessa catena ha regole riprese = 1 e vita
   della regola = numero di tick.
"""

from scripts.analisi_ancoraggio import (
    ancoraggio,
    controllo_lag2,
    coppie_consecutive,
    persistenza_governatore,
    persistenza_regole_amministratore,
    terne_consecutive,
)

CELLE_A = [[1, 1], [1, 2]]
CELLE_B = [[5, 5]]


def _decisione(distretto, celle, riscrive, regole=None):
    return {
        "district": distretto,
        "cells": celle,
        "accepted_government_policy": not riscrive,
        "rewrite_without_effect": False,
        "rules": regole if riscrive else [],
    }


def _tornate(sequenze):
    """sequenze: {distretto: [(riscrive, regole), ...]} una voce per tornata."""
    n = len(next(iter(sequenze.values())))
    tornate = []
    for i in range(n):
        tornate.append({
            "step": 25 * (i + 1),
            "last_round": [
                _decisione(d, CELLE_A if d == 0 else CELLE_B, seq[i][0], seq[i][1])
                for d, seq in sequenze.items()
            ],
        })
    return tornate


def _morti(passi, cella):
    return [{"death_step": p, "y": cella[0], "x": cella[1]} for p in passi]


def test_the_anchored_administrator_scores_plus_one_regardless_of_evidence():
    # Distretto 0 riscrive sempre, distretto 1 accetta sempre; i morti cadono a caso.
    tornate = _tornate({0: [(True, ["r"])] * 6, 1: [(False, None)] * 6})
    morti = _morti([30, 80, 130], CELLE_A[0]) + _morti([55, 105], CELLE_B[0])
    a = ancoraggio(coppie_consecutive(tornate, morti))
    assert a["p_riscrive_dato_riscriveva"] == 1.0 and a["p_riscrive_dato_accettava"] == 0.0
    assert a["ancoraggio_grezzo"] == 1.0
    assert a["ancoraggio_depurato"] == 1.0


def test_the_evidence_driven_administrator_has_zero_anchoring_within_strata():
    # Riscrive esattamente quando ha visto morti nella finestra precedente,
    # e i morti alternano fra i due distretti: ieri non predice oggi a parita' di evidenza.
    passi = [25, 50, 75, 100, 125, 150, 175, 200]
    morti_0 = {1, 2, 5, 6}   # tornate in cui il distretto 0 vede morti nella finestra
    morti_1 = {3, 4, 7}
    morti = []
    seq0, seq1 = [], []
    for i, p in enumerate(passi):
        if i in morti_0:
            morti += _morti([p - 10], CELLE_A[0])
        if i in morti_1:
            morti += _morti([p - 10], CELLE_B[0])
        seq0.append((i in morti_0, ["s"]))
        seq1.append((i in morti_1, ["s"]))
    tornate = _tornate({0: seq0, 1: seq1})
    a = ancoraggio(coppie_consecutive(tornate, morti))
    assert a["effetto_evidenza"] == 1.0
    # In ogni strato compaiono entrambe le decisioni di ieri, e ieri non conta.
    assert a["strati"]["morti=0"]["ancoraggio"] == 0.0
    assert a["strati"]["morti>0"]["ancoraggio"] == 0.0
    assert a["ancoraggio_depurato"] == 0.0
    # Il grezzo, da solo, qui vale zero per costruzione; in generale segue
    # l'autocorrelazione dei morti, ed e' per questo che si stratifica.
    assert a["ancoraggio_grezzo"] == 0.0


def test_rule_persistence_for_administrators_and_governor():
    tornate = _tornate({0: [(True, ["a", "b"]), (True, ["a", "b"]), (True, ["a", "c"]), (False, None)]})
    coppie = coppie_consecutive(tornate, [])
    p = persistenza_regole_amministratore(coppie)
    assert p["coppie_riscrive_riscrive"] == 2
    assert p["quota_regole_riprese"] == 0.75          # (1.0 + 0.5) / 2
    assert p["quota_catene_identiche"] == 0.5

    regola = {"if": {"indicator": "food_per_occupant", "op": "<", "value": 1.5}, "weights": {"sustenance": 3.0}}
    altra = {"if": None, "weights": {"build": 2.0}}
    decisioni = [
        {"tick": 0, "policy": {"rules": [regola]}},
        {"tick": 1, "policy": {"rules": [regola]}},
        {"tick": 2, "policy": {"rules": [regola, altra]}},
        {"tick": 3, "policy": {"rules": [altra]}},
    ]
    g = persistenza_governatore(decisioni)
    assert g["tick"] == 4
    assert g["quota_regole_riprese"] == round((1.0 + 0.5 + 1.0) / 3, 4)
    assert g["quota_catene_identiche"] == round(1 / 3, 4)
    assert g["vita_max_regola_tick"] == 3            # `regola` vive ai tick 0, 1, 2
    assert g["vita_media_regola_tick"] == 2.5          # (3 + 2) / 2


def test_a_district_that_appears_only_now_has_no_pair():
    tornate = _tornate({0: [(True, ["r"]), (True, ["r"])]})
    tornate[1]["last_round"].append(_decisione(7, CELLE_B, True, ["z"]))
    coppie = coppie_consecutive(tornate, [])
    assert [c["distretto"] for c in coppie] == [0]


def test_lag2_control_separates_prompt_memory_from_district_state():
    # Ancorato al PROMPT: riscrive oggi se e solo se riscriveva ieri. Dato che
    # ieri ha accettato, oggi accetta qualunque cosa avesse fatto l'altro ieri:
    # dipendenza dal lag 2 = 0.
    seq = [True, True, True, False, False, False, False]   # una volta accettato, accetta per sempre
    tornate = _tornate({0: [(s, ["r"]) for s in seq]})
    l2 = controllo_lag2(terne_consecutive(tornate))
    assert l2["terne"] > 0
    assert l2["p_riscrive_dato_lag2_riscrive"] == 0.0 and l2["p_riscrive_dato_lag2_accetta"] == 0.0
    assert l2["dipendenza_lag2"] == 0.0
    # Stato CRONICO: due distretti, uno riscrive quasi sempre (ma ha accettato
    # una volta), l'altro accetta quasi sempre (ma ha riscritto una volta).
    # Dato che ieri hanno accettato, chi riscriveva l'altro ieri riscrive oggi.
    cronico = [True, True, False, True, True, True]
    quieto = [False, False, False, False, False, True]
    tornate = _tornate({0: [(s, ["r"]) for s in cronico], 1: [(s, ["r"]) for s in quieto]})
    l2 = controllo_lag2(terne_consecutive(tornate))
    assert l2["dipendenza_lag2"] is not None and l2["dipendenza_lag2"] > 0
