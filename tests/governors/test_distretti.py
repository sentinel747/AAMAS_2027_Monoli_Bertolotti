# -*- coding: utf-8 -*-
"""I distretti e lo strato amministrativo.

Il modo in cui questo strato puo' rompersi in silenzio e' doppio, e i due modi
si annullano a vicenda nell'osservazione: se le maschere si sovrappongono, i
pilastri vengono moltiplicati due volte e l'amministratore *rincara* il governo
invece di correggerlo; se non copre nessuna cella, il decentramento non esiste
e la run e' indistinguibile da quella centralizzata. Nessuno dei due da' un
errore.
"""

import numpy as np
import pytest

from src.governors.apply import ELSE_KEY, apply_layered, apply_policy
from src.governors.districts import SENZA_DISTRETTO, Distretti
from src.governors.policy import Bounds, Condition, Policy, Rule


# --------------------------------------------------------------- i distretti


def test_la_cella_madre_non_entra_in_nessun_distretto():
    d = Distretti(celle_per_distretto=3)
    d.imposta_madre((5, 5))
    occupate = np.zeros((10, 10), dtype=bool)
    occupate[5, 5] = occupate[5, 6] = occupate[6, 5] = True
    d.aggiorna(occupate)
    assegnate = {c for celle in d.mappa().values() for c in celle}
    assert (5, 5) not in assegnate
    assert assegnate == {(5, 6), (6, 5)}


def test_un_amministratore_ogni_tre_celle():
    d = Distretti(celle_per_distretto=3)
    d.imposta_madre((0, 0))
    occupate = np.zeros((10, 10), dtype=bool)
    for k in range(7):
        occupate[1, k] = True
    d.aggiorna(occupate)
    mappa = d.mappa()
    assert d.numero_distretti() == 3          # 3 + 3 + 1
    assert sorted(len(c) for c in mappa.values()) == [1, 3, 3]


def test_una_cella_orfana_fa_nascere_un_amministratore():
    d = Distretti(celle_per_distretto=3)
    d.imposta_madre((0, 0))
    occupate = np.zeros((5, 5), dtype=bool)
    occupate[1, 1] = True
    assert d.aggiorna(occupate) == [0]
    occupate[2, 2] = True
    assert d.aggiorna(occupate) == []          # entra nel distretto esistente
    assert len(d.celle_di(0)) == 2


def test_l_assegnazione_non_si_rimescola():
    """Un distretto che cambiasse territorio renderebbe illeggibile la serie."""
    d = Distretti(celle_per_distretto=2)
    d.imposta_madre((0, 0))
    occupate = np.zeros((5, 5), dtype=bool)
    occupate[1, 1] = occupate[1, 2] = True
    d.aggiorna(occupate)
    primo = d.mappa()
    for _ in range(5):
        occupate[3, 3] = True
        d.aggiorna(occupate)
    assert d.celle_di(0) == primo[0]


def test_l_ordine_e_deterministico():
    def costruisci() -> dict:
        d = Distretti(celle_per_distretto=2)
        d.imposta_madre((0, 0))
        occupate = np.zeros((6, 6), dtype=bool)
        for y, x in ((4, 1), (1, 4), (2, 2), (3, 3)):
            occupate[y, x] = True
        d.aggiorna(occupate)
        return d.mappa()

    assert costruisci() == costruisci()


def test_la_griglia_marca_meno_uno_dove_non_c_e_distretto():
    d = Distretti(celle_per_distretto=3)
    d.imposta_madre((0, 0))
    occupate = np.zeros((4, 4), dtype=bool)
    occupate[0, 0] = occupate[1, 1] = True
    d.aggiorna(occupate)
    ids = d.griglia((4, 4))
    assert ids[0, 0] == SENZA_DISTRETTO       # la madre
    assert ids[3, 3] == SENZA_DISTRETTO       # cella vuota
    assert ids[1, 1] == 0


def test_la_madre_dichiarata_tardi_libera_il_posto():
    d = Distretti(celle_per_distretto=3)
    occupate = np.zeros((4, 4), dtype=bool)
    occupate[2, 2] = True
    d.aggiorna(occupate)
    assert d.celle_di(0) == [(2, 2)]
    d.imposta_madre((2, 2))
    assert d.celle_di(0) == []


# ----------------------------------------------------- l'applicazione a strati


class _CelleFinte:
    def __init__(self, occupancy):
        self.occupancy = occupancy


def _policy(peso: float) -> Policy:
    from src.agents import pillars

    return Policy((Rule(None, {pillars.P_BUILD: peso}),), "prova")


def _indici_build() -> tuple[int, ...]:
    from src.agents import pillars

    return tuple(pillars.ACTION_INDEX[a] for a in pillars.PILLAR_ACTIONS[pillars.P_BUILD])


@pytest.fixture
def scena():
    occupancy = np.array([[3, 3], [3, 0]], dtype=np.int32)
    ids = np.array([[SENZA_DISTRETTO, 0], [1, SENZA_DISTRETTO]], dtype=np.int16)
    priority = np.ones((2, 2, 30), dtype=np.float64)
    return _CelleFinte(occupancy), ids, priority


def test_ogni_cella_riceve_una_sola_catena(scena):
    """Sovrapporre governo e distretto moltiplicherebbe due volte gli stessi pilastri."""
    cells, ids, priority = scena
    apply_layered(priority, _policy(2.0), {0: _policy(3.0), 1: _policy(5.0)}, ids, cells)
    a = _indici_build()[0]
    assert priority[0, 0, a] == pytest.approx(2.0)   # madre -> governo
    assert priority[0, 1, a] == pytest.approx(3.0)   # distretto 0
    assert priority[1, 0, a] == pytest.approx(5.0)   # distretto 1
    assert priority[1, 1, a] == pytest.approx(1.0)   # vuota -> intatta


def test_un_distretto_che_si_astiene_ricade_sul_governo(scena):
    cells, ids, priority = scena
    apply_layered(priority, _policy(2.0), {0: _policy(3.0)}, ids, cells)
    a = _indici_build()[0]
    assert priority[0, 1, a] == pytest.approx(3.0)   # ha riscritto
    assert priority[1, 0, a] == pytest.approx(2.0)   # astenuto -> governo


def test_senza_distretti_e_identico_ad_apply_policy(scena):
    """La baseline deve restare bit-exact per costruzione, non per verifica."""
    cells, ids, priority = scena
    riferimento = np.ones_like(priority)
    apply_policy(riferimento, _policy(2.0), cells)
    apply_layered(priority, _policy(2.0), {}, ids, cells)
    assert np.array_equal(priority, riferimento)


def test_lo_scope_restringe_anche_il_conteggio(scena):
    cells, ids, priority = scena
    hits: dict = {}
    apply_policy(priority, Policy((), ""), cells, hits=hits, scope=ids == 0)
    assert hits[ELSE_KEY] == 1


def test_i_conteggi_distinguono_il_distretto_dal_governo(scena):
    cells, ids, priority = scena
    hits: dict = {}
    apply_layered(priority, _policy(2.0), {0: _policy(3.0)}, ids, cells, hits=hits)
    chiavi_distretto = [k for k in hits if k.startswith("[distretto 0]")]
    assert chiavi_distretto, hits
    assert sum(hits[k] for k in chiavi_distretto) == 1
