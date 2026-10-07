# -*- coding: utf-8 -*-
"""Lo strato psicosociale ha un punto fisso, e le sue costanti sono ricavate.

Fino al 2026-08-31 le quattro variabili dello strato erano integratori puri:
`x += spinte * passo`, senza alcun termine proporzionale a `x`. Una variabile
che integra una costante non ha punto fisso, e i `clip` a [0,1] non sono un
equilibrio ma un muro: misurato, fra l'87% e il 100% dei coloni stava
esattamente su 0 o su 1.

Queste prove sorvegliano le tre cose che rendono la correzione una correzione e
non una taratura: che le linee di base siano quelle che il modello dichiara
altrove, che i tassi risolvano l'equazione che li ricava, e che il punto fisso
sia davvero dove l'algebra dice.
"""
from __future__ import annotations

import numpy as np

from src.agents import vitals as V
from src.agents.base_agent import BaseAgent
from src.world.cell import Cell
from src.world.grid import GridWorld
from src.world.terrain import TerrainType

PASSO = V.GIORNI_PER_PASSO_RIFERIMENTO


def _mondo(lato: int = 5) -> GridWorld:
    celle = [[Cell(x=x, y=y, terrain=TerrainType.REGOLITH_PLAIN) for x in range(lato)]
             for y in range(lato)]
    return GridWorld(width=lato, height=lato, cells=celle)


def test_le_linee_di_base_sono_quelle_dichiarate_da_base_agent():
    """Un solo valore per la condizione normale, non due che possono divergere.

    Le linee di base non sono numeri nuovi: sono i default con cui nasce un
    colono. Se qualcuno li cambia in `BaseAgent` senza cambiarli qui, lo strato
    richiamerebbe ogni colono verso un valore che il modello non dichiara piu'.
    """
    a = BaseAgent(agent_id="a", name="a", role="colonist", x=0, y=0)
    assert a.stress_index == V.LINEA_BASE_STRESS
    assert a.morale == V.LINEA_BASE_MORALE
    assert a.cooperation == V.LINEA_BASE_COOPERAZIONE
    assert a.protocol_compliance == V.LINEA_BASE_ADERENZA


def test_ogni_tasso_porta_la_spinta_massima_esattamente_al_tetto():
    """L'equazione che ricava i tassi, riscritta al contrario.

    Il tasso non e' scelto: risolve `spinta_massima / tasso = spazio fino al
    tetto`. Quindi il punto fisso sotto la spinta massima deve valere
    esattamente 1,0 -- ne meno (la scala resterebbe inutilizzata e la soglia di
    danno a 0,82 diventerebbe codice morto) ne piu' (si tornerebbe al muro).
    """
    for spinta, linea, tasso in (
        (V._SPINTA_STRESS_MASSIMA, V.LINEA_BASE_STRESS, V.TASSO_RILASSAMENTO_STRESS),
        (V._SPINTA_MORALE_MASSIMA, V.LINEA_BASE_MORALE, V.TASSO_RILASSAMENTO_MORALE),
        (V._SPINTA_COOPERAZIONE_MASSIMA, V.LINEA_BASE_COOPERAZIONE,
         V.TASSO_RILASSAMENTO_COOPERAZIONE),
        (V._SPINTA_ADERENZA_MASSIMA, V.LINEA_BASE_ADERENZA, V.TASSO_RILASSAMENTO_ADERENZA),
    ):
        assert np.isclose(linea + spinta / tasso, 1.0, atol=1e-12)


def test_la_soglia_di_danno_da_stress_resta_raggiungibile():
    """Il ramo che degrada la salute sopra 0,82 non deve diventare irraggiungibile.

    E' il rischio proprio di questa correzione: un richiamo troppo forte
    schiaccia lo stress vicino alla linea di base e spegne un ramo del modello
    senza cancellarne una riga. La soglia deve stare dentro la scala che la
    spinta massima raggiunge, con margine.
    """
    massimo_raggiungibile = V.LINEA_BASE_STRESS + V._SPINTA_STRESS_MASSIMA / V.TASSO_RILASSAMENTO_STRESS
    assert massimo_raggiungibile > 0.82
    # e la quota di spinta che serve per arrivarci e' meno del massimo, cioe'
    # non serve la condizione peggiore in assoluto per farlo scattare
    spinta_alla_soglia = (0.82 - V.LINEA_BASE_STRESS) * V.TASSO_RILASSAMENTO_STRESS
    assert spinta_alla_soglia < V._SPINTA_STRESS_MASSIMA


def test_a_condizioni_ferme_lo_stress_converge_al_punto_fisso_previsto():
    """Il punto fisso e' `linea_di_base + spinte / tasso`, e si misura.

    Un colono solo, in una cella senza strutture e senza polvere, con i vitali
    pieni: le spinte sono note a mano, quindi l'equilibrio e' calcolabile prima
    di far girare il modello. Duecento passi bastano: il tempo caratteristico
    dello stress e' 3,5 passi.
    """
    mondo = _mondo()
    a = BaseAgent(agent_id="a", name="a", role="colonist", x=2, y=2)
    cella = mondo.get_cell(2, 2)
    cella.radiation_level = 0.0
    cella.dust_level = 0.0
    mondo.place_agent(a.agent_id, a.x, a.y)
    config = {"social": {"earth_mars_delay_minutes": 12.0}}

    for _ in range(200):
        # solo lo strato psicosociale: i vitali muoverebbero la privazione e
        # con essa le spinte, e l'equilibrio non sarebbe piu' calcolabile
        a.oxygen_level = a.hydration = a.satiety = 1.0
        a.fatigue = 0.0
        V.tick_agent_psychosocial(a, cella, mondo, PASSO, config=config)

    ritardo = min(0.055, 12.0 / 22.0 * 0.035) * (1.0 - a.autonomy_preference * 0.45)
    spinte = 0.045 + ritardo  # isolamento (solo) + ritardo; niente privazione,
    #                           niente ambiente, nessun sostegno
    atteso = V.LINEA_BASE_STRESS + spinte / V.TASSO_RILASSAMENTO_STRESS
    assert np.isclose(a.stress_index, atteso, atol=1e-6)


def test_lo_stress_imposto_torna_verso_la_linea_di_base():
    """Il richiamo agisce, e agisce nella direzione giusta da entrambi i lati.

    E' la proprieta' che l'integratore non aveva: da qualunque valore si parta,
    a condizioni ferme si finisce nello stesso punto. Due partenze opposte, uno
    stesso arrivo.
    """
    config = {"social": {"earth_mars_delay_minutes": 12.0}}
    arrivi = []
    for partenza in (0.0, 1.0):
        mondo = _mondo()
        a = BaseAgent(agent_id="a", name="a", role="colonist", x=2, y=2)
        a.stress_index = partenza
        cella = mondo.get_cell(2, 2)
        cella.radiation_level = 0.0
        cella.dust_level = 0.0
        mondo.place_agent(a.agent_id, a.x, a.y)
        for _ in range(200):
            a.oxygen_level = a.hydration = a.satiety = 1.0
            a.fatigue = 0.0
            V.tick_agent_psychosocial(a, cella, mondo, PASSO, config=config)
        arrivi.append(a.stress_index)
    assert np.isclose(arrivi[0], arrivi[1], atol=1e-6)
    assert 0.0 < arrivi[0] < 1.0


def test_l_equilibrio_non_dipende_dalla_durata_del_passo():
    """Due run con passo diverso devono descrivere lo stesso mondo.

    E' la ragione per cui il ritardo si integra in forma esatta invece che con
    lo schema esplicito. Con quello, sopra `tasso x passo = 1` la variabile
    veniva prima riportata di colpo sulla linea di base e poi spinta di un
    intero passo di pressione, finendo contro il muro: misurato, una prova a
    365 giorni per passo portava lo stress a 1,000 e uccideva l'equipaggio col
    ramo di danno, mentre l'equilibrio corretto per le stesse spinte vale 0,75.
    """
    spinta = 0.06
    equilibrio = V.LINEA_BASE_STRESS + spinta / V.TASSO_RILASSAMENTO_STRESS
    for passo in (0.5, 1.0, 7.0, 52.0, 500.0):
        valore = 0.0
        for _ in range(2000):
            valore += V.verso_l_equilibrio(
                valore, V.LINEA_BASE_STRESS, spinta, V.TASSO_RILASSAMENTO_STRESS, passo
            )
        assert np.isclose(valore, equilibrio, atol=1e-9), (passo, valore, equilibrio)


def test_l_avvicinamento_non_scavalca_mai_l_equilibrio():
    """Da qualunque parte si arrivi, ci si ferma sull'equilibrio.

    E' la proprieta' che rende lo schema stabile senza bisogno di alcun freno:
    il fattore `1 - exp(-tasso x passo)` sta in [0, 1) per ogni passo positivo,
    quindi il valore si avvicina all'equilibrio senza mai superarlo, per quanto
    lungo sia il passo.
    """
    spinta = 0.06
    equilibrio = V.LINEA_BASE_STRESS + spinta / V.TASSO_RILASSAMENTO_STRESS
    for passo in (1.0, 52.0, 10_000.0):
        for partenza in (0.0, 1.0):
            arrivo = partenza + V.verso_l_equilibrio(
                partenza, V.LINEA_BASE_STRESS, spinta, V.TASSO_RILASSAMENTO_STRESS, passo
            )
            assert min(partenza, equilibrio) <= arrivo <= max(partenza, equilibrio)
