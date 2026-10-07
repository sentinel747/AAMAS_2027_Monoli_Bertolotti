# -*- coding: utf-8 -*-
"""Il governatore deve sapere dove stanno le persone, e chi muore.

**Il fatto (2026-09-05).** A fine run il 77-84 per cento dei coloni vive
nella cella madre e il 94-100 per cento dei morti cade nei distretti. Il
quadro del governatore aggregava pero' gli indicatori PER CELLA: su un
centinaio di celle occupate, la madre con milleduecento coloni contava quanto
un avamposto di tre, e il governatore scriveva leggi per la cella mediana, che
e' un avamposto. Inoltre non riceveva i morti: gli amministratori li avevano
dall'02/09, lui no.

Tre aggiunte, nessuna delle quali toglie nulla: gli stessi indicatori pesati
per colono accanto a quelli per cella; la cella piu' popolata con la sua quota
e i suoi valori; i decessi dall'ultimo tick per cella e causa. Sono le
grandezze con cui un governo tara una legge nazionale sul Nord che ha un
milione di abitanti senza dimenticare il Sud che ne ha tre.
"""

import numpy as np
import pytest

from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.governors.llm_arm import build_governor_prompt
from src.governors.observation import build_picture
from src.governors.policy import Bounds

BOUNDS = Bounds(0.25, 4.0)
MADRE = (2, 3)
AVAMPOSTO = (0, 1)


def _colonia(nella_madre: int = 100, nell_avamposto: int = 1):
    """La madre affamata (1 cibo a testa) e un avamposto sazio (10 a testa)."""
    cells = CellArrays(4, 5)
    cells.cell_res[MADRE[0], MADRE[1], C.R["food"]] = float(nella_madre)
    cells.cell_res[AVAMPOSTO[0], AVAMPOSTO[1], C.R["food"]] = 10.0 * nell_avamposto
    n = nella_madre + nell_avamposto
    agents = AgentArrays(n)
    for row in range(n):
        agents.alive[row] = True
        agents.y[row], agents.x[row] = MADRE if row < nella_madre else AVAMPOSTO
    agents.n = n
    return cells, agents, np.arange(n, dtype=np.int64)


def test_gli_indicatori_per_cella_restano_e_arrivano_quelli_per_colono():
    cells, agents, rows = _colonia()
    quadro = build_picture(1, {}, cells, agents, rows)
    # Per cella: (1 + 10) / 2. Per colono: (100 x 1 + 1 x 10) / 101.
    assert quadro.indicators["food_per_occupant"]["mean"] == pytest.approx(5.5)
    assert quadro.indicators_weighted["food_per_occupant"]["mean"] == pytest.approx(110 / 101, abs=1e-3)
    assert set(quadro.indicators_weighted) == set(quadro.indicators)


def test_la_deviazione_pesata_e_quasi_zero_quando_quasi_tutti_stanno_in_una_cella():
    cells, agents, rows = _colonia()
    quadro = build_picture(1, {}, cells, agents, rows)
    assert quadro.indicators["food_per_occupant"]["std"] == pytest.approx(4.5)
    assert quadro.indicators_weighted["food_per_occupant"]["std"] < 1.0


def test_la_cella_piu_popolata_e_dichiarata_con_la_sua_quota():
    cells, agents, rows = _colonia()
    quadro = build_picture(1, {}, cells, agents, rows)
    principale = quadro.principale
    assert principale["cell"] == list(MADRE)
    assert principale["occupants"] == 100
    assert principale["share"] == pytest.approx(100 / 101, abs=1e-3)
    assert principale["indicators"]["food_per_occupant"] == pytest.approx(1.0)


def test_una_colonia_vuota_non_ha_una_cella_principale():
    cells = CellArrays(4, 5)
    agents = AgentArrays(1)
    quadro = build_picture(1, {}, cells, agents, np.zeros(0, dtype=np.int64))
    assert quadro.principale == {}
    assert quadro.indicators_weighted == {}


def test_i_morti_dall_ultimo_tick_entrano_nel_quadro():
    cells, agents, rows = _colonia()
    morti = {AVAMPOSTO: {"starvation": 2}, MADRE: {"exposure": 1}}
    quadro = build_picture(1, {}, cells, agents, rows, morti_per_cella=morti)
    assert quadro.deaths == {AVAMPOSTO: {"starvation": 2}, MADRE: {"exposure": 1}}
    assert build_picture(1, {}, cells, agents, rows).deaths == {}


# ------------------------------------------------------------------ il prompt


def test_il_prompt_dice_dove_stanno_i_coloni_e_da_i_pesati():
    cells, agents, rows = _colonia()
    quadro = build_picture(1, {}, cells, agents, rows)
    prompt = build_governor_prompt(quadro, BOUNDS)
    assert "99%" in prompt, "la quota della cella piu' popolata va detta in chiaro"
    assert "pesati per colono" in prompt.lower()


def test_il_prompt_porta_i_morti_con_cella_e_causa():
    cells, agents, rows = _colonia()
    quadro = build_picture(1, {}, cells, agents, rows, morti_per_cella={AVAMPOSTO: {"starvation": 2}})
    prompt = build_governor_prompt(quadro, BOUNDS)
    assert "starvation: 2" in prompt
    assert "[0, 1]" in prompt


def test_senza_morti_il_prompt_lo_dice_invece_di_tacere():
    cells, agents, rows = _colonia()
    prompt = build_governor_prompt(build_picture(1, {}, cells, agents, rows), BOUNDS)
    assert "nessuno" in prompt.lower()


def test_al_gradino_cieco_le_cause_di_morte_sono_mascherate_ma_il_conto_resta():
    cells, agents, rows = _colonia()
    quadro = build_picture(1, {}, cells, agents, rows, morti_per_cella={AVAMPOSTO: {"starvation": 2}})
    prompt = build_governor_prompt(quadro, BOUNDS, livello="cieco")
    assert "starvation" not in prompt
    assert ": 2" in prompt


# ---------------------------------------------- la shell conta i morti una volta


def test_governatore_e_amministratori_ricevono_gli_stessi_morti(tmp_path, monkeypatch):
    """Il contatore dei morti avanza a ogni lettura: se ciascun livello lo
    leggesse da se', il secondo riceverebbe sempre un elenco vuoto."""
    import src.simulation.agent_coupled_runner as shell
    from src.governors.administration import Amministrazione
    from scripts.parity_harness import realistic_config

    al_governo: list = []
    agli_amministratori: list = []
    vero_quadro = shell.build_picture
    vero_advance = Amministrazione.advance

    def spia_quadro(*a, **k):
        if "morti_per_cella" in k:  # il quadro del governatore; quello di colonia non li porta
            al_governo.append(k["morti_per_cella"])
        return vero_quadro(*a, **k)

    def spia_advance(self, *a, **k):
        agli_amministratori.append(k.get("morti_per_cella"))
        return vero_advance(self, *a, **k)

    monkeypatch.setattr(shell, "build_picture", spia_quadro)
    monkeypatch.setattr(Amministrazione, "advance", spia_advance)
    config = realistic_config(60, 4, 0)
    config["governors"] = {"arm": "scripted", "cadence_steps": 2, "administrators": {"enabled": True}}
    shell.AgentCoupledRunner(config).run(days=4, output_dir=tmp_path / "run")

    # Un tick di cadenza = un quadro al governatore e una tornata agli
    # amministratori, con lo STESSO oggetto dei morti: contati una volta.
    assert len(al_governo) >= 2
    assert len(al_governo) == len(agli_amministratori)
    for governo, amministratori in zip(al_governo, agli_amministratori):
        assert governo is not None, "il governatore riceve i morti"
        assert governo is amministratori
