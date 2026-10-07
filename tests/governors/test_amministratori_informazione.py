# -*- coding: utf-8 -*-
"""Che cosa un amministratore puo' sapere prima di decidere.

**Perche' questo file esiste.** Nella campagna del 2026-09-02 gli
amministratori LLM hanno accettato la politica del governo nel 95 per cento dei
casi sotto un governo, e nel 100 per cento senza governo. Le motivazioni erano
di merito --- non un si' automatico --- ma erano date su un quadro che non
conteneva nessuna delle grandezze su cui l'accettazione poteva sbagliare:

- nessun ESITO. Il quadro portava giacenze e medie, cioe' lo stato del
  magazzino. In questo mondo si muore di fame mentre il margine alimentare di
  colonia resta al tetto per l'intera run: un distretto puo' avere indicatori
  tranquilli e coloni che muoiono, e l'amministratore non poteva vederlo.
- solo MEDIE dei coloni, mentre gli indicatori di cella arrivavano gia' con
  min e max. La media non uccide nessuno: uccide la coda.
- nessuna MEMORIA. Il governatore riceve il consuntivo delle proprie regole;
  l'amministratore ripartiva da zero a ogni tornata e non poteva accorgersi di
  aver accettato dieci volte mentre il distretto peggiorava.
- il NON-INTERVENTO travestito da politica. Le catene del governo finiscono
  quasi sempre con una regola incondizionata a pesi tutti 1 --- dal 18 al 24
  per cento delle celle-passo "governate" ricade li'. Stampata come le altre,
  quella riga sembra una decisione, e accettarla sembra avallare una politica.

Ognuno di questi buchi si richiude in silenzio: nessuno da' errore, e l'unico
sintomo e' un tasso di accettazione alto che si legge come consenso.
"""

import numpy as np
import pytest

from src.core.arrays import AgentArrays, CellArrays
from src.core.shell_common import morti_per_cella
from src.governors.administration import (
    _riga_morti,
    _riga_precedente,
    DecisioneAmministratore,
    costruisci_prompt,
    politica_risolta,
    quadro_di_distretto,
)
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Condition, Policy, Rule
from src.governors.testi_prompt import frasi
from src.agents import pillars

#: La riga dei morti si compone dalla tabella delle frasi. Qui serve quella
#: italiana: il test verifica l'assemblaggio, non la traduzione.
FRASI_IT = frasi("it")

BOUNDS = Bounds(0.25, 4.0)
CELLE = [(1, 1), (1, 2)]


# ------------------------------------------------ il non-intervento travestito


def _politica(pesi: dict) -> Policy:
    return Policy((Rule(None, pesi),), "prova")


def test_una_regola_a_pesi_tutti_uno_si_legge_come_nessun_intervento():
    righe = politica_risolta(
        _politica({pillars.P_SUSTENANCE: 1.0, pillars.P_BUILD: 1.0}), None, CELLE
    )
    assert all("NESSUN INTERVENTO" in r for r in righe)
    assert all("tutti i pesi valgono 1" in r for r in righe)


def test_una_regola_che_pesa_davvero_resta_una_politica():
    righe = politica_risolta(_politica({pillars.P_SUSTENANCE: 3.0}), None, CELLE)
    assert all("NESSUN INTERVENTO" not in r for r in righe)
    assert all("sustenance x3" in r for r in righe)


def test_nessuna_politica_del_governo_e_dichiarata_tale():
    righe = politica_risolta(None, None, CELLE)
    assert len(righe) == len(CELLE)


# ------------------------------------------------------------------ gli esiti


def test_i_morti_del_distretto_arrivano_nel_prompt_con_la_causa():
    quadro = ColonyPicture(step=7, population=40, metrics={}, indicators={},
                           population_stats={}, structures={}, n_cells=2)
    prompt = costruisci_prompt(
        quadro, ["cella [1, 1]: NESSUN INTERVENTO"], BOUNDS, 0,
        morti={"starvation": 6, "dehydration": 2},
    )
    assert "Morti nel tuo distretto" in prompt
    assert "starvation: 6" in prompt
    assert "dehydration: 2" in prompt
    # La quota sui propri coloni: otto morti su quaranta non e' lo stesso
    # numero di otto morti su mille, e il valore assoluto da solo non lo dice.
    assert "20%" in prompt


def test_un_distretto_senza_morti_lo_dice_invece_di_tacere():
    """"Nessun morto" e "non misurato" non devono finire nella stessa riga."""
    assert "nessuno" in _riga_morti({}, 10, FRASI_IT)
    assert "nessuno" in _riga_morti(None, 10, FRASI_IT)


def test_la_quota_non_esplode_su_un_distretto_svuotato():
    assert "starvation: 3" in _riga_morti({"starvation": 3}, 0, FRASI_IT)


# ----------------------------------------------------------------- la memoria


def test_la_decisione_precedente_torna_nel_prompt():
    accettata = DecisioneAmministratore(distretto=0, policy=None,
                                        rationale="andava bene cosi'")
    riga = _riga_precedente(accettata)
    assert "ACCETTATO" in riga and "andava bene" in riga

    riscritta = DecisioneAmministratore(
        distretto=0, policy=_politica({pillars.P_SUSTENANCE: 2.0}), accettata=False
    )
    assert "RISCRITTO" in _riga_precedente(riscritta)


def test_alla_prima_tornata_non_si_inventa_un_precedente():
    assert _riga_precedente(None) == ""


# ------------------------------------------------------- la coda dei vitali


def _mondo(satieta: list[float], stress: list[float]):
    cells = CellArrays(4, 4)
    for y, x in CELLE:
        cells.occupancy[y, x] = len(satieta) // 2 or 1
    agents = AgentArrays(len(satieta))
    for i, (s, st) in enumerate(zip(satieta, stress)):
        agents.alive[i] = True
        agents.y[i], agents.x[i] = CELLE[i % len(CELLE)]
        agents.satiety[i] = s
        agents.stress[i] = st
    return cells, agents


def test_il_quadro_porta_la_coda_pericolosa_non_solo_la_media():
    """Un colono che sta morendo di fame in mezzo a nove sazi deve vedersi."""
    cells, agents = _mondo(satieta=[0.9] * 9 + [0.02], stress=[0.1] * 9 + [0.95])
    quadro = quadro_di_distretto(1, {}, cells, agents, np.arange(10), CELLE)
    stat = quadro.population_stats
    assert stat["satiety_mean"] > 0.7
    assert stat["satiety_min"] == pytest.approx(0.02)
    # Per fatica e stress la coda pericolosa e' quella ALTA: riportare il min
    # darebbe il colono piu' tranquillo del distretto proprio dove serve il
    # piu' provato.
    assert stat["stress_max"] == pytest.approx(0.95)
    assert "stress_min" not in stat


# ---------------------------------------------- il conteggio comune alle shell


def test_si_contano_solo_i_morti_nuovi():
    morti = [
        {"y": 1, "x": 1, "cause": "starvation"},
        {"y": 1, "x": 1, "cause": "starvation"},
        {"y": 2, "x": 3, "cause": "dehydration"},
    ]
    prima, cursore = morti_per_cella(morti, 0)
    assert prima == {(1, 1): {"starvation": 2}, (2, 3): {"dehydration": 1}}
    assert cursore == 3

    # Nessun morto nuovo: il cumulato racconterebbe la stessa storia due volte.
    dopo, cursore = morti_per_cella(morti, cursore)
    assert dopo == {}
    assert cursore == 3

    morti.append({"y": 1, "x": 1, "cause": "exposure"})
    nuovi, _ = morti_per_cella(morti, cursore)
    assert nuovi == {(1, 1): {"exposure": 1}}


def test_un_morto_senza_posizione_non_ferma_il_conteggio():
    morti = [{"cause": "starvation"}, {"y": 0, "x": 0, "cause": "starvation"}]
    conteggio, cursore = morti_per_cella(morti, 0)
    assert conteggio == {(0, 0): {"starvation": 1}}
    assert cursore == 2


# ------------------------------------- l'amministratore scritto sa dire di no


def _prompt_scritto(distretto: dict, colonia: dict) -> str:
    quadro = ColonyPicture(step=1, population=9, metrics={}, indicators=distretto,
                           population_stats={}, structures={}, n_cells=3)
    return costruisci_prompt(quadro, ["cella [1, 1]: NESSUN INTERVENTO"], BOUNDS, 0,
                             indicatori_colonia=colonia)


def _scritto(distretto: dict, colonia: dict) -> dict:
    from src.governors.admin_arms import AmministratoreScritto

    return AmministratoreScritto(BOUNDS).propose_text(_prompt_scritto(distretto, colonia))["raw"]


def test_sotto_la_media_ma_dentro_una_sigma_non_e_un_caso_anomalo():
    """Il caso che il criterio a percentuale sbagliava.

    Un distretto al 30% sotto la media, quando la colonia varia molto, e' un
    distretto normale. Il criterio vecchio --- massimo su otto indicatori dello
    scarto relativo oltre il 25% --- lo dichiarava anomalo, ed era vero quasi
    ovunque per costruzione: 98,7% di interventi sui dati veri.
    """
    esito = _scritto(
        distretto={"food_per_occupant": {"mean": 0.7, "max": 1.0}},
        colonia={"food_per_occupant": {"mean": 1.0, "std": 0.8}},
    )
    assert esito["accept"] is True
    assert "sigma" in esito["rationale"]


def test_oltre_una_sigma_sotto_la_media_fa_intervenire():
    esito = _scritto(
        distretto={"food_per_occupant": {"mean": 0.7, "max": 1.0}},
        colonia={"food_per_occupant": {"mean": 1.0, "std": 0.2}},
    )
    assert esito["accept"] is False
    assert esito["policy"][0]["if"]["indicator"] == "food_per_occupant"
    assert "sustenance" in esito["policy"][0]["weights"]


def test_la_soglia_dello_scritto_puo_scattare_anche_in_un_distretto_di_una_cella():
    """Il difetto A11: soglia = massimo del distretto con `<` stretto.

    In un distretto di una cella sola massimo e media coincidono, quindi
    `valore < massimo` e' falso per costruzione: la riscrittura sostituiva la
    catena del governo con una regola impossibile, e la cella restava senza
    alcuna politica. Misurato sulla campagna del 02/09: dal 19 al 33 per cento
    delle riscritture dello scritto non scattava mai. La soglia e' la media di
    colonia: la cella che sta sotto la media nazionale e' esattamente quella
    che ha reso il distretto anomalo, e la regola scatta li'.
    """
    esito = _scritto(
        distretto={"food_per_occupant": {"mean": 0.7, "max": 0.7, "min": 0.7}},
        colonia={"food_per_occupant": {"mean": 1.0, "std": 0.2}},
    )
    assert esito["accept"] is False
    soglia = esito["policy"][0]["if"]["value"]
    assert soglia > 0.7, "una soglia uguale al valore della cella non scatta mai"
    assert soglia == pytest.approx(1.0)


def test_un_indicatore_uguale_ovunque_non_rende_nessuno_anomalo():
    """Senza code non c'e' coda: dividere per zero renderebbe anomalo tutto."""
    esito = _scritto(
        distretto={"food_per_occupant": {"mean": 0.99, "max": 1.0}},
        colonia={"food_per_occupant": {"mean": 1.0, "std": 0.0}},
    )
    assert esito["accept"] is True


def test_un_distretto_sopra_la_media_non_interviene_mai():
    esito = _scritto(
        distretto={"food_per_occupant": {"mean": 2.0, "max": 3.0}},
        colonia={"food_per_occupant": {"mean": 1.0, "std": 0.2}},
    )
    assert esito["accept"] is True


# --------------------------- un registro appartiene a UNA run


def test_il_registro_del_governatore_non_accoda_due_run(tmp_path):
    """La ripresa di una campagna riesegue un seme nella stessa cartella.

    Aperto in aggiunta, il registro finiva per contenere le righe di due run
    sovrapposte con lo stesso numero di tick: misurato il 2026-09-03, settantotto
    tornate in un file che ne poteva contenere quaranta. Un registro cosi' non e'
    rieseguibile e non e' leggibile, e nulla lo segnala.
    """
    from src.governors.record import GovernorRecorder

    percorso = tmp_path / "governor_decisions.jsonl"
    percorso.write_text('{"tick": 0, "vecchio": true}\n', encoding="utf-8")
    GovernorRecorder(percorso)
    assert percorso.read_text(encoding="utf-8") == ""
