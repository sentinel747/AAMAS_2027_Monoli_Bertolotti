# -*- coding: utf-8 -*-
"""La tornata di distretto parte tutta insieme, e torna in ordine di distretto.

Sono due proprieta' distinte e nessuna delle due da' errore se si rompe.

La prima e' un costo: gli amministratori deliberano sulla stessa politica dello
stesso tick, quindi non c'e' nulla che uno debba sapere da un altro. In fila, un
tick costerebbe la latenza moltiplicata per il numero di distretti --- che
cresce con la colonia, cioe' proprio quando la run comincia a essere
interessante. Una regressione qui non si vede: la run resta corretta, diventa
soltanto impraticabile, e ce ne si accorge dopo ore di orologio.

La seconda e' la riproducibilita': se le decisioni venissero raccolte in ordine
di arrivo, due esecuzioni della stessa run scriverebbero registri diversi
perche' la rete ha risposto in ordine diverso.
"""

import asyncio
import time
from types import SimpleNamespace

import pytest

import src.governors.administration as administration
from src.governors.administration import Amministrazione
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds

RITARDO = 0.25
DISTRETTI = 5


class _DistrettiFinti:
    def __init__(self, quanti: int) -> None:
        self._mappa = {d: [(d, 0)] for d in range(quanti)}

    def mappa(self) -> dict:
        return dict(self._mappa)

    def numero_distretti(self) -> int:
        return len(self._mappa)


class _Lento:
    """Un amministratore che attende, e registra quando ha cominciato."""

    def __init__(self, distretto: int, istanti: list) -> None:
        self._distretto = distretto
        self._istanti = istanti

    async def propose_text_async(self, prompt: str) -> dict:
        self._istanti.append(time.perf_counter())
        await asyncio.sleep(RITARDO)
        return {"raw": {"accept": True, "rationale": f"distretto {self._distretto}"}}


class _LentoSincrono:
    """Solo `propose_text`: non deve rimettere in fila la tornata."""

    def __init__(self, distretto: int, istanti: list) -> None:
        self._distretto = distretto
        self._istanti = istanti

    def propose_text(self, prompt: str) -> dict:
        self._istanti.append(time.perf_counter())
        time.sleep(RITARDO)
        return {"raw": {"accept": True, "rationale": f"distretto {self._distretto}"}}


class _AlContrario:
    """Risponde tanto piu' tardi quanto piu' basso e' il numero di distretto."""

    def __init__(self, distretto: int) -> None:
        self._distretto = distretto

    async def propose_text_async(self, prompt: str) -> dict:
        await asyncio.sleep(0.02 * (DISTRETTI - self._distretto))
        return {"raw": {"accept": True, "rationale": f"distretto {self._distretto}"}}


@pytest.fixture(autouse=True)
def _quadri_finti(monkeypatch):
    """Il soggetto e' la tornata, non la costruzione del quadro."""
    quadro = ColonyPicture(
        step=1, population=1, metrics={}, indicators={},
        population_stats={}, structures={}, n_cells=1,
    )
    monkeypatch.setattr(administration, "quadro_di_distretto", lambda *a, **k: quadro)
    monkeypatch.setattr(administration, "politica_risolta", lambda *a, **k: [])


def _amministrazione(fabbrica, quanti: int = DISTRETTI) -> Amministrazione:
    return Amministrazione(fabbrica, _DistrettiFinti(quanti), Bounds(0.25, 4.0))


def _avanza(amministrazione) -> list:
    import numpy as np

    cells = SimpleNamespace(occupancy=np.ones((DISTRETTI, 2), dtype=float))
    amministrazione.advance(1, {}, cells, SimpleNamespace(), [], None)
    return amministrazione.ultime


def test_gli_amministratori_partono_tutti_nello_stesso_istante():
    istanti: list[float] = []
    amministrazione = _amministrazione(lambda d: _Lento(d, istanti))

    inizio = time.perf_counter()
    decisioni = _avanza(amministrazione)
    durata = time.perf_counter() - inizio

    assert len(decisioni) == DISTRETTI
    assert len(istanti) == DISTRETTI
    # In fila sarebbero DISTRETTI x RITARDO; insieme, poco piu' di RITARDO.
    assert durata < RITARDO * 2, f"tornata sequenziale: {durata:.2f}s"
    assert max(istanti) - min(istanti) < RITARDO / 2


def test_anche_un_proponente_solo_sincrono_resta_concorrente():
    istanti: list[float] = []
    amministrazione = _amministrazione(lambda d: _LentoSincrono(d, istanti))

    inizio = time.perf_counter()
    _avanza(amministrazione)
    durata = time.perf_counter() - inizio

    assert durata < RITARDO * 2, f"tornata sequenziale: {durata:.2f}s"


def test_le_decisioni_tornano_in_ordine_di_distretto_non_di_arrivo():
    amministrazione = _amministrazione(_AlContrario)
    decisioni = _avanza(amministrazione)
    assert [d.distretto for d in decisioni] == list(range(DISTRETTI))
    assert [d.rationale for d in decisioni] == [
        f"distretto {d}" for d in range(DISTRETTI)
    ]


def test_un_distretto_che_esplode_non_annulla_la_tornata():
    class _Esplode:
        async def propose_text_async(self, prompt: str) -> dict:
            raise RuntimeError("fornitore giu'")

    def fabbrica(distretto: int):
        return _Esplode() if distretto == 2 else _AlContrario(distretto)

    decisioni = _avanza(_amministrazione(fabbrica))
    assert len(decisioni) == DISTRETTI
    assert "chiamata fallita" in decisioni[2].rationale
    assert decisioni[2].policy is None


def test_un_arm_storico_non_aggiunge_campi_semantici_all_artifact():
    decisione = _avanza(_amministrazione(_AlContrario, quanti=1))[0]
    payload = decisione.to_json()
    assert "semantic_decision" not in payload
    assert "candidate_profile" not in payload
    assert "semantic_escalated" not in payload
