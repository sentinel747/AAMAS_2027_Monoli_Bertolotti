"""La pianificazione del governatore: cadenza, promozione, silenzio, attesa.

Ereditata dal consiglio, e le promesse restano le stesse: la chiamata non sta
mai sul percorso critico (salvo regime bloccante scelto), il passo di
applicazione dipende dalla sola configurazione, un mancato aggiornamento non
cancella la policy in vigore. Nuova e' la semantica del silenzio: una proposta
malformata (`policy=None`) NON sostituisce la policy precedente, una policy
vuota si'.
"""

import asyncio
import threading
import time

import pytest

from src.governors.arms import GovernorProposal
from src.governors.governor import Governor
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Policy, Rule

BOUNDS = Bounds(0.25, 4.0)


def _picture(step):
    return ColonyPicture(step=step, population=10)


def _policy(tag):
    """Una policy distinguibile: il tag vive nella rationale."""
    return Policy((Rule(None, {2: 2.0}),), rationale=tag)


class _Proposer:
    """Risponde subito, con la policy in coda; None = proposta malformata."""

    def __init__(self, policies):
        self._policies = list(policies)
        self.pictures = []

    async def propose(self, picture, bounds):
        self.pictures.append(picture)
        policy = self._policies.pop(0) if self._policies else Policy()
        return GovernorProposal(policy=policy, rationale="test")


class _SlowProposer:
    async def propose(self, picture, bounds):
        await asyncio.sleep(30.0)
        return GovernorProposal(policy=_policy("tardi"))


class _ProposerCheRifiuta:
    """Quello che il braccio LLM restituisce su un 503: subito, e vuoto.

    E' la forma esatta osservata il 18 settembre 2026 rilanciando OSS-55 su una
    farm satura --- `policy=None`, la rationale col codice HTTP, e nessuna
    attesa, perche' il rifiuto e' istantaneo.
    """

    async def propose(self, picture, bounds):
        return GovernorProposal(
            policy=None,
            rationale="chiamata fallita: HTTPStatusError: 503 server busy",
            guasto_fornitore=True,
        )


def test_boundaries_and_application_steps_come_from_the_config_alone():
    governor = Governor(_Proposer([]), BOUNDS, cadence_steps=20)
    try:
        assert governor.is_tick_boundary(1)
        assert governor.is_tick_boundary(21)
        assert not governor.is_tick_boundary(2)
        assert governor.tick_index(1) == 0 and governor.tick_index(21) == 1
        assert governor.application_step(0) == 21, "regime non bloccante: il confine dopo"
    finally:
        governor.close()

    blocking = Governor(_Proposer([]), BOUNDS, cadence_steps=20, wait_seconds=5.0)
    try:
        assert blocking.application_step(0) == 1, "regime bloccante: lo stesso confine"
    finally:
        blocking.close()


def test_the_policy_of_tick_t_enters_at_the_next_boundary():
    proposer = _Proposer([_policy("primo"), _policy("secondo")])
    governor = Governor(proposer, BOUNDS, cadence_steps=2)
    try:
        assert governor.advance(1, _picture(1)) is None, "al passo 1 non c'e' ancora nulla"
        assert governor.advance(2, _picture(2)) is None, "fra i confini niente cambia"
        for _ in range(200):
            if governor._pending.done():
                break
            time.sleep(0.005)
        adopted = governor.advance(3, _picture(3))
        assert adopted is not None and adopted.rationale == "primo"
        assert proposer.pictures[0].step == 1, (
            "la proposta adottata al passo 3 deve venire dal quadro del passo 1"
        )
    finally:
        governor.close()


def test_a_malformed_proposal_keeps_the_previous_policy_in_force():
    proposer = _Proposer([_policy("buona"), None, _policy("dopo")])
    governor = Governor(proposer, BOUNDS, cadence_steps=1)
    try:
        governor.advance(1, _picture(1))
        _wait(governor)
        assert governor.advance(2, _picture(2)).rationale == "buona"
        _wait(governor)
        in_force = governor.advance(3, _picture(3))
        assert in_force.rationale == "buona", (
            "policy=None e' silenzio: la politica non svanisce per un errore di rete"
        )
        assert governor.misses == 0, "una risposta malformata NON e' un mancato arrivo"
    finally:
        governor.close()


def test_an_empty_policy_replaces_the_previous_one():
    proposer = _Proposer([_policy("piena"), Policy()])
    governor = Governor(proposer, BOUNDS, cadence_steps=1)
    try:
        governor.advance(1, _picture(1))
        _wait(governor)
        assert governor.advance(2, _picture(2)).rationale == "piena"
        _wait(governor)
        adopted = governor.advance(3, _picture(3))
        assert adopted is not None and adopted.rules == (), (
            "la policy vuota e' un'adozione esplicita del non-intervento"
        )
    finally:
        governor.close()


def test_a_late_call_counts_as_a_miss_and_keeps_the_policy():
    governor = Governor(_SlowProposer(), BOUNDS, cadence_steps=1)
    try:
        governor.advance(1, _picture(1))
        assert governor.advance(2, _picture(2)) is None
        assert governor.misses == 1
        governor.advance(3, _picture(3))
        assert governor.misses == 2
    finally:
        governor.close()


def test_three_consecutive_misses_raise_the_alarm_once(capsys):
    governor = Governor(_SlowProposer(), BOUNDS, cadence_steps=1)
    try:
        for step in range(1, 6):
            governor.advance(step, _picture(step))
        out = capsys.readouterr().out
        assert out.count("ALLARME") == 1
        assert "mancati aggiornamenti consecutivi" in out
    finally:
        governor.close()


def test_blocking_regime_adopts_in_the_same_step():
    proposer = _Proposer([_policy("subito")])
    governor = Governor(proposer, BOUNDS, cadence_steps=2, wait_seconds=5.0)
    try:
        adopted = governor.advance(1, _picture(1))
        assert adopted is not None and adopted.rationale == "subito"
    finally:
        governor.close()


def test_blocking_regime_times_out_into_a_miss():
    governor = Governor(_SlowProposer(), BOUNDS, cadence_steps=1, wait_seconds=0.05)
    try:
        assert governor.advance(1, _picture(1)) is None
        assert governor.misses == 1
    finally:
        governor.close()


def test_un_rifiuto_immediato_vale_quanto_un_timeout():
    """Un 503 e' un mancato aggiornamento, anche se torna in un millesimo.

    Prima contava solo il timeout: un fornitore saturo rispondeva subito, la
    proposta arrivava regolare con `policy=None`, e la tornata finiva nel
    registro come riuscita. Misurato su una run vera: diciassette tornate,
    nessuna legge adottata, `governor_misses` a zero --- cioe' la condizione
    principale del cancello di qualita' cieca proprio sul guasto piu' grosso.
    """
    governor = Governor(_ProposerCheRifiuta(), BOUNDS, cadence_steps=1)
    try:
        governor.advance(1, _picture(1))
        _wait(governor)
        governor.advance(2, _picture(2))
        assert governor.misses == 1, (
            "un fornitore che rifiuta non ha consegnato nessuna legge: "
            "che lo faccia in fretta non lo rende una tornata riuscita"
        )
    finally:
        governor.close()


def test_un_rifiuto_non_cancella_la_legge_in_vigore():
    """Il guasto conta, ma non abroga: e' la promessa di sempre."""
    proposer = _Proposer([_policy("buona")])
    governor = Governor(proposer, BOUNDS, cadence_steps=1)
    try:
        governor.advance(1, _picture(1))
        _wait(governor)
        assert governor.advance(2, _picture(2)).rationale == "buona"
    finally:
        governor.close()

    governor = Governor(_ProposerCheRifiuta(), BOUNDS, cadence_steps=1,
                        wait_seconds=5.0)
    try:
        governor._in_force = _policy("buona")
        in_force = governor.advance(1, _picture(1))
        assert in_force is not None and in_force.rationale == "buona", (
            "un guasto del fornitore non deve far svanire la politica"
        )
        assert governor.misses == 1
    finally:
        governor.close()


def test_tre_rifiuti_di_fila_fanno_scattare_l_allarme(capsys):
    """La stessa guardia del timeout, sullo stesso numero."""
    governor = Governor(_ProposerCheRifiuta(), BOUNDS, cadence_steps=1,
                        wait_seconds=5.0)
    try:
        for step in range(1, 6):
            governor.advance(step, _picture(step))
        out = capsys.readouterr().out
        assert out.count("ALLARME") == 1, (
            "tre rifiuti di fila sono una configurazione che non puo' riuscire, "
            "e finora la run proseguiva in silenzio fino in fondo"
        )
    finally:
        governor.close()


def test_una_risposta_malformata_resta_fuori_dal_conteggio():
    """La distinzione decisa apposta, e che va protetta da un test.

    Il JSON rotto e' il modello che si comporta in modo erratico: e' un dato
    sul modello, non un guasto dell'infrastruttura. Contarlo fra i mancati
    aggiornamenti scarterebbe via le run in cui un modello e' meno stabile, e
    distorcerebbe il confronto fra modelli --- che e' lo scopo della campagna.
    """
    governor = Governor(_Proposer([None, None, None]), BOUNDS, cadence_steps=1)
    try:
        for step in range(1, 5):
            governor.advance(step, _picture(step))
            _wait(governor)
        assert governor.misses == 0
    finally:
        governor.close()


def test_the_recorder_gets_one_row_per_processed_boundary():
    rows = []

    class Recorder:
        def record(self, **kwargs):
            rows.append(kwargs)

    proposer = _Proposer([_policy("a"), _policy("b")])
    governor = Governor(proposer, BOUNDS, cadence_steps=1, recorder=Recorder())
    try:
        governor.advance(1, _picture(1))
        _wait(governor)
        governor.advance(2, _picture(2))
        _wait(governor)
        governor.advance(3, _picture(3))
    finally:
        governor.close()
    assert len(rows) == 2
    assert rows[0]["tick"] == 0 and rows[0]["application_step"] == 2
    assert rows[0]["picture"].step == 1, (
        "il registro porta il quadro CHE HA PRODOTTO la proposta, non quello del "
        "passo di adozione"
    )
    assert rows[0]["policy"].rationale == "a"


def test_close_is_idempotent_and_stops_the_thread():
    governor = Governor(_Proposer([]), BOUNDS, cadence_steps=1)
    governor.advance(1, _picture(1))
    governor.close()
    governor.close()
    assert "governors" not in {t.name for t in threading.enumerate()}


def test_invalid_construction_is_refused():
    with pytest.raises(ValueError):
        Governor(_Proposer([]), BOUNDS, cadence_steps=0)
    with pytest.raises(ValueError):
        Governor(_Proposer([]), BOUNDS, cadence_steps=1, wait_seconds=-1.0)


def _wait(governor, timeout=2.0):
    """Attende che la chiamata in volo sia conclusa, senza toccare l'API."""
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        pending = governor._pending
        if pending is None or pending.done():
            return
        time.sleep(0.005)
    raise AssertionError("la proposta non e' mai arrivata")
