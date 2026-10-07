# -*- coding: utf-8 -*-
"""Il terzo esito, quello che nessuno aveva dichiarato.

**Il fatto.** La catena dell'amministratore SOSTITUISCE quella del governo sulle
sue celle. Se la regola riscritta non scatta su nessuna di esse, o scatta con
pesi tutti 1, le celle non ricevono ne' il governo ne' una correzione: ricadono
alle preferenze pure. Non e' ACCETTA e non e' RISCRIVI: e' un decentramento che
toglie la legge senza metterne un'altra, e finora il registro lo scriveva come
una riscrittura qualunque.

Misurato sui registri della campagna del 2026-09-02, dopo il fatto e a mano:
nello scripted+amm dal 19 al 33 per cento delle riscritture non scattava mai in
tutta la run. Questi test pretendono che il registro lo dica da solo, tornata
per tornata, cosi' che l'analisi lo conti invece di scoprirlo.

La marcatura e' istantanea --- «su queste celle, adesso, la tua catena non fa
niente» --- e non predice il futuro: una regola puo' scattare piu' tardi. E'
la stessa informazione che `politica_risolta` da' all'amministratore sulla
politica del governo, applicata alla sua.
"""

import numpy as np

from src.agents import pillars
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.governors.administration import Amministrazione, riscrittura_a_vuoto
from src.governors.policy import Bounds, Condition, Policy, Rule

BOUNDS = Bounds(0.25, 4.0)
CELLE = [(1, 1), (1, 2)]


class _DistrettiFinti:
    def mappa(self) -> dict:
        return {0: list(CELLE)}

    def numero_distretti(self) -> int:
        return 1


class _Riscrive:
    """Risponde sempre con la stessa politica grezza."""

    def __init__(self, policy: list) -> None:
        self._policy = policy

    async def propose_text_async(self, prompt: str) -> dict:
        return {"raw": {"accept": False, "policy": self._policy, "rationale": "prova"}}


class _Accetta:
    async def propose_text_async(self, prompt: str) -> dict:
        return {"raw": {"accept": True, "rationale": "va bene"}}


def _mondo():
    cells = CellArrays(4, 4)
    agents = AgentArrays(2)
    for i, (y, x) in enumerate(CELLE):
        cells.occupancy[y, x] = 1
        cells.cell_res[y, x, C.R["food"]] = 10.0  # food_per_occupant = 10
        agents.alive[i] = True
        agents.y[i], agents.x[i] = y, x
    return cells, agents


def _tornata(proponente):
    amministrazione = Amministrazione(lambda d: proponente, _DistrettiFinti(), BOUNDS)
    cells, agents = _mondo()
    amministrazione.advance(1, {}, cells, agents, np.arange(2), None)
    return amministrazione


IMPOSSIBILE = [{"if": {"indicator": "food_per_occupant", "op": "<", "value": 0.0},
                "weights": {"sustenance": 3.0}}]
SCATTA = [{"if": {"indicator": "food_per_occupant", "op": ">", "value": 1.0},
           "weights": {"sustenance": 3.0}}]
PESI_UNO = [{"if": None, "weights": {"sustenance": 1.0, "build": 1.0}}]


# ------------------------------------------------------------ il predicato


def test_una_catena_che_non_scatta_su_nessuna_cella_e_a_vuoto():
    cells, _ = _mondo()
    policy = Policy((Rule(Condition("food_per_occupant", "<", 0.0), {pillars.P_SUSTENANCE: 3.0}),))
    assert riscrittura_a_vuoto(policy, cells, CELLE) is True


def test_una_catena_che_scatta_anche_su_una_sola_cella_non_e_a_vuoto():
    cells, _ = _mondo()
    cells.cell_res[1, 2, C.R["food"]] = 0.0  # la seconda cella e' affamata
    policy = Policy((Rule(Condition("food_per_occupant", "<", 1.0), {pillars.P_SUSTENANCE: 3.0}),))
    assert riscrittura_a_vuoto(policy, cells, CELLE) is False


def test_una_catena_a_pesi_tutti_uno_e_a_vuoto_anche_se_scatta():
    cells, _ = _mondo()
    policy = Policy((Rule(None, {pillars.P_SUSTENANCE: 1.0, pillars.P_BUILD: 1.0}),))
    assert riscrittura_a_vuoto(policy, cells, CELLE) is True


def test_una_catena_in_cui_la_prima_regola_a_pesi_uno_copre_tutto_e_a_vuoto():
    """Prima regola che scatta vince: una chiusura a pesi 1 messa PRIMA affama la seconda."""
    cells, _ = _mondo()
    policy = Policy((
        Rule(None, {pillars.P_SUSTENANCE: 1.0}),
        Rule(Condition("food_per_occupant", ">", 1.0), {pillars.P_SUSTENANCE: 3.0}),
    ))
    assert riscrittura_a_vuoto(policy, cells, CELLE) is True


# ------------------------------------------------------------- il registro


def test_il_registro_marca_la_riscrittura_a_vuoto():
    amministrazione = _tornata(_Riscrive(IMPOSSIBILE))
    decisione = amministrazione.ultime[0]
    assert decisione.policy is not None, "e' una riscrittura, non un'astensione"
    assert decisione.a_vuoto is True
    assert decisione.to_json()["rewrite_without_effect"] is True


def test_il_registro_non_marca_una_riscrittura_che_morde():
    amministrazione = _tornata(_Riscrive(SCATTA))
    decisione = amministrazione.ultime[0]
    assert decisione.a_vuoto is False
    assert decisione.to_json()["rewrite_without_effect"] is False


def test_accettare_non_e_mai_a_vuoto():
    amministrazione = _tornata(_Accetta())
    decisione = amministrazione.ultime[0]
    assert decisione.policy is None
    assert decisione.a_vuoto is False


def test_il_riassunto_conta_le_riscritture_a_vuoto_a_parte():
    amministrazione = _tornata(_Riscrive(PESI_UNO))
    riassunto = amministrazione.riassunto()
    assert riassunto["interventions"] == 1
    assert riassunto["rewrites_without_effect"] == 1


# ------------------------------------------------- la memoria della volta scorsa


def test_la_volta_scorsa_dice_se_la_riscrittura_era_a_vuoto():
    """Senza questa riga l'amministratore puo' ripetere dieci volte una regola inerte."""
    from src.governors.administration import _riga_precedente

    amministrazione = _tornata(_Riscrive(IMPOSSIBILE))
    riga = _riga_precedente(amministrazione.ultime[0])
    assert "RISCRITTO" in riga
    assert "senza politica" in riga.lower()


def test_la_volta_scorsa_non_accusa_una_riscrittura_che_mordeva():
    from src.governors.administration import _riga_precedente

    amministrazione = _tornata(_Riscrive(SCATTA))
    riga = _riga_precedente(amministrazione.ultime[0])
    assert "senza politica" not in riga.lower()


# ------------------------------------------- la popolazione del distretto nel registro


def test_il_registro_porta_la_popolazione_del_distretto():
    """Senza, la misura dentro la run divide per celle-passo e non per coloni-passo:
    un distretto di tre celle con due coloni e uno con duecento pesano uguale."""
    amministrazione = _tornata(_Accetta())
    decisione = amministrazione.ultime[0]
    assert decisione.popolazione == 2
    assert decisione.to_json()["population"] == 2


# ------------------------------------ la risposta ambigua e gli scarti del parser


class _AccettaMaScrive:
    """`accept: true` e insieme una politica: il modello ha detto due cose."""

    async def propose_text_async(self, prompt: str) -> dict:
        return {"raw": {"accept": True, "policy": SCATTA, "rationale": "la politica ignora la carenza, aggiungo una regola"}}


class _RiscriveFuoriVocabolario:
    async def propose_text_async(self, prompt: str) -> dict:
        return {"raw": {"accept": False, "policy": [
            {"if": {"indicator": "morale_index", "op": "<", "value": 0.5}, "weights": {"sustenance": 3.0}},
            {"if": {"indicator": "food_per_occupant", "op": "<", "value": 1.0}, "weights": {"felicita": 2.0, "build": 9.0}},
        ], "rationale": "prova"}}


def test_accettare_con_una_politica_allegata_e_marcato_come_ambiguo():
    """Il 43% delle accettazioni della v3 motiva con carenze e modifiche: se il
    modello allega anche una politica, oggi la si ignora in silenzio. Va detto
    nel registro, cosi' si misura quante volte succede."""
    amministrazione = _tornata(_AccettaMaScrive())
    decisione = amministrazione.ultime[0]
    assert decisione.policy is None, "accept: true vince, la politica non entra in vigore"
    assert decisione.accettata is True
    assert decisione.to_json()["accepted_with_policy"] is True
    assert amministrazione.riassunto()["accepted_with_policy"] == 1


def test_un_accettare_pulito_non_e_ambiguo():
    decisione = _tornata(_Accetta()).ultime[0]
    assert decisione.to_json()["accepted_with_policy"] is False


def test_gli_scarti_del_parser_finiscono_nel_registro():
    """Il governatore registra `drops`; l'amministratore no, e una regola con un
    indicatore inventato spariva senza traccia."""
    decisione = _tornata(_RiscriveFuoriVocabolario()).ultime[0]
    scarti = decisione.to_json()["drops"]
    assert scarti["unknown_indicators"] == 1
    assert scarti["unknown_pillars"] == 1
    assert scarti["clamped_weights"] == 1
    assert len(decisione.policy.rules) == 1, "resta la sola regola nel vocabolario"
