# -*- coding: utf-8 -*-
"""L'amministratore legge la legge intera, non solo l'articolo che lo tocca.

**Il buco.** Fino al 2026-09-05 il prompt dell'amministratore portava la
politica del governo *risolta* sulle sue celle: quale regola scatta e con che
pesi. Era pensato per non fargli dedurre nulla, ed e' giusto tenerlo. Ma da
solo non basta a RISCRIVERE: una regola che oggi non scatta sulle sue celle non
compariva affatto, e nemmeno l'ordine della catena. L'amministratore poteva
accettare o ripartire da zero, mai adattare --- cambiare una soglia, un peso,
l'ordine --- che e' la forma di correzione che il disegno chiede ("riscrivono
la policy nello stesso identico linguaggio per le loro celle").

Nessuno di questi test tocca il modello: il soggetto e' che cosa arriva nel
prompt, e lo si legge con un proponente finto che lo cattura.
"""

import numpy as np

from src.agents import pillars
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.governors.administration import Amministrazione, costruisci_prompt
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, Condition, Policy, Rule, rule_text

BOUNDS = Bounds(0.25, 4.0)
CELLA = (1, 1)

#: Scatta solo con poco cibo: sulla cella di prova, piena di cibo, NON scatta.
REGOLA_CIBO = Rule(Condition("food_per_occupant", "<", 1.0), {pillars.P_SUSTENANCE: 3.0})
#: Scatta solo con troppa gente: nemmeno questa scatta sulla cella di prova.
REGOLA_FOLLA = Rule(Condition("occupants", ">", 150.0), {pillars.P_EXPLORE: 2.0})
#: La chiusura incondizionata a pesi 1, che e' quella che scatta davvero.
REGOLA_CHIUSURA = Rule(None, {pillars.P_SUSTENANCE: 1.0})

POLITICA = Policy((REGOLA_CIBO, REGOLA_FOLLA, REGOLA_CHIUSURA), "prova")


class _DistrettiFinti:
    def mappa(self) -> dict:
        return {0: [CELLA]}

    def numero_distretti(self) -> int:
        return 1


class _Cattura:
    """Accetta sempre, ma tiene il prompt che ha ricevuto."""

    def __init__(self) -> None:
        self.prompt = ""

    async def propose_text_async(self, prompt: str) -> dict:
        self.prompt = prompt
        return {"raw": {"accept": True, "rationale": "prova"}}


def _mondo_con_cibo():
    cells = CellArrays(4, 4)
    cells.occupancy[CELLA] = 2
    cells.cell_res[CELLA[0], CELLA[1], C.R["food"]] = 100.0
    agents = AgentArrays(2)
    for i in range(2):
        agents.alive[i] = True
        agents.y[i], agents.x[i] = CELLA
    return cells, agents


def _prompt_ricevuto(policy) -> str:
    cattura = _Cattura()
    amministrazione = Amministrazione(lambda d: cattura, _DistrettiFinti(), BOUNDS)
    cells, agents = _mondo_con_cibo()
    amministrazione.advance(1, {}, cells, agents, np.arange(2), policy)
    return cattura.prompt


def test_il_prompt_porta_anche_le_regole_che_non_scattano_sulle_sue_celle():
    prompt = _prompt_ricevuto(POLITICA)
    # La vista risolta mostra solo la chiusura; le altre due regole devono
    # comunque esserci, perche' sono cio' che l'amministratore puo' adattare.
    assert rule_text(REGOLA_CIBO) in prompt
    assert rule_text(REGOLA_FOLLA) in prompt


def test_la_catena_e_nell_ordine_in_cui_il_governo_l_ha_scritta():
    prompt = _prompt_ricevuto(POLITICA)
    posizioni = [prompt.index(rule_text(r)) for r in POLITICA.rules]
    assert posizioni == sorted(posizioni)
    # Ed e' numerata: l'ordine e' cio' che decide chi affama chi.
    assert f"1. {rule_text(REGOLA_CIBO)}" in prompt
    assert f"3. {rule_text(REGOLA_CHIUSURA)}" in prompt


def test_il_prompt_dice_che_si_puo_ripartire_dalla_catena():
    prompt = _prompt_ricevuto(POLITICA)
    assert "ripartire" in prompt.lower()


def test_senza_governo_la_catena_lo_dichiara_invece_di_sparire():
    prompt = _prompt_ricevuto(None)
    assert "nessuna politica" in prompt.lower()


def test_costruisci_prompt_accetta_la_catena_come_testo():
    quadro = ColonyPicture(step=1, population=2, metrics={}, indicators={},
                           population_stats={}, structures={}, n_cells=1)
    prompt = costruisci_prompt(
        quadro, ["cella [1, 1]: NESSUN INTERVENTO"], BOUNDS, 0,
        catena="1. se food_per_occupant < 1 -> sustenance x3",
    )
    assert "1. se food_per_occupant < 1 -> sustenance x3" in prompt
