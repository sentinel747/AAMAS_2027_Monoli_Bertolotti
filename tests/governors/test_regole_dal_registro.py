# -*- coding: utf-8 -*-
"""La riga «se food_per_occupant < 1.5 -> build x1, ..., sustenance x3» e' cio'
che il registro e i frontend MOSTRANO, non cio' che il modello scrive.

**Il dubbio (2026-09-06).** Guardando le policy nell'analysis frontend ci si e'
chiesti se quel «se» davanti potesse confondere il parser, o se le celle
applicassero davvero una politica scritta in quella forma. La risposta sta nel
disegno: il modello risponde in JSON (`{"if": {...}, "weights": {...}}`), il
parser legge SOLO quello schema, e la riga in italiano e' `rule_text`, generata
DOPO il parser dalla regola gia' tipizzata, la stessa che `apply_policy`
valuta. Non esiste un percorso in cui il testo venga ri-letto.

Questi test fissano le tre cose che rendono vero il ragionamento:

1. una proposta REALE del registro (v3, seme 3, tick 3) passa dal parser senza
   scarti, si stampa esattamente come la mostra il frontend, e scatta sulle
   celle che la sua condizione descrive;
2. una policy scritta come TESTO e' fuori schema per il governatore
   (`None`: resta in vigore la precedente) — non viene ne' applicata ne'
   interpretata a meta';
3. per l'amministratore la stessa risposta a testo e' marcata MALFORMATA nel
   registro, distinta da un'accettazione: le celle restano al governo, ma il
   conteggio delle accettazioni non la assorbe piu' in silenzio.

Piu' una quarta: da oggi il registro conserva la risposta grezza
(`proposal_raw`), perche' i contatori di scarto delle campagne precedenti
dicevano "23 pilastri fuori vocabolario" senza poter dire quali.
"""

import json

import numpy as np

from src.agents import pillars
from src.agents.pillars import ACTION_INDEX, N_ACTIONS, PILLAR_ACTIONS
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays
from src.governors.administration import Amministrazione
from src.governors.apply import apply_policy
from src.governors.arms import GovernorProposal
from src.governors.observation import ColonyPicture
from src.governors.policy import Bounds, PolicyDrops, parse_policy, rule_text
from src.governors.record import GovernorRecorder, load_records

BOUNDS = Bounds(0.25, 4.0)

#: Copiata dal registro `runs/campagna_scarsa_v3/llm_completo_amm/
#: llm_completo+amm_seed3/governor_decisions.jsonl`, tick 3 (gpt-oss:20b).
PROPOSTA_REALE = {
    "policy": [
        {"if": {"indicator": "food_per_occupant", "op": "<", "value": 0.5},
         "weights": {"sustenance": 1.5, "resources": 1.5, "build": 2.0, "explore": 1.0}},
        {"if": {"indicator": "food_per_occupant", "op": ">", "value": 0.5},
         "weights": {"sustenance": 1.2, "resources": 1.5, "build": 1.5, "explore": 2.0}},
        {"if": None,
         "weights": {"sustenance": 1.0, "resources": 1.0, "build": 1.0, "explore": 1.0}},
    ],
    "rationale": "prova dal registro",
}

#: La stessa politica, scritta come la si LEGGE nel frontend. Non e' un
#: formato di ingresso: nessun parser la accetta.
POLITICA_A_TESTO = {
    "policy": [
        "se food_per_occupant < 0.5 -> sustenance x1.5, resources x1.5, build x2, explore x1",
        "sempre -> sustenance x1, resources x1, build x1, explore x1",
    ],
    "rationale": "scritta a parole",
}


def test_a_real_registered_proposal_parses_clean_and_reads_back_as_shown():
    policy, drops = parse_policy(PROPOSTA_REALE, BOUNDS)
    assert policy is not None
    assert drops == PolicyDrops()
    assert [rule_text(r) for r in policy.rules] == [
        "se food_per_occupant < 0.5 -> sustenance x1.5, resources x1.5, build x2, explore x1",
        "se food_per_occupant > 0.5 -> sustenance x1.2, resources x1.5, build x1.5, explore x2",
        "sempre -> sustenance x1, resources x1, build x1, explore x1",
    ]


def test_the_parsed_real_proposal_fires_on_the_cells_its_condition_describes():
    policy, _ = parse_policy(PROPOSTA_REALE, BOUNDS)
    cells = CellArrays(3, 3)
    cells.occupancy[0, 0] = 4                    # cibo 0 / 4 = 0 < 0.5 -> regola 1
    cells.occupancy[2, 2] = 2
    cells.cell_res[2, 2, C.R["food"]] = 10.0     # 5 a testa > 0.5 -> regola 2
    priority = np.ones((3, 3, N_ACTIONS))
    hits: dict = {}
    apply_policy(priority, policy, cells, hits=hits)

    build = ACTION_INDEX[next(iter(PILLAR_ACTIONS[pillars.P_BUILD]))]
    explore = ACTION_INDEX[next(iter(PILLAR_ACTIONS[pillars.P_EXPLORE]))]
    assert priority[0, 0, build] == 2.0 and priority[0, 0, explore] == 1.0
    assert priority[2, 2, build] == 1.5 and priority[2, 2, explore] == 2.0
    # Le chiavi dei contatori sono le stesse righe che il frontend mostra.
    assert hits[rule_text(policy.rules[0])] == 1
    assert hits[rule_text(policy.rules[1])] == 1
    # Celle vuote: mai toccate.
    assert np.all(priority[1, 1] == 1.0)


def test_a_policy_written_as_text_is_out_of_schema_for_the_governor():
    policy, drops = parse_policy(POLITICA_A_TESTO, BOUNDS)
    assert policy is None            # "scarta e tieni in vigore la precedente"
    assert drops == PolicyDrops()    # non e' uno scarto locale: e' un'altra lingua


# --- amministratori ---------------------------------------------------------

CELLE = [(1, 1), (1, 2)]


class _Distretti:
    def mappa(self) -> dict:
        return {0: list(CELLE)}

    def numero_distretti(self) -> int:
        return 1


class _Risponde:
    def __init__(self, raw) -> None:
        self._raw = raw

    async def propose_text_async(self, prompt: str) -> dict:
        return {"raw": self._raw, "provider": "finto", "model": "m"}


def _tornata(raw):
    amministrazione = Amministrazione(lambda d: _Risponde(raw), _Distretti(), BOUNDS)
    cells = CellArrays(4, 4)
    agents = AgentArrays(2)
    for i, (y, x) in enumerate(CELLE):
        cells.occupancy[y, x] = 1
        agents.alive[i] = True
        agents.y[i], agents.x[i] = y, x
    amministrazione.advance(1, {}, cells, agents, np.arange(2), None)
    return amministrazione


def test_an_administrator_rewrite_written_as_text_is_marked_malformed_not_accepted():
    amm = _tornata({"accept": False, **POLITICA_A_TESTO})
    [decisione] = amm.ultime
    assert decisione.policy is None          # le celle restano al governo...
    assert decisione.malformata is True      # ...ma NON e' un'accettazione
    riga = decisione.to_json()
    assert riga["malformed"] is True
    assert riga["accepted_government_policy"] is True
    assert riga["proposal_raw"] == POLITICA_A_TESTO["policy"]
    assert amm.riassunto()["malformed"] == 1


def test_a_well_formed_administrator_rewrite_is_not_malformed_and_keeps_its_raw():
    amm = _tornata({"accept": False, **PROPOSTA_REALE})
    [decisione] = amm.ultime
    assert decisione.policy is not None and decisione.malformata is False
    riga = decisione.to_json()
    assert riga["malformed"] is False
    assert riga["proposal_raw"] == PROPOSTA_REALE["policy"]
    assert riga["rules"][0] == rule_text(decisione.policy.rules[0])
    assert amm.riassunto()["malformed"] == 0


def test_a_plain_acceptance_is_neither_malformed_nor_raw():
    amm = _tornata({"accept": True, "rationale": "va bene"})
    [decisione] = amm.ultime
    riga = decisione.to_json()
    assert riga["malformed"] is False and riga["proposal_raw"] is None


def test_a_non_json_answer_is_malformed_for_the_administrator():
    amm = _tornata("se food_per_occupant < 0.5 -> sustenance x3")
    [decisione] = amm.ultime
    assert decisione.malformata is True and decisione.policy is None
    assert decisione.to_json()["proposal_raw"] == "se food_per_occupant < 0.5 -> sustenance x3"


# --- il registro del governatore --------------------------------------------

def test_the_governor_register_keeps_the_raw_proposal_when_there_is_one(tmp_path):
    policy, drops = parse_policy(PROPOSTA_REALE, BOUNDS)
    proposal = GovernorProposal(
        policy=policy, rationale="prova", provider="finto", model="m",
        drops=drops, raw_policy=PROPOSTA_REALE["policy"],
    )
    recorder = GovernorRecorder(tmp_path / "governor_decisions.jsonl")
    recorder.record(0, 3, ColonyPicture(step=3, population=10), proposal, policy, False)
    recorder.close()
    [row] = load_records(tmp_path / "governor_decisions.jsonl")
    assert row["governor"]["proposal_raw"] == PROPOSTA_REALE["policy"]
    # E la riga scritta resta UNA riga JSON valida.
    assert len((tmp_path / "governor_decisions.jsonl").read_text(encoding="utf-8").strip().splitlines()) == 1


def test_a_proposal_without_raw_writes_no_raw_key(tmp_path):
    policy, _ = parse_policy(PROPOSTA_REALE, BOUNDS)
    proposal = GovernorProposal(policy=policy, provider="scripted", model="none")
    recorder = GovernorRecorder(tmp_path / "g.jsonl")
    recorder.record(0, 3, ColonyPicture(step=3, population=10), proposal, policy, False)
    recorder.close()
    [row] = load_records(tmp_path / "g.jsonl")
    assert "proposal_raw" not in row["governor"]
    json.dumps(row)  # serializzabile
