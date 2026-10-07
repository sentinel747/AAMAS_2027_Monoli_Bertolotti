# -*- coding: utf-8 -*-
"""Il pannello React e `ManualConfigOptions` devono partire dagli stessi valori.

**Perche' esiste.** Il pannello React costruisce il config a mano, in
TypeScript, e non puo' importare la dataclass Python: i due elenchi di default
sono percio' due copie della stessa cosa, cioe' la famiglia 10 in forma pura.
Il 2026-08-31 avevano gia' divergito su cinque campi, e uno era grave:

  - `development_build_priority` valeva **0,07** nel pannello contro **0,30**
    in Python — e 0,07 e' proprio il valore che il sorgente Python documenta
    come rotto («a `cell_proposal_top_k = 5` tiene laboratori e infermerie
    fuori dal menu: hanno domanda insoddisfatta e non vengono mai
    costruiti»). Ogni run lanciata dalla GUI girava dunque una simulazione
    diversa, e peggiore, da ogni run lanciata dal terminale;
  - i quattro parametri di redistribuzione (borraccia acqua/ossigeno, tasso e
    serre pro capite) divergevano a loro volta, e mordono appena qualcuno
    accende la redistribuzione.

Un confronto a mano non li aveva visti in mesi. Questo test li confronta a
ogni esecuzione, campo per campo, leggendo direttamente i due sorgenti.
"""
from __future__ import annotations

import dataclasses
import re
from pathlib import Path

import pytest

from src.experiments.manual_config import ManualConfigOptions

PANNELLO = Path(__file__).resolve().parents[1] / "frontend/src/components/ScenarioPanel.tsx"

#: Campi il cui default React vive sotto un altro nome, oppure che il pannello
#: deliberatamente non espone. Ogni voce e' una DEROGA e va motivata qui: senza
#: questa disciplina la mappa diventa il posto dove si nascondono le divergenze.
ALIAS = {
    "governors_cadence_steps": "governorsCadence",
    "governors_wait_seconds": "governorsWait",
    "role_preference_randomness": "roleRandomness",
}
SENZA_CONTROLLO = {
    # Non esposti nel pannello: la GUI non li invia e vale il default del
    # backend, che e' esattamente il comportamento voluto.
    "cell_proposal_top_k",
    "role_distribution",
    # Il sito colonia parte in modalita' "auto" e le coordinate si spediscono
    # SOLO in modalita' "manual" (ScenarioPanel.tsx, ramo `colonySiteMode`):
    # il 180/90 del pannello e' il centro mappa proposto all'utente, non un
    # default che sovrascriva la scelta automatica del sito sicuro.
    "colony_start_x",
    "colony_start_y",
    # Stringhe di identita' della run, senza un valore "giusto" condiviso.
    "run_name",
    "selected_scenario_id",
    "assignments",
}


def _stati_react() -> dict[str, str]:
    testo = PANNELLO.read_text(encoding="utf-8")
    return {
        nome: valore.strip()
        for nome, valore in re.findall(
            r"const \[(\w+), set\w+\] = useState(?:<[^>]*>)?\(([^;]*?)\);", testo
        )
    }


def _camel(nome: str) -> str:
    parti = nome.split("_")
    return parti[0] + "".join(p.capitalize() for p in parti[1:])


def _numero(grezzo: str) -> float | None:
    try:
        return float(grezzo)
    except ValueError:
        return None


def _atteso(valore) -> str:
    if isinstance(valore, bool):
        return "true" if valore else "false"
    return str(valore)


@pytest.mark.parametrize(
    "campo",
    [
        f.name
        for f in dataclasses.fields(ManualConfigOptions)
        if f.name not in SENZA_CONTROLLO
    ],
)
def test_il_pannello_react_parte_dal_default_python(campo: str) -> None:
    stati = _stati_react()
    py = getattr(ManualConfigOptions(), campo)
    nome_react = ALIAS.get(campo, _camel(campo))
    if nome_react not in stati:
        pytest.skip(f"{campo}: nessuno stato React omonimo ({nome_react})")
    grezzo = stati[nome_react].strip().strip('"')

    if py is None or (isinstance(py, int) and py == -1 and campo.startswith("initial_")):
        # `None` (nessun valore) e DOTAZIONE_DERIVATA (-1, «ricavala dalla
        # popolazione») non hanno una controparte letterale: il pannello
        # mostra la stringa vuota o chiama `dotazioneIniziale`, che e'
        # verificata dal proprio test.
        return

    atteso = _numero(_atteso(py))
    ottenuto = _numero(grezzo)
    if atteso is not None and ottenuto is not None:
        assert ottenuto == pytest.approx(atteso), (
            f"{campo}: python={py!r} ma il pannello React parte da {grezzo!r}"
        )
    else:
        assert grezzo == _atteso(py), (
            f"{campo}: python={py!r} ma il pannello React parte da {grezzo!r}"
        )


def test_la_dotazione_derivata_usa_gli_stessi_rapporti() -> None:
    """`dotazioneIniziale` in TSX deve rispecchiare `dotazione_iniziale` in Python."""
    from src.agents.build_policy import COLONISTS_PER_STRUCTURE
    from src.experiments.manual_config import MARGINE_DOTAZIONE_INIZIALE
    from src.world.structures import StructureType

    testo = PANNELLO.read_text(encoding="utf-8")
    margine = float(
        re.search(r"const MARGINE_DOTAZIONE_INIZIALE = ([\d.]+);", testo).group(1)
    )
    assert margine == pytest.approx(MARGINE_DOTAZIONE_INIZIALE)

    attesi = {
        "habitat": COLONISTS_PER_STRUCTURE[StructureType.HABITAT],
        "greenhouse": COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE],
        "solarArray": COLONISTS_PER_STRUCTURE[StructureType.SOLAR_ARRAY],
        "oxygenPlant": COLONISTS_PER_STRUCTURE[StructureType.OXYGEN_PLANT],
        "storageDepot": COLONISTS_PER_STRUCTURE[StructureType.STORAGE_DEPOT],
    }
    for chiave, divisore in attesi.items():
        trovato = re.search(rf"{chiave}: Math\.max\(1, Math\.ceil\(\w+ / (\d+)\)\)|{chiave}: Math\.ceil\(\w+ / (\d+)\)", testo)
        assert trovato, f"{chiave}: rapporto non trovato nel pannello"
        valore = int(trovato.group(1) or trovato.group(2))
        assert valore == divisore, (
            f"{chiave}: il pannello divide per {valore}, COLONISTS_PER_STRUCTURE dice {divisore}"
        )
