# -*- coding: utf-8 -*-
"""Il prompt italiano di oggi, blindato parola per parola.

**Perche' un test che confronta testo.** La campagna incrociata sui prompt
(parola, gerarchia, lingua) confronta bracci che differiscono per una cosa sola.
Il braccio di riferimento e' il prompt gia' usato da tutte le campagne
dell'archivio: se il lavoro di parametrizzazione lo cambia anche di una virgola,
i suoi numeri smettono di essere confrontabili con quelli gia' raccolti e la
cosa non si vede da nessuna parte --- una run finisce, scrive i suoi artifact e
sembra identica a quelle di ieri.

Questi confronti falliscono rumorosamente in quel caso. Quando il cambiamento e'
voluto, si rigenerano i file d'oro con

    python tests/governors/test_prompt_invariato.py --rigenera

e il diff che ne esce e' il documento di cio' che e' cambiato: va letto, non
accettato d'ufficio.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ORO = Path(__file__).parent / "oro"

from dataclasses import dataclass  # noqa: E402

from src.agents import pillars  # noqa: E402
from src.governors.administration import (  # noqa: E402
    _riga_precedente,
    _voce,
    catena_del_governo,
    costruisci_prompt,
    politica_risolta,
)
from src.governors.llm_arm import build_governor_prompt  # noqa: E402
from src.governors.observation import ColonyPicture  # noqa: E402
from src.governors.policy import (  # noqa: E402
    INDICATORS,
    Bounds,
    Condition,
    Policy,
    Rule,
)
from src.governors.testi_prompt import frasi  # noqa: E402
from src.governors.varianti_prompt import SIGLE, variante_da_sigla  # noqa: E402

LIVELLI_ATTESI = ("completo", "senza_aiuti", "nomi_veri", "cieco")

#: Le sei caselle del disegno incrociato. Congelarne una sola lasciava le altre
#: cinque --- quindici run --- senza alcun testo di riferimento: una modifica ai
#: testi fra il seme 3 e il seme 7 della stessa casella non sarebbe stata vista
#: da niente, e i cinque semi avrebbero smesso di essere repliche.
CASELLE = tuple(sorted(SIGLE.values()))


def _quadro() -> ColonyPicture:
    """Un quadro fisso: qui conta che sia sempre lo stesso, non che sia realistico."""
    return ColonyPicture(
        step=180,
        population=59,
        n_cells=4,
        metrics={"colony_prosperity_index": 0.71, "food_stock": 212.0},
        indicators={
            nome: {"mean": 1.0, "std": 0.5, "min": 0.1, "max": 2.0}
            for nome in INDICATORS
        },
        population_stats={"health": 0.91, "hydration": 0.77},
        structures={"greenhouse": 12, "shelter": 48},
    )


def _bounds() -> Bounds:
    return Bounds(weight_min=0.25, weight_max=4.0)


@dataclass
class _DecisionePrecedente:
    """Quel tanto di decisione che serve a `_riga_precedente`."""

    policy: Policy | None
    rationale: str = "il distretto reggeva"
    a_vuoto: bool = False


def _politica_di_prova() -> Policy:
    return Policy(rules=(
        Rule(condition=Condition("food_per_occupant", "<", 2.0),
             weights={pillars.P_SUSTENANCE: 2.0}),
        Rule(condition=None,
             weights={pillars.P_SUSTENANCE: 1.0, pillars.P_BUILD: 1.0}),
    ))


def _prompt_amministratore(sigla: str = "0", livello: str = "completo") -> str:
    """Il prompt dell'amministratore montato dai suoi VERI produttori.

    La versione precedente passava un `risolta` e una `catena` scritti a mano e
    un `precedente` vuoto: le tre funzioni che generano quei pezzi non venivano
    mai chiamate, ed erano esattamente le tre che stampavano italiano dentro il
    prompt inglese e nomi veri dentro quello cieco.
    """
    variante = variante_da_sigla(sigla)
    politica = _politica_di_prova()
    return costruisci_prompt(
        quadro=_quadro(),
        risolta=politica_risolta(None, None, [(10, 20), (10, 21)], livello,
                                 _voce(frasi(variante.lingua), "amm_cella",
                                       livello == "cieco"),
                                 variante.lingua),
        bounds=_bounds(),
        distretto=2,
        indicatori_colonia={n: {"mean": 1.0} for n in INDICATORS},
        # Le cause sono quelle che il motore scrive davvero (`dead_agents.jsonl`),
        # non etichette inventate: al gradino cieco vanno mascherate.
        morti={"starvation": 3, "dehydration": 1},
        precedente=_riga_precedente(
            _DecisionePrecedente(policy=politica, a_vuoto=True),
            variante.lingua, livello),
        catena=catena_del_governo(politica, livello, variante.lingua),
        variante=variante,
        livello=livello,
    )


def prompt_attuali() -> dict[str, str]:
    """Tutti i testi che un modello riceve davvero, per nome di file d'oro."""
    testi = {
        f"governatore_{liv}.txt": build_governor_prompt(_quadro(), _bounds(), liv)
        for liv in LIVELLI_ATTESI
    }
    for sigla in CASELLE:
        testi[f"governatore_casella_{sigla}.txt"] = build_governor_prompt(
            _quadro(), _bounds(), "completo", variante_da_sigla(sigla))
    testi["amministratore.txt"] = _prompt_amministratore()
    # Le quattro combinazioni in cui i difetti di oggi si vedevano: la lingua
    # del testo e il gradino che toglie il dominio, incrociati.
    for sigla in ("0", "D"):
        for livello in ("completo", "cieco"):
            testi[f"amministratore_{sigla}_{livello}.txt"] = _prompt_amministratore(
                sigla, livello)
    return testi


@pytest.mark.parametrize("nome", sorted(prompt_attuali()))
def test_il_prompt_non_e_cambiato(nome: str) -> None:
    atteso = ORO / nome
    assert atteso.exists(), (
        f"manca il file d'oro {atteso}: rigenera con "
        "`python tests/governors/test_prompt_invariato.py --rigenera`"
    )
    ottenuto = prompt_attuali()[nome]
    assert ottenuto == atteso.read_text(encoding="utf-8"), (
        f"il prompt {nome} e' cambiato. Se il cambiamento e' voluto rigenera i "
        "file d'oro e LEGGI il diff prima di accettarlo: i bracci gia' eseguiti "
        "sono stati misurati con il testo vecchio."
    )


def test_i_quattro_gradini_restano_diversi_fra_loro() -> None:
    """Un errore di parametrizzazione che li appiattisse non si vedrebbe altrove."""
    testi = {liv: build_governor_prompt(_quadro(), _bounds(), liv) for liv in LIVELLI_ATTESI}
    assert len(set(testi.values())) == len(LIVELLI_ATTESI)


def _rigenera() -> None:
    ORO.mkdir(exist_ok=True)
    for nome, testo in prompt_attuali().items():
        (ORO / nome).write_text(testo, encoding="utf-8")
        print(f"scritto {ORO / nome} ({len(testo)} caratteri)")


if __name__ == "__main__":
    if "--rigenera" in sys.argv:
        _rigenera()
    else:
        print(__doc__)


def test_nessuna_casella_inglese_contiene_italiano() -> None:
    """Il difetto del 14 settembre: tre testi cablati in italiano.

    La politica risolta, la catena del governo e la riga della decisione
    precedente nascevano da f-string dentro il codice invece che dal dizionario
    delle frasi, quindi nel prompt inglese uscivano in italiano --- e
    l'istruzione «if above you read NO INTERVENTION» puntava a un marcatore che
    in quel prompt non compariva.
    """
    import re

    spie = {"nessuna", "regola", "regole", "politica", "governo", "preferenze",
            "volta", "scorsa", "avevi", "riscritto", "celle", "cella", "coloni",
            "scatta", "intervento", "emanato", "lasciate", "quindi", "tutti",
            "valgono", "cambia", "niente", "motivo", "dato"}
    for sigla in CASELLE:
        variante = variante_da_sigla(sigla)
        if variante.lingua != "en":
            continue
        for livello in ("completo", "cieco"):
            testo = _prompt_amministratore(sigla, livello).lower()
            parole = set(re.findall("[a-zà-ÿ']+", testo))
            assert not (spie & parole), (
                f"la casella {sigla} a livello {livello} contiene italiano: "
                f"{sorted(spie & parole)}"
            )


def test_il_gradino_cieco_non_restituisce_i_nomi_veri() -> None:
    """La fuga stava nella riga della decisione precedente.

    La proposta dell'amministratore viene ritradotta nei nomi veri prima del
    parser, quindi la policy conservata li contiene: ristampata grezza alla
    tornata dopo, restituiva il dominio che il gradino aveva tolto a quella
    prima. Un braccio cieco solo al primo giro non misura la cecita'.
    """
    for sigla in ("0", "D"):
        testo = _prompt_amministratore(sigla, "cieco")
        for nome in list(INDICATORS)[:5]:
            assert nome not in testo, (
                f"il prompt cieco della casella {sigla} nomina {nome}")
        for pilastro in ("sustenance", "explore", "build"):
            assert pilastro not in testo, (
                f"il prompt cieco della casella {sigla} nomina {pilastro}")
