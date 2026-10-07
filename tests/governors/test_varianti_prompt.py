# -*- coding: utf-8 -*-
"""Le sei caselle del disegno incrociato sui prompt.

I modi in cui questa campagna puo' rompersi in silenzio sono tre, e sono tutti
invisibili nel registro di una run finita:

1. la variante non arriva fino al prompt --- la run si chiama «A» e manda il
   testo di «0»;
2. governo e amministratori usano parole diverse, e il braccio misura la
   discordanza fra i due testi invece della parola;
3. il contratto con la macchina cambia insieme alla prosa, e allora non e' piu'
   la lingua a essere confrontata ma due protocolli diversi.

Qui falliscono rumorosamente.
"""

from __future__ import annotations

import re

import pytest

from src.governors.administration import costruisci_prompt
from src.governors.config import read_settings
from src.governors.llm_arm import build_governor_prompt
from src.governors.observation import ColonyPicture
from src.governors.policy import INDICATORS, PILLAR_BY_NAME, Bounds
from src.governors.varianti_prompt import (
    SIGLE,
    VariantePrompt,
    variante_da_sigla,
)

IMPLEMENTATE = ("0", "A", "B", "F", "C", "D")


@pytest.fixture
def quadro() -> ColonyPicture:
    return ColonyPicture(
        step=180,
        population=59,
        n_cells=4,
        metrics={"colony_prosperity_index": 0.71},
        indicators={
            nome: {"mean": 1.0, "std": 0.5, "min": 0.1, "max": 2.0} for nome in INDICATORS
        },
        population_stats={"health": 0.91},
        structures={"greenhouse": 12},
    )


@pytest.fixture
def bounds() -> Bounds:
    return Bounds(weight_min=0.25, weight_max=4.0)


def test_la_variante_predefinita_e_quella_storica() -> None:
    """Chi non sa che esistono le varianti riceve il prompt di sempre."""
    assert VariantePrompt().storica
    assert VariantePrompt().sigla == "0"
    assert variante_da_sigla("0") == VariantePrompt()


def test_ogni_sigla_va_e_torna() -> None:
    for chiave, sigla in SIGLE.items():
        variante = variante_da_sigla(sigla)
        assert variante.sigla == sigla
        assert (variante.lingua, variante.termine, variante.gerarchia) == chiave


@pytest.mark.parametrize("sbagliata", ["", "Z", "a b", "it"])
def test_una_sigla_inventata_e_un_errore(sbagliata: str) -> None:
    with pytest.raises(ValueError):
        variante_da_sigla(sbagliata)


@pytest.mark.parametrize("sigla", IMPLEMENTATE)
def test_i_prompt_delle_caselle_sono_tutti_diversi(sigla, quadro, bounds) -> None:
    testi = {
        s: build_governor_prompt(quadro, bounds, "completo", variante_da_sigla(s))
        for s in IMPLEMENTATE
    }
    assert len(set(testi.values())) == len(IMPLEMENTATE)


def test_la_parola_nuova_sostituisce_la_vecchia_ovunque(quadro, bounds) -> None:
    """«Categoria» non convive con «pilastro» ne' con «leva»: e' il punto."""
    testo = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla("A")).lower()
    assert "pilastr" not in testo
    assert "leva" not in testo and "leve" not in testo
    assert "categori" in testo


def test_la_parola_vecchia_resta_incoerente_come_era(quadro, bounds) -> None:
    """Il braccio di riferimento non va ripulito: e' il testo gia' misurato."""
    testo = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla("0")).lower()
    assert "pilastr" in testo
    assert "leve pesabili" in testo


def test_la_gerarchia_compare_solo_dove_e_chiesta(quadro, bounds) -> None:
    senza = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla("A"))
    con = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla("B"))
    assert "amministrator" not in senza.lower()
    assert "amministrator" in con.lower()
    # Dichiara un fatto, non detta una condotta: un'istruzione misurerebbe
    # l'obbedienza al prompt invece dell'effetto di sapere.
    for imperativo in ("scrivi regole piu'", "sii piu'", "lascia agli amministratori"):
        assert imperativo not in con.lower()


def test_la_gerarchia_non_entra_nei_gradini_senza_dominio(quadro, bounds) -> None:
    """Nominare distretti e amministratori al gradino cieco lo renderebbe vedente."""
    for livello in ("nomi_veri", "cieco"):
        testo = build_governor_prompt(quadro, bounds, livello, variante_da_sigla("B"))
        assert "amministrator" not in testo.lower()
        assert "distretto" not in testo.lower()


def test_il_gradino_cieco_resta_cieco_con_ogni_variante(quadro, bounds) -> None:
    for sigla in IMPLEMENTATE:
        testo = build_governor_prompt(quadro, bounds, "cieco", variante_da_sigla(sigla)).lower()
        for parola in ("marzian", "colon", "cella", "celle", "categorie di preferenza"):
            assert parola not in testo, f"{sigla} lascia passare «{parola}» al gradino cieco"


@pytest.mark.parametrize("sigla", IMPLEMENTATE)
def test_il_contratto_con_la_macchina_non_cambia_mai(sigla, quadro, bounds) -> None:
    """Nomi, operatori e chiavi del JSON sono gli stessi in ogni casella.

    E' cio' che rende confrontabili le caselle: parser, validatore e script di
    analisi non devono vedere la differenza.
    """
    testo = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla(sigla))
    for nome in INDICATORS:
        assert nome in testo
    for leva in PILLAR_BY_NAME:
        assert leva in testo
    for chiave in ('"policy"', '"if"', '"indicator"', '"op"', '"value"', '"weights"'):
        assert chiave in testo


def test_governo_e_amministratori_parlano_la_stessa_lingua(quadro, bounds) -> None:
    """Due vocabolari diversi misurerebbero la discordanza, non la parola."""
    variante = variante_da_sigla("A")
    amministratore = costruisci_prompt(
        quadro=quadro,
        risolta=["(10, 20): sustenance x2.0"],
        bounds=bounds,
        distretto=2,
        indicatori_colonia={n: {"mean": 1.0} for n in INDICATORS},
        morti={"fame": 3},
        catena="1. se food_per_occupant < 2.0 -> sustenance x2.0",
        variante=variante,
    ).lower()
    assert "pilastr" not in amministratore
    assert "categori" in amministratore


@pytest.mark.parametrize("sigla", ["F", "C", "D"])
def test_nel_prompt_inglese_non_resta_italiano(sigla, quadro, bounds) -> None:
    """Un prompt misto non misurerebbe la lingua ma la miscela.

    La costituzione di riferimento e' il posto in cui l'italiano e' rimasto piu'
    a lungo: entra nel prompt al solo gradino completo, che e' esattamente il
    gradino su cui gira questa campagna, e le sue due parole di collegamento
    («se», «altrimenti ogni colono...») venivano da un'altra funzione.
    """
    testo = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla(sigla)).lower()
    # Parole INTERE, non sottostringhe: la parola italiana puo' stare dentro
    # una inglese (coloni dentro colonists, celle dentro cells). Cercandola
    # come sottostringa il test accuserebbe l'inglese di essere italiano,
    # fallirebbe per il motivo sbagliato, e un test che fallisce a torto
    # viene disattivato --- portandosi via il controllo vero.
    presenti = set(re.findall("[a-zà-ÿ']+", testo))
    for parola in ("colono", "coloni", "governatore", "politica", "regola",
                   "peso", "unita'", "altrimenti", "soglia", "celle"):
        assert parola not in presenti, (
            f"{sigla} lascia passare l'italiano: {parola}")


@pytest.mark.parametrize("sigla", ["F", "C", "D"])
def test_il_prompt_inglese_dice_le_stesse_cose(sigla, quadro, bounds) -> None:
    """Traduzione, non riscrittura: gli stessi blocchi nello stesso ordine."""
    italiano = {"F": "0", "C": "A", "D": "B"}[sigla]
    en = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla(sigla))
    it = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla(italiano))
    # Stessa struttura: stesso numero di paragrafi e stessi blocchi di dati.
    assert en.count("\n\n") == it.count("\n\n")
    # Una traduzione molto piu' corta o molto piu' lunga sarebbe una riscrittura.
    assert 0.8 < len(en) / len(it) < 1.25


def test_la_gerarchia_esiste_in_tutte_e_due_le_lingue(quadro, bounds) -> None:
    con = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla("D"))
    senza = build_governor_prompt(quadro, bounds, "completo", variante_da_sigla("C"))
    assert "administrator" in con.lower()
    assert "administrator" not in senza.lower()


def test_anche_gli_amministratori_parlano_inglese(quadro, bounds) -> None:
    """Un governo inglese con distretti italiani sarebbe un sistema misto.

    E' il difetto piu' facile da non vedere: il prompt del governatore si
    guarda, quello degli amministratori no, e la campagna avrebbe misurato la
    lingua del solo livello centrale.
    """
    testo = costruisci_prompt(
        quadro=quadro,
        risolta=["(10, 20): sustenance x2.0"],
        bounds=bounds,
        distretto=2,
        indicatori_colonia={n: {"mean": 1.0} for n in INDICATORS},
        morti={"starvation": 3},
        catena="1. if food_per_occupant < 2.0 -> sustenance x2.0",
        variante=variante_da_sigla("D"),
    ).lower()
    presenti = set(re.findall("[a-zà-ÿ']+", testo))
    for parola in ("amministratore", "distretto", "coloni", "politica",
                   "governo", "celle", "regola", "peso", "soglia"):
        assert parola not in presenti, (
            f"l'amministratore inglese dice {parola}")
    assert "administrator" in testo


def test_le_due_lingue_hanno_le_stesse_chiavi() -> None:
    """Una chiave in meno lascerebbe un blocco vuoto in una lingua sola."""
    from src.governors.testi_prompt import FRASI

    assert set(FRASI["it"]) == set(FRASI["en"])


def test_la_variante_arriva_dalla_configurazione_al_proponente() -> None:
    """Il salto che non si vede: `--variante-prompt A` deve cambiare il testo."""
    config = {
        "governors": {
            "arm": "llm",
            "cadence_steps": 25,
            "prompt_variant": "B",
            "assignments": [{"provider": "fallback", "model": ""}],
        }
    }
    assert read_settings(config).prompt_variant == "B"
    assert read_settings({"governors": {"arm": "llm"}}).prompt_variant == "0"
