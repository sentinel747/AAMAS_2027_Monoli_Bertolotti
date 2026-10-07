# -*- coding: utf-8 -*-
"""Quanto contesto riceve il governatore: il contesto come variabile sperimentale.

**Perche' esiste.** Il braccio `llm` che batte la baseline non dice ancora
*che cosa* abbia aggiunto il modello. Puo' aver ragionato sui numeri, oppure
puo' aver riconosciuto il dominio --- una colonia che muore di sete --- e
applicato ricette che gia' conosceva. Sono due capacita' diverse e un solo
prompt non le separa. Qui il prompt diventa una scala a quattro gradini, dal
piu' informato al piu' cieco, e la differenza fra i gradini e' misurabile.

    completo   ruolo, come decidono i coloni, glossario dei pilastri,
               costituzione di riferimento, avvertenze sullo scenario
    senza_aiuti  come sopra, ma senza costituzione e senza avvertenze:
               il dominio resta, i suggerimenti spariscono
    nomi_veri  niente ruolo, niente spiegazione, niente glossario. Restano i
               nomi reali di indicatori e pilastri: il dominio e' inferibile,
               nessuno lo dichiara
    cieco      indicatori e pilastri rinominati in etichette neutre. Il
               modello vede nove numeri e cinque leve, e non sa di che cosa
               parlino

**Il gradino cieco richiede una traduzione, non solo un prompt.** Il modello
risponde nelle etichette neutre e il kernel conosce solo i nomi veri: la
proposta va ritradotta *prima* di `parse_policy`, altrimenti ogni regola
verrebbe scartata come "indicatore sconosciuto" e il braccio misurerebbe la
traduzione mancante invece della cecita'.

Le etichette neutre sono deliberatamente **prive di ordine suggestivo**: non
`indicatore_1..9` in ordine di importanza, ma un rimescolamento fisso, cosi'
che la posizione nell'elenco non riveli nulla. Fisso e non casuale, perche'
due run con lo stesso seme devono ricevere lo stesso prompt.
"""

from __future__ import annotations

from src.governors.policy import INDICATORS, PILLAR_BY_NAME

#: I quattro gradini, dal piu' informato al piu' cieco.
LIVELLI = ("completo", "senza_aiuti", "nomi_veri", "cieco")

#: Nome vero -> etichetta neutra. L'ordine e' rimescolato una volta e congelato.
ALIAS_INDICATORI: dict[str, str] = {
    "power_coverage": "voce_1",
    "food_per_occupant": "voce_2",
    "structure_integrity": "voce_3",
    "minerals_per_occupant": "voce_4",
    "water_per_occupant": "voce_5",
    "occupants": "voce_6",
    "ice_per_occupant": "voce_7",
    "oxygen_per_occupant": "voce_8",
    "material_per_occupant": "voce_9",
}

#: Nome vero -> etichetta neutra, per i cinque pilastri pesabili.
ALIAS_PILASTRI: dict[str, str] = {
    "build": "leva_A",
    "sustenance": "leva_B",
    "explore": "leva_C",
    "life": "leva_D",
    "resources": "leva_E",
}

#: Le tabelle inverse, per ritradurre la risposta.
NOME_DA_ALIAS_INDICATORE = {alias: nome for nome, alias in ALIAS_INDICATORI.items()}
NOME_DA_ALIAS_PILASTRO = {alias: nome for nome, alias in ALIAS_PILASTRI.items()}

assert set(ALIAS_INDICATORI) == set(INDICATORS), "alias e vocabolario divergono"
assert set(ALIAS_PILASTRI) == set(PILLAR_BY_NAME), "alias e pilastri divergono"


def normalizza(livello: str | None) -> str:
    """Un livello sconosciuto vale `completo`, che e' il comportamento storico."""
    valore = (livello or "completo").strip().lower()
    return valore if valore in LIVELLI else "completo"


def e_cieco(livello: str | None) -> bool:
    return normalizza(livello) == "cieco"


def mostra_dominio(livello: str | None) -> bool:
    """Il prompt dichiara che si tratta di una colonia marziana?"""
    return normalizza(livello) in ("completo", "senza_aiuti")


def mostra_aiuti(livello: str | None) -> bool:
    """Il prompt include costituzione di riferimento e avvertenze?"""
    return normalizza(livello) == "completo"


def maschera_indicatori(indicatori: dict[str, object], livello: str | None) -> dict[str, object]:
    """Il quadro con le chiavi che il modello vedra'."""
    if not e_cieco(livello):
        return dict(indicatori)
    return {
        ALIAS_INDICATORI.get(nome, nome): valore
        for nome, valore in indicatori.items()
    }


def maschera_dizionario(valori: dict[str, object], prefisso: str, livello: str | None) -> dict[str, object]:
    """Rinomina le chiavi in `prefisso_1..n`, conservando i valori.

    Serve alle metriche di colonia, alle medie di popolazione e ai totali di
    struttura, che nel gradino cieco non possono restare con i loro nomi ---
    `food_stock` o `colony_prosperity_index` dichiarano il dominio quanto una
    frase --- ma che nemmeno vanno tolte: toglierle confonderebbe *cieco* con
    *meno informato*, e la differenza fra i due gradini smetterebbe di essere
    attribuibile al solo contesto. L'ordine e' alfabetico e quindi stabile fra
    run: due esecuzioni con lo stesso seme ricevono lo stesso prompt.
    """
    if not e_cieco(livello):
        return dict(valori)
    return {
        f"{prefisso}_{indice}": valori[chiave]
        for indice, chiave in enumerate(sorted(valori), start=1)
    }


def traduci_proposta(grezza: object, livello: str | None) -> object:
    """Riporta ai nomi veri una proposta scritta nelle etichette neutre.

    Tollerante per progetto: una chiave che non e' un alias passa invariata e
    verra' scartata piu' avanti da `parse_policy`, che e' il posto dove gli
    scarti si contano. Qui non si valida, si traduce.
    """
    if not e_cieco(livello) or not isinstance(grezza, dict):
        return grezza

    regole = grezza.get("policy")
    if not isinstance(regole, list):
        return grezza

    tradotte = []
    for regola in regole:
        if not isinstance(regola, dict):
            tradotte.append(regola)
            continue
        nuova = dict(regola)
        condizione = nuova.get("if")
        if isinstance(condizione, dict):
            nome = condizione.get("indicator")
            if isinstance(nome, str) and nome in NOME_DA_ALIAS_INDICATORE:
                condizione = dict(condizione)
                condizione["indicator"] = NOME_DA_ALIAS_INDICATORE[nome]
                nuova["if"] = condizione
        pesi = nuova.get("weights")
        if isinstance(pesi, dict):
            nuova["weights"] = {
                NOME_DA_ALIAS_PILASTRO.get(chiave, chiave): valore
                for chiave, valore in pesi.items()
            }
        tradotte.append(nuova)

    fuori = dict(grezza)
    fuori["policy"] = tradotte
    return fuori


def maschera_nomi(testo: str, livello: str | None) -> str:
    """Sostituisce i nomi veri con le etichette neutre dentro un testo gia' reso.

    **Perche' serve solo all'amministratore.** Il governatore riceve i dati come
    dizionari e bastano `maschera_indicatori` e `maschera_dizionario`.
    L'amministratore riceve in piu' due testi gia' composti --- la catena del
    governo e la politica risolta sulle sue celle --- che contengono i nomi veri
    degli indicatori e delle categorie perche' sono stampe di regole. Cieco sui
    dati e vedente sulla legge sarebbe un gradino che non esiste: gli basterebbe
    leggere la propria legge per sapere che `food_per_occupant` parla di cibo.

    Fuori dal gradino cieco e' l'identita'. Sostituisce parole intere, cosi' che
    `occupants` non tocchi `occupants_per_cell` se un giorno esistesse.
    """
    if not e_cieco(livello) or not testo:
        return testo
    import re as _re

    tabella = {**ALIAS_INDICATORI, **ALIAS_PILASTRI}
    # I nomi piu' lunghi per primi: `material_per_occupant` non deve essere
    # toccato dalla sostituzione di `occupants`.
    schema = _re.compile(
        r"\b(" + "|".join(sorted(map(_re.escape, tabella), key=len, reverse=True)) + r")\b"
    )
    return schema.sub(lambda m: tabella[m.group(1)], testo)
