# -*- coding: utf-8 -*-
"""Lo strato amministrativo: fra il governo e le celle.

**Che cosa fa un amministratore.** E' l'assessore di una provincia che conta
piu' citta'. Il governo emana la politica; l'amministratore la legge gia'
*risolta sulle proprie celle* — quali regole vi scattano davvero e con quali
pesi — insieme alle metriche delle sole celle di sua competenza. Se gli va
bene, non tocca nulla e le celle applicano la politica del governo. Se non gli
va bene, riscrive per le sue celle e la sua catena prende il posto di quella
del governo, li' e soltanto li'.

**La cadenza non e' indipendente.** L'amministratore parla subito dopo il
governo, sullo stesso confine di tick: la politica non va dal governo alle
celle, va dal governo agli amministratori e da questi alle celle. Un
amministratore che deliberasse a un ritmo proprio starebbe correggendo una
politica diversa da quella che vede.

**Che cosa questo strato NON e'.** Non e' un secondo governatore con meno
celle. Riceve un input che il governatore non ha — la politica in vigore — e la
sua uscita e' una *modifica*, con l'astensione come esito legittimo e
frequente. Il conteggio delle astensioni e' esso stesso una misura: un
amministratore che non tocca mai nulla e' indistinguibile dal governo
centralizzato, ed e' proprio la domanda che il disegno pone.
"""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import asdict, dataclass, field, is_dataclass, replace

import numpy as np

from src.governors.apply import indicator_values
from src.governors.observation import ColonyPicture, _aggregate
from src.governors.context import ALIAS_INDICATORI, ALIAS_PILASTRI, e_cieco, \
    maschera_dizionario, maschera_indicatori, \
    maschera_nomi, normalizza as normalizza_livello, traduci_proposta
from src.governors.testi_prompt import frasi
from src.governors.varianti_prompt import VariantePrompt, parole
from src.governors.policy import (
    INDICATORS,
    indicatori_con_significato,
    MAX_RULES,
    PILLAR_BY_NAME,
    PILLAR_NAME_BY_INDEX,
    Bounds,
    Policy,
    parse_policy,
    rule_text,
)


def _esegui(coro):
    """Esegue una corutine da codice sincrono, ovunque questo giri.

    Il ciclo di simulazione e' sincrono e non ha un loop proprio, quindi la via
    normale e' `asyncio.run`. Ma le stesse shell girano anche dentro un server
    che un loop ce l'ha gia' (`state_store`), e li' `asyncio.run` solleva: in
    quel caso la tornata prende un loop suo su un thread dedicato. Senza questo
    ramo un guasto comparirebbe solo dall'interfaccia e non dalla riga di
    comando, cioe' nel posto dove si guarda di meno.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    scatola: dict[str, object] = {}

    def bersaglio() -> None:
        try:
            scatola["valore"] = asyncio.run(coro)
        except BaseException as errore:  # noqa: BLE001 - rilanciata sotto
            scatola["errore"] = errore

    filo = threading.Thread(target=bersaglio, name="amministratori")
    filo.start()
    filo.join()
    if "errore" in scatola:
        raise scatola["errore"]  # type: ignore[misc]
    return scatola.get("valore")


@dataclass(frozen=True)
class DecisioneAmministratore:
    """Che cosa un amministratore ha risposto su questo tick."""

    distretto: int
    #: `None` quando l'amministratore si astiene: le sue celle restano al governo.
    policy: Policy | None = None
    accettata: bool = True
    rationale: str = ""
    provider: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_s: float = 0.0
    celle: tuple[tuple[int, int], ...] = ()
    #: Riscrittura che, sulle celle del distretto e in questo istante, non fa
    #: niente: nessuna regola scatta, o scattano solo regole a pesi tutti 1. E'
    #: il terzo esito --- ne' ACCETTA ne' una correzione --- perche' la catena
    #: dell'amministratore sostituisce quella del governo, e una catena inerte
    #: lascia le celle alle preferenze pure. Sempre falso per un'astensione.
    a_vuoto: bool = False
    #: Coloni nel distretto al momento della decisione. E' il denominatore che
    #: alla misura dentro la run mancava: senza, si divide per celle-passo e
    #: un distretto di tre celle con due coloni pesa quanto uno con duecento.
    popolazione: int = 0
    #: `accept: true` E una politica allegata: il modello ha detto due cose, e
    #: la prima vince. Va scritto, perche' il 43 per cento delle accettazioni
    #: della v3 motiva con carenze e modifiche, e senza questo campo non si
    #: puo' sapere quante volte una riscrittura intesa e' stata ignorata.
    accettata_con_politica: bool = False
    #: Che cosa il parser ha scartato o riportato nei limiti (indicatori e
    #: pilastri fuori vocabolario, pesi fuori intervallo, regole cadute). Il
    #: governatore lo registra da sempre; l'amministratore no, e una regola
    #: con un indicatore inventato spariva senza traccia.
    scarti: dict = field(default_factory=dict)
    #: Guasto del fornitore: la chiamata non e' arrivata a destinazione o e'
    #: tornata con un errore. Le celle restano al governo, come per
    #: un'accettazione, ma NON e' ne' un'accettazione ne' una riscrittura: e'
    #: una tornata in cui quell'amministratore non ha parlato. Va contata a
    #: parte, altrimenti un endpoint spento si traveste da stile di governo.
    guasto: bool = False
    #: Risposta fuori schema: non un JSON, o un `accept: false` con una
    #: `policy` che il parser non ha potuto leggere (per esempio regole scritte
    #: come TESTO, «se food_per_occupant < 1.5 -> ...», invece che come
    #: oggetti). Le celle restano al governo, come per un'accettazione, ma
    #: NON e' un'accettazione: e' un amministratore che voleva riscrivere e
    #: non e' stato capito. Fino al 2026-09-06 le due cose erano indistinte.
    malformata: bool = False
    #: Il campo `policy` della risposta come il modello l'ha scritto, prima del
    #: parser (o la risposta intera se non era un dizionario). `None` per le
    #: accettazioni senza politica allegata e per i guasti del fornitore.
    grezza: object = None
    #: Decisione bounded redatta, presente solo negli arm SemIf/ibridi.
    semantic_decision: dict | None = None
    candidate_profile: str = ""
    #: Tutte le decisioni SemIf della tornata quando sono piu' di una (profilo
    #: a due livelli); `semantic_decision` resta l'ultima.
    semantic_decisions: list | None = None
    #: Vero soltanto quando SemIf ha scelto esplicitamente `deep_reasoning`.
    semantic_escalated: bool = False

    def __post_init__(self) -> None:
        # L'invariante sta qui e non nei chiamanti: un guasto non e' MAI
        # un'accettazione, e chi costruisce la decisione non deve poter
        # dimenticarsene. `frozen=True`, quindi si scrive per `object`.
        if self.guasto and self.accettata:
            object.__setattr__(self, "accettata", False)

    def to_json(self) -> dict:
        payload = {
            "district": self.distretto,
            "accepted_government_policy": bool(self.accettata),
            "provider_failure": bool(self.guasto),
            "accepted_with_policy": bool(self.accettata_con_politica),
            "rewrite_without_effect": bool(self.a_vuoto),
            "malformed": bool(self.malformata),
            "proposal_raw": self.grezza,
            "drops": dict(self.scarti),
            "population": int(self.popolazione),
            "rules": [rule_text(r) for r in self.policy.rules] if self.policy else [],
            "rationale": self.rationale,
            "provider": self.provider,
            "model": self.model,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "latency_s": round(self.latency_s, 3),
            "cells": [list(c) for c in self.celle],
        }
        if self.semantic_decision is not None:
            payload["semantic_decision"] = self.semantic_decision
            payload["candidate_profile"] = self.candidate_profile
            payload["semantic_escalated"] = bool(self.semantic_escalated)
        if self.semantic_decisions:
            payload["semantic_decisions"] = list(self.semantic_decisions)
        return payload


#: Quale coda di ogni vitale e' quella pericolosa. Non e' una scelta di stile:
#: per sazieta', idratazione, salute e morale il rischio sta in basso, per
#: fatica e stress in alto, e riportare la coda sbagliata darebbe all'
#: amministratore il caso MIGLIORE del suo distretto proprio dove serve il
#: peggiore.
_VITALI = {
    "health": "min",
    "hydration": "min",
    "satiety": "min",
    "morale": "min",
    "fatigue": "max",
    "stress": "max",
}


def quadro_di_distretto(
    step: int,
    metrics: dict,
    cells,
    agents,
    rows: np.ndarray,
    celle: list[tuple[int, int]],
) -> ColonyPicture:
    """Le stesse grandezze del quadro di colonia, ristrette a un distretto.

    Gli aggregati di colonia restano quelli veri e non ricalcolati sul solo
    distretto: un assessore legge i conti della provincia *e* quelli dello
    Stato, e senza i secondi non saprebbe se il suo problema e' locale.
    Cambiano la distribuzione degli indicatori, la popolazione e il conto delle
    celle, che sono le grandezze su cui le sue regole verranno valutate.
    """
    rows = np.asarray(rows, dtype=np.int64)
    insieme = set(celle)
    presenti = [
        int(r)
        for r in rows
        if (int(agents.y[int(r)]), int(agents.x[int(r)])) in insieme
    ]
    conteggio: dict[tuple[int, int], int] = {}
    for r in presenti:
        chiave = (int(agents.y[r]), int(agents.x[r]))
        conteggio[chiave] = conteggio.get(chiave, 0) + 1

    indicatori: dict[str, dict[str, float]] = {}
    if celle:
        ys = np.array([c[0] for c in celle], dtype=np.intp)
        xs = np.array([c[1] for c in celle], dtype=np.intp)
        for nome in INDICATORS:
            valori = np.asarray(indicator_values(nome, cells))[ys, xs]
            indicatori[nome] = _aggregate([float(v) for v in valori])

    statistiche: dict[str, float] = {}
    if presenti:
        indici = np.array(presenti, dtype=np.int64)
        for colonna, coda in _VITALI.items():
            valori = getattr(agents, colonna, None)
            if valori is None:
                continue
            fetta = np.asarray(valori)[indici]
            statistiche[f"{colonna}_mean"] = round(float(fetta.mean()), 3)
            # **La media non uccide nessuno: uccide la coda.** Gli indicatori di
            # cella arrivano gia' con min e max, i coloni arrivavano con la sola
            # media --- e in questo mondo si muore di fame mentre il margine
            # alimentare di colonia sta al tetto, cioe' proprio quando la media
            # e' tranquilla. Un amministratore che vede solo medie non puo'
            # vedere il proprio distretto morire, e la sua accettazione non e'
            # un giudizio ma un'assenza di dati.
            statistiche[f"{colonna}_{coda}"] = round(
                float(fetta.min() if coda == "min" else fetta.max()), 3
            )

    return ColonyPicture(
        step=int(step),
        population=len(presenti),
        metrics={str(k): float(v) for k, v in (metrics or {}).items()},
        indicators=indicatori,
        population_stats=statistiche,
        structures={},
        n_cells=len(celle),
    )


def _regola_applicata(policy: Policy, cells, y: int, x: int):
    """La PRIMA regola della catena che scatta su questa cella, o `None`.

    E' la stessa semantica di `apply_policy` (prima che scatta vince), scritta
    per una cella sola: serve a raccontare la catena, non ad applicarla.
    """
    for regola in policy.rules:
        if regola.condition is None:
            return regola
        valori = np.asarray(indicator_values(regola.condition.indicator, cells))
        valore = float(valori[y, x])
        scatta = (
            valore < regola.condition.value
            if regola.condition.op == "<"
            else valore > regola.condition.value
        )
        if scatta:
            return regola
    return None


def _pesi_tutti_uno(regola) -> bool:
    return all(abs(float(w) - 1.0) < 1e-9 for w in regola.weights.values())


def riscrittura_a_vuoto(policy: Policy | None, cells, celle) -> bool:
    """Vero se la catena, su queste celle e adesso, non cambia niente.

    **Perche' va detto.** La catena dell'amministratore sostituisce quella del
    governo: una catena in cui nessuna regola scatta, o scattano solo regole a
    pesi tutti 1, lascia le celle alle preferenze pure. Non e' un'accettazione
    e non e' una correzione; e' un decentramento che toglie la legge senza
    metterne un'altra. Misurato a mano sui registri della campagna del
    2026-09-02: dal 19 al 33 per cento delle riscritture dello scritto. Il
    giudizio e' istantaneo e non predice: una regola puo' scattare piu' tardi.
    """
    if policy is None or not policy.rules or not celle:
        return True
    for y, x in celle:
        applicata = _regola_applicata(policy, cells, int(y), int(x))
        if applicata is not None and not _pesi_tutti_uno(applicata):
            return False
    return True


def _voce(f: dict, chiave: str, cieco: bool) -> str:
    """La frase vedente o quella neutra, che al gradino cieco e' un'altra frase.

    Non basta mascherare i nomi degli indicatori: queste righe nominavano i
    coloni e il governo, cioe' il dominio, in chiaro.
    """
    return f[chiave + ("_neutro" if cieco else "")]


def politica_risolta(policy: Policy | None, cells, celle: list[tuple[int, int]],
                     livello: str = "completo", parola: str = "cella",
                     lingua: str = "it") -> list[str]:
    """Che cosa il governo dice, cella per cella, in questo distretto.

    Non la politica intera: l'amministratore non deve dedurre quale regola lo
    riguardi. Riceve gia' l'esito della catena sulle SUE celle, che e' l'unica
    parte che dovra' eventualmente correggere.
    """
    f = frasi(lingua)
    cieco = e_cieco(livello)
    if policy is None or not policy.rules or not celle:
        return [_voce(f, "vista_pura", cieco).format(parola=parola, cella=list(c))
                for c in celle]

    righe: list[str] = []
    for y, x in celle:
        applicata = _regola_applicata(policy, cells, y, x)
        if applicata is None:
            righe.append(_voce(f, "vista_nulla", cieco).format(
                parola=parola, cella=[y, x]))
        elif _pesi_tutti_uno(applicata):
            # **Un moltiplicatore 1 non e' una politica, e non deve leggersi
            # come tale.** Le catene del governo finiscono quasi sempre con una
            # regola incondizionata a pesi tutti 1: misurato sulla campagna,
            # dal 18 al 24 per cento delle celle-passo "governate" ricade li'.
            # Stampata come le altre --- "sempre -> sustenance x1, resources x1,
            # ..." --- quella riga sembra una decisione, e l'amministratore che
            # la accetta crede di avallare una politica mentre sta avallando il
            # nulla.
            righe.append(_voce(f, "vista_pesi_uno", cieco).format(
                parola=parola, cella=[y, x],
                regola=maschera_nomi(rule_text(applicata, lingua), livello)))
        else:
            pesi = ", ".join(
                f"{PILLAR_NAME_BY_INDEX[p]} x{w:g}"
                for p, w in sorted(applicata.weights.items())
            )
            righe.append(_voce(f, "vista_regola", cieco).format(
                parola=parola, cella=[y, x],
                regola=maschera_nomi(rule_text(applicata, lingua), livello),
                pesi=maschera_nomi(pesi, livello)))
    return righe


def catena_del_governo(policy: Policy | None, livello: str = "completo",
                       lingua: str = "it") -> str:
    """La legge intera, numerata, nell'ordine in cui il governo l'ha scritta.

    **Perche' oltre alla vista risolta e non al suo posto.** La vista risolta
    dice all'amministratore che cosa gli succede; questa gli dice da che cosa
    puo' ripartire. Fino al 2026-09-05 mancava: una regola che oggi non scatta
    sulle sue celle non compariva affatto, e nemmeno l'ordine della catena.
    Poteva accettare o riscrivere da zero, mai ADATTARE --- cambiare una soglia,
    un peso, l'ordine --- che e' la correzione che il disegno chiede. L'ordine
    e' numerato perche' e' l'ordine a decidere chi affama chi.
    """
    if policy is None or not policy.rules:
        return _voce(frasi(lingua), "catena_vuota", e_cieco(livello))
    return "\n".join(
        f"  {i}. {maschera_nomi(rule_text(regola, lingua), livello)}"
        for i, regola in enumerate(policy.rules, start=1)
    )


def _riga_morti(morti: dict | None, popolazione: int, f: dict, coda: str = "",
                livello: str = "completo") -> str:
    """Chi e' morto nel distretto dall'ultima tornata, e di che cosa.

    **E' l'unica voce di ESITO che l'amministratore riceve.** Tutto il resto
    sono giacenze e medie, cioe' lo stato del magazzino: un distretto puo'
    avere indicatori tranquilli e coloni che muoiono di fame, ed e' esattamente
    quello che succede in questo mondo --- il margine alimentare di colonia
    resta al tetto per tutta la run mentre le morti per fame si contano a
    migliaia. Senza questa riga, accettare la politica del governo non e' un
    giudizio sbagliato: e' l'unico giudizio possibile con i dati forniti.
    """
    if not morti:
        return f["amm_morti_nessuno" + coda]
    totale = sum(int(v) for v in morti.values())
    # Le cause sono dominio: al gradino cieco diventano `causa_1..n`, come gia'
    # accade nel quadro del governatore. L'ordine e' alfabetico e quindi stabile
    # fra esecuzioni, altrimenti due run con lo stesso seme riceverebbero due
    # prompt diversi.
    morti = maschera_dizionario(dict(morti), "causa", livello)
    dettaglio = ", ".join(f"{c}: {int(n)}" for c, n in sorted(morti.items(), key=lambda kv: -kv[1]))
    quota = (f["amm_morti_quota" + coda].format(quota="%.0f%%" % (totale / popolazione * 100))
             if popolazione else "")
    return f["amm_morti" + coda].format(totale=totale, quota=quota, dettaglio=dettaglio)


def _morti_del_distretto(morti_per_cella: dict | None, celle: list) -> dict:
    """Somma per causa i decessi delle celle di un distretto."""
    if not morti_per_cella:
        return {}
    totale: dict[str, int] = {}
    for cella in celle:
        for causa, quanti in (morti_per_cella.get(tuple(cella)) or {}).items():
            totale[str(causa)] = totale.get(str(causa), 0) + int(quanti)
    return totale


def _riga_precedente(decisione, lingua: str = "it",
                     livello: str = "completo") -> str:
    """Che cosa questo amministratore aveva deciso la volta scorsa.

    **Senza memoria non c'e' correzione.** Il governatore riceve il consuntivo
    delle proprie regole e puo' accorgersi che una non cattura niente; l'
    amministratore ripartiva ogni volta da zero, quindi non poteva accorgersi
    di aver accettato per dieci tornate mentre il distretto peggiorava. Una
    riga, e la decisione diventa una serie invece di un istante.
    """
    if decisione is None:
        return ""
    f = frasi(lingua)
    cieco = e_cieco(livello)
    if decisione.policy is None:
        return _voce(f, "prec_accettato", cieco).format(
            motivo=decisione.rationale[:200])
    # **La maschera serve anche qui, ed e' il punto in cui mancava.** La
    # proposta dell'amministratore viene ritradotta nei nomi veri prima del
    # parser, quindi la policy conservata li contiene: ristampata grezza,
    # questa riga restituiva alla tornata dopo il dominio che il gradino cieco
    # aveva tolto a quella prima.
    regole = "; ".join(maschera_nomi(rule_text(r, lingua), livello)
                       for r in decisione.policy.rules)
    riga = _voce(f, "prec_riscritto", cieco).format(regole=regole)
    if getattr(decisione, "a_vuoto", False):
        # Detto subito, perche' e' l'unico modo in cui l'amministratore puo'
        # accorgersi di aver tolto la legge senza metterne un'altra.
        riga += _voce(f, "prec_a_vuoto", cieco)
    return riga


def costruisci_prompt(
    quadro: ColonyPicture,
    risolta: list[str],
    bounds: Bounds,
    distretto: int,
    indicatori_colonia: dict | None = None,
    morti: dict | None = None,
    precedente: str = "",
    catena: str = "",
    variante: VariantePrompt | None = None,
    livello: str = "completo",
) -> str:
    """Il prompt dell'amministratore: il governo, le sue celle, e la facolta' di dire di no.

    La variante del prompt arriva fin qui perche' il vocabolario dei due livelli
    deve restare uno solo: un governatore che pesa «categorie» e amministratori
    che pesano «pilastri» misurerebbero la discordanza fra i due testi invece
    della parola. La gerarchia non li riguarda --- che sopra di loro ci sia un
    governo lo sanno gia', ed e' la prima riga del loro prompt.
    """
    variante = variante or VariantePrompt()
    p = parole(variante)
    f = frasi(variante.lingua)
    # **Il gradino cieco vale per tutte e due le parti del prompt.** Al livello
    # locale il dominio non sta solo nei dati: sta nella legge che
    # l'amministratore riceve gia' stampata, e sta nelle parole del ruolo. Cieco
    # sui numeri e vedente sulla legge non e' un gradino, e' una svista.
    livello = normalizza_livello(livello)
    cieco = e_cieco(livello)
    coda = "_neutro" if cieco else ""
    # Al gradino cieco la cosa pesata si chiama «leva», come per il governatore:
    # le etichette neutre sono `leva_A..leva_E` e sono congelate. Le chiavi
    # dell'amministratore dicono «pilastro» anche nella variante storica, quindi
    # qui si prendono quelle del governo, che sono la famiglia «leva»: dire
    # «un pilastro che vuoi proteggere» accanto a `leva_A` sarebbe incoerente.
    if cieco:
        base = parole(VariantePrompt(lingua=variante.lingua))
        p = dict(base)
        for chiave in ("pesabili", "una_da_proteggere", "va_alzata", "segnaposto"):
            p[chiave + "_amm"] = base[chiave]
    nomi = (
        {a: "" for a in sorted(ALIAS_INDICATORI.values())} if cieco
        else indicatori_con_significato(variante.lingua)
    )
    indicatori = "\n".join(
        (f"- {n}" if cieco else f"- {n}: {m}") for n, m in nomi.items()
    )
    categorie = ", ".join(sorted(ALIAS_PILASTRI.values()) if cieco else sorted(PILLAR_BY_NAME))
    return (
        (f["amm_ruolo_neutro"].format(celle=quadro.n_cells) if cieco
         else f["amm_ruolo"].format(distretto=distretto, celle=quadro.n_cells))
        # La legge intera prima del suo effetto: senza, l'amministratore vede
        # solo l'articolo che lo tocca e non puo' correggere gli altri.
        + f["amm_catena" + coda].format(catena=catena)
        + "\n".join(f"  {r}" for r in risolta)
        + "\n\n"
        # **Le due strade vanno presentate alla pari.** La versione precedente
        # apriva con "se ti va bene, non toccare nulla": accettare era la prima
        # opzione, la piu' breve e quella senza lavoro da fare. Misurato sulla
        # campagna: 95 per cento di accettazioni sotto un governo, 100 per cento
        # senza governo. Qui le due scelte sono simmetriche e nessuna e' il
        # default --- il che NON e' un invito a riscrivere: accettare una
        # politica adeguata resta l'esito giusto, e va detto.
        + f["amm_scelte" + coda].format(step=quadro.step, popolazione=quadro.population)
        + _riga_morti(morti, quadro.population, f, coda, livello)
        + (f"{precedente}\n" if precedente else "")
        # Il confronto con il paese e' la sola grandezza che giustifica un
        # livello locale: senza, un amministratore puo' solo ripetere in
        # piccolo il ragionamento del governo, e il decentramento non aggiunge
        # informazione ma solo chiamate.
        + f["amm_dati" + coda].format(
            metriche=json.dumps(
                maschera_dizionario(quadro.metrics, "metrica", livello),
                indent=1, sort_keys=True),
            indicatori=json.dumps(
                maschera_indicatori(quadro.indicators, livello), indent=1, sort_keys=True),
            indicatori_colonia=json.dumps(
                maschera_indicatori(indicatori_colonia or {}, livello),
                indent=1, sort_keys=True),
            stato=json.dumps(
                maschera_dizionario(quadro.population_stats, "media", livello),
                indent=1, sort_keys=True),
        )
        + f["amm_vocabolario" + coda].format(
            indicatori=indicatori,
            pesabili=p["pesabili_amm"],
            categorie=categorie,
            peso_min="%g" % bounds.weight_min,
            peso_max="%g" % bounds.weight_max,
            una_da_proteggere=p["una_da_proteggere_amm"],
            va_alzata=p["va_alzata_amm"],
            max_regole=MAX_RULES,
            segnaposto=p["segnaposto_amm"],
        )
    )

class Amministrazione:
    """Gli amministratori di distretto, interrogati dopo il governo.

    `fabbrica_proponente` costruisce un proponente per distretto. E' iniettata
    per la stessa ragione per cui lo e' quella del governatore: il braccio si
    verifica con proponenti finti, senza consumare credito.
    """

    def __init__(
        self,
        fabbrica_proponente,
        distretti,
        bounds: Bounds,
        attivo: bool = True,
        modelli: list[str] | None = None,
        variante: VariantePrompt | None = None,
        livello: str = "completo",
    ) -> None:
        self._fabbrica = fabbrica_proponente
        self.distretti = distretti
        self.bounds = bounds
        self.attivo = bool(attivo)
        #: La variante del prompt, che qui e' sempre quella del governo.
        self.variante = variante or VariantePrompt()
        #: Quanto del dominio vede il livello LOCALE. Indipendente da quello del
        #: governo: il braccio che li toglie il dominio a tutti e due e' una
        #: casella a se', e finora non esisteva.
        self.livello = normalizza_livello(livello)
        #: I modelli fra cui i distretti pescano (`provider:model`), per il
        #: registro: chi legge una run deve sapere se lo strato era omogeneo.
        self.modelli = list(modelli or [])
        self._proponenti: dict[int, object] = {}
        #: L'ultima tornata, per il pannello live e per il registro.
        self.ultime: list[DecisioneAmministratore] = []
        #: L'ultima decisione di CIASCUN distretto, che non e' la stessa cosa:
        #: un distretto vuoto salta la tornata, e senza questa memoria la sua
        #: decisione precedente sparirebbe dal prompt appena si ripopola.
        self._precedenti: dict[int, DecisioneAmministratore] = {}
        self.astensioni = 0
        self.interventi = 0
        #: Tornate in cui il fornitore di quell'amministratore non ha risposto.
        self.guasti = 0
        #: Fra gli interventi, quelli che sulle proprie celle non facevano
        #: niente al momento della decisione. Contati a parte perche' sono
        #: l'esito che la coppia accetta/riscrive nascondeva.
        self.riscritture_a_vuoto = 0
        #: Accettazioni con una politica allegata e ignorata: vedi
        #: `DecisioneAmministratore.accettata_con_politica`.
        self.accettate_con_politica = 0
        #: Risposte fuori schema (vedi `DecisioneAmministratore.malformata`).
        self.malformate = 0

    def _proponente(self, distretto: int):
        if distretto not in self._proponenti:
            self._proponenti[distretto] = self._fabbrica(distretto)
        return self._proponenti[distretto]

    def policies_in_vigore(self) -> dict[int, Policy]:
        return {
            d.distretto: d.policy
            for d in self.ultime
            if d.policy is not None and d.policy.rules
        }

    def advance(
        self,
        step: int,
        metrics: dict,
        cells,
        agents,
        rows,
        policy_governo: Policy | None,
        indicatori_colonia: dict | None = None,
        morti_per_cella: dict | None = None,
    ) -> dict[int, Policy]:
        """Una tornata: ogni amministratore legge il governo e decide.

        `morti_per_cella` mappa (y, x) -> {causa: quanti} per i soli decessi
        avvenuti DALL'ULTIMA TORNATA. E' l'unica voce di esito che lo strato
        riceve: senza, decide su giacenze e medie, e in questo mondo si muore
        di fame con il magazzino pieno.
        """
        if not self.attivo:
            self.ultime = []
            return {}
        indicatori_colonia = indicatori_colonia or {}

        mappa = self.distretti.mappa()
        occupata = np.asarray(cells.occupancy) > 0
        # **Prima si costruiscono tutti i prompt, poi si chiama.** I due passi
        # erano un ciclo solo, e quel ciclo metteva le chiamate in fila: con
        # dieci distretti il tick costava dieci latenze invece di una. Qui si
        # separano perche' la costruzione e' pura e istantanea, e cio' che resta
        # --- l'attesa --- si puo' sovrapporre.
        richieste: list[tuple] = []
        catena = catena_del_governo(policy_governo, self.livello,
                                   self.variante.lingua)
        for distretto, celle in sorted(mappa.items()):
            # **Un distretto svuotato non si interroga.** L'assegnazione e'
            # permanente per progetto, quindi un distretto le cui celle sono
            # state tutte abbandonate resta in elenco per sempre; chiedergli una
            # politica costerebbe una chiamata al modello a ogni tick per un
            # territorio dove non vive nessuno, e `apply_layered` la scarterebbe
            # comunque perche' ambisce alle sole celle occupate. Misurato sulla
            # run di prova: 11 distretti su 12 celle occupate, cioe' la meta'
            # dei distretti gia' vuota al passo 400.
            vive = [c for c in celle if bool(occupata[c[0], c[1]])]
            if not vive:
                continue
            quadro = quadro_di_distretto(step, metrics, cells, agents, rows, vive)
            risolta = politica_risolta(
                policy_governo, cells, vive, self.livello,
                _voce(frasi(self.variante.lingua), "amm_cella", e_cieco(self.livello)),
                self.variante.lingua)
            prompt = costruisci_prompt(
                quadro, risolta, self.bounds, distretto, indicatori_colonia,
                morti=_morti_del_distretto(morti_per_cella, vive),
                precedente=_riga_precedente(self._precedenti.get(distretto),
                                            self.variante.lingua,
                                            self.livello),
                catena=catena,
                variante=self.variante,
                livello=self.livello,
            )
            from src.governors.admin_semif_arm import AdminSemanticContext

            context = AdminSemanticContext(
                district=distretto,
                picture=quadro,
                governor_policy=policy_governo,
                colony_indicators=dict(indicatori_colonia),
                prompt=prompt,
                # Letti solo dal profilo SemIf v4: gli altri li ignorano.
                deaths=_morti_del_distretto(morti_per_cella, vive),
            )
            richieste.append((distretto, vive, prompt, int(quadro.population), context))

        decisioni = [
            replace(d, a_vuoto=riscrittura_a_vuoto(d.policy, cells, d.celle))
            if d.policy is not None
            else d
            for d in self.tornata(richieste)
        ]
        self.ultime = decisioni
        for d in decisioni:
            self._precedenti[d.distretto] = d
        # Un guasto non e' ne' un'astensione ne' un intervento: se entrasse
        # fra le astensioni, un endpoint spento abbasserebbe la quota di
        # riscritture del modello che serve, che e' esattamente la grandezza
        # con cui i modelli vengono confrontati.
        self.guasti += sum(1 for d in decisioni if d.guasto)
        self.astensioni += sum(1 for d in decisioni if d.policy is None and not d.guasto)
        self.interventi += sum(1 for d in decisioni if d.policy is not None)
        self.riscritture_a_vuoto += sum(1 for d in decisioni if d.a_vuoto)
        self.accettate_con_politica += sum(1 for d in decisioni if d.accettata_con_politica)
        self.malformate += sum(1 for d in decisioni if d.malformata)
        return self.policies_in_vigore()

    def tornata(self, richieste) -> list[DecisioneAmministratore]:
        """Tutti gli amministratori insieme, nello stesso istante.

        **Perche' insieme.** Deliberano sulla stessa politica dello stesso
        tick: nessuno ha bisogno della risposta di un altro, e nessuno vede un
        mondo diverso. In fila, il costo di orologio di un tick sarebbe la
        latenza moltiplicata per il numero di distretti --- che cresce con la
        colonia, cioe' proprio quando la run e' interessante. E' l'opposto del
        governo, che di chiamate ne fa **una** per tick di cadenza perche' di
        governi ce n'e' uno.

        L'ordine dei risultati resta quello dei distretti, non quello di
        arrivo: due esecuzioni della stessa run devono produrre lo stesso
        registro anche se la rete risponde in ordine diverso.
        """
        if not richieste:
            return []
        return _esegui(self._raccogli(richieste))

    async def _raccogli(self, richieste) -> list[DecisioneAmministratore]:
        normalized = [
            (*request[:4], request[4] if len(request) > 4 else None)
            for request in richieste
        ]
        esiti = await asyncio.gather(
            *(
                self._interroga(d, celle, prompt, pop, semantic_context=context)
                for d, celle, prompt, pop, context in normalized
            ),
            return_exceptions=True,
        )
        decisioni: list[DecisioneAmministratore] = []
        for (distretto, celle, _, popolazione, _), esito in zip(normalized, esiti):
            if isinstance(esito, BaseException):
                # `_interroga` cattura gia' i guasti del fornitore; qui si
                # atterra solo se a sollevare e' stato il resto. Un distretto
                # che esplode non deve annullare la tornata degli altri.
                decisioni.append(
                    DecisioneAmministratore(
                        distretto=distretto, policy=None, accettata=True,
                        rationale=f"tornata fallita: {esito}", celle=tuple(celle),
                        popolazione=popolazione,
                    )
                )
            else:
                decisioni.append(esito)
        return decisioni

    async def _interroga(
        self, distretto, celle, prompt, popolazione: int = 0, semantic_context=None
    ) -> DecisioneAmministratore:
        proponente = self._proponente(distretto)
        try:
            semantico = getattr(proponente, "propose_semantic_async", None)
            asincrono = getattr(proponente, "propose_text_async", None)
            if semantico is not None and semantic_context is not None:
                risposta = await semantico(semantic_context)
            elif asincrono is not None:
                risposta = await asincrono(prompt)
            else:
                # Un proponente solo sincrono non deve rimettere in fila la
                # tornata: va su un thread, e la concorrenza resta.
                risposta = await asyncio.to_thread(proponente.propose_text, prompt)
            if asyncio.iscoroutine(risposta):
                risposta = await risposta
        except Exception as errore:  # noqa: BLE001
            # Stessa garanzia del governatore: un guasto del fornitore non
            # ferma la simulazione. Non e' pero' un'accettazione: e' un
            # amministratore che non ha parlato, e il registro lo dice.
            return DecisioneAmministratore(
                distretto=distretto,
                policy=None,
                accettata=False,
                guasto=True,
                rationale=f"chiamata fallita: {errore}",
                celle=tuple(celle),
                popolazione=popolazione,
            )

        semantic_fields = {
            "semantic_decision": (
                risposta.get("semantic_decision") if isinstance(risposta, dict) else None
            ),
            "candidate_profile": str(
                risposta.get("candidate_profile", "") if isinstance(risposta, dict) else ""
            ),
            "semantic_escalated": bool(
                risposta.get("semantic_escalated", False)
                if isinstance(risposta, dict) else False
            ),
            "semantic_decisions": (
                risposta.get("semantic_decisions") if isinstance(risposta, dict) else None
            ),
        }

        if isinstance(risposta, dict) and risposta.get("failed"):
            # Il braccio LLM non solleva mai: segnala il guasto cosi'.
            return DecisioneAmministratore(
                distretto=distretto,
                policy=None,
                accettata=False,
                guasto=True,
                rationale=str(risposta.get("rationale", "")),
                provider=str(risposta.get("provider", "")),
                model=str(risposta.get("model", "")),
                latency_s=float(risposta.get("latency_s", 0.0) or 0.0),
                celle=tuple(celle),
                popolazione=popolazione,
                **semantic_fields,
            )

        grezza = risposta.get("raw") if isinstance(risposta, dict) else None
        if not isinstance(grezza, dict) or grezza.get("accept", True):
            politica_allegata = (
                isinstance(grezza, dict)
                and isinstance(grezza.get("policy"), list)
                and len(grezza["policy"]) > 0
            )
            return DecisioneAmministratore(
                distretto=distretto,
                policy=None,
                accettata=True,
                accettata_con_politica=bool(politica_allegata),
                # Una risposta che non e' nemmeno un dizionario non ha detto
                # "accetto": non e' stata capita.
                malformata=not isinstance(grezza, dict),
                grezza=(grezza.get("policy") if isinstance(grezza, dict) else grezza),
                rationale=str((grezza or {}).get("rationale", "")) if isinstance(grezza, dict) else "",
                provider=str(risposta.get("provider", "")),
                model=str(risposta.get("model", "")),
                tokens_in=int(risposta.get("tokens_in", 0) or 0),
                tokens_out=int(risposta.get("tokens_out", 0) or 0),
                latency_s=float(risposta.get("latency_s", 0.0) or 0.0),
                celle=tuple(celle),
                popolazione=popolazione,
                **semantic_fields,
            )

        policy, scarti = parse_policy(
            traduci_proposta(grezza, self.livello), self.bounds)
        return DecisioneAmministratore(
            distretto=distretto,
            policy=policy,
            # `None` = fuori schema: le celle restano al governo, ma il
            # registro deve dire che NON e' un'accettazione (`malformata`).
            accettata=policy is None,
            malformata=policy is None,
            grezza=grezza.get("policy"),
            scarti=asdict(scarti) if is_dataclass(scarti) else dict(getattr(scarti, "__dict__", {})),
            rationale=str(grezza.get("rationale", "")),
            provider=str(risposta.get("provider", "")),
            model=str(risposta.get("model", "")),
            tokens_in=int(risposta.get("tokens_in", 0) or 0),
            tokens_out=int(risposta.get("tokens_out", 0) or 0),
            latency_s=float(risposta.get("latency_s", 0.0) or 0.0),
            celle=tuple(celle),
            popolazione=popolazione,
            **semantic_fields,
        )

    def riassunto(self) -> dict:
        madre = getattr(self.distretti, "madre", None)
        return {
            "enabled": self.attivo,
            "districts": self.distretti.numero_distretti(),
            # La cella madre sta fuori da ogni distretto per progetto: sulla
            # mappa va disegnata come tale, e senza questa voce il frontend
            # non puo' distinguerla da una cella non ancora assegnata.
            "mother_cell": list(madre) if madre is not None else None,
            "abstentions": self.astensioni,
            "interventions": self.interventi,
            "rewrites_without_effect": self.riscritture_a_vuoto,
            "accepted_with_policy": self.accettate_con_politica,
            "malformed": self.malformate,
            "provider_failures": self.guasti,
            "models": list(self.modelli),
            "last_round": [d.to_json() for d in self.ultime],
        }
