from __future__ import annotations

"""Il governatore LLM: prompt, chiamata asincrona, interpretazione.

Il provider arriva dal costruttore e non dal registro: e' quell'iniezione che
rende il braccio verificabile con provider finti, senza consumare credito. Chi
costruisce un provider vero passa comunque dall'interlock di consenso in
`src.llm.provider_registry`.

Nessun errore esce da `propose`. Una rete giu', un JSON malformato o una
risposta fuori schema producono una proposta **senza policy** (`None`), che la
shell legge come "scarta e tieni in vigore la precedente". Un'eccezione qui
fermerebbe la simulazione per un problema che non e' della simulazione.
"""

import json
import time

from src.governors.apply import ELSE_KEY
from src.governors.arms import GovernorProposal, reference_rules_text
from src.governors.context import (
    ALIAS_INDICATORI,
    ALIAS_PILASTRI,
    e_cieco,
    maschera_dizionario,
    maschera_indicatori,
    mostra_aiuti,
    mostra_dominio,
    normalizza,
    traduci_proposta,
)
from src.governors.districts import CELLE_PER_DISTRETTO
from src.governors.observation import ColonyPicture
from src.governors.testi_prompt import frasi
from src.governors.varianti_prompt import VariantePrompt, parole, testo_gerarchia
from src.governors.policy import (
    Bounds,
    INDICATORS,
    indicatori_con_significato,
    MAX_RULES,
    PILLAR_BY_NAME,
    parse_policy,
    testo_regola_tradotto,
)

def _glossario(p: dict, f: dict) -> str:
    """Che cosa copre ciascuna categoria pesabile.

    E' il glossario al livello a cui la policy agisce: la lezione del glossario
    delle azioni (2026-08-21) --- il modello non deve inferire il significato di
    una leva dal suo nome --- vale identica qui. Le due frasi che nominano la
    cosa pesata vengono dalla variante (`varianti_prompt.py`), il resto dalla
    tabella dei testi (`testi_prompt.py`): qui non c'e' piu' prosa, solo
    l'assemblaggio, che e' scritto una volta sola per tutte le lingue.
    """
    return f["glossario"].format(
        glossario_apertura=p["glossario_apertura"],
        glossario_fame=p["glossario_fame"],
    )


def _token_count(value) -> int:
    """Un conteggio di token che non e' un numero vale zero, non un'eccezione.

    I token servono alla contabilita' del registro, non alla simulazione: se un
    provider consegna un campo di forma inattesa, `int()` solleverebbe **dopo**
    una chiamata riuscita, buttando via una policy valida e violando la
    garanzia dichiarata nel docstring del modulo.
    """
    try:
        return int(value or 0)
    except (TypeError, ValueError, OverflowError):
        return 0


def _indicator_warnings(picture: ColonyPicture, f: dict) -> str:
    """Le avvertenze sugli indicatori che ORA non possono discriminare.

    Misurato sulle run reali: sulla cella equatoriale `ice_per_occupant` vale
    zero ovunque, e finche' la colonia sta in una cella sola OGNI indicatore e'
    piatto — `occupants` per primo, costante alla popolazione. Un modello che
    scrivesse una condizione su un indicatore piatto crederebbe di selezionare
    celle e otterrebbe un interruttore tutto-o-niente; su uno a zero, una `<`
    con soglia positiva scatta ovunque e una `>` mai. Sono proprieta' dello
    SCENARIO corrente, non del vocabolario, quindi si dicono qui — calcolate
    dal quadro, non cablate — e spariscono da sole quando la colonia si
    espande o il ghiaccio compare. E' la lezione della quota 0: mai lasciare
    che il modello scopra da solo che una leva non fa cio' che sembra.
    """
    warnings = []
    for name, stats in sorted(picture.indicators.items()):
        massimo = float(stats.get("max", 0.0))
        minimo = float(stats.get("min", 0.0))
        if massimo == 0.0:
            warnings.append(f["avvertenza_zero"].format(nome=name))
        elif minimo == massimo and picture.n_cells > 1:
            warnings.append(f["avvertenza_piatta"].format(nome=name, valore="%g" % massimo))
    if picture.n_cells == 1:
        warnings.append(f["avvertenza_una_cella"])
    if not warnings:
        return ""
    return f["avvertenze_intestazione"] + "\n".join(warnings) + "\n\n"


def _consuntivo_regole(picture: ColonyPicture, cieco: bool, f: dict,
                       lingua: str = "it") -> str:
    """Che cosa ha fatto la politica precedente, regola per regola.

    Una regola a zero non e' una regola inutile: e' una regola **in ombra**,
    cioe' scritta dopo una piu' larga che le ha portato via tutte le celle. E'
    la sola informazione che permette a chi scrive la catena di accorgersene, e
    non e' contesto di dominio — riguarda la propria legge — quindi compare a
    ogni gradino, con i nomi delle regole nascosti solo dove tutto il resto lo e'.
    """
    if not picture.rule_hits:
        return ""
    righe = []
    for indice, (testo, quante) in enumerate(picture.rule_hits, start=1):
        e_altrimenti = testo == ELSE_KEY
        etichetta = (
            f["consuntivo_altrimenti"] if e_altrimenti
            else (f["consuntivo_regola"].format(indice=indice) if cieco
                  else f["consuntivo_regola_con_testo"].format(
                      indice=indice, testo=testo_regola_tradotto(testo, lingua)))
        )
        coda = f["consuntivo_ombra"] if quante == 0 and not e_altrimenti else ""
        righe.append(f["consuntivo_riga"].format(etichetta=etichetta, quante=quante, coda=coda))
    return f["consuntivo_intestazione"] + "\n".join(righe) + "\n\n"


def _parole(livello: str, f: dict) -> dict[str, str]:
    """Le parole del dominio, o le loro versioni neutre al gradino cieco.

    Il gradino cieco promette di non nominare Marte ne' i coloni; le righe del
    quadro pesato le nominavano in chiaro e il test dei livelli lo ha colto.
    """
    coda = "_cieco" if normalizza(livello) == "cieco" else ""
    return {n: f[n + coda] for n in ("colono", "coloni", "colonia")}


def _dove_stanno_i_coloni(picture: ColonyPicture, livello: str, f: dict) -> str:
    """La unita' piu' popolata, con la sua quota e i suoi valori.

    E' il Nord con un milione di abitanti: una regola che non scatta li' non
    tocca la maggioranza dei coloni, e il governatore deve poterlo vedere
    senza dedurlo da una media.
    """
    p = picture.principale
    if not p:
        return ""
    w = _parole(livello, f)
    valori = maschera_indicatori(p.get("indicators", {}), livello)
    return f["dove_stanno"].format(
        coloni=w["coloni"], colonia=w["colonia"], cella=p["cell"],
        occupanti=p["occupants"], quota="%.0f%%" % (p["share"] * 100),
        valori=json.dumps(valori, sort_keys=True),
    )


def _morti_dall_ultimo_tick(picture: ColonyPicture, livello: str, f: dict) -> str:
    """L'unica voce di esito: chi e' morto dall'ultimo tick, dove e di che cosa."""
    if not picture.deaths:
        return f["morti_nessuno"]
    totale = sum(int(n) for cause in picture.deaths.values() for n in cause.values())
    righe = []
    for cella, cause in sorted(picture.deaths.items(), key=lambda kv: -sum(kv[1].values())):
        mascherate = maschera_dizionario(dict(cause), "causa", livello)
        dettaglio = ", ".join(f"{c}: {n}" for c, n in sorted(mascherate.items()))
        righe.append(f["morti_riga"].format(cella=list(cella), dettaglio=dettaglio))
    return f["morti_intestazione"].format(totale=totale) + "\n".join(righe) + "\n\n"


def build_governor_prompt(
    picture: ColonyPicture,
    bounds: Bounds,
    livello: str = "completo",
    variante: VariantePrompt | None = None,
) -> str:
    """Il prompt, composto per gradini di contesto.

    Niente elenco di celle a nessun gradino: il governatore scrive condizioni,
    non coordinate, e riceve la distribuzione degli indicatori --- la stessa
    grandezza su cui le sue condizioni verranno valutate, cella per cella, a
    ogni passo.

    Che cosa cade, salendo verso la cecita', e' descritto in
    `src/governors/context.py`. Qui resta invariato cio' che non e' contesto ma
    *meccanica*: il vocabolario, i limiti dei pesi, la regola della prima che
    scatta, l'avvertenza che i pesi sono relativi. Toglierla renderebbe il
    braccio cieco non meno informato ma piu' ignorante delle regole del gioco,
    e la differenza smetterebbe di misurare il contesto.
    """
    livello = normalizza(livello)
    cieco = e_cieco(livello)
    variante = variante or VariantePrompt()
    # Al gradino cieco la cosa pesata si chiama «leva» qualunque sia la
    # variante, perche' le sue etichette neutre sono `leva_A..leva_E` e sono
    # congelate: la traduzione inversa della risposta ci si appoggia. Dire
    # «categorie pesabili: leva_A» sarebbe un prompt incoerente. La campagna
    # incrociata gira al gradino completo, dove la questione non si pone.
    p = parole(variante if not cieco else VariantePrompt(lingua=variante.lingua))
    f = frasi(variante.lingua)

    nomi_indicatori = ALIAS_INDICATORI if cieco else {n: n for n in INDICATORS}
    if cieco:
        # Elencati in ordine di etichetta: il rimescolamento sta nella tabella
        # degli alias, non nella lista, quindi la posizione non suggerisce nulla.
        indicatori = "\n".join(f"- {a}" for a in sorted(ALIAS_INDICATORI.values()))
        pilastri = ", ".join(sorted(ALIAS_PILASTRI.values()))
    else:
        indicatori = "\n".join(
            f"- {n}: {m}"
            for n, m in indicatori_con_significato(variante.lingua).items()
        )
        pilastri = ", ".join(sorted(PILLAR_BY_NAME))

    pezzi: list[str] = []

    if mostra_dominio(livello):
        # La gerarchia e' un fatto del disegno e sta qui, subito dopo il ruolo:
        # e' il posto in cui un governo apprende di averne uno sotto. Non
        # compare ai gradini che non dichiarano il dominio, dove nominare
        # distretti e amministratori lo reintrodurrebbe.
        pezzi.append(
            f["ruolo"].format(uno_fra_sei=p["uno_fra_sei"], i_plurale=p["i_plurale"])
            + testo_gerarchia(variante, CELLE_PER_DISTRETTO)
        )
    else:
        pezzi.append(f["ruolo_neutro"])

    if mostra_aiuti(livello):
        pezzi.append(f["aiuti"].format(
            regole=reference_rules_text(variante.lingua),
            plurale_diversi=p["plurale_diversi"]))

    # **Le due viste insieme.** Le condizioni scattano per unita', quindi la
    # distribuzione per unita' dice DOVE una regola morde; la vista pesata dice
    # su QUANTE PERSONE. Con la maggioranza dei coloni in una unita' sola, la
    # media per unita' e' la media degli avamposti.
    w = _parole(livello, f)
    pezzi.append(
        f["dati"].format(
            step=picture.step,
            popolazione=picture.population,
            celle=picture.n_cells,
            metriche=json.dumps(
                maschera_dizionario(picture.metrics, "metrica", livello),
                indent=1, sort_keys=True),
            indicatori=json.dumps(
                maschera_indicatori(picture.indicators, livello),
                indent=1, sort_keys=True),
            colono=w["colono"],
            coloni=w["coloni"],
            indicatori_pesati=json.dumps(
                maschera_indicatori(picture.indicators_weighted, livello),
                indent=1, sort_keys=True),
        )
        + _dove_stanno_i_coloni(picture, livello, f)
        + _morti_dall_ultimo_tick(picture, livello, f)
        + f["code_dati"].format(
            medie=json.dumps(
                maschera_dizionario(picture.population_stats, "media", livello),
                indent=1, sort_keys=True),
            strutture=json.dumps(
                maschera_dizionario(picture.structures, "totale", livello),
                indent=1, sort_keys=True),
        )
    )

    if mostra_aiuti(livello):
        pezzi.append(_indicator_warnings(picture, f))

    pezzi.append(_consuntivo_regole(picture, cieco, f, variante.lingua))

    pezzi.append(f["vocabolario"].format(
        indicatori=indicatori, pesabili=p["pesabili"], categorie=pilastri))

    if mostra_dominio(livello):
        pezzi.append(_glossario(p, f))

    # L'esempio concreto della categoria che nessuno alza --- l'esplorazione,
    # senza cui la colonia smette di fondare avamposti senza averlo deciso ---
    # e' domanda di dominio: al gradino cieco diventerebbe un indizio, e la
    # frase resta generale. L'avvertenza in se' non e' contesto ma meccanica, e
    # resta a ogni gradino.
    esempio = f["esempio_da_proteggere"] if mostra_dominio(livello) else ""

    pezzi.append(f["chiusura"].format(
        peso_min="%g" % bounds.weight_min,
        peso_max="%g" % bounds.weight_max,
        fra_le_plurale=p["fra_le_plurale"],
        tre_plurale=p["tre_plurale"],
        una_da_proteggere=p["una_da_proteggere"],
        esempio=esempio,
        va_alzata=p["va_alzata"],
        max_regole=MAX_RULES,
        segnaposto=p["segnaposto"],
    ))
    return "".join(pezzi)


class LLMProposer:
    """Un governatore che chiede a un modello."""

    def __init__(
        self,
        provider,
        model: str = "",
        cost_tracker=None,
        livello_contesto: str = "completo",
        variante: VariantePrompt | None = None,
    ) -> None:
        self._provider = provider
        self._model = model or getattr(provider, "model", "")
        self._cost_tracker = cost_tracker
        self._livello = normalizza(livello_contesto)
        self._variante = variante or VariantePrompt()

    @property
    def livello_contesto(self) -> str:
        return self._livello

    def _record_cost(self, response, provider_id: str, success: bool) -> None:
        """Iscrive la chiamata nella contabilita' della run, se ce n'e' una.

        Senza questo, `api_usage.json` dichiarava `"calls": 0` per una run in
        cui il governatore aveva chiamato davvero decine di volte. Non solleva
        mai, per la stessa ragione per cui non solleva `propose`.
        """
        if self._cost_tracker is None:
            return
        try:
            self._cost_tracker.record(
                cost_usd=float(getattr(response, "cost_usd", 0.0) or 0.0),
                success=success and not getattr(response, "error", ""),
                provider=provider_id,
                model=self._model,
                tokens_in=_token_count(getattr(response, "tokens_in", 0)),
                tokens_out=_token_count(getattr(response, "tokens_out", 0)),
                tokens_total=_token_count(getattr(response, "tokens_total", 0)),
                pricing_source=str(getattr(response, "pricing_source", "") or ""),
            )
        except Exception:  # noqa: BLE001 - la contabilita' non ferma la run
            pass

    async def propose(self, picture: ColonyPicture, bounds: Bounds) -> GovernorProposal:
        provider_id = getattr(self._provider, "provider_id", "sconosciuto")
        prompt = build_governor_prompt(picture, bounds, self._livello, self._variante)
        started = time.perf_counter()
        try:
            response = await self._provider.async_complete_json(prompt)
        except Exception as error:  # noqa: BLE001 - vedi il docstring del modulo
            self._record_cost(None, provider_id, success=False)
            return GovernorProposal(
                policy=None,
                rationale=f"chiamata fallita: {error}",
                provider=provider_id,
                model=self._model,
                latency_s=time.perf_counter() - started,
                guasto_fornitore=True,
            )
        latency = time.perf_counter() - started
        self._record_cost(response, provider_id, success=True)

        errore = getattr(response, "error", "")
        if errore:
            # **Il fornitore non solleva.** Quando la chiamata fallisce
            # restituisce il payload di ripiego pensato per la decisione di un
            # colono: JSON valido, senza chiave `policy`, che il parser
            # leggerebbe come catena vuota e che il governo adotterebbe,
            # abrogando la legge in vigore e registrando l'abrogazione come una
            # scelta. Va intercettato qui, prima del parser, come gia' fa il
            # livello locale. La contabilita' e' gia' stata fatta: `_record_cost`
            # marca da se' come fallita una risposta con `error` valorizzato,
            # ed e' per questo che `failed_calls` vedeva il guasto mentre la
            # politica veniva svuotata lo stesso.
            return GovernorProposal(
                policy=None,
                rationale=f"chiamata fallita: {errore}",
                provider=provider_id,
                model=self._model,
                tokens_in=_token_count(getattr(response, "tokens_in", 0)),
                tokens_out=_token_count(getattr(response, "tokens_out", 0)),
                latency_s=latency,
                guasto_fornitore=True,
            )

        # `getattr` e non `response.text`: una risposta senza quel campo — un
        # `dict`, una stringa nuda, un oggetto troncato da un SDK — e' fuori
        # schema quanto un JSON malformato, e un `AttributeError` non sarebbe
        # ne' `TypeError` ne' `ValueError`, quindi uscirebbe da `propose` DOPO
        # una chiamata riuscita. Cosi' invece finisce in `json.loads(None)`,
        # che solleva `TypeError` ed e' gia' intercettato qui sotto.
        try:
            raw = json.loads(getattr(response, "text", None))
        except (TypeError, ValueError):
            return GovernorProposal(
                policy=None,
                rationale="risposta non interpretabile: non e' JSON",
                provider=provider_id,
                model=self._model,
                tokens_in=_token_count(getattr(response, "tokens_in", 0)),
                tokens_out=_token_count(getattr(response, "tokens_out", 0)),
                latency_s=latency,
            )

        policy, drops = parse_policy(traduci_proposta(raw, self._livello), bounds)
        rationale = (
            policy.rationale
            if policy is not None and policy.rationale
            else ("risposta non interpretabile: fuori schema" if policy is None else "")
        )
        return GovernorProposal(
            policy=policy,
            rationale=rationale,
            provider=provider_id,
            model=self._model,
            tokens_in=_token_count(getattr(response, "tokens_in", 0)),
            tokens_out=_token_count(getattr(response, "tokens_out", 0)),
            # **Il costo viaggiava solo verso la contabilita' di run.** Il campo
            # esisteva in `GovernorProposal`, il recorder lo scriveva nel
            # registro per tick, e restava a zero per costruzione: una campagna
            # a pagamento riportava `"cost": 0.0` su ogni riga di
            # `governor_decisions.jsonl` mentre `api_usage.json` diceva il vero.
            # Due posti per una quantita' sola, di cui uno muto.
            cost=float(getattr(response, "cost_usd", 0.0) or 0.0),
            latency_s=latency,
            drops=drops,
            raw_policy=raw.get("policy") if isinstance(raw, dict) else raw,
        )
