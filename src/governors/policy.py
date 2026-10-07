from __future__ import annotations

"""La policy del governatore: il vocabolario, i tipi e la validazione.

**Che cosa e' una policy, nel disegno a governatore unico (2026-08-24).** Non
piu' una direttiva per celle nominate, ma la **funzione di scelta** che tutti i
coloni useranno: una lista ordinata di regole `se <indicatore di cella> <sopra/
sotto> <soglia> -> pesa questi pilastri`, chiusa da un *else* implicito — dove
nessuna regola scatta, il colono sceglie con le sole proprie preferenze, cioe'
con la funzione identica alla baseline. Il governatore non nomina mai una
coordinata: scrive condizioni, e il kernel le valuta su ogni cella a ogni passo.

Due scale temporali, ed e' il punto del disegno: il governatore riscrive la
policy alla propria cadenza guardando gli aggregati; le condizioni della policy
reagiscono cella per cella a ogni passo, vettorialmente, senza chiamate.

**Prima regola che scatta vince, per cella.** E' la struttura if/elif/else che
la policy dichiara di essere: comporre tutte le regole che scattano renderebbe
l'ordine irrilevante e il testo della policy una somma di pesi, non una
funzione leggibile. La regola senza condizione (`"if": null`) scatta sempre e
chiude la catena: e' il modo di scrivere un peso incondizionato.

Come per la direttiva che questa policy sostituisce: un valore fuori scala
viene **riportato** nell'intervallo e contato (un modello che esagera esprime
comunque una direzione); un non-numero — `NaN` compreso — viene scartato; una
proposta **malformata** (non un dizionario, regole che non sono mappe) non
produce alcuna policy e restituisce ``None``, che il chiamante legge come
"scarta e tieni in vigore la precedente". Una policy **vuota** (zero regole) e'
invece legittima e significa "nessun intervento": le due cose sono diverse e
devono restare diverse.

Nessun I/O, nessuna rete, nessun NumPy: qui si decide soltanto se una proposta
e' ammissibile e quale forma prende. La valutazione vettoriale sta in
`src/governors/apply.py`.
"""

import math
from dataclasses import dataclass, field

from src.agents import pillars

#: Peso che lascia il pilastro come sta. Il valore dell'*else* implicito.
NEUTRAL_WEIGHT = 1.0

#: Quante regole al massimo una policy puo' contenere. Otto bastano a scrivere
#: una condizione per indicatore con margine; oltre, il costo per passo cresce
#: linearmente e la policy smette di essere leggibile. Le eccedenti vengono
#: troncate e contate, non rifiutate in blocco: una policy lunga esprime
#: comunque una politica nelle sue prime righe.
MAX_RULES = 8

#: I pilastri che una policy puo' pesare, per nome. `social` e' escluso di
#: proposito: non ha azioni (`PILLAR_ACTIONS[P_SOCIAL]` e' vuota — la socialita'
#: e' un servizio di cella, non un'azione di lavoro), quindi un peso su di esso
#: sarebbe una leva morta offerta al modello. E' la lezione della quota 0: mai
#: mettere nel vocabolario qualcosa che non fa cio' che sembra fare.
PILLAR_BY_NAME: dict[str, int] = {
    "sustenance": pillars.P_SUSTENANCE,
    "resources": pillars.P_RESOURCES,
    "build": pillars.P_BUILD,
    "life": pillars.P_LIFE,
    "explore": pillars.P_EXPLORE,
}
PILLAR_NAME_BY_INDEX: dict[int, str] = {
    index: name for name, index in PILLAR_BY_NAME.items()
}

#: Grafie del modello che significano un pilastro del vocabolario. Misurato
#: sulla prima campagna con la risposta grezza nel registro (2026-09-06,
#: `runs/mondi/scarce_resources`, 487 riscritture): «sustainance» 117 volte,
#: «sustain» 8. Scartarle in silenzio toglieva alla regola proprio il pilastro
#: della fame, e il contatore diceva soltanto "pilastro sconosciuto". Gli alias
#: si contano a parte (`aliased_pillars`): sono una correzione, non un silenzio.
PILLAR_ALIASES: dict[str, str] = {
    "sustainance": "sustenance",
    "sustainence": "sustenance",
    "sustenence": "sustenance",
    "sustanance": "sustenance",
    "sustain": "sustenance",
    "resource": "resources",
    "building": "build",
    "exploration": "explore",
}

#: Gli indicatori di cella su cui una condizione puo' essere scritta, con il
#: loro significato. E' il vocabolario intero: un nome fuori da questa tabella
#: fa scartare la regola (contata), non la proposta. Sono pro capite o rapporti,
#: mai valori assoluti, perche' una soglia assoluta direbbe cose diverse a celle
#: di taglia diversa — la stessa ragione per cui `material_per_capita` sostitui'
#: `material_margin` nei mandati.
#:
#: **Sono le statistiche che un'amministrazione pubblicherebbe davvero**, ed e'
#: un vincolo di progetto e non un abbellimento: la politica che ne discende
#: deve poter essere letta come una politica, non come la manopola di un
#: simulatore. Riserve pro capite di cibo, acqua e ossigeno; ghiaccio e minerali
#: estraibili pro capite; il margine di riserva della rete elettrica; lo stato
#: di manutenzione delle opere pubbliche; la popolazione residente. Un governo
#: che dice "dove la riserva idrica scende sotto due giorni a testa, priorita'
#: alla costruzione" sta facendo cio' che i governi fanno.
#:
#: I quattro aggiunti il 2026-08-25 — acqua, ossigeno, copertura elettrica,
#: manutenzione — sono le grandezze diventate vincolanti con la ritaratura del
#: mondo: la razione vitale si paga in massa, il carico elettrico si paga, e
#: l'usura morde. Prima il governatore vedeva la salute media scendere e non
#: poteva scrivere una regola su nessuna delle cause.
INDICATORS: dict[str, str] = {
    "food_per_occupant": (
        "cibo in giacenza nella cella, per occupante"
    ),
    "water_per_occupant": (
        "acqua potabile in giacenza nella cella, per occupante"
    ),
    "oxygen_per_occupant": (
        "ossigeno in giacenza nella cella, per occupante"
    ),
    "ice_per_occupant": (
        "ghiaccio raccoglibile (giacenza piu' superficie), per occupante"
    ),
    "material_per_occupant": (
        "construction_material nel magazzino della cella, per occupante"
    ),
    "minerals_per_occupant": (
        "minerali nel giacimento della cella, per occupante"
    ),
    "power_coverage": (
        "copertura elettrica: energia disponibile diviso il carico degli "
        "impianti della cella (1,0 = il fabbisogno e' coperto esatto)"
    ),
    "structure_integrity": (
        "stato medio di manutenzione delle strutture della cella, da 0 a 1"
    ),
    "occupants": "quanti coloni stanno nella cella",
}

#: Gli stessi significati in inglese, per il prompt tradotto. I NOMI non si
#: traducono mai --- sono il contratto con la macchina, e il parser conosce solo
#: quelli --- ma la spiegazione accanto a ciascuno e' prosa, e in un prompt
#: inglese deve essere inglese: era l'ultimo pezzo d'italiano rimasto nel
#: braccio tradotto, e stava nel vocabolario, cioe' nel posto piu' letto.
INDICATORS_EN: dict[str, str] = {
    "food_per_occupant": (
        "food in stock in the cell, per occupant"
    ),
    "water_per_occupant": (
        "drinkable water in stock in the cell, per occupant"
    ),
    "oxygen_per_occupant": (
        "oxygen in stock in the cell, per occupant"
    ),
    "ice_per_occupant": (
        "collectable ice (stock plus surface), per occupant"
    ),
    "material_per_occupant": (
        "construction_material in the cell's warehouse, per occupant"
    ),
    "minerals_per_occupant": (
        "minerals in the cell's seam, per occupant"
    ),
    "power_coverage": (
        "power coverage: available energy divided by the load of the cell's "
        "installations (1.0 = demand is met exactly)"
    ),
    "structure_integrity": (
        "average maintenance state of the cell's structures, from 0 to 1"
    ),
    "occupants": "how many colonists are in the cell",
}

assert set(INDICATORS_EN) == set(INDICATORS), (
    "il vocabolario diverge fra le due lingue: "
    f"solo it {sorted(set(INDICATORS) - set(INDICATORS_EN))}, "
    f"solo en {sorted(set(INDICATORS_EN) - set(INDICATORS))}"
)


def indicatori_con_significato(lingua: str = "it") -> dict[str, str]:
    """Nome -> significato, nella lingua del prompt."""
    return INDICATORS_EN if lingua == "en" else INDICATORS

#: Gli operatori ammessi. Due soli: una condizione e' una soglia, e una soglia
#: si attraversa in due direzioni.
OPS = ("<", ">")


@dataclass(frozen=True)
class Bounds:
    """Limiti entro cui un peso puo' muoversi."""

    weight_min: float
    weight_max: float


@dataclass(frozen=True)
class Condition:
    """`<indicatore> <op> <soglia>`, valutata per cella."""

    indicator: str
    op: str
    value: float


@dataclass(frozen=True)
class Rule:
    """Una riga della policy: condizione (o sempre-vero) e pesi sui pilastri."""

    #: `None` = scatta sempre: e' la regola incondizionata, che chiude la catena.
    condition: Condition | None
    #: Indice di pilastro -> peso moltiplicativo, gia' nei limiti.
    weights: dict[int, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Policy:
    """Cio' che la simulazione riceve: una funzione di scelta, per tutti."""

    rules: tuple[Rule, ...] = ()
    rationale: str = ""


@dataclass(frozen=True)
class PolicyDrops:
    """Che cosa e' stato scartato o riportato nell'intervallo, e quanto.

    Contato e non ignorato: uno scarto silenzioso farebbe sembrare che il
    governatore non avesse chiesto nulla.
    """

    unknown_indicators: int = 0
    unknown_pillars: int = 0
    clamped_weights: int = 0
    #: Regole scartate intere: condizione fuori vocabolario, operatore ignoto,
    #: soglia non numerica, o nessun peso sopravvissuto.
    dropped_rules: int = 0
    #: Regole oltre `MAX_RULES`, troncate.
    excess_rules: int = 0
    #: Pesi scritti con una grafia alias (`PILLAR_ALIASES`) e ricondotti al
    #: pilastro giusto: applicati, ma contati, perche' il modello non parla
    #: esattamente la lingua del vocabolario.
    aliased_pillars: int = 0


def _finite_float(value) -> float | None:
    """Un numero finito, o `None`. `json.loads` accetta `NaN` e `Infinity`,
    e nessuno dei due e' una soglia o un peso: il primo non ordina, il secondo
    scatta sempre o mai. `OverflowError` perche' un intero JSON abbastanza lungo
    non entra in un `float` e `float()` solleva."""
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(number):
        return None
    return number


#: Le due sole parole di sintassi di una regola stampata. Il resto della riga
#: --- nomi di indicatori, di categorie, operatori, numeri --- e' contratto
#: macchina e non si traduce in nessuna variante.
_SE = {"it": "se", "en": "if"}
_SEMPRE = {"it": "sempre", "en": "always"}


def testo_regola_tradotto(testo: str, lingua: str = "it") -> str:
    """Il testo stabile di una regola, reso nella lingua del prompt.

    Serve dove si ha in mano la riga gia' formattata e non piu' la regola: il
    consuntivo degli scatti, che arriva dai contatori. La chiave resta italiana,
    la vista no.
    """
    if lingua == "it" or not testo:
        return testo
    for tabella in (_SE, _SEMPRE):
        testa = tabella["it"] + " "
        if testo.startswith(testa):
            return tabella[lingua] + " " + testo[len(testa):]
    return testo


def rule_text(rule: Rule, lingua: str = "it") -> str:
    """La regola in una riga leggibile e STABILE: e' la chiave dei contatori.

    Serve al conteggio degli scatti per regola (`apply_policy`), dove regole di
    policy diverse ma identiche nel testo devono sommare sullo stesso rigo — la
    costituzione dello scripted viene riproposta a ogni tick, e contare ogni
    tick a parte direbbe quaranta volte la stessa cosa. I pesi entrano nella
    chiave: due regole con la stessa condizione ma pesi diversi sono politiche
    diverse e vanno contate separate.
    """
    pesi = ", ".join(
        f"{PILLAR_NAME_BY_INDEX[pillar]} x{weight:g}"
        for pillar, weight in sorted(rule.weights.items())
    )
    if rule.condition is None:
        return f"{_SEMPRE[lingua]} -> {pesi}"
    condition = rule.condition
    return (f"{_SE[lingua]} {condition.indicator} {condition.op} "
            f"{condition.value:g} -> {pesi}")


def parse_policy(raw, bounds: Bounds) -> tuple[Policy | None, PolicyDrops]:
    """Interpreta e valida una proposta grezza.

    La distinzione fra "scarta la voce" e "scarta tutto" e' la stessa del
    vecchio `parse_proposal`: una voce fuori vocabolario e' un errore locale e
    si conta; una struttura fuori schema (non-dizionario dove serve un
    dizionario) e' una risposta che non parla la lingua, e non se ne puo'
    salvare meta'.
    """
    if not isinstance(raw, dict):
        return None, PolicyDrops()
    if "policy" not in raw:
        # **Chiave assente e lista vuota non sono la stessa cosa.** Una lista
        # vuota e' un non-intervento dichiarato e sostituisce la catena in
        # vigore; una risposta che la chiave non ce l'ha affatto --- il payload
        # di ripiego di un fornitore, la risposta a un'altra domanda --- non ha
        # legiferato niente, e trattarla come non-intervento ABROGA la legge
        # per conto di un modello che non ha parlato.
        return None, PolicyDrops()
    entries = raw["policy"]
    if not isinstance(entries, list):
        return None, PolicyDrops()

    unknown_ind = unknown_pil = clamped = dropped = excess = aliased = 0
    rules: list[Rule] = []
    for entry in entries:
        if entry is None:
            # **Una regola nulla non e' una risposta in un'altra lingua.** Il
            # modello chiude la lista con `null` (33 riscritture su 34 fuori
            # schema nella campagna del 2026-09-06): si scarta la voce e si
            # conta, come per una regola fuori vocabolario. Prima faceva
            # cadere la riscrittura intera, e le celle restavano al governo
            # contro l'intenzione dichiarata dell'amministratore.
            dropped += 1
            continue
        if not isinstance(entry, dict):
            return None, PolicyDrops()

        if len(rules) >= MAX_RULES:
            excess += 1
            continue

        raw_condition = entry.get("if", None)
        if raw_condition is None:
            condition = None
        elif isinstance(raw_condition, dict):
            indicator = str(raw_condition.get("indicator", ""))
            op = str(raw_condition.get("op", ""))
            value = _finite_float(raw_condition.get("value"))
            if indicator not in INDICATORS:
                unknown_ind += 1
                dropped += 1
                continue
            if op not in OPS or value is None:
                dropped += 1
                continue
            condition = Condition(indicator, op, value)
        else:
            return None, PolicyDrops()

        raw_weights = entry.get("weights") or {}
        if not isinstance(raw_weights, dict):
            return None, PolicyDrops()
        weights: dict[int, float] = {}
        for name, value in raw_weights.items():
            chiave = str(name).strip().lower()
            if chiave in PILLAR_ALIASES:
                chiave = PILLAR_ALIASES[chiave]
                aliased += 1
            pillar = PILLAR_BY_NAME.get(chiave)
            if pillar is None:
                unknown_pil += 1
                continue
            number = _finite_float(value)
            if number is None:
                unknown_pil += 1
                continue
            bounded = min(max(number, bounds.weight_min), bounds.weight_max)
            if bounded != number:
                clamped += 1
            weights[pillar] = bounded

        if not weights:
            dropped += 1
            continue
        rules.append(Rule(condition, weights))

    drops = PolicyDrops(unknown_ind, unknown_pil, clamped, dropped, excess, aliased)
    if entries and not rules:
        # **Ha provato a legiferare e non e' stato capito.** Tutte le regole
        # scartate con nessuna superstite non e' un non-intervento: e' silenzio,
        # e il silenzio lascia in piedi la catena precedente invece di
        # cancellarla. La differenza si vede nei contatori, che restano.
        return None, drops
    return Policy(tuple(rules), str(raw.get("rationale", "") or "")), drops
