from __future__ import annotations

"""I bracci di controllo del protocollo, entrambi senza rete.

`RandomProposer` produce policy qualsiasi entro lo stesso schema: isola
l'effetto di **ricevere** una policy da quello di riceverne una buona.
`ScriptedProposer` e' un governo competente ma non LLM: isola "l'LLM governa
bene" da "governare aiuta". Senza il secondo, un braccio LLM che batte la
baseline non dice quale delle due cose e' stata dimostrata.

**Lo `ScriptedProposer` emette sempre la stessa policy: una costituzione.**
Nel disegno a direttive per cella il braccio scripted rileggeva il quadro a
ogni tick e decideva se e quanto parlare; nel disegno a policy quella
adattivita' sta DENTRO la policy — le condizioni si valutano cella per cella a
ogni passo — quindi il governo competente non-LLM e' una regola fissa scritta
una volta: `REFERENCE_RULES`. Dove la colonia sta bene nessuna condizione
scatta e la funzione di scelta resta la baseline; dove una cella e' sotto
soglia, il peso si accende da solo. Il confronto con il braccio LLM diventa
cosi' pulito: entrambi scrivono nello stesso linguaggio, e cio' che l'LLM puo'
aggiungere e' adattare le regole al quadro che vede — soglie, pesi e pilastri
diversi nel tempo — mentre lo scripted non puo'.

**Il braccio LLM riceve `REFERENCE_RULES` nel prompt.** E' la stessa scelta
gia' fatta per i mandati (2026-08-21): i due bracci devono ricevere lo stesso
obiettivo operativo, altrimenti il confronto misura anche la differenza fra un
obiettivo scritto e uno da indovinare.

Sono asincroni come il braccio LLM, cosi' che il governatore non debba
distinguerli.
"""

import random
import zlib
from dataclasses import dataclass, field

from src.governors.observation import ColonyPicture
from src.governors.policy import (
    Bounds,
    Condition,
    INDICATORS,
    PILLAR_BY_NAME,
    Policy,
    PolicyDrops,
    Rule,
)

#: La politica di riferimento: la costituzione del braccio `scripted`, e il
#: mandato operativo che il prompt del braccio `llm` riporta parola per parola.
#:
#: **Riscelta sui dati del mondo attuale (sweep del 2026-08-25,
#: `docs/benchmarks/2026-08-25-scenario-ammissibile-e-indicatori.md`).** La
#: costituzione precedente — cibo sotto due unita' per occupante -> sostentamento
#: x3, piu' di 150 occupanti -> esplorazione x2 — era stata scelta il 2026-08-24
#: su un'economia che non conservava la massa. Dopo la ritaratura non era solo
#: obsoleta:
#:
#: - **era inerte**: 1.464 scatti sui tre semi e delta ESATTAMENTE zero su
#:   popolazione e celle, cioe' il braccio `scripted` era diventato una seconda
#:   baseline. Con un controllo competente indistinguibile dal nessun controllo,
#:   il disegno a quattro bracci non distingue piu' "l'LLM governa bene" da
#:   "governare aiuta": collassa;
#: - **in crisi puntava dalla parte sbagliata**: al 50% di distruzione il cibo
#:   per occupante scende sotto due e la regola triplicava il SOSTENTAMENTO,
#:   cioe' mandava a foraggiare un magazzino vuoto, mentre la risposta era
#:   ricostruire.
#:
#: Sei candidate, tre semi, scenario ammesso dal criterio di
#: `scripts/cerca_dilemma.py` (ISRU 0,10: pressione con venti morti e varianza
#: di esito da quattro a dieci celle abitate):
#:
#:   attuale (cibo + affollamento)        pop +0/+0/+0  celle +0/+0/+0  1464 scatti, INERTE
#:   corrente < fabbisogno -> build       pop +4/-1/-3  celle +1/+0/-4   161 scatti, discorde
#:   manutenzione < 0,70 -> build         pop +0/+0/+0  celle +0/+0/+0   148 scatti, inerte
#:   riserva idrica < 1 -> sustenance     pop +0/+0/+0  celle +0/+0/-1  1577 scatti, inerte
#:   giacimento < 1 -> explore            pop +4/-4/+1  celle +1/+0/-6  2540 scatti, discorde
#:   corrente + giacimento                pop +2/-7/+2  celle +1/-2/-6  2385 scatti, discorde
#:
#: **Il criterio di scelta e' che la costituzione non sia inerte**, non che
#: vinca: con tre semi e una varianza fra semi che domina l'effetto, il SEGNO
#: non e' stabilito e va misurato nella campagna. Cio' che si puo' affermare qui
#: e' quali candidate muovono la colonia e quali no, ed e' la sola cosa che
#: serve per non mandare in campagna un braccio di controllo muto.
#:
#: ====================================================================
#: **RISCELTA il 2026-09-02, e la precedente era dannosa.** Dieci candidate,
#: tre semi, 120 coloni, 400 passi, scenario ISRU 0,10. Baseline: pop
#: 268/213/234, celle 39/25/26.
#:
#:   costituzione                         pop delta      celle delta   ripartizione   ammessa
#:   attuale (cibo + affollamento)        +49/+48/-7     +16/+22/-1    58% / 1%       SI
#:   corrente < fabbisogno -> build       -88/-32/-54    -38/-24/-25   100%           no, COSTANTE
#:   manutenzione < 0,70 -> build         -80/+40/-66    -27/+1/-22    37%            SI
#:   riserva idrica < 1 -> sustenance     -112/-120/-31  -20/-5/-12    42%            no, tutti -
#:   giacimento < 1 -> explore            -85/+17/-25    -3/+0/-1      65%            SI
#:   corrente + giacimento (in vigore)    -88/-32/-54    -38/-24/-25   100% / 0%      no, COSTANTE
#:   corrente + giacimento, explore prot. -57/+48/-19    -22/+24/+3    74% / 7%       SI
#:   corrente, explore e cibo protetti    -34/+24/-5     -18/+17/-4    76%            SI
#:   manutenzione, explore protetto       -27/-138/+0    +5/-10/+5     37% / 35%      no, tutti -
#:   acqua, explore protetto              -227/-107/-37  -25/-12/-8    46% / 38%      no, tutti -
#:
#: **La costituzione in vigore fino a oggi era la peggiore delle dieci**, ed e'
#: stato il criterio nuovo a smascherarla: la sua prima regola catturava il
#: **100 per cento** delle celle-passo e la seconda lo **zero**. Non era una
#: catena a due regole ma una sola, e la sua condizione era vera ovunque —
#: cioe' una costante travestita da condizione. Su punteggi rinormalizzati una
#: costante non aiuta nessuna cella: sposta l'allocazione globale e penalizza
#: cio' che non nomina, qui `explore`. Misurato a parte, su tre semi a 400
#: passi: espansione da 69/116/86 celle (nessun governo) a 3/1/1.
#:
#: Che il difetto stesse nei PESI e non nell'ordine ne' nelle soglie e' stato
#: isolato con tre esperimenti: una regola incondizionata in coda cattura il
#: 2-11 per cento e non cambia l'esito; stringere la soglia da 1,0 a 0,6 da'
#: numeri identici cifra per cifra; aggiungere `explore x3` **dentro la regola
#: che scatta** riporta l'espansione a 44/101/97 celle.
#:
#: I due criteri nuovi (in `scripts/scegli_costituzione.py`): **discrimina** —
#: nessuna regola al 100 per cento (costante) ne' a zero (in ombra dietro una
#: precedente) — e **non nuoce**, cioe' il delta non e' negativo su tutti i
#: semi. Un braccio di controllo competente puo' perdere; se perde sempre e' un
#: handicap, e il confronto fra bracci misurerebbe quello.
#:
#: **Le due regole scelte**, nell'ordine (prima-regola-vince, quindi una cella
#: affamata riceve `sustenance` e non viene piu' considerata dalla seconda):
#:
#: - **cibo per occupante sotto due razioni -> sostentamento x3.** La carenza
#:   che una colonia sente per prima e su cui un governo si gioca la
#:   legittimita'. Cattura il 58 per cento delle celle-passo: discrimina.
#: - **oltre centocinquanta abitanti -> esplorazione x2.** L'affollamento come
#:   segnale di fondazione: quando una cella e' piena, la risposta non e'
#:   costruirci dentro ma andarsene. Cattura l'1 per cento — rara, non in
#:   ombra: e' una regola di soglia alta, e scatta quando serve.
#:
#: E' la costituzione che era in vigore fino al 2026-08-25, sostituita allora
#: per un'obiezione teorica — "in crisi triplica il sostentamento e manda a
#: foraggiare un magazzino vuoto" — mai verificata su dati. Misurata su dieci
#: candidate e' l'unica con delta positivo su due semi in entrambe le variabili
#: di esito. **L'obiezione teorica ha perso contro la misura**, ed e' la ragione
#: per cui questo file porta la tabella e non solo il risultato.
#:
#: I pesi restano FISSI: la condizione decide dove il peso agisce, e un peso
#: costante rende la costituzione leggibile e il braccio riproducibile.
REFERENCE_RULES: tuple[tuple[str, str, float, str, float], ...] = (
    ("food_per_occupant", "<", 2.0, "sustenance", 3.0),
    ("occupants", ">", 150.0, "explore", 2.0),
)


def reference_policy(bounds: Bounds) -> Policy:
    """`REFERENCE_RULES` in forma eseguibile, coi pesi riportati nei limiti.

    Il clamp non e' pro forma: i limiti sono configurabili
    (`priority_multiplier_range`), e una costituzione che li scavalcasse
    darebbe al braccio scripted uno spazio che il braccio LLM non ha.
    """
    rules = tuple(
        Rule(
            Condition(indicator, op, threshold),
            {
                PILLAR_BY_NAME[pillar]: min(
                    max(weight, bounds.weight_min), bounds.weight_max
                )
            },
        )
        for indicator, op, threshold, pillar, weight in REFERENCE_RULES
    )
    return Policy(rules, "politica di riferimento: " + reference_rules_text())


#: Le due parole di collegamento della costituzione di riferimento, nelle due
#: lingue del prompt. Il resto della riga --- nomi di indicatore, operatori,
#: nomi di categoria, pesi --- non si traduce: e' il contratto con la macchina.
_SE = {"it": "se", "en": "if"}
_ALTRIMENTI = {
    "it": "altrimenti ogni colono sceglie con le proprie preferenze",
    "en": "otherwise every colonist chooses on their own preferences",
}


def reference_rules_text(lingua: str = "it") -> str:
    """Le regole di riferimento in una riga leggibile, per prompt e registro.

    **Va tradotta, e per poco.** Questa riga entra nel prompt al solo gradino
    completo, che e' il gradino su cui gira la campagna incrociata sulle lingue:
    lasciarla in italiano dentro un prompt inglese avrebbe reso il braccio F un
    testo misto, e la differenza fra i bracci non sarebbe piu' stata la lingua.
    Le uniche parole italiane erano «se» e la chiusa sull'else implicito.
    """
    se = _SE.get(lingua, _SE["it"])
    parts = [
        f"{se} {indicator} {op} {threshold:g} -> {pillar} x{weight:g}"
        for indicator, op, threshold, pillar, weight in REFERENCE_RULES
    ]
    return "; ".join(parts) + "; " + _ALTRIMENTI.get(lingua, _ALTRIMENTI["it"])


def _seed_for(seed: int, step: int) -> int:
    """Seme stabile FRA PROCESSI per il braccio casuale.

    Non `hash(...)`: l'hash di una stringa e' randomizzato a ogni processo
    (PYTHONHASHSEED), quindi lo stesso braccio darebbe policy diverse a ogni
    esecuzione — e l'harness di parita' gira in sottoprocessi. `zlib.crc32` e'
    stabile fra processi, piattaforme e versioni di Python.
    """
    return zlib.crc32(f"{int(seed)}|{int(step)}".encode("utf-8"))


#: Intervalli plausibili delle soglie per il braccio casuale, per indicatore.
#: Coprono la banda che gli indicatori percorrono davvero in una run di tesi:
#: un braccio casuale con soglie a 10^6 non scatterebbe mai e sarebbe una
#: seconda baseline, non un controllo.
#: **Ogni indicatore del vocabolario deve comparire qui.** La parita' espressiva
#: fra i bracci e' un criterio di disegno dichiarato: se il braccio casuale non
#: potesse scrivere una condizione che il braccio linguistico scrive, ogni
#: proprieta' di quella condizione si presenterebbe come una proprieta' del
#: modello. Un test lo verifica sul vocabolario intero, cosi' che aggiungere un
#: indicatore e dimenticare la sua banda rompa subito invece che alla campagna.
_RANDOM_THRESHOLDS: dict[str, tuple[float, float]] = {
    "food_per_occupant": (0.0, 4.0),
    # Il tetto di magazzino e' 4 unita' per occupante: le bande di acqua e
    # ossigeno seguono quella, come quella del cibo.
    "water_per_occupant": (0.0, 4.0),
    "oxygen_per_occupant": (0.0, 4.0),
    "ice_per_occupant": (0.0, 4.0),
    "material_per_occupant": (0.0, 5.0),
    "minerals_per_occupant": (0.0, 5.0),
    # La copertura elettrica ha il suo punto di rottura a 1,0 (fabbisogno
    # coperto esatto): la banda lo attraversa in entrambi i versi.
    "power_coverage": (0.0, 2.0),
    # L'integrita' vive nell'intervallo unitario, e la soglia di intervento
    # della manutenzione e' 0,6.
    "structure_integrity": (0.0, 1.0),
    "occupants": (1.0, 300.0),
}


@dataclass
class GovernorProposal:
    """Che cosa il governatore consegna alla shell, registro compreso."""

    policy: Policy | None
    rationale: str = ""
    provider: str = "none"
    model: str = "none"
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    latency_s: float = 0.0
    drops: PolicyDrops = field(default_factory=PolicyDrops)
    #: Il campo `policy` della risposta COME IL MODELLO L'HA SCRITTO, prima del
    #: parser. Fino al 2026-09-06 il registro conservava solo la catena gia'
    #: parsata: i contatori dicevano "23 pilastri scartati" e nessuno poteva
    #: piu' sapere QUALI nomi il modello avesse usato. `None` per i bracci che
    #: non passano da un testo (scripted, random, replay).
    raw_policy: object = None
    #: **Il fornitore non ha risposto**, distinto da «ha risposto una cosa che
    #: non si puo' leggere». Il primo e' infrastruttura e vale come mancato
    #: aggiornamento: se capita sempre, la configurazione non puo' riuscire. Il
    #: secondo e' un dato sul modello e NON va contato, perche' scartare le run
    #: in cui un modello e' piu' erratico distorce il confronto fra modelli.
    #: Senza questo campo i due casi arrivavano al governatore identici, come
    #: `policy=None`, e un 503 immediato passava per tornata riuscita.
    guasto_fornitore: bool = False
    #: Decisione bounded completa e redatta, presente solo per gli arm
    #: semantici. Non contiene prompt, endpoint o chiavi.
    semantic_decision: dict | None = None
    #: Versione immutabile del profilo di candidate usato in questa tornata.
    candidate_profile: str = ""
    #: Tutte le decisioni SemIf della tornata, in ordine, quando sono piu' di
    #: una (profilo a due livelli). `semantic_decision` resta l'ultima, per i
    #: registri e i lettori che ne conoscono una sola. `None` altrimenti.
    semantic_decisions: list | None = None


class RandomProposer:
    """Policy entro i limiti, riproducibili dal seme."""

    def __init__(self, seed: int) -> None:
        self._seed = int(seed)

    async def propose(self, picture: ColonyPicture, bounds: Bounds) -> GovernorProposal:
        rng = random.Random(_seed_for(self._seed, picture.step))
        names = sorted(INDICATORS)
        pillars_by_name = sorted(PILLAR_BY_NAME)
        rules = []
        for _ in range(rng.randint(1, 3)):
            indicator = rng.choice(names)
            low, high = _RANDOM_THRESHOLDS[indicator]
            pillar = rng.choice(pillars_by_name)
            rules.append(
                Rule(
                    Condition(indicator, rng.choice(("<", ">")), rng.uniform(low, high)),
                    {
                        PILLAR_BY_NAME[pillar]: rng.uniform(
                            bounds.weight_min, bounds.weight_max
                        )
                    },
                )
            )
        rationale = f"policy casuale, seme {self._seed}"
        return GovernorProposal(
            policy=Policy(tuple(rules), rationale),
            rationale=rationale,
        )


class ScriptedProposer:
    """Il governo competente non-LLM: riscrive sempre la stessa costituzione."""

    async def propose(self, picture: ColonyPicture, bounds: Bounds) -> GovernorProposal:
        del picture  # la costituzione non guarda il quadro: le sue condizioni si'.
        policy = reference_policy(bounds)
        return GovernorProposal(policy=policy, rationale=policy.rationale)
