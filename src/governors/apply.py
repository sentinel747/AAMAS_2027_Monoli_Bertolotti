from __future__ import annotations

"""Applicazione della policy al menu che ogni cella pubblica.

La policy **inclina** cio' che la cella ha calcolato, non lo sostituisce: la
priorita' di una cella codifica un bisogno reale, ricavato da `compute_needs`,
e il governo la piega senza cancellarla. Un peso non puo' rendere possibile
un'azione che la cella vieta — la maschera di ammissibilita' azzera prima, e su
zero un moltiplicatore e' un non-intervento. La cella resta il filtro; il
governo, un'inclinazione.

**Dove agisce.** Sulla `priority` per azione, PRIMA del taglio top-k, nello
stesso punto d'innesto della vecchia direttiva (`priority_tilt` dentro
`compute_cell_proposals`): applicato dopo, un peso potrebbe solo riordinare un
menu gia' composto, mai far entrare un'azione ammissibile che il taglio aveva
escluso. Pesare tutte le azioni di un pilastro moltiplica esattamente per quel
peso la `cell_priority` del pilastro nel prodotto dei sei fattori — cioe' e' il
"peso sulle preferenze globali" chiesto dal disegno, senza toccare i tre
backend bit-exact di `decision_scoring`.

**Semantica: prima regola che scatta vince, per cella.** La catena e' un
if/elif/else valutato in blocco: `remaining` parte dalle celle occupate — il
governo agisce su chi c'e', non sul terreno vuoto — e ogni regola consuma le
celle che ha preso. Dove nessuna regola scatta, il menu resta quello della
baseline e il colono sceglie con le sole proprie preferenze.

Gli array vengono mutati **in posto**, per la stessa ragione della vecchia
`tilt_priority`: sono appena stati costruiti da `compute_cell_proposals`,
nessun altro li ha visti, e la funzione corre dentro `kernel.step`. Le due
forme — griglia intera `[H,W,A]` e sottoinsieme compatto `[1,K,A]` — passano
dalla stessa identica aritmetica: gli indicatori si calcolano dagli array di
`cells`, che hanno la stessa forma della `priority`, quindi non serve alcuna
doppia indicizzazione.
"""

import numpy as np

from src.agents import pillars
from src.core import constants as C
from src.core.kernel_biology import _PESI_CARICO
from src.governors.policy import Condition, Policy, rule_text

#: Indici di azione per pilastro, risolti una volta all'import: dentro il passo
#: sarebbero risoluzioni di dizionario ripetute.
_PILLAR_ACTION_INDICES: dict[int, tuple[int, ...]] = {
    pillar: tuple(pillars.ACTION_INDEX[action] for action in actions)
    for pillar, actions in pillars.PILLAR_ACTIONS.items()
    if actions
}


def _power_coverage(cells) -> np.ndarray:
    """Energia disponibile diviso il carico degli impianti della cella.

    E' il margine di riserva che un gestore di rete pubblica, ed e' anche la
    grandezza che governa davvero la produzione: sotto 1,0 gli impianti
    producono in proporzione (vedi `kernel_biology`). Dove non c'e' carico la
    copertura e' piena per definizione — una cella senza impianti non ha un
    problema di corrente.
    """
    if getattr(cells, "copertura_vera_attiva", False):
        # Interruttore `copertura elettrica vera` (2026-09-26): la copertura
        # calcolata dal kernel nell'ultimo passo (energia usata / carico).
        # Prima del primo passo nessuna cella e' ancora stata alimentata:
        # copertura piena, come per le celle senza carico.
        vera = getattr(cells, "copertura_elettrica", None)
        if vera is not None:
            return vera
        return np.ones(cells.occupancy.shape, dtype=np.float64)
    carico = cells.struct_count @ _PESI_CARICO
    energia = cells.cell_res[:, :, C.R["energy"]]
    return np.divide(
        energia, carico, out=np.ones_like(carico), where=carico > 0.0
    )


def _structure_integrity(cells) -> np.ndarray:
    """Integrita' media delle strutture della cella, da 0 a 1.

    `struct_integrity` e' la SOMMA sulle istanze di ciascun tipo: la media si
    ottiene dividendo per il conteggio, ed e' 1,0 dove non c'e' niente da
    manutenere (nessun arretrato).
    """
    conteggi = cells.struct_count.astype(np.float64)
    totale = conteggi.sum(axis=-1)
    integrita = cells.struct_integrity.sum(axis=-1)
    return np.divide(
        integrita, totale, out=np.ones_like(totale), where=totale > 0.0
    )


def indicator_values(name: str, cells) -> np.ndarray:
    """Il valore dell'indicatore, per cella, nella forma degli array di `cells`.

    Il denominatore e' `max(occupancy, 1)`: nelle celle vuote non decide
    nessuno e `apply_policy` non le tocca comunque, ma un indicatore non deve
    poter dividere per zero. `ice_per_occupant` somma giacenza e superficie
    perche' e' la somma che decide se `collect_ice` passa — lo stesso conto che
    faceva il quadro delle celle.
    """
    occupants = np.maximum(cells.occupancy, 1).astype(np.float64)
    if name == "occupants":
        return cells.occupancy.astype(np.float64)
    if name == "food_per_occupant":
        return cells.cell_res[:, :, C.R["food"]] / occupants
    if name == "ice_per_occupant":
        return (cells.cell_res[:, :, C.R["ice"]] + cells.water_ice) / occupants
    if name == "material_per_occupant":
        return cells.cell_res[:, :, C.R["construction_material"]] / occupants
    if name == "minerals_per_occupant":
        return cells.cell_res[:, :, C.R["minerals"]] / occupants
    if name == "water_per_occupant":
        return cells.cell_res[:, :, C.R["water"]] / occupants
    if name == "oxygen_per_occupant":
        return cells.cell_res[:, :, C.R["oxygen"]] / occupants
    if name == "power_coverage":
        return _power_coverage(cells)
    if name == "structure_integrity":
        return _structure_integrity(cells)
    raise KeyError(
        f"indicatore {name!r} fuori vocabolario: parse_policy non doveva lasciarlo passare"
    )


def _condition_mask(condition: Condition, cells) -> np.ndarray:
    values = indicator_values(condition.indicator, cells)
    if condition.op == "<":
        return values < condition.value
    return values > condition.value


#: La chiave dell'else nei contatori di scatto. Con un trattino davanti, cosi'
#: non puo' collidere con `rule_text` di nessuna regola reale.
ELSE_KEY = "- nessuna regola (preferenze pure)"


def apply_policy(
    priority: np.ndarray,
    policy: Policy,
    cells,
    hits: dict | None = None,
    scope: np.ndarray | None = None,
) -> None:
    """Valuta la policy su ogni cella occupata e pesa la `priority` in posto.

    Con una policy senza regole non tocca niente: il chiamante (il kernel) non
    la invoca affatto quando la policy e' `None`, quindi la baseline resta
    bit-exact per costruzione e non per verifica.

    ``hits``, se passato, accumula quante (cella, passo) ciascuna regola ha
    catturato — chiave `rule_text` — piu' l'else sotto `ELSE_KEY`. E' la
    risposta al problema dell'ombra: con la semantica prima-regola-vince una
    regola larga scritta per prima affama le successive, e senza questo conto
    lo si scopre solo strumentando a mano. Il conteggio e' pura osservazione:
    non tocca ne' gli array ne' l'ordine delle operazioni, quindi la parita'
    bit-exact non lo vede — ed e' un test a garantirlo, non questa frase.
    """
    dentro = cells.occupancy > 0
    if scope is not None:
        dentro = dentro & np.asarray(scope, dtype=np.bool_)
    if not policy.rules:
        if hits is not None:
            hits[ELSE_KEY] = hits.get(ELSE_KEY, 0) + int(dentro.sum())
        return
    remaining = dentro
    for rule in policy.rules:
        if not remaining.any():
            break
        if rule.condition is None:
            hit = remaining
        else:
            hit = remaining & _condition_mask(rule.condition, cells)
            if not hit.any():
                continue
        if hits is not None:
            key = rule_text(rule)
            hits[key] = hits.get(key, 0) + int(hit.sum())
        for pillar, weight in rule.weights.items():
            for action in _PILLAR_ACTION_INDICES.get(pillar, ()):
                priority[hit, action] *= weight
        remaining = remaining & ~hit
    if hits is not None:
        rimaste = int(remaining.sum())
        if rimaste:
            hits[ELSE_KEY] = hits.get(ELSE_KEY, 0) + rimaste


def consuntivo_per_regola(policy, hits_ora: dict | None, hits_prima: dict | None) -> tuple:
    """Quante celle ogni regola ha catturato DALL'ULTIMO tick, in ordine di scrittura.

    I contatori del kernel sono cumulativi sull'intera run: la differenza fra
    due tick e' cio' che la politica attualmente in vigore ha fatto da quando e'
    entrata in vigore, ed e' l'unica forma utile a chi deve riscriverla. Il
    cumulato direbbe anche cio' che hanno fatto le politiche precedenti, che non
    sono piu' sue.

    L'ordine e' quello in cui il governatore ha scritto le regole, non quello
    dei conteggi: e' l'ordine che decide chi affama chi, e leggerlo in ordine
    decrescente nasconderebbe proprio il fatto che si vuole rendere visibile.
    """
    if policy is None or not getattr(policy, "rules", ()):
        return ()
    ora = hits_ora or {}
    prima = hits_prima or {}
    fuori = []
    for regola in policy.rules:
        chiave = rule_text(regola)
        fuori.append((chiave, int(ora.get(chiave, 0)) - int(prima.get(chiave, 0))))
    fuori.append((ELSE_KEY, int(ora.get(ELSE_KEY, 0)) - int(prima.get(ELSE_KEY, 0))))
    return tuple(fuori)


def apply_layered(
    priority: np.ndarray,
    governor_policy: Policy | None,
    district_policies: dict,
    district_ids: np.ndarray,
    cells,
    hits: dict | None = None,
) -> None:
    """La catena del governo, con i distretti che la riscrivono sulle proprie celle.

    **L'ordine e' quello dell'amministrazione reale, e non e' commutativo.**
    Una cella appartenente a un distretto riceve la catena del suo
    amministratore *invece* di quella del governo, non in aggiunta: applicarle
    entrambe moltiplicherebbe due volte gli stessi pilastri e l'amministratore
    non potrebbe piu' *correggere* il governo, solo rincarare. Dove
    l'amministratore ha deciso di non intervenire, il suo distretto non compare
    in `district_policies` e la cella ricade sotto la catena del governo.

    La cella madre e le celle non ancora assegnate hanno `district_ids == -1` e
    restano sempre di competenza diretta del governo.

    Le maschere sono disgiunte per costruzione --- un identificativo per cella
    --- quindi nessuna priorita' viene toccata due volte, e con
    `district_policies` vuoto l'effetto e' identico ad `apply_policy` sulla sola
    catena del governo: la baseline resta bit-exact per costruzione.
    """
    ids = np.asarray(district_ids)
    coperte = np.zeros_like(ids, dtype=np.bool_)
    for distretto, policy in sorted(district_policies.items()):
        if policy is None:
            continue
        scope = ids == int(distretto)
        if not scope.any():
            continue
        coperte |= scope
        parziali = {} if hits is not None else None
        apply_policy(priority, policy, cells, hits=parziali, scope=scope)
        if hits is not None:
            for chiave, quante in parziali.items():
                etichetta = f"[distretto {distretto}] {chiave}"
                hits[etichetta] = hits.get(etichetta, 0) + quante
    if governor_policy is not None:
        apply_policy(priority, governor_policy, cells, hits=hits, scope=~coperte)
