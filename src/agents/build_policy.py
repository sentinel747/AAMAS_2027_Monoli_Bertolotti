from __future__ import annotations

import os

import numpy as np

from src.world.occupancy import housing_slots_from_structures
from src.world.structures import StructureType

# Toggle di sola misura, come gli altri di questo filone: con `0` i conteggi
# tornano a essere ricostruiti scorrendo le viste. I due percorsi danno lo stesso
# dizionario -- sono conteggi interi della stessa colonna -- e il toggle serve a
# poter rimisurare il guadagno con un confronto A/B interlacciato.
_FAST_STRUCTURE_COUNTS = os.environ.get("MARSABM_STRUCT_COUNTS", "1") != "0"

# Colonists served by one building of each type on the same cell (ratios from
# docs/MARS_ABM_NOTE_CONFIGURAZIONE.md). Beyond the ratio the need is met and
# new sites are refused, so materials flow to what is actually missing.
#: Ogni quanti coloni la colonia tira un dado di natalita' in un passo.
#:
#: Vive qui e non in `population` perche' `population` importa questo modulo e
#: non viceversa: una costante sola, letta da chi fa nascere e da chi costruisce.
#: `tests/test_population_growth.py` verifica che i due usi restino allineati.
POPOLAZIONE_PER_DADO_DI_NATALITA = 50


def posti_di_riserva(occupanti: int | float) -> int:
    """I posti che una cella deve tenere liberi perche' una nascita sia possibile.

    **Perche' una riserva serve, e perche' e' QUESTA (2026-08-30).** La
    saturazione era «coperto esattamente il fabbisogno di chi c'e' ora», e una
    nascita richiede `capienza - occupanti > 0`: le due regole si chiudevano
    l'una sull'altra. Misurato su cento coloni fondatori a trecento passi, la
    cella madre finiva con 108 occupanti e capienza 108, e la colonia restava
    ferma per sempre con il 45% dei coloni senza alcuna azione ammissibile —
    **la colonia non poteva costruire la casa che le serviva per crescere,
    perche' non era ancora cresciuta.**

    La riserva non e' un numero scelto: e' quante nascite un passo puo'
    produrre, cioe' un dado ogni `POPOLAZIONE_PER_DADO_DI_NATALITA` coloni. E'
    la piu' piccola riserva che rende una nascita strutturalmente possibile, ed
    e' piccola apposta. Un margine proporzionale generoso (il 15% della
    dotazione iniziale) e' stato provato e **ritirato**: teneva le costruzioni a
    priorita' massima per sempre, e con `top_k = 5` scacciavano dal menu la
    raccolta dalle serre mature — la regressione che
    `test_mature_greenhouses_publish_one_routine_cultivation_job_each`
    sorveglia da prima di questa correzione.
    """
    return max(1, -(-int(max(0, occupanti)) // POPOLAZIONE_PER_DADO_DI_NATALITA))


COLONISTS_PER_STRUCTURE: dict[StructureType, int] = {
    StructureType.HABITAT: 5,
    StructureType.GREENHOUSE: 7,
    # Il pozzo serve quante persone la serra che lo affianca: acqua e cibo
    # scalano insieme, e la resa del pozzo e' ricavata da questo numero.
    StructureType.WATER_EXTRACTOR: 7,
    # Solar tracks population like the greenhouse it powers: the previous
    # local demand formula saturated cell by cell while the colony-wide power
    # margin sank to 0.57 (28 arrays vs ~49 needed) — the load lives in
    # greenhouses and infirmaries scattered across outpost cells.
    StructureType.SOLAR_ARRAY: 7,
    StructureType.OXYGEN_PLANT: 10,
    StructureType.HEATER: 15,
    StructureType.INFIRMARY: 20,
    StructureType.STORAGE_DEPOT: 25,
    StructureType.WEATHER_STATION: 25,
    StructureType.RESEARCH_LAB: 25,
}


def structure_counts(cell) -> dict[StructureType, int]:
    """Quante strutture per tipo ci sono nella cella.

    Questa funzione ha bisogno solo dei CONTEGGI, che nel motore ad array sono
    gia' una colonna (`struct_count`). Ricavarli materializzando e scorrendo una
    vista per struttura costava, misurato sulla configurazione reale, 22,6 us a
    chiamata su 74.868 chiamate -- il 6,4% dell'intera run -- perche' la cella di
    colonia arriva a ~180 strutture e il ciclo le percorreva tutte ogni volta.

    La via veloce non e' una cache: e' leggere il dato al posto di ricostruirlo,
    quindi non c'e' nessuna invalidazione da tenere coerente. Il risultato e'
    identico per costruzione -- il conteggio per tipo E' la colonna, e sono
    interi -- e resta la via lenta per i chiamanti che passano una cella del
    motore a oggetti, che quella colonna non ce l'ha.
    """
    fast = getattr(cell, "structure_counts", None) if _FAST_STRUCTURE_COUNTS else None
    if fast is not None:
        return fast()
    counts = {structure_type: 0 for structure_type in StructureType}
    for structure in cell.structures:
        counts[structure.type] += 1
    return counts


def termini_di_capienza(cell) -> tuple[int, ...]:
    """I quattro termini il cui minimo E' la capienza vitale della cella.

    Alloggi, cibo, energia, ossigeno e acqua: requisiti indipendenti, e finche'
    ne manca uno solo la capienza vale zero. Sono estratti qui perche' due
    domande diverse leggono la stessa quaterna -- «quanti ne sostiene?»
    (`local_life_support_capacity`) e «quanti gliene mancano?»
    (`posti_di_cantiere`) -- e scriverla due volte e' la famiglia di difetto che
    questo modello produce piu' spesso.
    """
    counts = structure_counts(cell)
    housing = int(
        housing_slots_from_structures(
            counts[StructureType.SHELTER],
            counts[StructureType.HABITAT],
            counts[StructureType.INFIRMARY],
        )
    )
    return (
        housing,
        counts[StructureType.GREENHOUSE]
        * COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE],
        counts[StructureType.SOLAR_ARRAY]
        * COLONISTS_PER_STRUCTURE[StructureType.SOLAR_ARRAY],
        counts[StructureType.OXYGEN_PLANT]
        * COLONISTS_PER_STRUCTURE[StructureType.OXYGEN_PLANT],
        # **E l'acqua (2026-09-01).** Mancava, e una cella poteva dichiararsi
        # operativa senza avere di che dissetare nessuno.
        counts[StructureType.WATER_EXTRACTOR]
        * COLONISTS_PER_STRUCTURE[StructureType.WATER_EXTRACTOR],
    )


def local_life_support_capacity(cell) -> int:
    """Residents jointly supported by housing, food, power and oxygen."""
    return max(0, min(termini_di_capienza(cell)))


def posti_di_cantiere(cell) -> int:
    """Quante persone puo' ospitare un avamposto che non e' ancora abitabile.

    **La squadra di fondazione, e perche' e' una squadra (2026-09-01).** Fino a
    oggi un avamposto ammetteva ESATTAMENTE un pioniere: la regola di arrivo
    chiedeva `popolazione == 0`. Ma perche' la cella diventi abitabile servono
    quattro strutture diverse -- alloggio, serra, pannelli, impianto d'ossigeno
    -- e finche' non ci sono tutte la capienza resta zero, quindi la cella
    continua a rifiutare chiunque. Una persona sola, con il solo kit del
    fondatore e su terreno senza giacimenti, non le finisce: misurato a 400
    passi su tre semi, **32 avamposti fermi con capienza zero**, a cui mancava
    l'impianto d'ossigeno in 31 casi, la serra in 25, i pannelli in 15. Molti
    avevano perfino un secondo ricovero costruito al posto di cio' che serviva,
    e ventidue erano gia' stati abbandonati.

    Il numero di posti non e' scelto: e' **quanti dei quattro requisiti
    mancano**, letto dalla stessa quaterna che calcola la capienza. Un fondatore
    per ogni cosa che c'e' da tirare su. Il permesso si stringe da solo mentre
    l'avamposto cresce e si spegne quando la capienza vera subentra.

    E' anche la regola che il selettore del bersaglio gia' applicava per conto
    suo (`_pick_settlement_target` accettava celle con meno di sei presenti):
    erano due regole per la stessa domanda, e si contraddicevano. Ora e' una.
    """
    return sum(1 for termine in termini_di_capienza(cell) if termine <= 0)


def permessi_di_viaggio(agent) -> tuple[bool, bool]:
    """I due permessi d'ingresso che dipendono dallo stato del viaggiatore.

    **Una definizione sola (2026-09-01).** Chi PIANIFICA la rotta e chi
    l'AMMETTE alla porta devono porsi la stessa domanda: erano due, e non
    coincidevano. `_move_toward_cell` sceglieva il passo successivo filtrando
    con `len(agents_present) < 10`, che non e' la regola di arrivo di nessuno,
    e il passo veniva poi respinto da `cell_accepts_arrival` -- misurato, il
    92,6% dei rifiuti erano ricognizioni, e un rifiuto costava al colono
    l'intero passo perche' il ciclo delle preferenze non ha un ripiego.

    Il permesso di sosta d'emergenza (`survival_return`) non sta qui: dipende
    dalla singola richiesta, non dallo stato del colono.
    """
    allow_pioneer = (
        getattr(agent, "settle_phase", None) == "out"
        or bool(getattr(agent, "founder_kit_reserved", False))
    )
    allow_transit = getattr(agent, "scout_phase", None) in {"out", "back"}
    return allow_pioneer, allow_transit


def cell_accepts_arrival(
    origin,
    target,
    pending_arrivals: int = 0,
    *,
    allow_pioneer: bool = False,
    allow_transit: bool = False,
    allow_survival_overflow: bool = False,
) -> bool:
    """Admit one pioneer, supported resident or bounded emergency return.

    A survival route may reserve one otherwise empty transit cell, just like a
    pioneer may reserve one virgin destination.  A physiologically endangered
    resident may additionally use exactly one temporary place above nominal
    capacity in an operational outpost.  The pending-arrival ledger bounds the
    exception, while ordinary migration remains strictly capacity-limited.
    """
    del origin
    target_population = len(target.agents_present) + int(pending_arrivals)
    prospective_population = target_population + 1
    life_support_capacity = float(local_life_support_capacity(target))
    housing_capacity = float(target.occupancy_capacity())
    operational_capacity = min(life_support_capacity, housing_capacity)
    if prospective_population <= operational_capacity:
        return True
    # **Il transito e' la stessa eccezione del rientro (2026-09-01).** Un posto
    # temporaneo sopra la capienza nominale era concesso solo a chi rientrava in
    # emergenza; chi era di PASSAGGIO poteva invece entrare unicamente in una
    # cella vuota. Ma un viandante non prende residenza piu' di quanto la prenda
    # un colono che rientra, e la conseguenza era misurabile: il 55% dei rifiuti
    # di arrivo aveva per bersaglio una cella **operativa e piena**, cioe'
    # esattamente il tipo di cella che una carovana attraversa. Il ricognitore
    # restava fermo, e restare fermo gli costava l'intero passo.
    #
    # Il registro delle prenotazioni tiene il limite: il posto temporaneo e'
    # uno, non uno a testa, quindi una folla non puo' riversarsi nella stessa
    # cella nello stesso passo.
    if (
        (allow_survival_overflow or allow_transit)
        and operational_capacity > 0.0
        and prospective_population <= operational_capacity + 1.0
    ):
        return True
    # **Il cantiere ospita la squadra che gli serve (2026-09-01).** Vale per chi
    # FONDA, non per chi passa: un avamposto non ancora abitabile e' un posto di
    # lavoro, e ci si va per costruirlo. Prima il permesso era `popolazione ==
    # 0`, cioe' una persona sola, e l'avamposto si chiudeva alle spalle del
    # primo arrivato.
    if allow_pioneer and prospective_population <= posti_di_cantiere(target):
        return True
    # Un viandante puo' sostare su terreno vergine, e uno solo per volta: e' la
    # regola di sempre, e resta invariata.
    return (
        target_population == 0
        and int(pending_arrivals) == 0
        and bool(allow_transit)
    )


def fabbisogno_da_coprire(colonists: int) -> int:
    """Le persone per cui la cella deve costruire: le presenti piu' la riserva.

    Una funzione sola, perche' la stessa domanda la pongono la maschera ad array
    (`cell_action_mask`), la seconda maschera in `cell_proposals` e la
    saturazione a oggetti qui sotto.
    """
    presenti = max(1, int(colonists))
    return presenti + posti_di_riserva(presenti)


def costruzione_satura(
    structure_type: StructureType, conteggio, occupanti, alloggi=None, riserva=True
):
    """La cella copre gia' il fabbisogno di questo tipo di struttura?

    **Una regola sola per tre percorsi (2026-08-30).** La stessa domanda era
    scritta tre volte — la maschera di cella (`cell_action_mask`), una SECONDA
    maschera dentro `cell_proposals`, e la saturazione a oggetti qui sotto — e
    proprio quella duplicazione ha reso invisibile la correzione del margine:
    aperto il cancello in due punti su tre, la simulazione non cambiava di un
    solo passo, perche' il terzo lo richiudeva.

    Accetta scalari e array numpy: `conteggio` sono le strutture di quel tipo,
    `occupanti` la popolazione della cella, `alloggi` i posti letto costruiti
    (serve solo per i rifugi, che si misurano in posti e non in unita').
    """
    presenti = np.maximum(1.0, np.asarray(occupanti, dtype=np.float64))
    if riserva:
        coperti = presenti + np.maximum(
            1.0, np.ceil(presenti / float(POPOLAZIONE_PER_DADO_DI_NATALITA))
        )
    else:
        coperti = presenti
    if structure_type == StructureType.SHELTER:
        return np.asarray(alloggi, dtype=np.float64) >= coperti
    ratio = COLONISTS_PER_STRUCTURE.get(structure_type)
    if ratio is None:
        return np.zeros_like(coperti, dtype=np.bool_)
    return np.asarray(conteggio, dtype=np.float64) >= np.maximum(
        1.0, np.ceil(coperti / float(ratio))
    )


def copertura_completa(
    structure: StructureType, cell, colonists_override: int | None = None
) -> bool:
    """La cella copre gia' il fabbisogno di CHI C'E' ORA, senza margine.

    **Non e' la stessa domanda di `structure_saturated`, ed e' la lezione di
    questa correzione (2026-08-30).** Una funzione sola rispondeva a due
    domande diverse: «posso costruirne un'altra?» e «questa cella e' matura
    abbastanza perche' la gente vada a fondarne un'altra?». Aggiungere il
    margine di crescita alla prima e' corretto; farlo anche alla seconda alza
    la soglia di maturita' e **spegne le spedizioni di insediamento** —
    misurato: celle occupate da dieci a cinque e da cinque a due sui semi di
    prova. Il codice conosceva gia' questo rischio: la valvola di sfogo del
    metodo `_maybe_start_settlement` fu scritta perche' con trecento fondatori
    la cella madre non copriva mai l'ossigeno e le spedizioni erano zero.
    """
    counts = structure_counts(cell)
    colonists = max(
        1,
        int(colonists_override)
        if colonists_override is not None
        else len(cell.agents_present),
    )
    alloggi = housing_slots_from_structures(
        counts.get(StructureType.SHELTER, 0),
        counts.get(StructureType.HABITAT, 0),
        counts.get(StructureType.INFIRMARY, 0),
    )
    return bool(
        costruzione_satura(
            structure, counts.get(structure, 0), colonists, alloggi, riserva=False
        )
    )


def structure_saturated(
    structure: StructureType, cell, colonists_override: int | None = None
) -> bool:
    """True when the cell already covers the need for this structure type
    (mirrors the capacity formulas in colony_dynamics). Shared by the
    rule-based policy (skip proposing) and the action executor (refuse opening
    new sites), because with cell-wide shared construction sites a whole crowd
    can otherwise chain-complete copies of the same building in a single step.
    """
    counts: dict[StructureType, int] = {}
    for existing in cell.structures:
        counts[existing.type] = counts.get(existing.type, 0) + 1
    colonists = max(
        1,
        int(colonists_override)
        if colonists_override is not None
        else len(cell.agents_present),
    )
    alloggi = housing_slots_from_structures(
        counts.get(StructureType.SHELTER, 0),
        counts.get(StructureType.HABITAT, 0),
        counts.get(StructureType.INFIRMARY, 0),
    )
    return bool(
        costruzione_satura(
            structure, counts.get(structure, 0), colonists, alloggi
        )
    )
