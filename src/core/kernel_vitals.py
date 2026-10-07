from __future__ import annotations

import numpy as np

from src.agents.vitals import (
    # **L'avvicinamento all'equilibrio e' importato, non trascritto.**
    # `verso_l_equilibrio` usa solo aritmetica e un `exp` fra scalari:
    # funziona identica su un float e su un ndarray, quindi i due motori
    # eseguono LA STESSA riga di codice invece di due copie che possono
    # divergere. E' la forma piu' forte di parita' disponibile qui.
    LINEA_BASE_ADERENZA,
    LINEA_BASE_COOPERAZIONE,
    LINEA_BASE_MORALE,
    LINEA_BASE_STRESS,
    TASSO_RILASSAMENTO_ADERENZA,
    TASSO_RILASSAMENTO_COOPERAZIONE,
    TASSO_RILASSAMENTO_MORALE,
    TASSO_RILASSAMENTO_STRESS,
    verso_l_equilibrio,
    CONSUMO_IDRATAZIONE_PER_PASSO,
    CONSUMO_OSSIGENO_PER_PASSO,
    CONSUMO_SAZIETA_PER_PASSO,
    FATICA_AMBIENTALE_PER_PASSO,
    FATICA_DA_INQUINAMENTO_PER_PASSO,
    FATICA_POLARE_PER_PASSO,
    GIORNI_PER_PASSO_RIFERIMENTO,
    LETHAL_STEPS_WITHOUT_FOOD,
    LETHAL_STEPS_WITHOUT_WATER,
    RISTORO_ACQUA,
    RISTORO_CIBO,
    RISTORO_OSSIGENO,
    OSSIGENO_POLARE_PER_PASSO,
    PASSI_SINTOMO_SENZA_ACQUA,
    PASSI_SINTOMO_SENZA_CIBO,
    QUOTA_OSSIGENO_DA_INQUINAMENTO,
    SALUTE_POLARE_PER_PASSO,
    USURA_RADIAZIONE_PER_PASSO,
    RATION_EPSILON,
    RAZIONE_ACQUA,
    RAZIONE_CIBO,
    RAZIONE_OSSIGENO,
    SUPPORT_FOOD_K,
    SUPPORT_HAB_WATER_K,
    SUPPORT_HEAL_K,
    SUPPORT_OXYGEN_K,
    SUPPORT_WATER_K,
)
from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays

# 1:1 vectorized transcription of src/agents/vitals.py:
#   tick_agent_vitals (lines 100-207), tick_agent_psychosocial (209-238),
#   consume_water_ration/consume_food_ration/consume_oxygen_ration (35-56),
#   death_cause (59-72).
# Every numeric constant below is copied UNCHANGED from that source. The
# object functions remain the truth: if this module and vitals.py ever
# diverge, fix the transcription here, never loosen the equivalence test.
#
# refresh_cell_alerts (vitals.py:75-97) is explicitly NOT part of this
# kernel — it is shell/GUI state (active_alerts), wired later against
# AgentSideState.memory.
#
# Scope note on deaths: both the vitals phase and the psychosocial phase
# (dehydration/starvation/hypoxia/health_collapse via a health<=0 check,
# ordered like death_cause) are reported through the same "deaths" list.
# Neither tick_agent_vitals nor tick_agent_psychosocial in the object model
# ever marks an agent dead or classifies a cause itself — that is entirely
# the runner's job (agent_coupled_runner.py, state_store.py), done via a
# single health<=0 sweep AFTER tick_all_agents has run BOTH phases for every
# agent. This module mirrors that: it sweeps once after the vitals phase and
# once more after the psychosocial phase (which can itself drain health to
# 0 via the high-stress branch), so a death from either phase is always
# reported in the SAME tick, never one tick late.
#
# tick_all_agents (vitals.py:241-252) calls tick_agent_vitals then (when
# enabled) tick_agent_psychosocial for EVERY agent in its input list, with no
# health check gating either call — an agent whose health hit 0 in the vitals
# phase still receives the psychosocial update this same tick, and that
# post-psychosocial stress_index/morale/cooperation/protocol_compliance is
# what agent.to_dict() later captures into dead_agents.jsonl. The world only
# removes the agent afterwards. This module mirrors that: the psychosocial
# phase below runs over the SAME `rows` the vitals phase used (the pre-tick
# alive set), not a re-filtered alive_rows(), so a row that died in the
# vitals phase still gets written stress/morale/cooperation/compliance/health
# here. The post-psychosocial death sweep then excludes rows already present
# in the vitals-phase `deaths` so such a row is never appended twice.




def _serialized_ration_draw(cell_res, res_idx, rows, ys, xs, razione) -> np.ndarray:
    """Una razione per colono, in ordine di riga; vero dove e' stata erogata.

    **Stessa trappola di `_serialized_reserve_draw`, stessa soluzione.** Nel
    motore a oggetti i coloni di una cella attingono uno dopo l'altro, e
    ciascuno vede la giacenza gia' ridotta da chi lo precede. Una lettura
    vettoriale ingenua — `cell_res[ay, ax, ri]` letta una volta e confrontata
    con tutti — servirebbe la stessa unita' a tutti gli occupanti della cella
    e fabbricherebbe massa dal nulla, esattamente il difetto che questo lavoro
    sta chiudendo.

    Le celle con un solo candidato (il caso di gran lunga piu' frequente) si
    risolvono in un passaggio vettoriale; quelle condivise con un ciclo breve
    in ordine di riga, che e' l'ordine di inserimento e quindi l'ordine di
    iterazione del motore a oggetti.
    """
    erogata = np.zeros(rows.size, dtype=bool)
    if rows.size == 0:
        return erogata
    assert np.all(rows[:-1] <= rows[1:]), (
        "_serialized_ration_draw richiede `rows` in ordine crescente (di "
        "inserimento) per riprodurre il prelievo sequenziale del motore a oggetti"
    )
    gruppi: dict[tuple[int, int], list[int]] = {}
    for i in range(rows.size):
        gruppi.setdefault((int(ys[i]), int(xs[i])), []).append(i)

    soli = [idxs[0] for idxs in gruppi.values() if len(idxs) == 1]
    if soli:
        si = np.asarray(soli, dtype=np.intp)
        yy, xx = ys[si], xs[si]
        ok = (cell_res[yy, xx, res_idx] + RATION_EPSILON) >= razione
        cell_res[yy, xx, res_idx] = np.where(
            ok,
            np.maximum(0.0, cell_res[yy, xx, res_idx] - razione),
            cell_res[yy, xx, res_idx],
        )
        erogata[si] = ok

    for (y, x), idxs in gruppi.items():
        if len(idxs) == 1:
            continue
        for i in idxs:
            giacenza = float(cell_res[y, x, res_idx])
            if giacenza + RATION_EPSILON < razione:
                continue
            cell_res[y, x, res_idx] = max(0.0, giacenza - razione)
            erogata[i] = True
    return erogata


def _lascia_le_scorte(agents, cells, righe) -> None:
    """Travasa nella cella l'inventario di chi muore, invece di cancellarlo.

    **Una falla di conservazione sopravvissuta all'audit (2026-08-28).** Il
    decesso si limitava a `agents.alive[riga] = False`, e poiche' ogni somma
    del modello scorre `alive_rows()`, tutto cio' che il colono portava
    *spariva*: misurato su un decesso forzato, `tools -0,999`, `minerals
    -1,000`, `med_kits -1,000`, `construction_material -2,027` contro un
    inventario di 1, 1, 1 e 2. Cibo e ossigeno non lo mostravano perche' la
    produzione delle serre nello stesso passo mascherava la perdita.
    Sull'orizzonte di una campagna e' massa vera: 55 decessi in 1500 passi
    facevano evaporare ~660 unita', e gli attrezzi — che quasi non si
    producono — scendevano da 50 a 24.
    
    Il difetto e' rimasto invisibile perche' il bilancio di massa era sempre
    stato eseguito su scenari **senza morti**: il residuo copre cio' che
    accade, non cio' che non e' stato provato.

    Fisicamente non c'e' nulla da decidere: l'acqua nella borraccia di un
    colono morto e' ancora nella base. Il tetto del magazzino resta quello di
    sempre, quindi un travaso che sfonda la capienza viene scartato dal
    passo successivo come qualunque altra eccedenza.
    """
    if righe.size == 0:
        return
    ys = agents.y[righe].astype(np.intp)
    xs = agents.x[righe].astype(np.intp)
    for indice_risorsa in range(C.NR):
        np.add.at(
            cells.cell_res[:, :, indice_risorsa],
            (ys, xs),
            agents.inv[righe, indice_risorsa],
        )
    agents.inv[righe, :] = 0.0


def _classify_death_cause(hydration: float, steps_without_water: int, satiety: float,
                           steps_without_food: int, oxygen: float) -> str:
    # vitals.py:59-72 death_cause, vectorized-call-site helper shared by both
    # the vitals-phase and the psychosocial-phase death sweeps below so the
    # ordering (dehydration, starvation, hypoxia, health_collapse) can never
    # drift between the two call sites.
    if hydration <= 0.01 or steps_without_water >= LETHAL_STEPS_WITHOUT_WATER:
        return "dehydration"
    if satiety <= 0.01 or steps_without_food >= LETHAL_STEPS_WITHOUT_FOOD:
        return "starvation"
    if oxygen <= 0.05:
        return "hypoxia"
    return "health_collapse"


def tick_vitals(
    agents: AgentArrays,
    cells: CellArrays,
    dt_days: float,
    psychosocial: bool,
    delay_minutes: float,
    drank_mask: np.ndarray,
    ate_mask: np.ndarray,
) -> dict:
    """Vectorized 1:1 transcription of tick_agent_vitals (+ tick_agent_psychosocial
    when psychosocial=True), matching the object pipeline's ordering in
    agent_coupled_runner.py / state_store.py: both phases run for every agent
    BEFORE any death is swept, so a row can die in either phase and still be
    reported in this same call's "deaths" list. When psychosocial=True, the
    psychosocial phase runs over ALL rows alive at the START of this tick,
    including rows the vitals phase just marked dead — mirroring
    tick_all_agents (vitals.py:241-252), which applies tick_agent_psychosocial
    to every agent in its list unconditionally, so a dying agent's final
    stress/morale/cooperation/compliance (later captured into
    dead_agents.jsonl) are the post-psychosocial values, not the
    post-vitals-only values.

    Returns a dict with:
      - "deaths": list[tuple[int, str]] of (row_index, death_cause) pairs.
        The vitals-phase deaths are appended first (in that sweep's row
        order), then the psychosocial-phase deaths (in that sweep's row
        order) — this is NOT necessarily the object engine's insertion order,
        which is a single sweep over agents after both phases run for every
        agent. Each death is independent and self-contained (row + cause), so
        consumers must not rely on cross-death ordering. death_cause follows
        vitals.py's death_cause() ordering (dehydration, starvation, hypoxia,
        health_collapse). A row can appear at most once, even though it may
        receive a psychosocial update after already being marked dead in the
        vitals phase.
      - "memory_events": list[tuple[int, str]] of (row_index, event_text)
        pairs mirroring agent.memory.add_event() calls in the object model
        (polar exposure, dehydration/starvation symptoms and deaths, high
        stress). Callers are responsible for writing these into the actual
        AgentSideState.memory of the corresponding agent.

    Side effects: always mutates `agents` in place (health/oxygen/hydration/
    satiety/fatigue/steps_without_water/steps_without_food/inv/alive columns,
    plus stress/morale/cooperation/compliance when psychosocial=True — written
    for every row alive at the start of the tick, dead-this-tick rows
    included). When psychosocial=True, ALSO overwrites `cells.occupancy` in
    place with the 3x3-neighborhood agent counts of the PRE-death alive set
    for this tick (matching the object model, where world.remove_agent runs
    only after the whole tick — see the "nearby" comment below). This
    occupancy rebuild is unconditional whenever psychosocial=True and the
    function reaches the psychosocial phase at all (i.e. whenever `rows` was
    non-empty at entry) — callers that need occupancy reflecting only
    post-death survivors must recompute it themselves after sweeping
    `deaths`.

    `drank_mask`/`ate_mask` must be boolean arrays indexed by the SAME global
    row convention as `agents` (i.e. length >= agents.n, one entry per agent
    row, not just per alive row) — this function indexes them with `rows`
    (`agents.alive_rows()`) internally.
    """
    deaths: list[tuple[int, str]] = []
    memory_events: list[tuple[int, str]] = []

    rows = agents.alive_rows()
    if rows.size == 0:
        return {"deaths": deaths, "memory_events": memory_events}

    dt = float(min(1.0, max(0.0, float(dt_days) / 3650.0)))
    # **Due scale, e sono deliberate (2026-08-25).** `passo` e' la durata del
    # passo rapportata a quella di riferimento: la usano i vitali di
    # PRIVAZIONE (sazieta', idratazione, ossigeno) e il supporto che li
    # compensa, perche' il consumo di un colono e' per passo e la sua finestra
    # letale si conta in passi. `dt` resta la scala lunga (dt_days/3650) per i
    # processi ambientali — usura da radiazione, esposizione polare, fatica,
    # inquinamento — che si misurano in anni e la cui taratura attuale produce
    # erosioni sensate su orizzonti lunghi.
    #
    # Su quella scala lunga quei termini restano di fatto immobili (la
    # radiazione toglie 0,0027 di salute in cento passi), ed e' un limite noto
    # e dichiarato: portarli sulla scala del passo con i coefficienti attuali
    # ucciderebbe l'intera colonia in un centinaio di passi, e richiede una
    # taratura propria che non e' stata fatta.
    passo = max(0.0, float(dt_days)) / GIORNI_PER_PASSO_RIFERIMENTO

    ay = agents.y[rows].astype(np.intp)
    ax = agents.x[rows].astype(np.intp)

    # ---- per-agent working state (local copies; written back at the end) ----
    health = agents.health[rows].copy()
    oxygen = agents.oxygen[rows].copy()
    hydration = agents.hydration[rows].copy()
    satiety = agents.satiety[rows].copy()
    fatigue = agents.fatigue[rows].copy()
    steps_without_water = agents.steps_without_water[rows].astype(np.int32)
    steps_without_food = agents.steps_without_food[rows].astype(np.int32)

    drank_this_step = np.asarray(drank_mask)[rows].astype(bool)
    ate_this_step = np.asarray(ate_mask)[rows].astype(bool)

    # ---- cell gathers (struct_fx aggregates local_effect over all structures
    # on the cell, same as oxygen_support/habitability_support/food_support/
    # water_support/healing_bonus in vitals.py:15-32) ----
    #
    # **Diviso per gli occupanti, ed e' la correzione dimensionale del
    # 2026-08-25.** `struct_fx` e' una somma su tutte le strutture della cella:
    # una quantita' ESTENSIVA. Leggerla come dose per singolo colono la
    # trattava da intensiva, e il risultato era che il supporto cresceva con la
    # colonia mentre il bisogno pro capite restava fermo. Misurato a piano
    # rispettato: un colono consuma 0,105 di sazieta' per passo e ne riceveva
    # 15,68 a trecento abitanti (149x), 31,36 a seicento (299x). Nessun evento,
    # nessuna scarsita' e nessuna politica poteva scalfire una colonia con quel
    # rapporto — ed e' il difetto che rendeva il mondo insensibile a tutto.
    #
    # Dividendo, il supporto pro capite diventa INDIPENDENTE dalla taglia
    # quando il piano di costruzione e' rispettato (0,213 di cibo per colono
    # da trecento abitanti in su), che e' la proprieta' che rende la taratura
    # valida a ogni scala invece che a una sola.
    occupanti = np.maximum(1.0, cells.occupancy[ay, ax].astype(np.float64))
    oxy_supp = cells.struct_fx[ay, ax, C.E["oxygen"]] / occupanti
    local_support = cells.struct_fx[ay, ax, C.E["habitability"]] / occupanti
    local_water_support = cells.struct_fx[ay, ax, C.E["water"]] / occupanti
    local_food_support = cells.struct_fx[ay, ax, C.E["food"]] / occupanti
    hb = cells.struct_fx[ay, ax, C.E["healing_bonus"]] / occupanti

    # **La razione si paga (2026-08-25).** Trascrizione di `preleva_razione`
    # in vitals.py: un effetto di struttura positivo dice che l'impianto c'e',
    # non che abbia qualcosa da erogare. Il prelievo dalla giacenza della
    # cella decide, e dove la giacenza e' finita il contatore di privazione
    # riparte. Le maschere di supporto vengono azzerate dove la razione non
    # e' stata erogata, cosi' l'aritmetica a valle resta identica al motore a
    # oggetti senza doverla duplicare ramo per ramo.
    razione_ossigeno = np.zeros(rows.size, dtype=bool)
    cand = oxy_supp > 0.0
    if np.any(cand):
        razione_ossigeno[cand] = _serialized_ration_draw(
            cells.cell_res, C.R["oxygen"], rows[cand], ay[cand], ax[cand],
            RAZIONE_OSSIGENO,
        )
    razione_acqua = np.zeros(rows.size, dtype=bool)
    cand = (local_water_support > 0.0) | (local_support > 0.0)
    if np.any(cand):
        razione_acqua[cand] = _serialized_ration_draw(
            cells.cell_res, C.R["water"], rows[cand], ay[cand], ax[cand],
            RAZIONE_ACQUA,
        )
    razione_cibo = np.zeros(rows.size, dtype=bool)
    cand = local_food_support > 0.0
    if np.any(cand):
        razione_cibo[cand] = _serialized_ration_draw(
            cells.cell_res, C.R["food"], rows[cand], ay[cand], ax[cand],
            RAZIONE_CIBO,
        )
    local_water_support = np.where(razione_acqua, local_water_support, 0.0)
    local_food_support = np.where(razione_cibo, local_food_support, 0.0)
    oxy_supp_erogato = np.where(razione_ossigeno, oxy_supp, 0.0)
    habitability_cell = cells.habitability[ay, ax]
    pollution = cells.pollution[ay, ax]
    liquid_water = cells.liquid_water[ay, ax]
    water_ice = cells.water_ice[ay, ax]
    radiation = cells.radiation[ay, ax]
    temp_mod = cells.temp_mod[ay, ax]
    polar_severity = cells.polar[ay, ax]
    struct_heat = cells.struct_fx[ay, ax, C.E["temperature"]]  # cell.structure_heat()

    # vitals.py:112
    hab = np.clip(habitability_cell + local_support, 0.0, 1.0)

    # vitals.py:108, 115
    o2_from_struct = oxy_supp_erogato * SUPPORT_OXYGEN_K * passo
    o2_drain = CONSUMO_OSSIGENO_PER_PASSO * passo * (1.10 - np.minimum(1.0, hab * 6.0 + o2_from_struct * 4.0))

    # vitals.py:117-119 (pollution_risk is never negative, so the unconditional
    # arithmetic form below is exactly equivalent to the source's guarded branch:
    # both added terms are 0.0 wherever pollution_risk == 0.0)
    o2_drain = o2_drain + (
        pollution * QUOTA_OSSIGENO_DA_INQUINAMENTO * CONSUMO_OSSIGENO_PER_PASSO * passo
    )
    fatigue = np.minimum(
        1.0, fatigue + pollution * FATICA_DA_INQUINAMENTO_PER_PASSO * passo
    )

    # vitals.py:121
    oxygen = np.clip(oxygen - o2_drain + o2_from_struct, 0.0, 1.0)

    # vitals.py:122-123 oxygen ration (consume_oxygen_ration, vitals.py:52-56)
    o2_ration_need = (o2_from_struct <= 0.0) & (oxygen < 0.72)
    inv_oxygen = agents.inv[rows, C.R["oxygen"]]
    ox_ok = o2_ration_need & ((inv_oxygen + RATION_EPSILON) >= 0.1)
    if np.any(ox_ok):
        idx = rows[ox_ok]
        agents.inv[idx, C.R["oxygen"]] = np.maximum(0.0, agents.inv[idx, C.R["oxygen"]] - 0.1)
        oxygen = np.where(ox_ok, np.minimum(1.0, oxygen + RISTORO_OSSIGENO), oxygen)

    # vitals.py:125-128
    # I due coefficienti di supporto (abitabilita' e acqua) sono scalati
    # insieme sul supporto normalizzato: l'acqua pro capite che il piano
    # garantisce e' 0,0274, molto piu' piccola del consumo di 0,126, quindi
    # richiede il fattore piu' alto dei tre. Le riserve di cella (acqua
    # liquida e ghiaccio) restano com'erano: sono giacenze, non supporto di
    # struttura, e non vanno divise per gli occupanti.
    supporto_idrico = np.where(
        razione_acqua,
        local_support * SUPPORT_HAB_WATER_K + local_water_support * SUPPORT_WATER_K,
        0.0,
    )
    hydration = np.clip(
        hydration
        - CONSUMO_IDRATAZIONE_PER_PASSO * passo
        + (liquid_water * 0.002 + water_ice * 0.0008 + supporto_idrico) * passo,
        0.0,
        1.0,
    )

    # vitals.py:130 — coefficiente ritarato sul supporto normalizzato
    # (2026-08-25): 0,2129 di cibo pro capite x 0,0916 x 7 = 0,1365 contro un
    # consumo di 0,105, cioe' un margine di 1,3. Col vecchio 0,035 la colonia
    # riceverebbe meta' del suo fabbisogno e morirebbe di fame.
    satiety = np.clip(satiety - CONSUMO_SAZIETA_PER_PASSO * passo + local_food_support * SUPPORT_FOOD_K * passo, 0.0, 1.0)

    # vitals.py:132
    fatigue = np.clip(
        fatigue + FATICA_AMBIENTALE_PER_PASSO * passo * (1.05 - hab * 1.8), 0.0, 1.0
    )

    # vitals.py:134-144 polar branch
    cold_modifier = np.maximum(0.0, -(temp_mod + struct_heat) - 10.0) / 20.0
    polar_exposure = np.maximum(polar_severity, np.minimum(1.0, cold_modifier))
    polar_mask = polar_exposure > 0.0
    life_support = np.minimum(0.85, local_support + oxy_supp * 0.25)
    exposure = polar_exposure * np.maximum(0.0, 1.0 - life_support)
    fatigue = np.where(
        polar_mask, np.minimum(1.0, fatigue + FATICA_POLARE_PER_PASSO * exposure * passo), fatigue
    )
    oxygen = np.where(
        polar_mask, np.maximum(0.0, oxygen - OSSIGENO_POLARE_PER_PASSO * exposure * passo), oxygen
    )
    health = np.where(
        polar_mask, np.maximum(0.0, health - SALUTE_POLARE_PER_PASSO * exposure * passo), health
    )
    severe_polar = polar_mask & (exposure >= 0.55)
    for i in np.flatnonzero(severe_polar):
        memory_events.append(
            (int(rows[i]), "Severe polar cold exposure: retreat or build life support before sustained work.")
        )

    # vitals.py:146-151
    shielding = np.minimum(0.75, local_support * 2.5)
    rad_wear = USURA_RADIAZIONE_PER_PASSO * passo * radiation * (1.0 - shielding)
    hypoxia = np.maximum(0.0, 0.35 - oxygen) * 0.04 * passo
    dehydrate = np.maximum(0.0, 0.3 - hydration) * 0.025 * passo
    starvation = np.maximum(0.0, 0.25 - satiety) * 0.03 * passo
    health = np.clip(health - rad_wear - hypoxia - dehydrate - starvation, 0.0, 1.0)

    # vitals.py:153-156
    water_need = (~razione_acqua) & (~drank_this_step)
    inv_water = agents.inv[rows, C.R["water"]]
    inv_ice = agents.inv[rows, C.R["ice"]]
    water_ok = water_need & ((inv_water + RATION_EPSILON) >= 0.1)
    ice_ok = water_need & (~water_ok) & ((inv_ice + RATION_EPSILON) >= 0.1)
    if np.any(water_ok):
        idx = rows[water_ok]
        agents.inv[idx, C.R["water"]] = np.maximum(0.0, agents.inv[idx, C.R["water"]] - 0.1)
    if np.any(ice_ok):
        idx = rows[ice_ok]
        agents.inv[idx, C.R["ice"]] = np.maximum(0.0, agents.inv[idx, C.R["ice"]] - 0.1)
    used_water_ration = water_ok | ice_ok
    # **La razione personale disseta e sfama davvero (2026-08-25).** Prima
    # azzerava soltanto il CONTATORE di privazione: la borraccia si svuotava,
    # il colono risultava "rifornito", e l'idratazione continuava a scendere.
    # Era invisibile finche' l'idratazione non si muoveva (2,9e-5 per passo);
    # sulla scala del passo diventa un colono che beve la sua scorta e muore
    # di sete lo stesso. La razione d'ossigeno gia' faceva la cosa giusta
    # (+0,25), ed e' quella che fa da modello: acqua e cibo usano gli stessi
    # cambi delle azioni corrispondenti, +0,35 di idratazione per 0,1 d'acqua
    # e +0,25 di sazieta' per 0,1 di cibo.
    hydration = np.where(used_water_ration, np.minimum(1.0, hydration + RISTORO_ACQUA), hydration)

    food_need = (~razione_cibo) & (~ate_this_step)
    inv_food = agents.inv[rows, C.R["food"]]
    food_ok = food_need & ((inv_food + RATION_EPSILON) >= 0.1)
    if np.any(food_ok):
        idx = rows[food_ok]
        agents.inv[idx, C.R["food"]] = np.maximum(0.0, agents.inv[idx, C.R["food"]] - 0.1)
    used_food_ration = food_ok
    satiety = np.where(used_food_ration, np.minimum(1.0, satiety + RISTORO_CIBO), satiety)

    # vitals.py:158-165
    water_reset = drank_this_step | used_water_ration | razione_acqua
    steps_without_water = np.where(water_reset, 0, steps_without_water + 1).astype(np.int32)
    hydration = np.where(
        water_reset,
        hydration,
        np.minimum(hydration, np.maximum(0.0, 1.0 - steps_without_water / LETHAL_STEPS_WITHOUT_WATER)),
    )

    # vitals.py:167-174
    food_reset = ate_this_step | used_food_ration | razione_cibo
    steps_without_food = np.where(food_reset, 0, steps_without_food + 1).astype(np.int32)
    satiety = np.where(
        food_reset,
        satiety,
        np.minimum(satiety, np.maximum(0.0, 1.0 - steps_without_food / LETHAL_STEPS_WITHOUT_FOOD)),
    )

    # vitals.py:176-184
    dehydration_death = hydration <= 0.0
    health = np.where(dehydration_death, 0.0, health)
    for i in np.flatnonzero(dehydration_death):
        memory_events.append(
            (int(rows[i]), f"Died from dehydration after {int(steps_without_water[i])} consecutive steps without drinking.")
        )
    dehydration_symptom = (~dehydration_death) & (steps_without_water >= PASSI_SINTOMO_SENZA_ACQUA)
    health = np.where(dehydration_symptom, np.maximum(0.0, health - 0.12 * dt), health)
    fatigue = np.where(dehydration_symptom, np.minimum(1.0, fatigue + 0.10 * dt), fatigue)
    for i in np.flatnonzero(dehydration_symptom):
        memory_events.append((
            int(rows[i]),
            f"Dehydration symptoms after {PASSI_SINTOMO_SENZA_ACQUA} "
            "consecutive steps without drinking.",
        ))

    # vitals.py:186-194
    starvation_death = satiety <= 0.0
    health = np.where(starvation_death, 0.0, health)
    for i in np.flatnonzero(starvation_death):
        memory_events.append(
            (int(rows[i]), f"Died from starvation after {int(steps_without_food[i])} consecutive steps without food.")
        )
    starvation_symptom = (~starvation_death) & (steps_without_food >= PASSI_SINTOMO_SENZA_CIBO)
    health = np.where(starvation_symptom, np.maximum(0.0, health - 0.08 * dt), health)
    fatigue = np.where(starvation_symptom, np.minimum(1.0, fatigue + 0.08 * dt), fatigue)
    for i in np.flatnonzero(starvation_symptom):
        memory_events.append((
            int(rows[i]),
            f"Starvation symptoms after {PASSI_SINTOMO_SENZA_CIBO} "
            "consecutive steps without food.",
        ))

    # vitals.py:196 refresh_cell_alerts — deliberately NOT reproduced here.

    # vitals.py:198-206 healing bonus, gated on the same "return" the source
    # takes for agents whose health already hit 0 this tick.
    alive_after = health > 0.0
    hb_mask = alive_after & (hb > 0.0)
    # La cura vale ora quanto una cura: con 0,0275 di `healing_bonus` pro
    # capite garantito dal piano (un'infermeria ogni venti coloni), il
    # coefficiente porta la rigenerazione a ~0,058 di salute per passo. Prima
    # della normalizzazione erano +1,73 per passo in una capitale, contro un
    # danno massimo di 0,10 da un brillamento: nessun evento e nessuna
    # carenza potevano competere, ed e' il motivo per cui non moriva mai
    # nessuno.
    # La cura e' per passo come il vitto (vedi vitals.py).
    health = np.where(hb_mask, np.minimum(1.0, health + hb * SUPPORT_HEAL_K * passo), health)
    oxygen = np.where(hb_mask, np.minimum(1.0, oxygen + hb * SUPPORT_HEAL_K * 0.667 * passo), oxygen)
    hydration = np.where(hb_mask, np.minimum(1.0, hydration + hb * SUPPORT_HEAL_K * 0.5 * passo), hydration)
    fatigue = np.where(hb_mask, np.maximum(0.0, fatigue - hb * SUPPORT_HEAL_K * passo), fatigue)

    # write back
    agents.health[rows] = health
    agents.oxygen[rows] = oxygen
    agents.hydration[rows] = hydration
    agents.satiety[rows] = satiety
    agents.fatigue[rows] = fatigue
    agents.steps_without_water[rows] = steps_without_water
    agents.steps_without_food[rows] = steps_without_food

    # ---- death determination (vitals.py:59-72 death_cause, vectorized) ----
    died_mask = health <= 0.0
    if np.any(died_mask):
        for i in np.flatnonzero(died_mask):
            row = int(rows[i])
            cause = _classify_death_cause(
                float(hydration[i]), int(steps_without_water[i]),
                float(satiety[i]), int(steps_without_food[i]), float(oxygen[i]),
            )
            deaths.append((row, cause))
        _lascia_le_scorte(agents, cells, rows[died_mask])
        agents.alive[rows[died_mask]] = False
        # Un morto esce da `agents_present`: invalida l'indice per-cella
        # (vedi `AgentArrays.positions_index`).
        agents.touch_positions()

    if not psychosocial:
        return {"deaths": deaths, "memory_events": memory_events}

    # ---- tick_agent_psychosocial (vitals.py:209-238), vectorized ----
    # tick_all_agents (vitals.py:241-252) calls tick_agent_psychosocial for
    # EVERY agent in its list unconditionally, right after tick_agent_vitals,
    # with no health check in between — an agent whose health hit 0 in the
    # vitals phase (this row's `died_mask` above) still gets the psychosocial
    # update this same tick, and that post-update stress/morale/cooperation/
    # compliance is what agent.to_dict() later captures into
    # dead_agents.jsonl. So rows_p is `rows` itself (the pre-tick alive set),
    # NOT a re-filtered alive_rows() — every row the vitals phase started
    # with, dead-this-tick rows included, is an update target here. The
    # death-cause re-check further below excludes rows already in
    # `died_mask` so such a row is never appended to `deaths` twice.
    rows_p = rows

    ay_p = ay
    ax_p = ax

    # nearby = occupancy of the 3x3 neighborhood centered on the agent, minus
    # self (vitals.py:219, world.get_agents_in_range(x, y, 1) minus 1).
    # cells.occupancy is a persistent field on CellArrays that the biology
    # kernel will also read once it exists; refresh it in place rather than
    # in a scratch array so it stays current for the rest of this step's
    # pipeline.
    #
    # Built from `rows` (the PRE-death alive set used by the vitals phase
    # above) — which, per the note above, is now also exactly the update
    # target set `rows_p`. In the object pipeline (agent_coupled_runner.py,
    # state_store.py) world.remove_agent runs only AFTER tick_all_agents has
    # finished BOTH phases for every agent, so an agent who died in THIS
    # tick's vitals phase is still present in world._agent_positions and
    # still counted by world.get_agents_in_range() for every other agent's
    # psychosocial pass this same step — matching `ay_p`/`ax_p` being the
    # same pre-death coordinate arrays used for the occupancy grid itself, so
    # the "-1" self-exclusion below is exactly correct for every row.
    cells.occupancy[:] = 0
    np.add.at(cells.occupancy, (ay, ax), 1)
    padded = np.zeros((cells.H + 2, cells.W + 2), dtype=np.int32)
    padded[1:-1, 1:-1] = cells.occupancy
    neigh_sum = np.zeros((cells.H, cells.W), dtype=np.int32)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            neigh_sum += padded[1 + dy : 1 + dy + cells.H, 1 + dx : 1 + dx + cells.W]
    nearby = np.maximum(0, neigh_sum[ay_p, ax_p] - 1)

    oxygen_p = agents.oxygen[rows_p]
    hydration_p = agents.hydration[rows_p]
    satiety_p = agents.satiety[rows_p]
    fatigue_p = agents.fatigue[rows_p]
    autonomy_p = agents.autonomy[rows_p]
    stress_p = agents.stress[rows_p].copy()
    morale_p = agents.morale[rows_p].copy()
    cooperation_p = agents.cooperation[rows_p].copy()
    compliance_p = agents.compliance[rows_p].copy()
    health_p = agents.health[rows_p].copy()

    # Stessa divisione del blocco fisiologico: il supporto psicosociale di una
    # struttura si spartisce fra gli occupanti come ogni altro.
    occupanti_p = np.maximum(1.0, cells.occupancy[ay_p, ax_p].astype(np.float64))
    local_support_p = cells.struct_fx[ay_p, ax_p, C.E["habitability"]] / occupanti_p
    hb_p = cells.struct_fx[ay_p, ax_p, C.E["healing_bonus"]] / occupanti_p
    radiation_p = cells.radiation[ay_p, ax_p]
    dust_p = cells.dust[ay_p, ax_p]

    # vitals.py:220
    compagnia = np.minimum(0.18, nearby * 0.045)
    support = local_support_p + hb_p * 0.5 + compagnia
    # vitals.py:221-226
    deprivation = (
        np.maximum(0.0, 0.55 - oxygen_p)
        + np.maximum(0.0, 0.55 - hydration_p)
        + np.maximum(0.0, 0.55 - satiety_p)
        + np.maximum(0.0, fatigue_p - 0.55)
    )
    # vitals.py:227
    environmental_pressure = np.maximum(0.0, radiation_p - 0.8) * 0.05 + dust_p * 0.035
    # vitals.py:228
    # La compagnia non va contata due volte, una in `support` e una qui col
    # segno meno: vedi la nota in `vitals.tick_agent_psychosocial`.
    isolation_pressure = np.where(nearby == 0, 0.045, 0.0)
    # vitals.py:229
    delay_pressure = min(0.055, float(delay_minutes) / 22.0 * 0.035) * (1.0 - autonomy_p * 0.45)
    # vitals.py:230-231
    # Anche qui la scala e' quella del passo (vedi vitals.py).
    spinta_stress = (
        isolation_pressure + deprivation * 0.045 + environmental_pressure
        + delay_pressure - support * 0.10
    )
    stress_p = np.clip(stress_p + verso_l_equilibrio(
        stress_p, LINEA_BASE_STRESS, spinta_stress,
        TASSO_RILASSAMENTO_STRESS, passo), 0.0, 1.0)
    # vitals.py:232-233
    # Trascrizione vettoriale del tetto sulla compagnia: vedi la nota in
    # `vitals.tick_agent_psychosocial`. Le due forme devono restare
    # identiche, ed e' un test di equivalenza a verificarlo.
    spinta_morale = (
        support * 0.075 + compagnia * (0.008 / 0.045)
        - stress_p * 0.045 - deprivation * 0.025
    )
    morale_p = np.clip(morale_p + verso_l_equilibrio(
        morale_p, LINEA_BASE_MORALE, spinta_morale,
        TASSO_RILASSAMENTO_MORALE, passo), 0.0, 1.0)
    # vitals.py:234
    spinta_cooperazione = (morale_p - 0.55) * 0.012 - stress_p * 0.008
    cooperation_p = np.clip(cooperation_p + verso_l_equilibrio(
        cooperation_p, LINEA_BASE_COOPERAZIONE, spinta_cooperazione,
        TASSO_RILASSAMENTO_COOPERAZIONE, passo), 0.0, 1.0)
    # vitals.py:235
    spinta_aderenza = (morale_p - stress_p - 0.15) * 0.006
    compliance_p = np.clip(compliance_p + verso_l_equilibrio(
        compliance_p, LINEA_BASE_ADERENZA, spinta_aderenza,
        TASSO_RILASSAMENTO_ADERENZA, passo), 0.0, 1.0)
    # vitals.py:236-238
    high_stress = stress_p > 0.82
    health_p = np.where(high_stress, np.maximum(0.0, health_p - 0.012 * passo), health_p)
    for i in np.flatnonzero(high_stress):
        memory_events.append(
            (int(rows_p[i]), "High stress is degrading performance; prioritize rest, communication, or habitat support.")
        )

    agents.stress[rows_p] = stress_p
    agents.morale[rows_p] = morale_p
    agents.cooperation[rows_p] = cooperation_p
    agents.compliance[rows_p] = compliance_p
    agents.health[rows_p] = health_p

    # ---- death determination AFTER the psychosocial phase too ----
    # The high-stress branch above (vitals.py:236-238) can drain health to
    # 0.0 by itself. The object pipeline sweeps for agent.health <= 0 once,
    # AFTER tick_all_agents has run both tick_agent_vitals and
    # tick_agent_psychosocial for every agent (agent_coupled_runner.py:220,
    # state_store.py:410) — so such an agent must be reported dead in THIS
    # same tick's return value, not one tick later. hydration/satiety/oxygen/
    # steps_without_* are untouched by the psychosocial phase, so the cause
    # is classified from the same values written back after the vitals
    # phase. rows_p is now `rows` itself (see the comment above the
    # psychosocial section), which INCLUDES every row the vitals-phase sweep
    # already marked dead and appended to `deaths` — those rows' `health_p`
    # starts at <=0.0 (written back by the vitals phase) and stays <=0.0
    # through the psychosocial update, so without an explicit exclusion they
    # would satisfy `health_p <= 0.0` again here and be double-reported.
    # `died_mask` (from the vitals-phase sweep above) is aligned index-for-
    # index with `rows`/`rows_p`, so `~died_mask` excludes exactly those rows.
    died_mask_p = (health_p <= 0.0) & (~died_mask)
    if np.any(died_mask_p):
        # hydration_p/satiety_p/oxygen_p (gathered above, before the
        # psychosocial writes) already hold the right values: neither is
        # touched by tick_agent_psychosocial. steps_without_* need a fresh
        # gather since rows_p wasn't indexed against them yet.
        sww_pd = agents.steps_without_water[rows_p]
        swf_pd = agents.steps_without_food[rows_p]
        for i in np.flatnonzero(died_mask_p):
            row = int(rows_p[i])
            cause = _classify_death_cause(
                float(hydration_p[i]), int(sww_pd[i]),
                float(satiety_p[i]), int(swf_pd[i]), float(oxygen_p[i]),
            )
            deaths.append((row, cause))
        _lascia_le_scorte(agents, cells, rows_p[died_mask_p])
        agents.alive[rows_p[died_mask_p]] = False
        agents.touch_positions()

    return {"deaths": deaths, "memory_events": memory_events}
