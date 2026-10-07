from __future__ import annotations

"""NumPy phase-B primitives for the preference decision policy.

The scalar implementation in :mod:`src.agents.preference_agent` remains the
readable oracle.  Operations below deliberately preserve its expression order
so equivalence tests can require bit-exact float64 results.
"""

import numpy as np

from src.agents import pillars
from src.agents.action_space import (
    ActionType,
    BUILD_ACTIONS,
    RESOURCE_EPSILON,
    WATER_RESERVE_CAP,
)
from src.agents.preference_agent import (
    CRITICAL_HEALTH,
    CRITICAL_HYDRATION,
    CRITICAL_SATIETY,
    URGENCY_VERSION,
    _BUILD_INDICES,
    _SURVIVAL_ACTIONS,
    RAZIONE_MINIMA,
    decide_preferences_precomputed,
)
from src.core import constants as C
from src.core import physio_gate
from src.world.structures import BUILD_COSTS


def agent_action_mask_batch(
    agents,
    rows: np.ndarray,
    cell_masks: np.ndarray,
    individual_survival_priority_enabled: bool = True,
    cells=None,
) -> np.ndarray:
    """Apply the scalar personal-feasibility mask to a batch of rows."""
    rows = np.asarray(rows, dtype=np.int64)
    masks = np.asarray(cell_masks, dtype=np.bool_).copy()
    inv = agents.inv[rows]
    if individual_survival_priority_enabled:
        masks[:, pillars.ACTION_INDEX[ActionType.PHYSIOLOGICAL_RECOVERY]] = False

    for action in pillars.AUTOMATIC_CELL_SERVICES:
        masks[:, pillars.ACTION_INDEX[action]] = False

    masks[:, pillars.ACTION_INDEX[ActionType.USE_MED_KIT]] &= (
        inv[:, C.R["med_kits"]] >= 1.0
    )
    masks[:, pillars.ACTION_INDEX[ActionType.EAT_FOOD]] &= (
        inv[:, C.R["food"]] + RESOURCE_EPSILON >= 0.1
    )
    drink_unavailable = (
        (inv[:, C.R["water"]] + RESOURCE_EPSILON < 0.1)
        & (inv[:, C.R["ice"]] + RESOURCE_EPSILON < 0.1)
        & ~masks[:, pillars.ACTION_INDEX[ActionType.COLLECT_ICE]]
    )
    masks[drink_unavailable, pillars.ACTION_INDEX[ActionType.DRINK_WATER]] = False
    reserve_full = (
        inv[:, C.R["water"]]
        + inv[:, C.R["ice"]]
        + RESOURCE_EPSILON
        >= WATER_RESERVE_CAP
    )
    masks[reserve_full, pillars.ACTION_INDEX[ActionType.REFILL_WATER]] = False

    for action, structure_type in BUILD_ACTIONS.items():
        cost = BUILD_COSTS[structure_type]
        affordable = np.ones(rows.size, dtype=np.bool_)
        for resource in C.RESOURCES:
            stock = inv[:, C.R[resource]]
            if cells is not None:
                stock = stock + cells.cell_res[
                    agents.y[rows], agents.x[rows], C.R[resource]
                ]
            affordable &= stock + RESOURCE_EPSILON >= float(
                getattr(cost, resource)
            )
        site_in_progress = np.zeros(rows.size, dtype=np.bool_)
        if cells is not None:
            site_in_progress = (
                cells.site_progress[
                    agents.y[rows], agents.x[rows], C.S[structure_type]
                ]
                >= 0.0
            )
        masks[:, pillars.ACTION_INDEX[action]] &= affordable | site_in_progress

    maintenance_stock = inv[:, C.R["construction_material"]]
    if cells is not None:
        maintenance_stock = maintenance_stock + cells.cell_res[
            agents.y[rows], agents.x[rows], C.R["construction_material"]
        ]
    masks[:, pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]] &= (
        maintenance_stock + RESOURCE_EPSILON >= 1.0
    )

    if individual_survival_priority_enabled:
        critical = (
            (agents.hydration[rows] < CRITICAL_HYDRATION)
            | (agents.satiety[rows] < CRITICAL_SATIETY)
            | (agents.health[rows] < CRITICAL_HEALTH)
        )
        # Trascrizione vettoriale della deroga scalare in
        # `preference_agent.agent_action_mask`: dove la cella non puo' erogare
        # ne' cibo ne' acqua, riparare l'impianto E' l'azione di
        # sopravvivenza, perche' e' l'unica che riapre quel canale. Le due
        # forme devono restare identiche: e' `_SURVIVAL_ACTIONS` la fonte
        # comune dell'insieme, e questa e' la sola eccezione, scritta con la
        # stessa condizione.
        if cells is not None:
            non_eroga = (
                cells.cell_res[agents.y[rows], agents.x[rows], C.R["food"]]
                < RAZIONE_MINIMA
            ) | (
                cells.cell_res[agents.y[rows], agents.x[rows], C.R["water"]]
                < RAZIONE_MINIMA
            )
        else:
            non_eroga = np.zeros(rows.size, dtype=np.bool_)
        for action in pillars.ACTION_ORDER:
            if action in _SURVIVAL_ACTIONS:
                continue
            indice = pillars.ACTION_INDEX[action]
            if action is ActionType.MAINTAIN_STRUCTURE:
                masks[critical & ~non_eroga, indice] = False
            else:
                masks[critical, indice] = False
    return masks


def compute_urgencies_batch(
    agents, rows: np.ndarray, masks_at_agent: np.ndarray,
    kit_fondatore=None,
) -> np.ndarray:
    """Compute the six scalar urgency formulas for all selected rows."""
    rows = np.asarray(rows, dtype=np.int64)
    masks = np.asarray(masks_at_agent, dtype=np.bool_)
    urgency = np.zeros((rows.size, pillars.N_PILLARS), dtype=np.float64)
    inv = agents.inv[rows]

    deprived = (agents.steps_without_water[rows] >= 2) | (
        agents.steps_without_food[rows] >= 6
    )
    urgency[:, pillars.P_SUSTENANCE] = np.minimum(
        1.0,
        np.maximum(1.0 - agents.hydration[rows], 1.0 - agents.satiety[rows])
        ** 2
        * 1.5
        + np.where(deprived, 0.3, 0.0),
    )
    urgency[:, pillars.P_RESOURCES] = np.clip(
        1.0
        - 0.5
        * (
            inv[:, C.R["construction_material"]] / 10.0
            + inv[:, C.R["minerals"]] / 6.0
        ),
        0.0,
        1.0,
    )
    has_buildable = np.any(masks[:, _BUILD_INDICES], axis=1)
    urgency[:, pillars.P_BUILD] = 0.6 * has_buildable.astype(np.float64) + 0.4 * np.minimum(
        1.0, inv[:, C.R["construction_material"]] / 10.0
    )
    urgency[:, pillars.P_LIFE] = np.minimum(
        1.0,
        np.maximum(1.0 - agents.health[rows], agents.fatigue[rows])
        ** 2
        * 1.3
        + 0.2 * agents.stress[rows],
    )
    urgency[:, pillars.P_SOCIAL] = np.minimum(
        1.0,
        0.8 * agents.stress[rows]
        + np.maximum(0.0, 0.82 - agents.morale[rows]),
    )
    # **Il kit riservato e' missione attiva quanto una fase.** La strada scalare
    # lo considera da sempre (`preference_agent._urgenze`); questa guardava solo
    # la colonna `mission`, dove il kit non c'e' perche' vive nello stato cold.
    # Risultato: lo stesso colono, nello stesso passo, riceveva 1,00 o 0,55 a
    # seconda di quale strada lo serviva.
    in_missione = agents.mission[rows] != 0
    if kit_fondatore is not None:
        in_missione = in_missione | np.asarray(kit_fondatore, dtype=bool)
    urgency[:, pillars.P_EXPLORE] = np.minimum(
        1.0,
        0.25 + 0.6 * agents.curiosity[rows] + 0.5 * in_missione,
    )
    return urgency


def pillar_availability_batch(action_masks: np.ndarray) -> np.ndarray:
    """Collapse action masks to the six pillar availability columns."""
    masks = np.asarray(action_masks, dtype=np.bool_)
    available = np.zeros((masks.shape[0], pillars.N_PILLARS), dtype=np.bool_)
    for pillar, actions in pillars.PILLAR_ACTIONS.items():
        indices = [pillars.ACTION_INDEX[action] for action in actions]
        available[:, pillar] = np.any(masks[:, indices], axis=1)
    return available


def pillar_priority_batch(
    action_masks: np.ndarray, action_priorities: np.ndarray
) -> np.ndarray:
    """Batch equivalent of the scalar cell-priority collapse."""
    masks = np.asarray(action_masks, dtype=np.bool_)
    priorities = np.asarray(action_priorities, dtype=np.float64)
    result = np.zeros((masks.shape[0], pillars.N_PILLARS), dtype=np.float64)
    for pillar, actions in pillars.PILLAR_ACTIONS.items():
        indices = [pillars.ACTION_INDEX[action] for action in actions]
        if not indices:
            continue
        values = np.where(masks[:, indices], priorities[:, indices], 0.0)
        result[:, pillar] = np.max(values, axis=1)
    return result


def score_pillars_batch(
    preferences: np.ndarray,
    urgencies: np.ndarray,
    availability: np.ndarray,
    cell_priority: np.ndarray | None = None,
    skills: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize preference x urgency row-wise after availability masking."""
    if cell_priority is None:
        cell_priority = np.ones_like(preferences, dtype=np.float64)
    if skills is None:
        skills = np.ones_like(preferences, dtype=np.float64)
    raw = (
        np.asarray(preferences, dtype=np.float64)
        * np.asarray(urgencies, dtype=np.float64)
        * np.asarray(cell_priority, dtype=np.float64)
        * np.asarray(skills, dtype=np.float64)
        * np.asarray(availability, dtype=np.bool_)
    )
    totals = raw.sum(axis=1)
    scores = np.zeros_like(raw, dtype=np.float64)
    positive = totals > 0.0
    scores[positive] = raw[positive] / totals[positive, None]
    return scores


def choose_pillar_batch(
    scores: np.ndarray, sampling: str, uniforms: np.ndarray
) -> np.ndarray:
    """Batch equivalent of ``choose_pillar`` with one uniform per row."""
    scores = np.asarray(scores, dtype=np.float64)
    if sampling == "greedy":
        return np.argmax(scores, axis=1).astype(np.int64, copy=False)
    if sampling != "softmax":
        raise ValueError(f"unknown decision sampling: {sampling}")
    cumulative = np.cumsum(scores, axis=1)
    totals = cumulative[:, -1]
    thresholds = np.asarray(uniforms, dtype=np.float64) * totals
    chosen = np.sum(cumulative <= thresholds[:, None], axis=1)
    chosen = np.minimum(chosen, pillars.N_PILLARS - 1)
    chosen[totals <= 0.0] = 0
    return chosen.astype(np.int64, copy=False)


def decide_batch(
    agents,
    cells,
    side,
    pre_rows: np.ndarray,
    cell_masks: np.ndarray,
    cell_priorities: np.ndarray,
    cell_quotas: np.ndarray,
    uniforms: np.ndarray,
    sampling: str,
    step_index: int,
    by_cell: dict[tuple[int, int], list[str]],
    *,
    world,
    agent_views: dict,
    individual_survival_priority_enabled: bool = True,
    semantic_pillar_factors: dict | None = None,
    semantic_action_factors: dict | None = None,
    pillar_margin_out: dict | None = None,
    semantic_counters: dict | None = None,
) -> dict:
    """Vectorize through pillar choice, then bind in deterministic row order.

    I quattro argomenti `semantic_*`/`pillar_margin_out` appartengono allo
    strato SemIf degli agenti. A `None` (default) il corpo esegue esattamente
    le operazioni di prima: nessun prodotto per uno, nessuna copia.
    """
    del side  # represented by ``world`` and the supplied live views.
    rows = np.asarray(pre_rows, dtype=np.int64)
    at_agent = cell_masks[agents.y[rows], agents.x[rows]]
    priorities_at_agent = cell_priorities[agents.y[rows], agents.x[rows]]
    quotas_at_agent = cell_quotas[agents.y[rows], agents.x[rows]]
    personal_masks = agent_action_mask_batch(
        agents,
        rows,
        at_agent,
        individual_survival_priority_enabled,
        cells,
    )
    available = pillar_availability_batch(personal_masks)
    cell_priority = pillar_priority_batch(personal_masks, priorities_at_agent)
    if individual_survival_priority_enabled:
        preferences = agents.pref[rows]
        # Il kit sta nello stato cold, non in una colonna: si legge dalle
        # viste vive, in ordine di riga, e costa trecento accessi a dizionario
        # per passo.
        kit = np.fromiter(
            (
                bool(getattr(agent_views.get(agents.ids[int(r)]),
                             "founder_kit_reserved", False))
                for r in rows
            ),
            dtype=bool,
            count=int(rows.size),
        )
        urgencies = compute_urgencies_batch(
            agents, rows, personal_masks, kit_fondatore=kit
        )
        skills = agents.skill[rows]
    else:
        preferences = agents.pref[rows]
        urgencies = np.ones_like(preferences)
        skills = agents.skill[rows]
    scores = score_pillars_batch(
        preferences, urgencies, available, cell_priority, skills
    )
    if pillar_margin_out is not None and scores.shape[0]:
        ordinati = np.sort(scores, axis=1)
        margini = ordinati[:, -1] - ordinati[:, -2]
        for index, row in enumerate(rows):
            pillar_margin_out[agents.ids[int(row)]] = float(margini[index])
    chosen = choose_pillar_batch(scores, sampling, uniforms)
    if semantic_pillar_factors is not None:
        # Il fattore semantico entra come un'abilita' in piu': stesso prodotto,
        # stessa normalizzazione, stesso uniforme. Un agente assente vale 1.
        fattori = np.ones_like(scores, dtype=np.float64)
        for index, row in enumerate(rows):
            vettore = semantic_pillar_factors.get(agents.ids[int(row)])
            if vettore is not None:
                fattori[index] = np.asarray(vettore, dtype=np.float64)
        scores_sem = score_pillars_batch(
            preferences, urgencies, available, cell_priority,
            np.asarray(skills, dtype=np.float64) * fattori,
        )
        chosen_sem = choose_pillar_batch(scores_sem, sampling, uniforms)
        if semantic_counters is not None:
            semantic_counters["agent_decisions"] = (
                semantic_counters.get("agent_decisions", 0) + int(rows.size)
            )
            semantic_counters["pillar_changed"] = (
                semantic_counters.get("pillar_changed", 0)
                + int(np.count_nonzero(chosen_sem != chosen))
            )
        scores, chosen = scores_sem, chosen_sem

    # Un solo attraversamento per passo, prima del ciclo: chi non puo' avere
    # un'emergenza fisiologica viene dimostrato in blocco, cosi' che il ciclo
    # sequenziale non chiami la funzione per poi scoprire che restituisce `None`
    # (cosa che fa nel 99,98% dei casi). `None` significa "filtro spento": in quel
    # caso ogni riga chiama, che e' il comportamento storico.
    skip_override = (
        physio_gate.skip_override(agents, cells, rows)
        if individual_survival_priority_enabled
        else None
    )

    requests = {}
    claims: dict[tuple[int, int, int], int] = {}
    # La spedizione la equipaggia la COMUNITA', non il solo magazzino:
    # `_equip_founder_kit_from_warehouse` ha bisogno dei compagni di cella
    # per attingere alle loro sacche. Chiave stringa, quindi non collide
    # con le chiavi a tupla delle prenotazioni.
    claims["__agenti__"] = agent_views
    for index, row in enumerate(rows):
        agent_id = agents.ids[int(row)]
        agent = agent_views[agent_id]
        riga_priorita = priorities_at_agent[index]
        # Argomenti dello strato semantico: vuoti senza strato, cosi' la
        # chiamata resta identica a quella storica.
        semantici: dict = {}
        if semantic_pillar_factors is not None:
            vettore = semantic_pillar_factors.get(agent_id)
            if vettore is not None:
                semantici["semantic_pillar_row"] = vettore
        if semantic_action_factors is not None:
            fattori_azione = semantic_action_factors.get(agent_id)
            if fattori_azione is not None:
                semantici["scoring_priority_row"] = riga_priorita
                riga_priorita = riga_priorita * np.asarray(
                    fattori_azione, dtype=np.float64
                )
        requests[agent_id] = decide_preferences_precomputed(
            agent,
            world,
            personal_masks[index],
            float(uniforms[index]),
            sampling,
            claims,
            int(step_index),
            precomputed_scores=scores[index],
            precomputed_chosen=int(chosen[index]),
            action_priority_row=riga_priorita,
            quota_row=quotas_at_agent[index],
            individual_survival_priority_enabled=individual_survival_priority_enabled,
            skip_physiological_override=(
                skip_override is not None and bool(skip_override[index])
            ),
            **semantici,
        )
    return requests
