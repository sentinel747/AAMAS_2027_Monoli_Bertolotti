from __future__ import annotations

"""Readable phase-A primitives for preference-based decisions.

These scalar formulas are the reference oracle for the vectorized phase-B
implementation in ``src.core.kernel_decision``.
"""

import os

import numpy as np

from src.agents import pillars
from src.agents.action_space import (
    ActionRequest,
    ActionType,
    BUILD_ACTIONS,
    RESOURCE_EPSILON,
    WATER_RESERVE_CAP,
    has_structure_at_local_position,
)
from src.agents.rule_based_agent import RuleBasedAgent, _has_food_source, _has_water_source
from src.agents.build_policy import (
    cell_accepts_arrival,
    local_life_support_capacity,
    permessi_di_viaggio,
)
from src.world.structures import BUILD_COSTS, StructureType

URGENCY_VERSION = 1

# Ordine di applicazione dei due filtri sui vicini in `_bind_explore` (vedi la
# nota estesa li'). Risolto una volta all'import: e' letto dentro un percorso
# eseguito migliaia di volte per step, dove una lettura di variabile d'ambiente
# per chiamata costerebbe piu' di quanto la scelta faccia risparmiare.
_CHEAP_FILTER_FIRST = (
    os.environ.get("MARSABM_FILTER_ORDER", "cheap_first").strip().lower() != "legacy"
)

# Memo per fase di decisione dentro `_bind_explore` (vedi `_decision_memo`).
# Come gli altri toggle di questo filone, `0` riporta al percorso storico: i due
# percorsi calcolano la stessa cosa, e il toggle serve a poter rimisurare il
# guadagno con un confronto A/B interlacciato invece di doverlo dichiarare.
_BIND_MEMO_ENABLED = os.environ.get("MARSABM_BIND_MEMO", "1") != "0"

# M2 tuning after the 300-agent diagnostic: at the original 0.35/0.30/0.30
# thresholds the seven-step dehydration clock was already unrecoverable for
# explorers several cells from support. These values engage the hard guard
# while a return path is still physically possible.
CRITICAL_HYDRATION = 0.75
CRITICAL_SATIETY = 0.70
CRITICAL_HEALTH = 0.55
WATER_GUARD_STEPS = 2
FOOD_GUARD_STEPS = 6
WATER_RETURN_RESERVE = 3.2
FOOD_RETURN_RESERVE = 1.0

_SURVIVAL_ACTIONS = frozenset(
    {
        ActionType.PHYSIOLOGICAL_RECOVERY,
        ActionType.DRINK_WATER,
        ActionType.REFILL_WATER,
        ActionType.EAT_FOOD,
        ActionType.COLLECT_ICE,
        ActionType.FORAGE,
        ActionType.USE_MED_KIT,
        ActionType.REST,
        # **`OBSERVE` NON e' un'azione di sopravvivenza (2026-08-31).** Il suo
        # esecutore restituisce `ActionResult(True, "observed area")` e non
        # tocca nulla: non puo' contribuire a tenere in vita nessuno. Restava
        # qui, e siccome e' sempre ammissibile e a quota illimitata era cio'
        # che un colono in stato critico finiva per fare quando ogni altra
        # azione di sopravvivenza era impraticabile. Misurato nella cella in
        # crisi della run `verifica3000_seed9`: **57,8% delle azioni**, contro
        # 33,6% di `drink_water` e ZERO manutenzioni, mentre l'integrita' del
        # parco scendeva a 0,072 e la gente moriva di fame. Fuori di qui un
        # colono critico senza nulla di praticabile riceve `DO_NOTHING`, che
        # dice il vero.
        ActionType.MOVE,
        ActionType.DO_NOTHING,
    }
)

#: La razione che un impianto eroga in un passo. Duplicare il numero preso da
#: `vitals` creerebbe la solita coppia destinata a divergere, ma importarlo
#: qui creerebbe un ciclo (`vitals` non importa questo modulo, ma
#: `action_space` si'): il valore vive in `action_space`, che questo modulo
#: gia' importa, e da li' lo si legge.
RAZIONE_MINIMA = 0.1


def _la_cella_non_eroga(cell_resources) -> bool:
    """La cella non puo' dare ne' da bere ne' da mangiare, questo passo.

    E' la condizione in cui le azioni di sopravvivenza ordinarie non possono
    piu' nulla: `forage` viene respinta su una dispensa vuota (misurato: 170
    rifiuti nella finestra di crisi), `eat_food` e' gia' mascherata su una
    sacca vuota, e bere ripaga esattamente il consumo di un passo senza mai
    far risalire l'idratazione sopra la soglia critica. Da li' non si esce, ed
    e' uno stato assorbente del livello decisionale.
    """
    if cell_resources is None:
        return False
    return (
        float(getattr(cell_resources, "food", 0.0)) < RAZIONE_MINIMA
        or float(getattr(cell_resources, "water", 0.0)) < RAZIONE_MINIMA
    )
#: Le azioni che rendono il pilastro `build` "utile qui e ora", e da cui discende
#: la sua urgenza. **Derivate dal pilastro, non dal catalogo delle costruzioni
#: (2026-08-31).** Prima erano le sole dieci azioni di erezione (`BUILD_ACTIONS`),
#: mentre `PILLAR_ACTIONS[P_BUILD]` contiene anche `MAINTAIN_STRUCTURE`: due
#: definizioni per lo stesso insieme, e quella usata dall'urgenza escludeva
#: proprio il lavoro che conta quando non c'e' piu' niente da erigere.
#:
#: **Cosa costava.** L'urgenza vale `0,6 x c_e_qualcosa_da_fare + 0,4 x
#: materiale_in_sacca/10`, e il punteggio di un pilastro e' un PRODOTTO: un
#: fattore nullo lo cancella, per quanto alta sia la priorita' che la cella gli
#: assegna. In una cella dove tutto e' gia' costruito ma tutto sta cadendo a
#: pezzi, con le sacche vuote, l'urgenza di `build` era esattamente ZERO.
#: Misurato sulla cella in crisi della run da 3000 passi: maschera ammessa,
#: priorita' di cella 1,0, quota 12 su 13 occupanti, e **nessuna manutenzione
#: eseguita**, mentre l'integrita' del parco scendeva a 0,072 e 132 coloni
#: morivano di fame. La cella chiedeva la cosa giusta e nessuno poteva sentirla.
_BUILD_INDICES = tuple(
    pillars.ACTION_INDEX[action] for action in pillars.PILLAR_ACTIONS[pillars.P_BUILD]
)


def compute_urgencies(agent, mask_row: np.ndarray) -> np.ndarray:
    """Compute bounded urgency for the six pillars for one agent."""
    urgency = np.zeros(pillars.N_PILLARS, dtype=np.float64)
    inventory = agent.inventory
    deprived = int(getattr(agent, "steps_without_water", 0)) >= 2 or int(
        getattr(agent, "steps_without_food", 0)
    ) >= 6
    urgency[pillars.P_SUSTENANCE] = min(
        1.0,
        max(1.0 - agent.hydration, 1.0 - agent.satiety) ** 2 * 1.5
        + (0.3 if deprived else 0.0),
    )
    urgency[pillars.P_RESOURCES] = np.clip(
        1.0
        - 0.5
        * (
            float(inventory.construction_material) / 10.0
            + float(inventory.minerals) / 6.0
        ),
        0.0,
        1.0,
    )
    has_buildable = any(bool(mask_row[index]) for index in _BUILD_INDICES)
    urgency[pillars.P_BUILD] = 0.6 * float(has_buildable) + 0.4 * min(
        1.0, float(inventory.construction_material) / 10.0
    )
    urgency[pillars.P_LIFE] = min(
        1.0,
        max(1.0 - agent.health, agent.fatigue) ** 2 * 1.3
        + 0.2 * float(agent.stress_index),
    )
    urgency[pillars.P_SOCIAL] = min(
        1.0,
        0.8 * float(agent.stress_index)
        + max(0.0, 0.82 - float(agent.morale)),
    )
    mission_active = (
        getattr(agent, "scout_phase", None) is not None
        or getattr(agent, "settle_phase", None) is not None
        or bool(getattr(agent, "founder_kit_reserved", False))
    )
    urgency[pillars.P_EXPLORE] = min(
        1.0,
        0.25
        + 0.6 * float(agent.curiosity)
        + (0.5 if mission_active else 0.0),
    )
    return urgency


def agent_action_mask(
    agent,
    cell_mask_row: np.ndarray,
    individual_survival_priority_enabled: bool = True,
    cell_resources=None,
    cell=None,
) -> np.ndarray:
    """Apply personal feasibility and the physiological safety guard."""
    mask = np.asarray(cell_mask_row, dtype=np.bool_).copy()
    if individual_survival_priority_enabled:
        mask[pillars.ACTION_INDEX[ActionType.PHYSIOLOGICAL_RECOVERY]] = False
    for action in pillars.AUTOMATIC_CELL_SERVICES:
        mask[pillars.ACTION_INDEX[action]] = False
    inventory = agent.inventory
    if float(getattr(inventory, "med_kits", 0.0)) < 1.0:
        mask[pillars.ACTION_INDEX[ActionType.USE_MED_KIT]] = False
    if float(getattr(inventory, "food", 0.0)) + RESOURCE_EPSILON < 0.1:
        mask[pillars.ACTION_INDEX[ActionType.EAT_FOOD]] = False
    # Cell ice availability is already represented by COLLECT_ICE. The action
    # validator remains the final authority for liquid-water/structure sources.
    if (
        float(getattr(inventory, "water", 0.0)) + RESOURCE_EPSILON < 0.1
        and float(getattr(inventory, "ice", 0.0)) + RESOURCE_EPSILON < 0.1
        and not bool(mask[pillars.ACTION_INDEX[ActionType.COLLECT_ICE]])
    ):
        mask[pillars.ACTION_INDEX[ActionType.DRINK_WATER]] = False
    if (
        float(getattr(inventory, "water", 0.0))
        + float(getattr(inventory, "ice", 0.0))
        + RESOURCE_EPSILON
        >= WATER_RESERVE_CAP
    ):
        mask[pillars.ACTION_INDEX[ActionType.REFILL_WATER]] = False
    for action, structure_type in BUILD_ACTIONS.items():
        index = pillars.ACTION_INDEX[action]
        cost = BUILD_COSTS[structure_type]
        site_in_progress = cell is not None and any(
            str(site_key).split("@", 1)[0] == structure_type.value
            for site_key in cell.construction_sites
        )
        can_afford = inventory.can_afford(cost)
        if not can_afford and cell_resources is not None:
            can_afford = all(
                float(getattr(inventory, field, 0.0))
                + float(getattr(cell_resources, field, 0.0))
                + RESOURCE_EPSILON
                >= float(getattr(cost, field, 0.0))
                for field in cost.__dataclass_fields__
            )
        # The full construction cost is paid only when a site is opened.
        # Continuing an existing site must therefore remain feasible even
        # after its founder inventory and the cell warehouse have been spent.
        if mask[index] and not (can_afford or site_in_progress):
            mask[index] = False
    maintain_index = pillars.ACTION_INDEX[ActionType.MAINTAIN_STRUCTURE]
    maintenance_stock = float(getattr(inventory, "construction_material", 0.0))
    if cell_resources is not None:
        maintenance_stock += float(
            getattr(cell_resources, "construction_material", 0.0)
        )
    if maintenance_stock + RESOURCE_EPSILON < 1.0:
        mask[maintain_index] = False

    critical = (
        float(agent.hydration) < CRITICAL_HYDRATION
        or float(agent.satiety) < CRITICAL_SATIETY
        or float(agent.health) < CRITICAL_HEALTH
    )
    if individual_survival_priority_enabled and critical:
        # **Riparare l'impianto E' un'azione di sopravvivenza, quando la cella
        # non eroga piu' (2026-08-31).** La guardia esiste per impedire a un
        # colono che sta morendo di andare a costruire altrove, ed e' giusta.
        # Ma bloccarlo su un insieme di azioni che nel suo caso non possono
        # nulla lo condanna: con la dispensa vuota `forage` viene respinta,
        # `eat_food` e' mascherata, e bere ripaga esattamente il consumo di un
        # passo senza far risalire l'idratazione sopra la soglia. Misurato: la
        # serra ferma a integrita' 0,072 e nessuno che la riparasse, perche'
        # tutti erano "critici". Riparare la serra e' l'unica cosa che riapre
        # il canale del cibo, e in quello stato e' letteralmente l'azione che
        # tiene in vita.
        consentite = _SURVIVAL_ACTIONS
        if _la_cella_non_eroga(cell_resources):
            consentite = _SURVIVAL_ACTIONS | {ActionType.MAINTAIN_STRUCTURE}
        for action in pillars.ACTION_ORDER:
            if action not in consentite:
                mask[pillars.ACTION_INDEX[action]] = False
    return mask


def pillar_availability(action_mask_row: np.ndarray) -> np.ndarray:
    """Collapse action-level feasibility into six pillar flags."""
    available = np.zeros(pillars.N_PILLARS, dtype=np.bool_)
    for pillar, actions in pillars.PILLAR_ACTIONS.items():
        available[pillar] = any(
            bool(action_mask_row[pillars.ACTION_INDEX[action]]) for action in actions
        )
    return available


def pillar_priority(
    action_mask_row: np.ndarray, action_priority_row: np.ndarray
) -> np.ndarray:
    """Collapse cell proposal priorities to one multiplier per pillar."""
    result = np.zeros(pillars.N_PILLARS, dtype=np.float64)
    for pillar, actions in pillars.PILLAR_ACTIONS.items():
        values = [
            float(action_priority_row[pillars.ACTION_INDEX[action]])
            for action in actions
            if bool(action_mask_row[pillars.ACTION_INDEX[action]])
        ]
        result[pillar] = max(values, default=0.0)
    return result


def score_pillars(
    preferences: np.ndarray,
    urgencies: np.ndarray,
    availability: np.ndarray,
    cell_priority: np.ndarray | None = None,
    skills: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize preference x urgency x cell priority after masking."""
    if cell_priority is None:
        cell_priority = np.ones(pillars.N_PILLARS, dtype=np.float64)
    if skills is None:
        skills = np.ones(pillars.N_PILLARS, dtype=np.float64)
    raw = (
        np.asarray(preferences, dtype=np.float64)
        * np.asarray(urgencies, dtype=np.float64)
        * np.asarray(cell_priority, dtype=np.float64)
        * np.asarray(skills, dtype=np.float64)
        * np.asarray(availability, dtype=np.bool_)
    )
    total = float(raw.sum())
    if total <= 0.0:
        return np.zeros(pillars.N_PILLARS, dtype=np.float64)
    return raw / total


def choose_pillar(scores: np.ndarray, sampling: str, u01: float) -> int:
    """Choose one pillar using stable greedy or one-uniform categorical draw."""
    if sampling == "greedy":
        return int(np.argmax(scores))
    if sampling != "softmax":
        raise ValueError(f"unknown decision sampling: {sampling}")
    cumulative = np.cumsum(np.asarray(scores, dtype=np.float64))
    if cumulative[-1] <= 0.0:
        return 0
    index = np.searchsorted(cumulative, float(u01) * cumulative[-1], side="right")
    return min(int(index), pillars.N_PILLARS - 1)


_MAINTAIN_CLAIMS_PER_CELL = 1
_COLLECT_ACTIONS = (
    ActionType.COLLECT_ICE,
    ActionType.COLLECT_MINERALS,
    ActionType.COLLECT_MATERIALS,
)
_COLLECTION_RESOURCE = {
    ActionType.COLLECT_ICE: ("ice", 2.0),
    ActionType.COLLECT_MINERALS: ("minerals", 1.0),
    ActionType.COLLECT_MATERIALS: ("construction_material", 1.0),
}


def _resource_claim_key(cell, resource: str) -> tuple:
    return ("resource", int(cell.y), int(cell.x), str(resource))


def _unclaimed_cell_resource(cell, resource: str, claims: dict) -> float:
    available = float(getattr(cell.resources, resource, 0.0))
    if resource == "ice":
        available += float(getattr(cell, "water_ice", 0.0))
    return max(0.0, available - float(claims.get(_resource_claim_key(cell, resource), 0.0)))


def _cell_resource_requirements(
    agent, action: ActionType, cell=None
) -> dict[str, float]:
    """Return the warehouse share a selected project must reserve this step."""
    if action in BUILD_ACTIONS:
        structure_type = BUILD_ACTIONS[action]
        # Construction inputs are paid atomically when a site is opened.
        # Requiring the full cost again in the intra-step claim ledger made
        # the personal feasibility mask say "continue" and this later layer
        # say "cannot afford": founders then observed forever beside a paid
        # 33/50% site.  An active site only reserves the finite worker quota.
        if cell is not None and any(
            str(site_key).split("@", 1)[0] == structure_type.value
            for site_key in cell.construction_sites
        ):
            return {}
        cost = BUILD_COSTS[structure_type]
    elif action == ActionType.MAINTAIN_STRUCTURE:
        from src.world.resources import ResourceBundle

        cost = ResourceBundle(construction_material=1.0)
    else:
        return {}
    requirements: dict[str, float] = {}
    for resource in cost.__dataclass_fields__:
        required = float(getattr(cost, resource, 0.0))
        personal = float(getattr(agent.inventory, resource, 0.0))
        cell_share = max(0.0, required - personal)
        if cell_share > RESOURCE_EPSILON:
            requirements[resource] = cell_share
    return requirements


def _request(agent, action: ActionType, *, target=None, message: str) -> ActionRequest:
    return ActionRequest(
        agent.agent_id, action, target=target, message=f"[pref] {message}"
    )


def _claim_cap(
    action: ActionType, cell, quota_row: np.ndarray | None = None
) -> int | None:
    if quota_row is not None:
        configured = int(quota_row[pillars.ACTION_INDEX[action]])
        return None if configured < 0 else configured
    if action == ActionType.MAINTAIN_STRUCTURE:
        return _MAINTAIN_CLAIMS_PER_CELL
    pools = {
        ActionType.COLLECT_ICE: float(cell.water_ice) + float(cell.resources.ice),
        ActionType.COLLECT_MINERALS: float(cell.resources.minerals),
        ActionType.COLLECT_MATERIALS: float(cell.resources.construction_material),
    }
    if action not in pools:
        return None
    return max(1, int(pools[action]))


def _apply_claim_limits(
    mask: np.ndarray,
    agent,
    cell,
    claims: dict,
    quota_row: np.ndarray | None = None,
) -> None:
    actions = pillars.ACTION_ORDER if quota_row is not None else (*_COLLECT_ACTIONS, ActionType.MAINTAIN_STRUCTURE)
    for action in actions:
        index = pillars.ACTION_INDEX[action]
        cap = _claim_cap(action, cell, quota_row)
        if cap is not None and claims.get((cell.y, cell.x, index), 0) >= cap:
            mask[index] = False
    # Quotas stop duplicate actions; these reservations stop different actions
    # from promising the same warehouse unit before sequential execution.  In
    # particular, a build/maintenance job and COLLECT_MATERIALS can no longer
    # both spend the final construction-material unit in one weekly step.
    for action, (resource, _units) in _COLLECTION_RESOURCE.items():
        index = pillars.ACTION_INDEX[action]
        if bool(mask[index]) and _unclaimed_cell_resource(cell, resource, claims) <= RESOURCE_EPSILON:
            mask[index] = False
    for action in (*BUILD_ACTIONS, ActionType.MAINTAIN_STRUCTURE):
        index = pillars.ACTION_INDEX[action]
        if not bool(mask[index]):
            continue
        requirements = _cell_resource_requirements(agent, action, cell)
        if any(
            _unclaimed_cell_resource(cell, resource, claims) + RESOURCE_EPSILON < amount
            for resource, amount in requirements.items()
        ):
            mask[index] = False


def _record_claim(
    agent,
    action: ActionType,
    cell,
    claims: dict,
    quota_row: np.ndarray | None = None,
) -> None:
    """Iscrive la scelta nei DUE registri, che sono libri diversi.

    Il tetto di teste esiste solo quando una quota lo fissa; la prenotazione sul
    magazzino della cella esiste **sempre**, perche' la risorsa e' finita
    comunque. Prima questa funzione usciva su `_claim_cap(...) is None` e
    portava via con se' anche la prenotazione: con `quota = -1` — che significa
    "nessun tetto di teste", non "nessuna contabilita'" — il registro restava
    vuoto, la guardia sulla risorsa in `_apply_claim_limits` non si chiudeva mai
    e l'intera cella proponeva la raccolta su uno stock che ne serve uno.

    Solo il braccio LLM raggiungeva il caso: `RandomProposer` estrae quote da 1
    in su, `ScriptedProposer` parte dal pavimento della cella, il mondo scrive
    numeri non negativi per ogni azione che consuma risorse — e il prompt dei
    governatori offre `-1` esplicitamente. Misurato: 25.168 `collect_materials`
    rifiutate con "no materials found in this cell" su un seme, zero in tutti i
    bracci di controllo su venti run. Un tetto numerico, anche altissimo, non ha
    mai avuto il problema: era la sola scorciatoia del `None`.
    """
    index = pillars.ACTION_INDEX[action]
    if _claim_cap(action, cell, quota_row) is not None:
        key = (cell.y, cell.x, index)
        claims[key] = claims.get(key, 0) + 1
    if action in _COLLECTION_RESOURCE:
        resource, units = _COLLECTION_RESOURCE[action]
        resource_key = _resource_claim_key(cell, resource)
        claims[resource_key] = float(claims.get(resource_key, 0.0)) + min(
            float(units), _unclaimed_cell_resource(cell, resource, claims)
        )
    else:
        for resource, amount in _cell_resource_requirements(
            agent, action, cell
        ).items():
            resource_key = _resource_claim_key(cell, resource)
            claims[resource_key] = float(claims.get(resource_key, 0.0)) + amount


def _arrival_claim_key(request: ActionRequest) -> tuple | None:
    if request.action not in {ActionType.MOVE, ActionType.EXPLORE}:
        return None
    target = request.target if isinstance(request.target, dict) else {}
    if "x" not in target or "y" not in target:
        return None
    return ("arrival", int(target["x"]), int(target["y"]))


def _claim_arrival_if_available(
    agent, request: ActionRequest, cell, world, claims: dict
) -> bool:
    """Reserve destination capacity while the batch is still deciding."""
    key = _arrival_claim_key(request)
    if key is None:
        return True
    _kind, x, y = key
    if (x, y) == (cell.x, cell.y):
        return True
    pending = int(claims.get(key, 0))
    target_data = request.target if isinstance(request.target, dict) else {}
    survival_return = bool(target_data.get("survival_return", False))
    # I due permessi si leggono da una definizione sola, condivisa con chi
    # traccia la rotta (`rule_based_agent._move_toward_cell`).
    pioneer_allowed, scouting_transit = permessi_di_viaggio(agent)
    scout_home = getattr(agent, "scout_home", None)
    returning_scout = (
        getattr(agent, "scout_phase", None) == "back"
        and scout_home is not None
        and (survival_return or (x, y) == tuple(scout_home))
    )
    if returning_scout:
        # A returning scout is in transit, not migrating into each intermediate
        # cell. Capacity claims must therefore cover the whole route, not just
        # the final home edge: otherwise a fixed late row in the decision batch
        # can lose the same saturated transit claim every week and starve in
        # place despite an active return mission.
        claims[key] = pending + 1
        return True
    if not cell_accepts_arrival(
        cell,
        world.get_cell(x, y),
        pending_arrivals=pending,
        allow_pioneer=pioneer_allowed,
        allow_transit=survival_return or scouting_transit,
        allow_survival_overflow=survival_return,
    ):
        return False
    claims[key] = pending + 1
    return True


def _first_allowed(mask: np.ndarray, actions) -> ActionType | None:
    return next(
        (action for action in actions if bool(mask[pillars.ACTION_INDEX[action]])),
        None,
    )


def _best_allowed(
    mask: np.ndarray, actions, action_priority: np.ndarray
) -> ActionType | None:
    candidates = [
        action for action in actions if bool(mask[pillars.ACTION_INDEX[action]])
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda action: float(action_priority[pillars.ACTION_INDEX[action]]),
    )


def _bind_sustenance(
    agent, mask: np.ndarray, action_priority: np.ndarray
) -> ActionRequest | None:
    immediate = []
    if float(agent.hydration) < 0.75:
        immediate.append(ActionType.DRINK_WATER)
    if float(agent.satiety) < 0.75:
        immediate.append(ActionType.EAT_FOOD)
    action = _first_allowed(mask, immediate)
    if action is None:
        action = _best_allowed(
            mask,
            (ActionType.COLLECT_ICE, ActionType.FORAGE),
            action_priority,
        )
    if action is None:
        action = _first_allowed(mask, (ActionType.DRINK_WATER, ActionType.EAT_FOOD))
    if action is None:
        return None
    return _request(agent, action, message="restoring food and water security")


def _bind_resources(
    agent, mask: np.ndarray, action_priority: np.ndarray
) -> ActionRequest | None:
    action = _best_allowed(
        mask,
        (ActionType.COLLECT_MATERIALS, ActionType.COLLECT_MINERALS),
        action_priority,
    )
    if action is None:
        return None
    return _request(agent, action, message="collecting priority resources")


def _build_action_for_site_key(key: str) -> ActionType | None:
    structure_name = str(key).split("@", 1)[0]
    return next(
        (
            action
            for action, structure_type in BUILD_ACTIONS.items()
            if structure_type.value == structure_name
        ),
        None,
    )


def _bind_build(
    agent, cell, mask: np.ndarray, action_priority: np.ndarray
) -> ActionRequest | None:
    founder_reserved = bool(getattr(agent, "founder_kit_reserved", False))
    settle_phase = getattr(agent, "settle_phase", None)
    settle_target = getattr(agent, "settle_target", None)
    # A nominated founder must not spend the earmarked greenhouse inputs on
    # ordinary mother-cell projects while its expedition kit is incomplete.
    if founder_reserved and settle_phase is None:
        return None
    # The same protection is required while travelling.  A two-cell mission
    # can interleave Build choices between movement weeks; treating those as
    # ordinary local projects consumed the shelter/solar/O2 reserve on a
    # random transit habitat before the founder reached its target.
    if (
        founder_reserved
        and settle_phase == "out"
        and settle_target != (agent.x, agent.y)
    ):
        return None
    coverage_target = {"coverage_population": len(cell.agents_present)}
    if (
        founder_reserved
        and settle_phase == "out"
        and settle_target == (agent.x, agent.y)
    ):
        # Keep the personal kit bound to its declared purpose.  Depositing it
        # in the destination warehouse exposed the solar/O2 remainder to
        # inter-cell redistribution; allowing unrelated builds here could
        # consume the same reserve without moving it.  Preferences still
        # choose the Build pillar, while this binder selects its feasible
        # bootstrap action from the cell proposal.
        bootstrap = agent._bootstrap_build_action(cell)
        if bootstrap is None:
            return None
        action_i = pillars.ACTION_INDEX[bootstrap.action]
        if not bool(mask[action_i]):
            return None
        bootstrap.target = coverage_target
        bootstrap.message = "completing reserved outpost life support"
        return bootstrap
    sites = sorted(
        cell.construction_sites.items(), key=lambda item: (-float(item[1]), item[0])
    )
    for site_key, _progress in sites:
        action = _build_action_for_site_key(site_key)
        structure_type = BUILD_ACTIONS.get(action) if action is not None else None
        if (
            action is not None
            and structure_type is not None
            and bool(mask[pillars.ACTION_INDEX[action]])
            and not has_structure_at_local_position(
                cell,
                structure_type,
                agent.local_x_m,
                agent.local_y_m,
            )
        ):
            return _request(
                agent,
                action,
                target=coverage_target,
                message=f"continuing {site_key}",
            )
    locally_feasible_builds = tuple(
        action
        for action, structure_type in BUILD_ACTIONS.items()
        if not has_structure_at_local_position(
            cell,
            structure_type,
            agent.local_x_m,
            agent.local_y_m,
        )
    )
    action = _best_allowed(
        mask,
        locally_feasible_builds + (ActionType.MAINTAIN_STRUCTURE,),
        action_priority,
    )
    if action is None:
        return None
    return _request(
        agent,
        action,
        target=coverage_target,
        message="supporting colony infrastructure",
    )


def _nearby_id(candidate) -> str:
    return str(getattr(candidate, "agent_id", candidate))


def _prefixed(request: ActionRequest) -> ActionRequest:
    request.message = f"[pref] {request.message or 'physiological override'}"
    if request.action in {ActionType.MOVE, ActionType.EXPLORE} and isinstance(
        request.target, dict
    ):
        request.target["survival_return"] = True
    return request


def _has_local_water_support(cell) -> bool:
    return (
        float(cell.liquid_water) >= 0.1
        or float(cell.water_ice) + float(cell.resources.ice) >= 0.1
        or any(
            float(structure.local_effect.get("water", 0.0)) > 0.0
            for structure in cell.structures
        )
    )


def _physiological_override(agent, world, cell, mask: np.ndarray) -> ActionRequest | None:
    """Return a hard survival action before stochastic preference scoring."""
    water_clock_critical = int(
        getattr(agent, "steps_without_water", 0)
    ) >= WATER_GUARD_STEPS
    food_clock_critical = int(
        getattr(agent, "steps_without_food", 0)
    ) >= FOOD_GUARD_STEPS

    def starvation_action() -> ActionRequest | None:
        if float(agent.inventory.food) + RESOURCE_EPSILON >= 0.1 and bool(
            mask[pillars.ACTION_INDEX[ActionType.EAT_FOOD]]
        ):
            return _request(
                agent, ActionType.EAT_FOOD, message="starvation emergency: eating now"
            )
        if _has_food_source(cell) and bool(mask[pillars.ACTION_INDEX[ActionType.FORAGE]]):
            return _request(
                agent, ActionType.FORAGE, message="starvation emergency: foraging now"
            )
        move = agent._move_toward_structure(
            world,
            {StructureType.GREENHOUSE},
            "starvation emergency: returning to food-support cell",
        )
        # With individual survival priority enabled this function is the
        # explicit safety layer above the cell menu. The proposal mask may
        # omit ordinary MOVE in an unsupported cell, but it must not veto the
        # only route to food once the starvation clock is critical.
        if move is not None:
            return _prefixed(move)
        return None

    # A food clock already inside its guard window outranks a merely low
    # hydration scalar. This prevents an off-base scout from spending every
    # weekly turn drinking while its starvation clock advances. A genuinely
    # critical water clock still wins because dehydration becomes lethal first.
    if food_clock_critical and not water_clock_critical:
        action = starvation_action()
        if action is not None:
            return action
    if (
        float(agent.hydration) < CRITICAL_HYDRATION
        or water_clock_critical
    ):
        if _has_water_source(agent, cell) and bool(
            mask[pillars.ACTION_INDEX[ActionType.DRINK_WATER]]
        ):
            return _request(
                agent, ActionType.DRINK_WATER, message="dehydration emergency: drinking now"
            )
        if (cell.water_ice > 0.1 or cell.resources.ice > 0.1) and bool(
            mask[pillars.ACTION_INDEX[ActionType.COLLECT_ICE]]
        ):
            return _request(
                agent, ActionType.COLLECT_ICE, message="dehydration emergency: collecting ice"
            )
        move = agent._move_toward_structure(
            world,
            {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
            "dehydration emergency: returning to water-support cell",
        )
        move = move or agent._move_toward_nearby_ice(world) or agent._move_toward_polar_ice(world)
        if move is not None and bool(mask[pillars.ACTION_INDEX[move.action]]):
            return _prefixed(move)
    # An actual starvation emergency must outrank a preventive canteen refill.
    # Otherwise a hydrated colonist standing on ice can spend successive weekly
    # turns drinking while satiety continues toward zero.
    if (
        float(agent.satiety) < CRITICAL_SATIETY
        or food_clock_critical
    ):
        action = starvation_action()
        if action is not None:
            return action
    return_move = agent._return_for_ration_budget(world)
    if return_move is not None and bool(
        mask[pillars.ACTION_INDEX[return_move.action]]
    ):
        return _prefixed(return_move)
    water_reserve = float(agent.inventory.water) + float(agent.inventory.ice)
    if water_reserve <= WATER_RETURN_RESERVE:
        if _has_local_water_support(cell) and bool(
            mask[pillars.ACTION_INDEX[ActionType.REFILL_WATER]]
        ):
            return _request(
                agent,
                ActionType.REFILL_WATER,
                message="low water reserve: replenishing canteen before further travel",
            )
        if (cell.water_ice > 0.1 or cell.resources.ice > 0.1) and bool(
            mask[pillars.ACTION_INDEX[ActionType.COLLECT_ICE]]
        ):
            return _request(
                agent,
                ActionType.COLLECT_ICE,
                message="low water reserve: extracting local ice",
            )
        move = agent._move_toward_structure(
            world,
            {StructureType.GREENHOUSE, StructureType.HABITAT, StructureType.INFIRMARY},
            "low water reserve: returning to life support",
        )
        move = move or agent._move_toward_nearby_ice(world) or agent._move_toward_polar_ice(world)
        if move is not None and bool(mask[pillars.ACTION_INDEX[move.action]]):
            return _prefixed(move)
    if float(agent.inventory.food) <= FOOD_RETURN_RESERVE:
        if _has_food_source(cell) and bool(mask[pillars.ACTION_INDEX[ActionType.FORAGE]]):
            return _request(
                agent, ActionType.FORAGE, message="low food reserve: replenishing rations"
            )
        move = agent._move_toward_structure(
            world,
            {StructureType.GREENHOUSE},
            "low food reserve: returning to greenhouse",
        )
        if move is not None and bool(mask[pillars.ACTION_INDEX[move.action]]):
            return _prefixed(move)
    # An ordinary resident already standing outside a supported settlement
    # must not become stranded merely because its current ration budget is
    # temporarily adequate. Active scout/founder missions own their return
    # state machines; everyone else keeps progressing toward known support.
    if (
        local_life_support_capacity(cell) <= 0
        and getattr(agent, "scout_phase", None) is None
        and getattr(agent, "settle_phase", None) is None
        and not bool(getattr(agent, "founder_kit_reserved", False))
    ):
        move = agent._move_toward_structure(
            world,
            {
                StructureType.GREENHOUSE,
                StructureType.HABITAT,
                StructureType.INFIRMARY,
            },
            "returning from unsupported ground to life support",
        )
        if move is not None and bool(mask[pillars.ACTION_INDEX[move.action]]):
            return _prefixed(move)
    if float(agent.health) < CRITICAL_HEALTH:
        action = _first_allowed(mask, (ActionType.USE_MED_KIT, ActionType.REST))
        if action is not None:
            return _request(agent, action, message="medical emergency override")
    return None


def _bind_life(agent, mask: np.ndarray) -> ActionRequest | None:
    # Questo binder non guarda i vicini: la logistica di cella li serve senza
    # consumare un turno. Fino al 2026-08-05 riceveva comunque una lista di
    # `AgentView` dei vicini e la scartava con `del`, mentre a monte quella lista
    # veniva costruita per ogni agente a ogni step con una scansione 3x3 della
    # mappa per cella. Il parametro e' stato rimosso lungo tutta la catena invece
    # di essere lasciato con un valore vuoto: una lista vuota avrebbe funzionato
    # allo stesso modo ma sarebbe stata una trappola per un futuro consumatore,
    # che avrebbe visto "nessun vicino" invece di un errore.
    if bool(mask[pillars.ACTION_INDEX[ActionType.PHYSIOLOGICAL_RECOVERY]]):
        return _request(
            agent,
            ActionType.PHYSIOLOGICAL_RECOVERY,
            message="following the cell physiological recovery proposal",
        )
    if float(agent.health) < 0.5 and bool(
        mask[pillars.ACTION_INDEX[ActionType.USE_MED_KIT]]
    ):
        return _request(agent, ActionType.USE_MED_KIT, message="treating poor health")
    if float(agent.fatigue) > 0.5 and bool(
        mask[pillars.ACTION_INDEX[ActionType.REST]]
    ):
        return _request(agent, ActionType.REST, message="recovering from fatigue")
    # A healthy, rested colonist does not spend a full seven-day turn on a
    # prophylactic REST merely because LIFE is their strongest preference.
    # Returning None lets the same normalized preference ranking try the next
    # feasible pillar; critical needs remain handled by the physiological
    # override above this binder.
    return None


def _decision_memo(world) -> tuple[dict, dict]:
    """I due dizionari di questo passo: vicini per origine, arrivi per bersaglio.

    Conteggio che ha motivato la memo (configurazione reale, 300 agenti, 30
    passi, 8.864 chiamate a ``_bind_explore``):

    ==========================  =========  ==========  ============
    voce                        chiamate   distinte    ripetizione
    ==========================  =========  ==========  ============
    ``world.neighbors``             8.864           1      8.864x
    ``cell_accepts_arrival``       70.912           8      8.864x
    ==========================  =========  ==========  ============

    Il costo di questo binder non e' calcolo: e' la stessa risposta ricalcolata
    migliaia di volte. Gli agenti si concentrano in pochissime celle, quindi
    l'origine e' quasi sempre la stessa e i vicini valutati sono sempre gli
    stessi otto.

    **Perche' e' esatto e non "quasi".** ``cell_accepts_arrival`` fa ``del
    origin``: non dipende dall'agente che chiede. I suoi altri ingressi --
    ``agents_present``, ``struct_count``, capacita' di alloggio -- cambiano
    soltanto nella fase di esecuzione, mai dentro la fase di decisione. Quindi
    dentro un passo la risposta per una cella bersaglio e' una costante, e
    memorizzarla non e' un'approssimazione ma il riconoscimento di una costante.

    Il dizionario e' appeso al mondo e buttato quando cambia lo step, come gia'
    fa ``_structure_targets``: nessuna invalidazione da tenere coerente, perche'
    non sopravvive al passo che l'ha prodotto.

    La memo copre **solo** il call-site di ``_bind_explore``, che chiama con gli
    argomenti di default. ``_claim_arrival_if_available`` passa invece arrivi
    pendenti e permessi che cambiano *durante* il passo, e resta fuori: una memo
    condivisa risponderebbe li' a una domanda diversa da quella memorizzata.
    """
    step = int(getattr(world, "step", 0))
    state = getattr(world, "_decision_memo", None)
    if state is None or state[0] != step:
        state = (step, {}, {})
        world._decision_memo = state
    return state[1], state[2]


def _bind_explore(
    agent,
    world,
    cell,
    mask: np.ndarray,
    action_priority: np.ndarray,
    claims: dict | None = None,
) -> ActionRequest | None:
    # Reaching the EXPLORE binder means the normalized preferences have
    # already selected exploration.  Let that pillar start the settlement
    # state machine as well as continue it: the old phase-only gate could
    # continue an expedition but no preference-policy decision could ever set
    # ``settle_phase`` in the first place.
    expedition = (
        agent._settlement_action(world, cell)
        if claims is None
        else agent._settlement_action(world, cell, claims)
    )
    scout_active = getattr(agent, "scout_phase", None) is not None
    supported_home = local_life_support_capacity(cell) >= max(
        1, len(cell.agents_present)
    )
    if (
        expedition is None
        and not bool(getattr(agent, "founder_kit_reserved", False))
        and (scout_active or supported_home)
    ):
        # Preference selection remains the gate: only an agent whose scored
        # Explore pillar reaches this binder may start a physical survey.
        # Previously preference agents could continue a scout mission but
        # never create one, so the shared map froze after the first founder
        # wave and almost every later Explore choice degraded to OBSERVE.
        expedition = agent._scouting_action(world, cell)
    if expedition is not None and bool(mask[pillars.ACTION_INDEX[expedition.action]]):
        expedition.message = f"[pref] {expedition.message or 'continuing expedition'}"
        return expedition
    if bool(getattr(agent, "founder_kit_reserved", False)):
        # The retry cadence is deliberately short, but stochastic preferences
        # may select EXPLORE before it is due.  Waiting in the nominated cell
        # preserves access to its warehouse and greenhouse; falling through
        # to ordinary frontier motion made the founder abandon its own supply
        # point before the kit could be completed.
        if bool(mask[pillars.ACTION_INDEX[ActionType.OBSERVE]]):
            return _request(
                agent,
                ActionType.OBSERVE,
                message="waiting in the colony while the founder kit is prepared",
            )
        return None
    if supported_home and not scout_active:
        # A selected Explore preference remains meaningful even when no safe
        # shared-frontier mission is due/reachable, but it must not degrade to
        # random physical wandering among already mapped cells. Waiting here
        # preserves the stochastic choice while the next scout cadence or a
        # newly operational relay outpost can expose new frontier work.
        #
        # **Provato e RITIRATO: cedere il turno agli altri pilastri
        # (2026-09-01).** Questa riga sola produce il 48,3% di TUTTE le
        # decisioni della colonia, quindi sembrava il posto dove si perde meta'
        # della manodopera. Restituire `None` -- cosi' che il ciclo delle
        # preferenze provi risorse, costruzione e vita -- e' stato misurato su
        # tre semi a 400 passi:
        #
        #                   observe    do_nothing    lavoro utile
        #   con l'attesa     51,7%         9,6%          38,6%
        #   cedendo il turno  6,8%        53,6%          39,7%
        #
        # Il ripiego NON trova lavoro: `observe` diventa `do_nothing` quasi uno
        # a uno e il lavoro utile guadagna 1,1 punti in media (+2,1 / +0,7 /
        # +0,4 sui tre semi), dentro il rumore fra semi. E' la misura che dice
        # cosa sia davvero questo ramo: **un termometro dell'ozio, non la sua
        # causa**. Il lavoro non c'e' perche' la cella pubblica poche quote per
        # molte mani, e la cura e' l'espansione, non un ripiego diverso.
        #
        # Fra due stati equivalenti si tiene quello che dice di piu': «aspetto
        # un incarico di frontiera» e' un'informazione, `do_nothing` no.
        # **Il lavoro viene prima dell'attesa (2026-09-01).** Questo ramo
        # restituiva `OBSERVE` -- «aspetto un incarico di frontiera» -- e una
        # richiesta restituita chiude la decisione: il colono non provava piu'
        # nessun altro pilastro, e questa riga sola produceva il 48,3% di TUTTE
        # le decisioni della colonia.
        #
        # **Provato, ritirato, e poi tenuto: la misura dipendeva dal regime.**
        # Il 2026-09-01, in un mondo dove il 93% delle celle non aveva nulla da
        # estrarre, cedere il turno guadagnava 1,1 punti di lavoro utile e
        # spostava soltanto l'ozio da `observe` a `do_nothing`: ritirato. Con il
        # fondo regolitico -- ogni cella ha materiale -- la stessa modifica
        # cambia esito, misurata su tre semi a 400 passi:
        #
        #                  popolazione        morti (seme 9)   integrita' finale
        #   con l'attesa    81 / 95 / 97           57               0,47
        #   lavoro prima    99 / 100 / 100         16               0,58
        #
        # Quarantuno coloni. La ragione e' che il turno ceduto finisce in
        # MANUTENZIONE, e sotto integrita' 0,4 l'efficienza di una struttura si
        # dimezza: il parco che regge e' cio' che tiene in vita la colonia.
        #
        # E' la stessa lezione della voce 35 della checklist, dove il freno sui
        # cantieri era stato rifiutato da una misura fatta in un'economia che
        # non permetteva di costruire abbastanza perche' il vincolo mordesse:
        # **un'ipotesi rifiutata sotto un regime va rimisurata quando il regime
        # cambia.**
        #
        # La ragione originaria resta e non e' toccata: una preferenza di
        # esplorazione non deve degradare in vagabondaggio fra celle gia'
        # mappate. Non degrada: cede il turno, e se nessun altro pilastro ha
        # lavoro il ripiego finale qui sotto restituisce comunque `OBSERVE` con
        # il messaggio che dice perche'.
        return None
    action = _best_allowed(
        mask,
        (ActionType.MOVE, ActionType.EXPLORE, ActionType.OBSERVE),
        action_priority,
    )
    if _BIND_MEMO_ENABLED:
        neighbor_memo, arrival_memo = _decision_memo(world)
        origin_key = (int(agent.x), int(agent.y))
        neighbors = neighbor_memo.get(origin_key)
        if neighbors is None:
            neighbors = world.neighbors(agent.x, agent.y, 1)
            neighbor_memo[origin_key] = neighbors
    else:
        neighbor_memo = arrival_memo = None
        neighbors = world.neighbors(agent.x, agent.y, 1)
    if action in {ActionType.MOVE, ActionType.EXPLORE} and neighbors:
        def frontier_score(candidate) -> tuple[float, int, int]:
            frontier = sum(
                1 for adjacent in world.neighbors(candidate.x, candidate.y, 1)
                if not adjacent.explored
            )
            score = (
                float(agent._cell_score(candidate))
                + (1.5 if not candidate.explored else 0.0)
                + 0.35 * frontier
                - 0.03 * len(candidate.agents_present)
            )
            # Stable coordinate tie-break keeps seeded runs reproducible.
            return score, -candidate.y, -candidate.x

        # I due filtri sono predicati puri sullo stesso elenco e preservano
        # l'ordine, quindi applicarli in un ordine o nell'altro produce la stessa
        # lista. L'ordine scelto e' quello per costo crescente.
        #
        # `cell_accepts_arrival` (src/agents/build_policy.py:54) fa `del origin`:
        # dipende solo dalla cella di destinazione ed e' una manciata di letture.
        # `_can_visit_cell_and_return` costa circa sette volte tanto, perche' fa
        # due `_return_ration_requirement`, ognuno dei quali risolve la distanza
        # di supporto e i passi per cella.
        #
        # Misura sulla config golden (300 agenti, 30 step): il filtro costoso
        # rispondeva True 70.032 volte su 70.032, cioe' non scartava nulla,
        # mentre quello economico scartava tutti i 70.032 candidati. Applicarlo
        # per primo elimina 140.064 delle 158.064 risoluzioni di distanza di
        # supporto dell'intera run.
        #
        # La proporzione dipende dal carico, l'ordinamento no: il filtro
        # economico gira comunque su tutti i candidati, quindi anteporlo non e'
        # mai peggio ed e' migliore appena scarta qualcosa.
        #
        # L'ordine storico resta selezionabile con MARSABM_FILTER_ORDER=legacy.
        # Non e' un'opzione di modellazione -- i due ordini producono la stessa
        # lista, verificato bit-exact sui seed golden -- ma uno strumento di
        # misura: il guadagno va poter essere rimisurato con un confronto A/B
        # interlacciato, e un numero che finisce in tesi senza poter essere
        # riprodotto non e' difendibile.
        def accepts_arrival(candidate) -> bool:
            # Stessa risposta per la stessa cella bersaglio dentro un passo:
            # vedi `_decision_memo` per la dimostrazione e per il conteggio che
            # l'ha motivata (8 risposte distinte per 70.912 chiamate).
            if arrival_memo is None:
                return cell_accepts_arrival(cell, candidate)
            key = (int(candidate.x), int(candidate.y))
            answer = arrival_memo.get(key)
            if answer is None:
                answer = cell_accepts_arrival(cell, candidate)
                arrival_memo[key] = answer
            return answer

        if _CHEAP_FILTER_FIRST:
            safe_neighbors = [
                candidate for candidate in neighbors if accepts_arrival(candidate)
            ]
            safe_neighbors = [
                candidate
                for candidate in safe_neighbors
                if agent._can_visit_cell_and_return(world, candidate.x, candidate.y)
            ]
        else:
            safe_neighbors = [
                candidate
                for candidate in neighbors
                if agent._can_visit_cell_and_return(world, candidate.x, candidate.y)
            ]
            safe_neighbors = [
                candidate for candidate in safe_neighbors if accepts_arrival(candidate)
            ]
        if not safe_neighbors:
            # Shared observation maps the local operational radius without
            # manufacturing a physical pioneer in every adjacent cell.
            if bool(mask[pillars.ACTION_INDEX[ActionType.OBSERVE]]):
                return _request(
                    agent,
                    ActionType.OBSERVE,
                    message="mapping the shared frontier from supported ground",
                )
            return None
        target = max(safe_neighbors, key=frontier_score)
        return _request(
            agent,
            action,
            target={"x": target.x, "y": target.y},
            message=(
                "mapping the best shared frontier"
                if action == ActionType.EXPLORE
                else "moving toward the best local opportunity"
            ),
        )
    if action is None:
        return None
    return _request(agent, action, message="surveying the local area")


def decide_preferences(
    agent,
    world,
    cell_mask_row: np.ndarray,
    u01: float,
    sampling: str,
    claims: dict,
    step: int,
    action_priority_row: np.ndarray | None = None,
    quota_row: np.ndarray | None = None,
    individual_survival_priority_enabled: bool = True,
) -> ActionRequest:
    """Choose and bind one executable action for a single agent."""
    cell = world.get_cell(agent.x, agent.y)
    mask = agent_action_mask(
        agent,
        cell_mask_row,
        individual_survival_priority_enabled,
        cell.resources,
        cell,
    )
    return decide_preferences_precomputed(
        agent,
        world,
        mask,
        u01,
        sampling,
        claims,
        step,
        action_priority_row=action_priority_row,
        quota_row=quota_row,
        individual_survival_priority_enabled=individual_survival_priority_enabled,
    )


def decide_preferences_precomputed(
    agent,
    world,
    personal_mask_row: np.ndarray,
    u01: float,
    sampling: str,
    claims: dict,
    step: int,
    *,
    precomputed_scores: np.ndarray | None = None,
    precomputed_chosen: int | None = None,
    action_priority_row: np.ndarray | None = None,
    quota_row: np.ndarray | None = None,
    individual_survival_priority_enabled: bool = True,
    skip_physiological_override: bool = False,
    semantic_pillar_row: np.ndarray | None = None,
    scoring_priority_row: np.ndarray | None = None,
) -> ActionRequest:
    """Bind from an agent mask, optionally reusing phase-B score columns.

    `semantic_pillar_row` e `scoring_priority_row` servono solo allo strato
    SemIf degli agenti e valgono `None` altrimenti, nel qual caso il corpo e'
    quello di prima. Il primo e' il fattore semantico sui pilastri, da
    riapplicare quando i claim costringono a ricalcolare i punteggi; il secondo
    e' la riga di priorita' NON scalata dal livello 2, perche' il ricalcolo dei
    pilastri non deve vedere i fattori per azione.

    Claims remain sequential.  If an earlier agent changes this row's mask,
    the scalar oracle is recomputed before sampling so the batch path remains
    exactly equivalent even at a saturated collection/maintenance cap.
    """
    cell = world.get_cell(agent.x, agent.y)
    mask = np.asarray(personal_mask_row, dtype=np.bool_).copy()
    if action_priority_row is None:
        action_priority_row = mask.astype(np.float64)
    else:
        action_priority_row = np.asarray(action_priority_row, dtype=np.float64)
    before_claims = mask.copy()
    _apply_claim_limits(mask, agent, cell, claims, quota_row)
    # `skip_physiological_override` non e' una scorciatoia: e' l'esito di una
    # condizione sufficiente dimostrata prima del ciclo, per tutto il batch
    # insieme (src/core/physio_gate.py). Vale sulla maschera **pre-claim**, che e'
    # un sovrainsieme di questa: i limiti di claim possono solo spegnere bit, e
    # tutti i rami di `_physiological_override` che leggono la maschera sono
    # guardati da un bit acceso. Se nessuno poteva scattare sulla maschera piu'
    # larga, nessuno puo' scattare su questa.
    if individual_survival_priority_enabled and not skip_physiological_override:
        override = _physiological_override(agent, world, cell, mask)
        if override is not None:
            if _claim_arrival_if_available(agent, override, cell, world, claims):
                _record_claim(agent, override.action, cell, claims, quota_row)
                return override
    claims_changed_mask = not np.array_equal(mask, before_claims)
    if precomputed_scores is None or claims_changed_mask:
        # Con `individual_survival_priority_enabled` spento le urgenze valgono
        # tutte 1: l'esperimento guidato dalla cella tiene preferenze e abilita'
        # come nucleo della decisione, ma disattiva l'urgenza dello stato vitale
        # personale. La cella continua a decidere quali azioni esistono e con
        # quale priorita'.
        scores, chosen = decision_scoring.score_decision(
            agent,
            mask,
            (
                action_priority_row
                if scoring_priority_row is None
                else np.asarray(scoring_priority_row, dtype=np.float64)
            ),
            u01,
            sampling,
            individual_survival_priority_enabled,
        )
        if semantic_pillar_row is not None:
            pesati = np.asarray(scores, dtype=np.float64) * np.asarray(
                semantic_pillar_row, dtype=np.float64
            )
            totale = float(pesati.sum())
            scores = pesati / totale if totale > 0.0 else np.zeros_like(pesati)
            chosen = choose_pillar(scores, sampling, u01)
    else:
        scores = np.asarray(precomputed_scores, dtype=np.float64)
        chosen = int(precomputed_chosen)
    if not scores.any():
        return _request(agent, pillars.FALLBACK_ACTION, message="no feasible action")
    remaining = [
        int(index)
        for index in np.argsort(-scores, kind="stable")
        if int(index) != chosen and scores[int(index)] > 0.0
    ]
    for pillar in [chosen, *remaining]:
        if pillar == pillars.P_SUSTENANCE:
            request = _bind_sustenance(agent, mask, action_priority_row)
        elif pillar == pillars.P_RESOURCES:
            request = _bind_resources(agent, mask, action_priority_row)
        elif pillar == pillars.P_BUILD:
            request = _bind_build(agent, cell, mask, action_priority_row)
        elif pillar == pillars.P_LIFE:
            request = _bind_life(agent, mask)
        elif pillar == pillars.P_SOCIAL:
            # This pillar is intentionally unavailable: its preference weight
            # will be consumed by cell-level coordination, not by an action.
            request = None
        else:
            request = _bind_explore(
                agent, world, cell, mask, action_priority_row, claims
            )
        if request is not None:
            if not _claim_arrival_if_available(agent, request, cell, world, claims):
                continue
            _record_claim(agent, request.action, cell, claims, quota_row)
            return request
    # **Il ripiego finale dice perche' (2026-09-01).** `DO_NOTHING` e `OBSERVE`
    # sono entrambi senza effetto, quindi la scelta fra i due non cambia la
    # simulazione di un passo: cambia cio' che il registro delle azioni permette
    # di leggere. Da quando il pilastro dell'esplorazione cede il turno invece
    # di restituire `OBSERVE`, tutto l'ozio della colonia arriverebbe qui e
    # verrebbe registrato come `do_nothing`, cioe' senza causa. Restituire
    # `OBSERVE` -- che la cella propone sempre, quota illimitata -- conserva
    # l'informazione: il colono ha guardato perche' non c'era altro da fare.
    if bool(mask[pillars.ACTION_INDEX[ActionType.OBSERVE]]):
        return _request(
            agent,
            ActionType.OBSERVE,
            message="no work available in this cell; observing",
        )
    return _request(agent, pillars.FALLBACK_ACTION, message="no bindable action")


class PreferenceAgent(RuleBasedAgent):
    """MRO marker exposing RuleBasedAgent's deterministic binder helpers."""

    pass


# In fondo, e per modulo invece che per nome: `decision_scoring` importa da qui
# le cinque funzioni che sono il suo oracolo (`compute_urgencies`,
# `score_pillars`, ...), quindi la dipendenza e' circolare. Importare il MODULO
# risolve il ciclo in entrambe le direzioni -- l'oggetto modulo esiste anche
# quando e' ancora a meta' dell'inizializzazione, e gli attributi vengono letti
# solo al momento della chiamata -- mentre `from ... import score_decision`
# fallirebbe se qualcuno importasse `decision_scoring` per primo.
import src.core.decision_scoring as decision_scoring  # noqa: E402
