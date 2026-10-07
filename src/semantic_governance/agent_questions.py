"""Domande SemIf per singolo agente e fattori moltiplicativi dalla risposta.

Livello 1: quale pilastro (cinque pesabili + unknown; `social` non e' mai
un'opzione, vedi `src/governors/policy.py`). Livello 2: quale azione dentro
`resources`, `build` o `explore`, gli unici pilastri il cui binder legge
`action_priority` per intero (spec, "Dove il livello 2 puo' agire davvero").
Nessuna soglia: la distribuzione intera diventa un fattore, e l'incertezza si
attenua da se'.
"""

from __future__ import annotations

import numpy as np

from src.agents import pillars
from src.agents.action_space import ActionType
from src.core import constants as C

from .schemas import FALLBACK_OPTIONS, SemanticDecision, SemanticDecisionRequest, SemanticOption

AGENT_PROFILE_VERSION = "jev-semif-agent-v1"
L1_QUESTION_ID = f"{AGENT_PROFILE_VERSION}:L1"
L2_QUESTION_ID = f"{AGENT_PROFILE_VERSION}:L2:{{pillar}}"
QUESTION = "What should this colonist prioritise this week?"
LEVEL2_QUESTION = "Within this priority, which action should this colonist take this week?"

_PILLAR_INDEX = {
    "sustenance": pillars.P_SUSTENANCE,
    "resources": pillars.P_RESOURCES,
    "build": pillars.P_BUILD,
    "life": pillars.P_LIFE,
    "explore": pillars.P_EXPLORE,
}
_L1_OPTIONS = (
    ("sustenance", "Secure food and water: drink, eat, forage, collect ice"),
    ("resources", "Collect construction materials or minerals"),
    ("build", "Build or maintain a colony structure"),
    ("life", "Recover health or rest: medical kit, rest, recovery"),
    ("explore", "Move, explore or observe new ground"),
)
LEVEL2_PILLARS = ("resources", "build", "explore")
_ACTION_TEXT = {
    ActionType.COLLECT_MINERALS: "Collect minerals",
    ActionType.COLLECT_MATERIALS: "Collect construction material",
    ActionType.BUILD_SHELTER: "Build a shelter",
    ActionType.BUILD_SOLAR_ARRAY: "Build a solar array",
    ActionType.BUILD_OXYGEN_PLANT: "Build an oxygen plant",
    ActionType.BUILD_GREENHOUSE: "Build a greenhouse",
    ActionType.BUILD_WATER_EXTRACTOR: "Build a water extractor",
    ActionType.BUILD_HEATER: "Build a heater",
    ActionType.BUILD_INFIRMARY: "Build an infirmary",
    ActionType.BUILD_HABITAT: "Build a habitat",
    ActionType.BUILD_RESEARCH_LAB: "Build a research lab",
    ActionType.BUILD_STORAGE_DEPOT: "Build a storage depot",
    ActionType.BUILD_WEATHER_STATION: "Build a weather station",
    ActionType.MAINTAIN_STRUCTURE: "Maintain or repair a structure",
    ActionType.MOVE: "Move to another cell",
    ActionType.EXPLORE: "Explore unknown ground",
    ActionType.OBSERVE: "Observe the surroundings",
}
_UNKNOWN = SemanticOption("unknown", "Insufficient information to decide")


def _r(value) -> float:
    return round(float(value), 2)


def colony_summary(agents, rows) -> dict:
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size == 0:
        return {"population": 0}
    return {
        "population": int(rows.size),
        "mean_health": _r(agents.health[rows].mean()),
        "mean_hydration": _r(agents.hydration[rows].mean()),
        "mean_satiety": _r(agents.satiety[rows].mean()),
        "mean_fatigue": _r(agents.fatigue[rows].mean()),
    }


def agent_record(agents, cells, row: int) -> dict:
    row = int(row)
    x, y = int(agents.x[row]), int(agents.y[row])
    inv = agents.inv[row]
    return {
        "health": _r(agents.health[row]),
        "hydration": _r(agents.hydration[row]),
        "satiety": _r(agents.satiety[row]),
        "fatigue": _r(agents.fatigue[row]),
        "stress": _r(agents.stress[row]),
        "oxygen": _r(agents.oxygen[row]),
        "cell_occupants": int(cells.occupancy[y, x]),
        "carrying": {
            "food": _r(inv[C.R["food"]]),
            "water": _r(inv[C.R["water"]]),
            "construction_material": _r(inv[C.R["construction_material"]]),
            "med_kits": _r(inv[C.R["med_kits"]]),
        },
    }


def level1_request(run_id, step, agent_id, colony, record, max_options=16):
    return SemanticDecisionRequest(
        run_id=str(run_id), step=int(step), actor=f"agent:{agent_id}",
        question_id=L1_QUESTION_ID,
        state={"colony": colony, "colonist": record},
        question=QUESTION,
        options=tuple(SemanticOption(i, d) for i, d in _L1_OPTIONS) + (_UNKNOWN,),
        max_options=max_options,
    )


def _pillar_actions(pillar_name: str) -> tuple[ActionType, ...]:
    return tuple(pillars.PILLAR_ACTIONS[_PILLAR_INDEX[pillar_name]])


def level2_request(run_id, step, agent_id, pillar_name, colony, record, max_options=16):
    if pillar_name not in LEVEL2_PILLARS:
        raise ValueError(f"level 2 is not asked for pillar {pillar_name!r}")
    options = tuple(
        SemanticOption(action.value, _ACTION_TEXT[action])
        for action in _pillar_actions(pillar_name)
    ) + (_UNKNOWN,)
    return SemanticDecisionRequest(
        run_id=str(run_id), step=int(step), actor=f"agent:{agent_id}",
        question_id=L2_QUESTION_ID.format(pillar=pillar_name),
        state={"colony": colony, "colonist": record, "priority": pillar_name},
        question=LEVEL2_QUESTION,
        options=options,
        max_options=max_options,
    )


def _weights(decision: SemanticDecision, real_ids: list[str]) -> dict[str, float] | None:
    # Nessun taglio per `route`: una risposta vera con `unknown` in testa porta
    # `route="fallback"` dal vivo ma `route="replay"` in replay, e tagliare per
    # route rendeva il replay diverso dalla run (revisione finale, C1). I
    # fallback d'errore (`fallback_decision`) mettono massa 1 su `unknown` e
    # restano neutri per costruzione.
    p = decision.probabilities
    p_unknown = sum(float(p.get(k, 0.0)) for k in FALLBACK_OPTIONS)
    mass = sum(float(p.get(k, 0.0)) for k in real_ids)
    if mass <= 0.0:
        return None
    m = len(real_ids)
    return {
        k: p_unknown + (1.0 - p_unknown) * m * float(p.get(k, 0.0)) / mass
        for k in real_ids
    }


def pillar_factors(decision: SemanticDecision) -> np.ndarray:
    factors = np.ones(pillars.N_PILLARS, dtype=np.float64)
    weights = _weights(decision, [i for i, _ in _L1_OPTIONS])
    if weights is not None:
        for name, value in weights.items():
            factors[_PILLAR_INDEX[name]] = value
    return factors


def action_factors(decision: SemanticDecision, pillar_name: str) -> np.ndarray:
    factors = np.ones(pillars.N_ACTIONS, dtype=np.float64)
    actions = _pillar_actions(pillar_name)
    weights = _weights(decision, [a.value for a in actions])
    if weights is not None:
        for action in actions:
            factors[pillars.ACTION_INDEX[action]] = weights[action.value]
    return factors
