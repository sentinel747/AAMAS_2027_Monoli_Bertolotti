from __future__ import annotations

"""Il punteggio dei sei pilastri per un agente, in bracci confrontabili.

Task C.2, secondo tempo. Il conteggio dopo la memo di ``_bind_explore`` ha
spostato il bersaglio: cinque funzioni numeriche pure valgono il **10,42% del
passo** (configurazione reale, 300 agenti), 44,2 us per decisione.

===========================  ============  ============
funzione                     % del passo   us/chiamata
===========================  ============  ============
``compute_urgencies``               3,75          15,9
``score_pillars``                   1,88           8,0
``pillar_priority``                 1,70           7,2
``choose_pillar``                   1,64           7,0
``pillar_availability``             1,45           6,2
===========================  ============  ============

Vengono chiamate **11.941 volte su 12.000 decisioni**: quasi sempre. Il motivo e'
che ``_apply_claim_limits`` spegne almeno un bit della maschera, e allora
``decide_preferences_precomputed`` ricalcola l'oracolo scalare invece di riusare
le colonne di fase B. La vettorizzazione a monte esiste ed e' corretta; su questo
carico il suo risultato viene quasi sempre buttato.

**Perche' NumPy qui e' sospetto.** Gli array sono da **sei** elementi. A quella
taglia il costo per chiamata di NumPy -- creazione dell'oggetto array, dispatch
del tipo, boxing degli scalari -- puo' superare l'aritmetica che esegue. Il
braccio di controllo giusto non e' quindi soltanto Rust: e' anche **Python puro
senza NumPy**. Se vincesse quello sarebbe un risultato onesto, e andrebbe detto.

**Il vincolo di parita' che non e' ovvio.** ``compute_urgencies`` eleva al
quadrato con ``x ** 2`` su un ``float`` Python, cioe' attraverso ``pow()`` della
libreria di sistema. Misurato su questa piattaforma: ``x ** 2 != x * x`` per
**556 valori su un milione** nel dominio vero (basi in ``[0, 1]``). Sostituire
l'elevamento con una moltiplicazione non e' quindi una riscrittura innocua ma una
divergenza all'ultimo bit ogni ~1800 agenti. Ogni braccio deve elevare al
quadrato nello stesso modo, e ``tests/core/test_decision_scoring.py`` lo verifica
sui valori che divergono invece di fidarsi.
"""

import os

import numpy as np

from src.agents import pillars
from src.agents.preference_agent import (
    _BUILD_INDICES,
    choose_pillar,
    compute_urgencies,
    pillar_availability,
    pillar_priority,
    score_pillars,
)
from src.core import constants as C
from src.core import native
from src.core.arrays import (
    MISSION_SCOUT_BACK,
    MISSION_SCOUT_OUT,
    MISSION_SETTLE_OUT,
)

VALID_BACKENDS = ("auto", "numpy", "python", "rust")

# Il default e' `auto`: `rust` se il kernel nativo c'e', altrimenti `python`.
#
# **Perche' qui il default dipende dall'ambiente e in `physio_gate` no.** La'
# i bracci differivano per meno del rumore, quindi un default fisso valeva piu'
# della velocita'. Qui la differenza e' misurata e reale (2,83x per decisione,
# 1,032x end-to-end), e i tre bracci sono verificati bit-exact: scegliere il
# piu' veloce cambia il tempo, non il risultato. Quale braccio ha girato finisce
# comunque nel manifest della run tramite `native.describe()`, quindi una misura
# archiviata resta interpretabile.
_BACKEND_ENV = "MARSABM_SCORING"

# Indici di azione per pilastro, risolti una volta all'import: dentro il ciclo
# sarebbero migliaia di risoluzioni di dizionario per passo.
_PILLAR_INDICES: tuple[tuple[int, ...], ...] = tuple(
    tuple(
        pillars.ACTION_INDEX[action]
        for action in pillars.PILLAR_ACTIONS.get(pillar, ())
    )
    for pillar in range(pillars.N_PILLARS)
)
_UNIFORM_PREFERENCE = 1.0 / pillars.N_PILLARS

# L'esponente delle urgenze, definito qui e passato al lato nativo invece di
# essere scritto due volte. Non e' un parametro di modellazione: e' il `2` di
# `x ** 2` in `compute_urgencies`, e viaggia perche' con un esponente costante
# LLVM riscrive `powf(2.0)` in una moltiplicazione, che differisce da `pow()`
# all'ultimo bit per 556 valori su un milione nel dominio vero.
URGENCY_EXPONENT = 2.0
_LAST_PILLAR = pillars.N_PILLARS - 1


def configured_backend() -> str:
    """Backend richiesto dall'ambiente, validato e risolto."""
    backend = os.environ.get(_BACKEND_ENV, "auto").strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV}={backend!r} non e' valido; "
            f"scegliere fra {', '.join(VALID_BACKENDS)}"
        )
    if backend == "auto":
        return "rust" if native.is_available() else "python"
    if backend == "rust" and not native.is_available():
        raise ValueError(
            f"{_BACKEND_ENV}=rust richiede il kernel nativo, non disponibile: "
            f"{native.unavailable_reason()}. Compilare con "
            "`python scripts/build_rust.py`."
        )
    return backend


def score_decision_numpy(
    agent,
    mask: np.ndarray,
    action_priority: np.ndarray,
    u01: float,
    sampling: str,
    survival_enabled: bool,
) -> tuple[np.ndarray, int]:
    """Il percorso storico, invariato: e' l'oracolo di riferimento."""
    available = pillar_availability(mask)
    cell_priority = pillar_priority(mask, action_priority)
    preferences = getattr(agent, "pillar_preferences", None)
    if preferences is None:
        preferences = np.full(pillars.N_PILLARS, _UNIFORM_PREFERENCE)
    skills = np.asarray(
        getattr(agent, "pillar_skills", np.ones(pillars.N_PILLARS, dtype=np.float64))
    )
    if survival_enabled:
        urgencies = compute_urgencies(agent, mask)
    else:
        urgencies = np.ones(pillars.N_PILLARS, dtype=np.float64)
    scores = score_pillars(
        np.asarray(preferences), urgencies, available, cell_priority, skills
    )
    return scores, choose_pillar(scores, sampling, u01)


def _as_floats(values, fill: float) -> list[float]:
    """Sei numeri come lista, da qualunque cosa il chiamante abbia passato.

    Il percorso storico faceva ``np.asarray(...)``, che accetta indifferentemente
    ``ndarray``, lista o tupla -- e i test del motore a oggetti passano tuple.
    Questa funzione conserva quella tolleranza tenendo la via veloce per gli
    ``ndarray``, che sono il caso del ciclo vero.
    """
    if values is None:
        return [fill] * pillars.N_PILLARS
    as_list = getattr(values, "tolist", None)
    return as_list() if as_list is not None else [float(value) for value in values]


def score_decision_python(
    agent,
    mask: np.ndarray,
    action_priority: np.ndarray,
    u01: float,
    sampling: str,
    survival_enabled: bool,
) -> tuple[np.ndarray, int]:
    """Lo stesso calcolo senza NumPy, per array da sei elementi.

    Ogni enunciato ricalca l'ordine dell'originale, incluso ``** 2``, che non e'
    sostituibile con una moltiplicazione (vedi la nota nel docstring del modulo).
    Le due maschere sono convertite in liste una volta: dentro i cicli, un
    accesso a lista costa una frazione di un accesso ad ``ndarray``, che
    restituisce uno scalare NumPy da scatolare.
    """
    if sampling not in ("greedy", "softmax"):
        raise ValueError(f"unknown decision sampling: {sampling}")
    mask_flags = mask.tolist()
    priorities = action_priority.tolist()

    preferences = _as_floats(
        getattr(agent, "pillar_preferences", None), _UNIFORM_PREFERENCE
    )
    skills = _as_floats(getattr(agent, "pillar_skills", None), 1.0)

    if survival_enabled:
        inventory = agent.inventory
        material = float(inventory.construction_material)
        minerals = float(inventory.minerals)
        deprived = (
            int(getattr(agent, "steps_without_water", 0)) >= 2
            or int(getattr(agent, "steps_without_food", 0)) >= 6
        )
        sustenance_base = max(1.0 - agent.hydration, 1.0 - agent.satiety)
        resources = 1.0 - 0.5 * (material / 10.0 + minerals / 6.0)
        life_base = max(1.0 - agent.health, agent.fatigue)
        stress = float(agent.stress_index)
        mission_active = (
            getattr(agent, "scout_phase", None) is not None
            or getattr(agent, "settle_phase", None) is not None
            or bool(getattr(agent, "founder_kit_reserved", False))
        )
        urgencies = [
            min(
                1.0,
                sustenance_base**2 * 1.5 + (0.3 if deprived else 0.0),
            ),
            0.0 if resources < 0.0 else (1.0 if resources > 1.0 else resources),
            0.6 * float(any(mask_flags[index] for index in _BUILD_INDICES))
            + 0.4 * min(1.0, material / 10.0),
            min(1.0, life_base**2 * 1.3 + 0.2 * stress),
            min(1.0, 0.8 * stress + max(0.0, 0.82 - float(agent.morale))),
            min(
                1.0,
                0.25 + 0.6 * float(agent.curiosity) + (0.5 if mission_active else 0.0),
            ),
        ]
    else:
        urgencies = [1.0] * pillars.N_PILLARS

    raw = [0.0] * pillars.N_PILLARS
    total = 0.0
    for pillar, indices in enumerate(_PILLAR_INDICES):
        best = 0.0
        available = False
        for index in indices:
            if mask_flags[index]:
                priority = priorities[index]
                # `max(values, default=0.0)` dell'originale: lo zero e' il valore
                # per lista VUOTA, non un minimo. Con una sola priorita' negativa
                # ammissibile, `if priority > best` la scarterebbe e restituirebbe
                # zero, cioe' un pilastro piu' attraente di quanto la cella dica.
                if not available or priority > best:
                    best = priority
                available = True
        if available:
            value = preferences[pillar] * urgencies[pillar] * best * skills[pillar]
            raw[pillar] = value
            total += value
    if total <= 0.0:
        return np.zeros(pillars.N_PILLARS, dtype=np.float64), 0

    scores = [value / total for value in raw]
    if sampling == "greedy":
        chosen = 0
        best_score = scores[0]
        for pillar in range(1, pillars.N_PILLARS):
            if scores[pillar] > best_score:
                best_score, chosen = scores[pillar], pillar
    else:
        cumulative = 0.0
        running = [0.0] * pillars.N_PILLARS
        for pillar, value in enumerate(scores):
            cumulative += value
            running[pillar] = cumulative
        if cumulative <= 0.0:
            chosen = 0
        else:
            threshold = float(u01) * cumulative
            chosen = 0
            for value in running:
                if value <= threshold:
                    chosen += 1
                else:
                    break
            if chosen > _LAST_PILLAR:
                chosen = _LAST_PILLAR
    return np.array(scores, dtype=np.float64), chosen


_SCORER = None


def _scorer():
    """Il valutatore nativo, costruito una volta per processo.

    Le tabelle costanti attraversano il confine qui e non a ogni decisione: sono
    costanti, e passarle ~300 volte per passo sarebbe costo puro. Vivono comunque
    solo lato Python -- una copia scritta in Rust sarebbe una seconda sorgente di
    verita' capace di divergere in silenzio.
    """
    global _SCORER
    if _SCORER is None:
        _SCORER = native.require().DecisionScorer(
            [list(indices) for indices in _PILLAR_INDICES],
            list(_BUILD_INDICES),
            (C.R["construction_material"], C.R["minerals"]),
            [MISSION_SCOUT_OUT, MISSION_SCOUT_BACK, MISSION_SETTLE_OUT],
            URGENCY_EXPONENT,
        )
    return _SCORER


def score_decision_rust(
    agent,
    mask: np.ndarray,
    action_priority: np.ndarray,
    u01: float,
    sampling: str,
    survival_enabled: bool,
) -> tuple[np.ndarray, int]:
    """Lo stesso calcolo nel kernel nativo, un attraversamento per decisione.

    Le colonne passano **intere** con l'indice di riga: ``rust-numpy`` le condivide
    per riferimento, quindi la selezione costa meno fatta dal lato nativo. E' la
    regola ricavata dal Task A, dove il 90,9% del costo osservato di un
    attraversamento era preparazione degli argomenti lato Python.
    """
    arrays = getattr(agent, "_a", None)
    if arrays is None:
        # Agente del motore a oggetti: non ha colonne da passare. Il braccio
        # Python puro e' l'equivalente esatto -- verificato bit-exact -- quindi
        # qui si degrada a quello invece di fallire su un percorso che i test del
        # motore a oggetti percorrono ancora.
        return score_decision_python(
            agent, mask, action_priority, u01, sampling, survival_enabled
        )
    if sampling not in ("greedy", "softmax"):
        raise ValueError(f"unknown decision sampling: {sampling}")
    scores, chosen = _scorer().score(
        int(agent.row),
        mask,
        action_priority,
        arrays.hydration,
        arrays.satiety,
        arrays.health,
        arrays.fatigue,
        arrays.stress,
        arrays.morale,
        arrays.curiosity,
        arrays.steps_without_water,
        arrays.steps_without_food,
        arrays.inv,
        arrays.pref,
        arrays.skill,
        arrays.mission,
        bool(getattr(agent, "founder_kit_reserved", False)),
        float(u01),
        sampling == "greedy",
        bool(survival_enabled),
    )
    return scores, int(chosen)


_ARMS = {
    "numpy": score_decision_numpy,
    "python": score_decision_python,
    "rust": score_decision_rust,
}


def score_decision(
    agent,
    mask: np.ndarray,
    action_priority: np.ndarray,
    u01: float,
    sampling: str,
    survival_enabled: bool,
    backend: str | None = None,
) -> tuple[np.ndarray, int]:
    """Punteggi normalizzati dei sei pilastri e pilastro scelto."""
    arm = _ARMS.get(backend or _RESOLVED)
    if arm is None:
        raise ValueError(
            f"backend {backend or _RESOLVED!r} non disponibile; "
            f"scegliere fra {', '.join(sorted(_ARMS))}"
        )
    return arm(agent, mask, action_priority, u01, sampling, survival_enabled)


# Risolto una volta sola: e' letto dentro un percorso eseguito ~300 volte per
# passo, dove una lettura di variabile d'ambiente per chiamata costerebbe piu' di
# quanto la scelta faccia risparmiare.
_RESOLVED = configured_backend()
