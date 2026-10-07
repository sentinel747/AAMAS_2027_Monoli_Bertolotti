from __future__ import annotations

"""Raggiungibilita' con rientro garantito, in tre implementazioni confrontabili.

Task 4 del piano di migrazione Rust. Il cluster
``_can_visit_cell_and_return`` / ``_return_ration_requirement`` /
``_support_distance_cells`` e' il primo bersaglio: dopo il Task 0b il profilo lo
misura a circa il 49% della run strumentata, con 70.032 valutazioni ogni 30 step.

Qui convivono deliberatamente **tre** implementazioni della stessa semantica:

``scalar``
    Ricalcola la distanza di supporto per ogni coppia (agente, cella), come fa
    oggi il percorso a oggetti. E' l'oracolo di correttezza e la baseline
    "algoritmo attuale".

``numpy``
    Riformula il calcolo come campo di distanza per l'intera griglia, una volta
    per step, piu' aritmetica vettoriale sulle coppie. E' il **braccio di
    controllo** richiesto dal punto R2.1 della revisione 2 del piano: senza di
    esso, il guadagno del riordino algoritmico verrebbe attribuito a Rust.

``rust``
    Lo stesso algoritmo del braccio NumPy, eseguito da ``mars_core``.

Il numero che quantifica il cambio di linguaggio e' ``numpy / rust``. Il rapporto
``scalar / rust`` misura invece riordino algoritmico e linguaggio insieme, ed e'
onesto solo se dichiarato come tale.
"""

import os

import numpy as np

from src.core import native

# Nessun supporto raggiungibile. Coincide con `mars_core::reachability::NO_SUPPORT`.
NO_SUPPORT = -1

# Backend del campo di distanza usato dal percorso decisionale reale.
#
# `scalar` conserva il comportamento storico: per ogni origine, minimo su tutti i
# target, con cache per (tipo, origine). `numpy` e `rust` calcolano invece
# l'intero campo di distanza una volta per step e servono ogni query come accesso
# ad array.
#
# **Il default resta `scalar`, ed e' una decisione basata su misura, non
# conservatorismo.** Sul microbenchmark del batch isolato la formulazione a campo
# vince 261x, ma end-to-end nel simulatore e' piu' LENTA: 0,862x per `numpy` e
# 0,922x per `rust` (300 agenti x 30 step, confronti interlacciati e appaiati).
# Il motivo e' che la cache per-origine ha un tasso di successo altissimo -- gli
# agenti si concentrano in poche celle -- quindi il ramo scalar calcola il minimo
# per pochissime origini distinte, mentre il campo lo calcola per tutte le 864
# celle della griglia. Il microbenchmark, misurando coppie tutte distinte,
# assumeva implicitamente un tasso di successo nullo.
#
# I backend a campo restano selezionabili perche' servono al confronto a tre
# bracci e perche' diventeranno convenienti quando il confine con Rust si
# spostera' piu' in alto (un'unica chiamata per step invece di una per query).
VALID_BACKENDS = ("scalar", "numpy", "rust")

# Il backend viene risolto una volta sola. `configured_backend()` sta sul percorso
# caldo (oltre 150.000 chiamate ogni 30 step con 300 agenti): rileggere e validare
# una variabile d'ambiente a ogni chiamata reintrodurrebbe, nel punto che si sta
# ottimizzando, proprio il tipo di costo per-chiamata che questa modifica elimina.
_RESOLVED_BACKEND: str | None = None


def _resolve_backend() -> str:
    requested = os.environ.get("MARSABM_REACH_BACKEND", "scalar").strip().lower()
    if requested not in VALID_BACKENDS:
        raise ValueError(
            f"MARSABM_REACH_BACKEND={requested!r} non valido; "
            f"attesi {', '.join(VALID_BACKENDS)}"
        )
    if requested == "rust" and not native.is_available():
        raise native.NativeKernelUnavailable(
            "MARSABM_REACH_BACKEND=rust richiede il kernel nativo: "
            f"{native.unavailable_reason()}. Eseguire `python scripts/build_rust.py`."
        )
    return requested


def configured_backend() -> str:
    """Backend risolto per questo processo, validato una volta sola."""
    global _RESOLVED_BACKEND
    if _RESOLVED_BACKEND is None:
        _RESOLVED_BACKEND = _resolve_backend()
    return _RESOLVED_BACKEND


def reset_backend_cache() -> None:
    """Rilegge l'ambiente alla prossima chiamata. Solo per i test.

    Non e' un interruttore di runtime: cambiare backend a meta' run lascerebbe in
    giro campi di distanza calcolati da un backend e consultati da un altro. I
    test che confrontano i backend lo fanno in processi separati; questa funzione
    serve ai test che verificano la validazione dell'ambiente.
    """
    global _RESOLVED_BACKEND
    _RESOLVED_BACKEND = None

# Costanti di razionamento, allineate a `src/agents/rule_based_agent.py`.
DEFAULT_RATION_PER_STEP = 0.1
DEFAULT_SAFETY_STEPS = 2


def wrapped_chebyshev(first_x, first_y, second_x, second_y, width):
    """Distanza Chebyshev con asse x periodico, in forma vettorizzabile.

    Accetta scalari o array: e' la stessa formula di
    ``src/world/navigation.py``, scritta con operazioni che NumPy propaga.
    """
    direct = np.abs(np.asarray(first_x) - np.asarray(second_x)) % width
    horizontal = np.minimum(direct, width - direct)
    vertical = np.abs(np.asarray(first_y) - np.asarray(second_y))
    return np.maximum(horizontal, vertical)


# --------------------------------------------------------------------------
# Braccio `scalar`: l'algoritmo attuale, una query per coppia.
# --------------------------------------------------------------------------

def support_distance_scalar(x, y, target_columns, target_rows, width):
    """Distanza minima da un supporto, ricalcolata per la singola cella."""
    if len(target_columns) == 0:
        return NO_SUPPORT
    best = None
    for target_x, target_y in zip(target_columns, target_rows):
        direct = abs(int(x) - int(target_x)) % width
        horizontal = min(direct, width - direct)
        distance = max(horizontal, abs(int(y) - int(target_y)))
        if best is None or distance < best:
            best = distance
    return int(best)


def reachability_scalar(
    pair_agent,
    pair_cell,
    grid_width,
    grid_height,
    food_targets,
    water_targets,
    agent_food,
    agent_water,
    agent_steps_per_cell,
    ration_per_step: float = DEFAULT_RATION_PER_STEP,
    safety_steps: int = DEFAULT_SAFETY_STEPS,
) -> np.ndarray:
    """Valuta ogni coppia in modo indipendente, ricalcolando ogni distanza."""
    del grid_height  # l'indice piatto porta gia' l'informazione di riga
    food_columns, food_rows = food_targets
    water_columns, water_rows = water_targets
    out = np.zeros(len(pair_agent), dtype=np.bool_)

    for index in range(len(pair_agent)):
        agent = int(pair_agent[index])
        cell = int(pair_cell[index])
        cell_x = cell % grid_width
        cell_y = cell // grid_width
        steps_per_cell = int(agent_steps_per_cell[agent])

        food_distance = support_distance_scalar(
            cell_x, cell_y, food_columns, food_rows, grid_width
        )
        water_distance = support_distance_scalar(
            cell_x, cell_y, water_columns, water_rows, grid_width
        )

        if food_distance == NO_SUPPORT and water_distance == NO_SUPPORT:
            out[index] = True
            continue
        if food_distance == NO_SUPPORT or water_distance == NO_SUPPORT:
            out[index] = False
            continue

        food_required = (
            food_distance * steps_per_cell + max(0, safety_steps)
        ) * ration_per_step
        water_required = (
            water_distance * steps_per_cell + max(0, safety_steps)
        ) * ration_per_step
        out[index] = (
            float(agent_food[agent]) - ration_per_step >= food_required
            and float(agent_water[agent]) - ration_per_step >= water_required
        )
    return out


# --------------------------------------------------------------------------
# Braccio `numpy`: campo di distanza una volta per step, poi aritmetica.
# --------------------------------------------------------------------------

def support_distance_field_numpy(
    width: int, height: int, target_columns, target_rows
) -> np.ndarray:
    """Campo delle distanze di supporto, in ordine di riga (layout C)."""
    cells = int(width) * int(height)
    targets_x = np.asarray(target_columns, dtype=np.int64).ravel()
    targets_y = np.asarray(target_rows, dtype=np.int64).ravel()
    if targets_x.size == 0:
        return np.full(cells, NO_SUPPORT, dtype=np.int32)

    grid_y, grid_x = np.divmod(np.arange(cells, dtype=np.int64), int(width))
    # Una passata per target invece di una matrice (celle x target): con poche
    # strutture di supporto e molte celle questa forma allora meno memoria e
    # sfrutta il minimo cumulativo in place.
    best = None
    for target_x, target_y in zip(targets_x, targets_y):
        distance = wrapped_chebyshev(grid_x, grid_y, target_x, target_y, int(width))
        best = distance if best is None else np.minimum(best, distance)
    return best.astype(np.int32, copy=False)


def reachability_batch_numpy(
    pair_agent,
    pair_cell,
    food_field,
    water_field,
    agent_food,
    agent_water,
    agent_steps_per_cell,
    ration_per_step: float = DEFAULT_RATION_PER_STEP,
    safety_steps: int = DEFAULT_SAFETY_STEPS,
) -> np.ndarray:
    """Maschera di raggiungibilita', valutata in blocco su tutte le coppie."""
    agents = np.asarray(pair_agent, dtype=np.int64)
    cells = np.asarray(pair_cell, dtype=np.int64)
    steps = np.asarray(agent_steps_per_cell, dtype=np.int64)[agents]

    food_distance = np.asarray(food_field, dtype=np.int64)[cells]
    water_distance = np.asarray(water_field, dtype=np.int64)[cells]

    food_missing = food_distance == NO_SUPPORT
    water_missing = water_distance == NO_SUPPORT

    margin = max(0, int(safety_steps))
    food_required = (food_distance * steps + margin) * ration_per_step
    water_required = (water_distance * steps + margin) * ration_per_step

    carried_food = np.asarray(agent_food, dtype=np.float64)[agents]
    carried_water = np.asarray(agent_water, dtype=np.float64)[agents]
    affordable = (carried_food - ration_per_step >= food_required) & (
        carried_water - ration_per_step >= water_required
    )

    # `np.select` invece di tre passaggi separati: l'ordine dei casi riproduce
    # quello del percorso scalare, dove l'assenza TOTALE di supporto e' un
    # permesso esplicito e l'assenza PARZIALE e' un divieto.
    return np.select(
        [food_missing & water_missing, food_missing | water_missing],
        [True, False],
        default=affordable,
    ).astype(np.bool_, copy=False)


# --------------------------------------------------------------------------
# Braccio `rust`.
# --------------------------------------------------------------------------

def support_distance_field_rust(
    width: int, height: int, target_columns, target_rows
) -> np.ndarray:
    module = native.require()
    return module.support_distance_field_py(
        int(width),
        int(height),
        np.ascontiguousarray(target_columns, dtype=np.int32),
        np.ascontiguousarray(target_rows, dtype=np.int32),
    )


def reachability_batch_rust(
    pair_agent,
    pair_cell,
    food_field,
    water_field,
    agent_food,
    agent_water,
    agent_steps_per_cell,
    ration_per_step: float = DEFAULT_RATION_PER_STEP,
    safety_steps: int = DEFAULT_SAFETY_STEPS,
) -> np.ndarray:
    module = native.require()
    return module.reachability_batch_py(
        np.ascontiguousarray(pair_agent, dtype=np.uint32),
        np.ascontiguousarray(pair_cell, dtype=np.uint32),
        np.ascontiguousarray(food_field, dtype=np.int32),
        np.ascontiguousarray(water_field, dtype=np.int32),
        np.ascontiguousarray(agent_food, dtype=np.float64),
        np.ascontiguousarray(agent_water, dtype=np.float64),
        np.ascontiguousarray(agent_steps_per_cell, dtype=np.int32),
        float(ration_per_step),
        int(safety_steps),
    )


# --------------------------------------------------------------------------
# Campo di distanza per il percorso decisionale reale.
# --------------------------------------------------------------------------

def support_distance_field(
    width: int, height: int, target_columns, target_rows, backend: str
) -> np.ndarray:
    """Campo di distanza col backend richiesto, in ordine di riga."""
    if backend == "rust":
        return np.asarray(
            support_distance_field_rust(width, height, target_columns, target_rows)
        )
    if backend == "numpy":
        return support_distance_field_numpy(width, height, target_columns, target_rows)
    raise ValueError(f"backend senza campo precomputato: {backend!r}")


def cached_support_distance_field(world, cache_key, targets, backend: str):
    """Campo di distanza memorizzato sul mondo, valido per lo step corrente.

    Riproduce la validita' della cache preesistente di `_support_distance_cells`:
    una voce per step, invalidata al cambio di `world.step`. In particolare una
    struttura costruita **durante** lo step non aggiorna il campo, esattamente
    come non aggiornava la cache per-origine che questo codice sostituisce.
    Conservarlo e' voluto: cambiarlo cambierebbe le decisioni, e questa modifica
    deve restare bit-exact.

    Il campo copre l'intera griglia, quindi una singola voce per step serve tutte
    le origini, mentre la cache precedente ne memorizzava una per ogni coppia
    (tipo, origine) e ricalcolava il minimo su tutti i target a ogni nuova
    origine.
    """
    step = int(getattr(world, "step", 0))
    cache_state = getattr(world, "_support_distance_field_cache", None)
    if not cache_state or cache_state[0] != step:
        cache_state = (step, {})
        world._support_distance_field_cache = cache_state

    field = cache_state[1].get(cache_key)
    if field is None:
        columns = np.fromiter(
            (int(target.x) for target in targets), dtype=np.int64, count=len(targets)
        )
        rows = np.fromiter(
            (int(target.y) for target in targets), dtype=np.int64, count=len(targets)
        )
        field = support_distance_field(
            int(world.width), int(world.height), columns, rows, backend
        )
        cache_state[1][cache_key] = field
    return field
