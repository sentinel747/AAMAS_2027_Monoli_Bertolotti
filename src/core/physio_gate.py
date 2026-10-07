from __future__ import annotations

"""Il filtro che dimostra, per un intero batch, chi non puo' avere un'emergenza.

Task C.1 del piano ``docs/superpowers/plans/2026-08-07-decision-loop-rust-port.md``.

``_physiological_override`` (src/agents/preference_agent.py) restituisce ``None``
nel **99,98%** delle chiamate: non spende il suo tempo a decidere, lo spende a
verificare per poi concludere che non c'e' niente da fare. Questo modulo esegue
quella verifica una volta per passo, per tutti gli agenti insieme, e dice al
ciclo sequenziale quali chiamate puo' saltare del tutto.

**Perche' e' lecito saltarle.** Il filtro non e' un'euristica: e' una condizione
sufficiente dimostrata. La funzione ha otto rami, ognuno protetto da una guardia,
e il salto richiede che tutte e otto le guardie siano false. Sette sono scalari
per-agente. L'ottavo -- ``_return_for_ration_budget``, che e' l'unico ramo **senza
guardia** -- e' quello che rende il filtro non banale:

``_return_for_ration_budget`` puo' restituire un movimento solo se la distanza
dal supporto e' ``> 0``. Se la cella dell'agente ha capacita' di supporto vitale
positiva allora, per la definizione di ``local_life_support_capacity``, quella
cella contiene almeno una serra; la serra appartiene a entrambi gli insiemi di
strutture interrogati (``{GREENHOUSE}`` e ``{GREENHOUSE, HABITAT, INFIRMARY}``);
quindi entrambe le distanze valgono **zero** e nessuno dei due rami puo' scattare.
La stessa condizione azzera anche il ramo "terreno non supportato".

Quindi una singola lettura di ``struct_count`` -- una colonna che il simulatore
gia' mantiene -- sostituisce due risoluzioni di distanza di supporto per agente
per passo, senza interrogare il mondo.

**Effetti collaterali.** Saltare la chiamata e' innocuo perche' i rami saltati non
scrivono nulla: ``_move_toward_*`` e ``_return_for_ration_budget`` costruiscono un
``ActionRequest`` leggendo il mondo, e l'unica memoria che toccano e' la cache
per-passo dei bersagli-struttura, che e' memoizzazione pura.

**Tre bracci, per il punto R2.1 del piano di migrazione.** ``scalar`` e' l'oracolo
leggibile; ``numpy`` e' il braccio di controllo, perche' la condizione e'
per-agente su colonne SoA e quindi vettorizzabile senza cambiare linguaggio;
``rust`` esegue lo stesso predicato nel kernel nativo. Il numero che quantifica il
cambio di linguaggio e' ``numpy / rust``, mai ``scalar / rust``.
"""

import os

import numpy as np

from src.agents.build_policy import COLONISTS_PER_STRUCTURE
from src.agents.preference_agent import (
    CRITICAL_HEALTH,
    CRITICAL_HYDRATION,
    CRITICAL_SATIETY,
    FOOD_GUARD_STEPS,
    FOOD_RETURN_RESERVE,
    WATER_GUARD_STEPS,
    WATER_RETURN_RESERVE,
)
from src.core import constants as C
from src.core import native
from src.world.structures import StructureType

VALID_BACKENDS = ("off", "scalar", "numpy", "rust")

# Il default e' `numpy`, e la ragione e' riproducibilita': e' l'unico braccio
# disponibile in ogni checkout, quindi il comportamento predefinito non dipende
# dall'aver compilato l'estensione. `rust` resta selezionabile ed e' il braccio
# che il Task D confronta. I tre bracci producono lo stesso vettore di booleani --
# e' verificato da `tests/core/test_physio_gate.py`, che li confronta fra loro e
# poi confronta il salto con l'esito reale della funzione scalare.
_BACKEND_ENV = "MARSABM_PHYSIO_GATE"

_R_WATER = C.R["water"]
_R_ICE = C.R["ice"]
_R_FOOD = C.R["food"]

_S_SHELTER = C.S[StructureType.SHELTER]
_S_HABITAT = C.S[StructureType.HABITAT]
_S_INFIRMARY = C.S[StructureType.INFIRMARY]
_S_GREENHOUSE = C.S[StructureType.GREENHOUSE]
_S_SOLAR = C.S[StructureType.SOLAR_ARRAY]
_S_OXYGEN = C.S[StructureType.OXYGEN_PLANT]

_RATIO_GREENHOUSE = int(COLONISTS_PER_STRUCTURE[StructureType.GREENHOUSE])
_RATIO_SOLAR = int(COLONISTS_PER_STRUCTURE[StructureType.SOLAR_ARRAY])
_RATIO_OXYGEN = int(COLONISTS_PER_STRUCTURE[StructureType.OXYGEN_PLANT])


def configured_backend() -> str:
    """Backend richiesto dall'ambiente, validato."""
    backend = os.environ.get(_BACKEND_ENV, "numpy").strip().lower()
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"{_BACKEND_ENV}={backend!r} non e' valido; "
            f"scegliere fra {', '.join(VALID_BACKENDS)}"
        )
    if backend == "rust" and not native.is_available():
        raise ValueError(
            f"{_BACKEND_ENV}=rust richiede il kernel nativo, non disponibile: "
            f"{native.unavailable_reason()}. Compilare con "
            "`python scripts/build_rust.py`."
        )
    return backend


def _support_capacity_positive(counts) -> bool:
    """``local_life_support_capacity(cell) > 0`` letto dai soli conteggi.

    Riscritta in interi: ``int(shelter + 2*habitat + 0.5*infirmary)`` e'
    esattamente ``(2*shelter + 4*habitat + infirmary) // 2`` per conteggi non
    negativi, e restare negli interi rende i tre bracci identici per costruzione
    invece che per fortuna di arrotondamento.
    """
    housing = (
        2 * int(counts[_S_SHELTER])
        + 4 * int(counts[_S_HABITAT])
        + int(counts[_S_INFIRMARY])
    ) // 2
    capacity = min(
        housing,
        int(counts[_S_GREENHOUSE]) * _RATIO_GREENHOUSE,
        int(counts[_S_SOLAR]) * _RATIO_SOLAR,
        int(counts[_S_OXYGEN]) * _RATIO_OXYGEN,
    )
    return max(0, capacity) > 0


def skip_override_scalar(agents, cells, rows: np.ndarray) -> np.ndarray:
    """L'oracolo leggibile: una riga alla volta, nell'ordine dei rami saltati."""
    rows = np.asarray(rows, dtype=np.int64)
    verdicts = np.zeros(rows.size, dtype=np.bool_)
    for position, row in enumerate(rows):
        index = int(row)
        y = int(agents.y[index])
        x = int(agents.x[index])
        inventory = agents.inv[index]
        verdicts[position] = (
            # rami D e G: la cella e' supportata, quindi entrambe le distanze
            # di supporto sono zero e il terreno non e' "non supportato"
            _support_capacity_positive(cells.struct_count[y, x])
            # rami A e C (orologio della fame) e B (orologio della sete)
            and int(agents.steps_without_food[index]) < FOOD_GUARD_STEPS
            and int(agents.steps_without_water[index]) < WATER_GUARD_STEPS
            # ramo B (idratazione), C (sazieta'), H (salute)
            and float(agents.hydration[index]) >= CRITICAL_HYDRATION
            and float(agents.satiety[index]) >= CRITICAL_SATIETY
            and float(agents.health[index]) >= CRITICAL_HEALTH
            # rami E ed F: le riserve trasportate sono sopra la soglia di rientro
            and float(inventory[_R_WATER]) + float(inventory[_R_ICE])
            > WATER_RETURN_RESERVE
            and float(inventory[_R_FOOD]) > FOOD_RETURN_RESERVE
        )
    return verdicts


def skip_override_numpy(agents, cells, rows: np.ndarray) -> np.ndarray:
    """Braccio di controllo: lo stesso predicato come algebra sulle colonne."""
    rows = np.asarray(rows, dtype=np.int64)
    ys = agents.y[rows].astype(np.int64, copy=False)
    xs = agents.x[rows].astype(np.int64, copy=False)
    counts = cells.struct_count[ys, xs].astype(np.int64, copy=False)
    housing = (
        2 * counts[:, _S_SHELTER]
        + 4 * counts[:, _S_HABITAT]
        + counts[:, _S_INFIRMARY]
    ) // 2
    capacity = np.minimum(
        np.minimum(housing, counts[:, _S_GREENHOUSE] * _RATIO_GREENHOUSE),
        np.minimum(
            counts[:, _S_SOLAR] * _RATIO_SOLAR,
            counts[:, _S_OXYGEN] * _RATIO_OXYGEN,
        ),
    )
    inventory = agents.inv[rows]
    return (
        (capacity > 0)
        & (agents.steps_without_food[rows] < FOOD_GUARD_STEPS)
        & (agents.steps_without_water[rows] < WATER_GUARD_STEPS)
        & (agents.hydration[rows] >= CRITICAL_HYDRATION)
        & (agents.satiety[rows] >= CRITICAL_SATIETY)
        & (agents.health[rows] >= CRITICAL_HEALTH)
        & (
            inventory[:, _R_WATER] + inventory[:, _R_ICE]
            > WATER_RETURN_RESERVE
        )
        & (inventory[:, _R_FOOD] > FOOD_RETURN_RESERVE)
    )


def skip_override_rust(agents, cells, rows: np.ndarray) -> np.ndarray:
    """Lo stesso predicato nel kernel nativo, con un solo attraversamento.

    Le colonne sono passate **intere** e la selezione delle righe e' lasciata al
    lato nativo: e' la regola di progetto ricavata dal Task A, dove il 90,9% del
    costo osservato di un attraversamento non era l'attraversamento ma la
    preparazione degli argomenti lato Python.
    """
    module = native.require()
    verdicts = module.physio_gate(
        np.ascontiguousarray(rows, dtype=np.int64),
        agents.x,
        agents.y,
        agents.hydration,
        agents.satiety,
        agents.health,
        agents.steps_without_water,
        agents.steps_without_food,
        agents.inv,
        cells.struct_count,
        (_R_WATER, _R_ICE, _R_FOOD),
        (
            _S_SHELTER,
            _S_HABITAT,
            _S_INFIRMARY,
            _S_GREENHOUSE,
            _S_SOLAR,
            _S_OXYGEN,
        ),
        (_RATIO_GREENHOUSE, _RATIO_SOLAR, _RATIO_OXYGEN),
        (
            CRITICAL_HYDRATION,
            CRITICAL_SATIETY,
            CRITICAL_HEALTH,
            WATER_RETURN_RESERVE,
            FOOD_RETURN_RESERVE,
        ),
        (int(WATER_GUARD_STEPS), int(FOOD_GUARD_STEPS)),
    )
    return np.asarray(verdicts, dtype=np.bool_)


_ARMS = {
    "scalar": skip_override_scalar,
    "numpy": skip_override_numpy,
    "rust": skip_override_rust,
}


def skip_override(agents, cells, rows: np.ndarray, backend: str | None = None):
    """Righe per cui ``_physiological_override`` restituisce ``None`` con certezza.

    Restituisce ``None`` quando il filtro e' disattivato: il chiamante deve
    trattarlo come "non lo so per nessuno" e chiamare la funzione scalare per
    tutti, che e' il comportamento storico.
    """
    resolved = configured_backend() if backend is None else backend
    if resolved == "off":
        return None
    if resolved not in _ARMS:
        raise ValueError(
            f"backend {resolved!r} non valido; "
            f"scegliere fra {', '.join(VALID_BACKENDS)}"
        )
    return _ARMS[resolved](agents, cells, rows)
