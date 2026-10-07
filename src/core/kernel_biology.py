from __future__ import annotations

"""1:1 vectorized transcription of the per-step cell pipeline used by BOTH
object-model engines:

  - biology            Cell.update_biology            (src/world/cell.py:69-153)
  - structure effects   apply_structure_effects         (src/simulation/step_effects.py:33-55)
  - colony feedback     apply_colony_resource_feedback  (src/simulation/step_effects.py:63-96)
  - structure wear      apply_structure_wear            (src/simulation/step_effects.py:113-147)
  - habitability        Cell.recompute_habitability     (src/world/cell.py:154-188)
  - knowledge gain       research_knowledge_gain          (src/simulation/step_effects.py:99-110)

The object functions remain the truth FOR THE BLOCKS LISTED ABOVE: if this
module and those sources ever diverge, fix the transcription here, never loosen
the equivalence test (tests/core/test_kernel_biology_equivalence.py).

**One block has no object counterpart, and saying so is the point (2026-09-14).**
The ISRU conversion (regolith -> construction material + minerals, see
`isru_material_rate` below) exists only here: `step_effects` never grew it. Any
run of this project goes through `kernel.step`, so no published number depends
on the object path for it --- but the oracle does not cover that block, and
whoever reads the line above would otherwise go looking for a truth that is not
there. Knowledge gain, by contrast, IS covered: `research_knowledge_gain`
(step_effects.py:422) exists and the equivalence test compares it.

SCOPE (review round 3, Important 1): a seventh source function belongs on
this list - `update_biology_cells` (src/simulation/biology_update.py), the
wrapper BOTH engines actually call biology through. It takes a `full_grid`
flag (default False, the "active" scope) that restricts `Cell.update_biology`
to cells passing `_biology_cell_can_change` instead of visiting the whole
grid - a cheap Python-loop skip for the object model, since an untouched
baseline cell has nothing to advance. `update_cells` below takes the same
`full_grid` flag and reproduces that scope with `_active_biology_mask`, the
vectorized transcription of `_biology_cell_can_change` (masking WRITES,
since numpy has no equivalent "skip the array element" shortcut - the
computation still runs over the whole grid either way, only the write to
`cells.*` is gated). Getting this wrong is invisible on any scenario built
entirely of already-active cells (occupied, structured, or already carrying
biomass/water/nutrients) - see `_active_biology_mask`'s own docstring for
the exact predicate.

ORDERING NOTE (important divergence trap): the object pipeline's real
per-step call order is biology -> structure effects -> colony feedback ->
wear, and `Cell.recompute_habitability` is called TWICE per step for a cell
that hosts structures (once inside `update_biology`, cell.py:153; once again
inside `apply_structure_effects`, step_effects.py:51, AFTER the greenhouse
fertilization has mutated vegetation/nutrients) and only ONCE (the
update_biology call) for a cell with none. Crucially, `apply_structure_wear`
NEVER calls recompute_habitability - wear reduces `structure.integrity` but
that only shows up in habitability on the FOLLOWING step, when the next
`update_biology`/`apply_structure_effects` call reads the now-lower
`structure.local_effect`. So the habitability score that "sticks" for a
step is always computed from PRE-wear structure integrities.

Below, the habitability+pollution-density-floor recompute therefore runs
right after structure effects (mirroring the second, sticking, object-model
call) and BEFORE wear - not after, even though wear is listed earlier in the
task's prose pipeline order. Placing it after wear would feed the
just-reduced struct_fx into the habitability structure term and permanently
diverge from the object engine by one step's wear on every cell that hosts
structures (verified against tests/core/test_kernel_biology_equivalence.py's
scenario, which builds a GREENHOUSE + SOLAR_ARRAY cell specifically to
exercise this). Colony feedback and wear do not touch any of
vegetation/water_ice/liquid_water/pollution/nutrients/struct_fx-habitability,
so doing the recompute once, at that point, reproduces the object model's
final per-cell habitability_score exactly for every cell (structured or not).

AGGREGATION SACRIFICES (documented divergences from the per-structure object
model, accepted so the whole grid can be updated with array ops instead of a
Python loop per structure):

- Efficiency threshold on the MEAN, not per-instance (`CellArrays.from_world`,
  src/core/arrays.py:171-178; reused here as `eff_adj`/`mean_int`, line ~319):
  the object model applies the 0.4 efficiency cliff to each structure's own
  integrity, then sums; the aggregate applies it to `mean(integrity)*count`.
  The two coincide when every same-type structure at a cell sits on the same
  side of 0.4, and diverge otherwise (e.g. two SHELTERs at 1.0 and 0.3: objects
  give efficiency 1.0 + 0.15 = 1.15; the aggregate gives eff(0.65) * 2 = 1.3).
- Wear's zero-clamp is on the SUM, not per-instance (line ~381,
  `cells.struct_integrity[...] = np.maximum(0.0, cells.struct_integrity -
  total_wear[..., None] * count_f)`): `apply_structure_wear` (step_effects.py:140)
  clamps each `s.integrity` at 0 individually. Two same-type structures at 0.05
  and 0.9 hit with wear 0.1: objects give max(0, 0.05-0.1) + max(0, 0.9-0.1) =
  0.0 + 0.8 = 0.8; the aggregate gives max(0, (0.05+0.9) - 2*0.1) = 0.75. Only
  visible when a same-type structure at a cell is wear-driven below zero while
  a same-type sibling is not (rare at default wear rates, reachable under
  heavy dust/wind/radiation or many steps).
- Structure-warning events are one-per-(cell, type) here (post_mean_int below
  0.2), vs one-per-STRUCTURE-instance in the object model (step_effects.py:141);
  a cell with two same-type structures where only one instance is individually
  below 0.2 can emit here (mean below 0.2) when the object model would not, or
  vice versa.

Neither sacrifice is exercised by the default scenario (`_scenario()`, one
structure per type per cell, so aggregate == per-instance); both require a
scenario with >=2 same-type structures at integrities straddling 0.4 or the
post-wear zero floor to observe, and are intentionally out of scope for the
equivalence tests here (documented instead of silently accepted).
"""

import math
import os

import numpy as np

from src.core import constants as C
from src.core.arrays import AgentArrays, CellArrays, TERRAIN_ORDER
from src.core.constants import (
    AGENT_ENERGY_RESERVE_CAP,
    AGENT_OXYGEN_RESERVE_CAP,
    AGENT_REFILL_PER_STEP,
)
from src.simulation.step_effects import (
    CAPIENZA_PER_DEPOSITO,
    GIORNI_PER_PASSO_RIFERIMENTO,
    PRELIEVO_CIBO_PERSONALE,
    PRELIEVO_MATERIALE_PERSONALE,
    RESA_ATTREZZI,
    RESA_CONOSCENZA,
    CAPIENZA_PER_STRUTTURA,
    CARICO_ELETTRICO,
    RESA_ACQUA,
    RESA_CIBO,
    RESA_ENERGIA,
    RESA_ENERGIA_PER_CARICO,
    RESA_KIT_MEDICI,
    RESA_OSSIGENO,
    RISORSE_A_TETTO,
    SCALA_USURA_PER_GIORNO,
    SCORTA_PRO_CAPITE,
)
#: La polvere del cielo marziano in condizioni ordinarie, sulla scala [0, 1]
#: di `cell.dust_level`. E' il punto in cui la resa solare vale uno: la
#: colonia e' dimensionata su questo cielo, e il fattore misura solo quanto
#: se ne discosta.
#:
#: **Misurato, non stimato.** Il primo valore scelto a occhio era 0,25, e
#: penalizzava la baseline del 6% (energia totale 342,6 contro 363,9) perche'
#: la polvere vera delle celle abitate sta piu' in alto: mediana 0,350 su
#: trecento coloni, con e senza eventi. A 0,35 l'energia di una run non
#: governata torna identica all'originale, cifra per cifra, e il fattore
#: agisce solo dove deve — quando una tempesta alza la polvere sopra il
#: cielo ordinario.
TYPICAL_DUST = 0.35

#: Quanta resa solare toglie una polvere satura rispetto a quella tipica.
#: Dal MCD: 57,8 W/m2 nello scenario di tempesta contro 527 in climatologia,
#: cioe' l'11% — resta dunque l'89% da perdere.
SOLAR_DUST_LOSS = 0.89

from src.world.cell import POLLUTION_DEGRADATION_CEILING, VEGETATION_CARRYING_CAPACITY
from src.world.occupancy import overcrowding_excess_from_housing
from src.world.structures import DOMANDA_MINERALI_SU_MATERIALE, StructureType
from src.world.terrain import TERRAIN_HABITABILITY

_TERRAIN_HAB_V = np.array([TERRAIN_HABITABILITY[t] for t in TERRAIN_ORDER], dtype=np.float64)

# Toggle di sola misura, come MARSABM_CELL_SUBSET: con `0` il blocco biologico
# torna a valutare ogni formula su tutta la griglia e a scegliere con
# `np.where(active_mask, ...)`. I due percorsi producono lo stesso stato
# bit-per-bit; il toggle serve a poter rimisurare il guadagno con un confronto
# A/B interlacciato invece di doverlo prendere sulla fiducia.
_BIOLOGY_SUBSET_ENABLED = os.environ.get("MARSABM_BIOLOGY_SUBSET", "1") != "0"


#: Pesi del carico elettrico in forma di vettore su `C.STRUCTURES`, gia'
#: moltiplicati per la resa. Un prodotto `struct_count @ pesi` costa una
#: passata sola; la versione con un ciclo sul dizionario allocava quattro array
#: di griglia per passo, ed e' il genere di dettaglio che in questo motore si
#: paga a ogni passo di ogni run.
_PESI_CARICO = np.array(
    [
        CARICO_ELETTRICO.get(tipo, 0.0) * RESA_ENERGIA_PER_CARICO
        for tipo in C.STRUCTURES
    ],
    dtype=np.float64,
)



#: **Interruttore di misura sull'ordine del tetto di magazzino.** Spento, il
#: kernel applica il tetto prima dei prelievi di riserva, com'e' sempre stato in
#: ogni campagna di questo progetto. Acceso, lo applica dopo, cioe' nell'ordine
#: del motore a oggetti (`apply_colony_resource_feedback`). Esiste per poter
#: eseguire lo stesso seme nei due ordini e misurare la differenza, non per
#: cambiare il regime: il valore predefinito e' bit-identico al passato.
TETTO_DOPO_PRELIEVI = os.environ.get("MARSABM_TETTO_DOPO_PRELIEVI") == "1"

def _active_biology_mask(cells: CellArrays, melt_active: bool) -> np.ndarray:
    """Vectorized transcription of `_biology_cell_can_change`
    (src/simulation/biology_update.py:50-60) - the predicate
    `update_biology_cells` filters cells through when called with
    `full_grid=False` (the "active" scope, the object engines' default -
    `_full_biology_update_enabled` in agent_coupled_runner.py:486-488/
    state_store.py:844-846). Every branch of that predicate is reproduced
    here against CellArrays' own columns/bookkeeping dicts:

      cell.agents_present / cell.structures / cell.construction_sites
        -> occupancy>0 / struct_count.sum(-1)>0 / site_progress>=0 (any
           structure type) or a non-empty `_site_extra` entry (the
           non-structural construction-site fallback, e.g. "ice_extraction" -
           see `ConstructionSitesProxy` in src/core/views.py).
      vegetation_biomass/proto_soil_development/organic_matter/liquid_water/
      pollution_risk > 0 -> the matching CellArrays column > 0.
      melt_active and water_ice > 0 -> same, `melt_active` passed in (a
        planetary-state-only scalar, matching biology_update.py's own
        module-level check - not per-cell).
      nutrients dict any > 0 -> nut_n/nut_p/nut_c > 0.

    NOT EXHAUSTIVE (known gap, upstream of this transcription): a cell whose
    ONLY activating feature is a non-structural construction site (e.g.
    "ice_extraction") loaded from the INITIAL world reads `obj=True,
    vec=False` here - `CellArrays.from_world` (src/core/arrays.py,
    `_site_key_to_type`) silently drops any construction-site key with no
    "@" (not a `StructureType`), so `site_progress` stays all -1 and
    `_site_extra` stays empty for that cell; this mask's `_site_extra`
    fallback only helps for sites created at runtime (via
    `ConstructionSitesProxy` in src/core/views.py), not ones present at
    `CellArrays.from_world` time. Fixing this is upstream of this function
    (`CellArrays.from_world` would need to capture non-structural sites too)
    and out of this task's scope; this predicate reproduces
    `_biology_cell_can_change` on the other 13 of 14 branches.
    """
    mask = (
        (cells.occupancy > 0)
        | (cells.struct_count.sum(axis=-1) > 0)
        | (cells.site_progress >= 0.0).any(axis=-1)
        | (cells.vegetation > 0.0)
        | (cells.proto_soil > 0.0)
        | (cells.organic > 0.0)
        | (cells.liquid_water > 0.0)
        | (cells.pollution > 0.0)
        | (cells.nut_n > 0.0)
        | (cells.nut_p > 0.0)
        | (cells.nut_c > 0.0)
    )
    if melt_active:
        mask = mask | (cells.water_ice > 0.0)
    extra_sites = getattr(cells, "_site_extra", None)
    if extra_sites:
        for (y, x), sites in extra_sites.items():
            if sites:
                mask[y, x] = True
    return mask


def recompute_habitability(
    cells: CellArrays,
    planetary_state: dict | None,
    cell_degradation: bool = True,
    ys=None,
    xs=None,
) -> None:
    """Vectorized 1:1 transcription of Cell.recompute_habitability (cell.py:154-188).

    Writes `cells.habitability` (and, when `cell_degradation`, the density-load
    floor into `cells.pollution`) in place. With `ys`/`xs` given (scalars or
    index arrays) only those cells are touched - used by
    `WorldView.add_structure` (src/core/views.py) to give a newly completed
    structure the exact same immediate full recompute the object engine
    performs, instead of an incremental approximation. Without arguments the
    whole grid is recomputed - used by `update_cells` below.

    Precondition (shared with `CellArrays.recompute_struct_fx`): `cells.struct_fx`
    must already reflect the current `struct_count`/`struct_integrity` - callers
    that just mutated structures must call `recompute_struct_fx` first.
    """
    full = ys is None

    def f(name):
        arr = getattr(cells, name)
        return arr if full else arr[ys, xs]

    terrain = f("terrain")
    score = np.array(_TERRAIN_HAB_V[terrain], dtype=np.float64, copy=True)
    water_ice = f("water_ice")
    liquid_water = f("liquid_water")
    vegetation = f("vegetation")
    radiation = f("radiation")
    dust = f("dust")
    polar = f("polar")
    temp_mod = f("temp_mod")
    struct_fx = f("struct_fx")
    struct_heat = struct_fx[..., C.E["temperature"]]

    score = (
        score
        + np.minimum(water_ice, 10.0) * 0.005
        + np.minimum(liquid_water, 5.0) * 0.02
        + np.minimum(vegetation, VEGETATION_CARRYING_CAPACITY) * 0.01
        - radiation * 0.03
        - dust * 0.02
        - polar * 0.11
    )
    effective_temp_mod = temp_mod + struct_heat
    score = score - np.maximum(0.0, -effective_temp_mod - 10.0) * 0.003

    pollution = f("pollution")
    if cell_degradation:
        area_m2 = np.maximum(1.0, f("area_m2"))
        occupancy = f("occupancy").astype(np.float64)
        struct_count_total = f("struct_count").sum(axis=-1).astype(np.float64)
        density_load = occupancy * 6.0 + struct_count_total * 20.0
        density_pollution = np.minimum(0.08, density_load / area_m2 * 25_000.0)
        pollution = np.maximum(pollution, density_pollution)
        if full:
            cells.pollution[...] = pollution
        else:
            cells.pollution[ys, xs] = pollution
    score = score - pollution * 0.45

    if planetary_state:
        shielding = float(planetary_state.get("radiation_shielding_index", 0.0))
        liquid_stability = float(planetary_state.get("liquid_water_stability", 0.0))
        vegetation_suitability = float(planetary_state.get("vegetation_suitability", 0.0))
        temp_c = float(planetary_state.get("mean_temperature_c", -63.0))
        temp_score = max(0.0, min(1.0, (temp_c + 60.0) / 80.0))
        ice_stock = f("cell_res")[..., C.R["ice"]]
        score = (
            score
            + shielding * 0.06
            + liquid_stability * np.minimum(water_ice + ice_stock, 10.0) * 0.01
            + vegetation_suitability * 0.08
            + temp_score * 0.03
            - (1.0 - shielding) * 0.04
        )

    score = score + struct_fx[..., C.E["habitability"]]
    result = np.clip(score, 0.0, 1.0).astype(np.float64)
    if full:
        cells.habitability[...] = result
    else:
        cells.habitability[ys, xs] = result


def _serialized_reserve_draw(cell_res, res_idx, inv, inv_idx, rows, ys, xs, cap, refill) -> None:
    """Draw up to `refill` units of a cell-stocked reserve (energy/oxygen) into
    each agent's inventory (capped at `cap`), one agent at a time in `rows`
    order for cells that host MORE THAN ONE candidate agent.

    Mass-conservation trap: the object model's `for agent in agents.values():
    ... cell.resources.energy -= draw; agent.inventory.energy += draw` loop
    (step_effects.py:83-96) lets agents sharing a cell draw sequentially, each
    seeing the stock already reduced by the previous one. A naive vectorized
    gather (`cells.cell_res[ay, ax, ri]` read once, applied to every agent at
    that cell) would let all of them draw from the SAME undrained stock and
    manufacture matter out of nothing. Cells with at most one candidate agent
    (the overwhelmingly common case) are handled by one vectorized pass;
    cells with more than one are resolved with a small Python loop, in `rows`
    order (== dict insertion order, since `AgentArrays.from_agents` assigns
    rows in that order) to match the object model's iteration order exactly.
    """
    if rows.size == 0:
        return
    # UNDEFENDED ROW-ORDER ASSUMPTION: the per-cell Python loop below only
    # reproduces the object model's `for agent in agents.values(): ...`
    # sequential draw order (step_effects.py:83-96) because `rows` is
    # guaranteed ascending == `AgentArrays.from_agents` insertion order
    # (arrays.py:60-75, `alive_rows()` returns `np.flatnonzero(...)` which is
    # always ascending). A future row-reuse allocator (e.g. one that recycles
    # a dead agent's low row index for a newly spawned agent inserted later)
    # could silently break that correspondence without touching this
    # function; assert the invariant instead of trusting it implicitly.
    assert np.all(rows[:-1] <= rows[1:]), (
        "_serialized_reserve_draw requires `rows` in ascending (insertion) order "
        "to match the object model's sequential per-agent draw"
    )
    groups: dict[tuple[int, int], list[int]] = {}
    for i in range(rows.size):
        key = (int(ys[i]), int(xs[i]))
        groups.setdefault(key, []).append(i)

    single_idx = [idxs[0] for idxs in groups.values() if len(idxs) == 1]
    if single_idx:
        si = np.asarray(single_idx, dtype=np.intp)
        rr, yy, xx = rows[si], ys[si], xs[si]
        stock = cell_res[yy, xx, res_idx]
        headroom = cap - inv[rr, inv_idx]
        can_draw = (stock > 0.0) & (headroom > 0.0)
        draw = np.where(can_draw, np.minimum(refill, np.minimum(stock, headroom)), 0.0)
        cell_res[yy, xx, res_idx] -= draw
        inv[rr, inv_idx] += draw

    for key, idxs in groups.items():
        if len(idxs) == 1:
            continue
        y, x = key
        for i in idxs:
            r = int(rows[i])
            stock = float(cell_res[y, x, res_idx])
            if stock <= 0.0:
                continue
            headroom = cap - float(inv[r, inv_idx])
            if headroom <= 0.0:
                continue
            draw = min(refill, stock, headroom)
            cell_res[y, x, res_idx] -= draw
            inv[r, inv_idx] += draw


def update_cells(
    cells: CellArrays,
    agents: AgentArrays,
    planetary_state: dict,
    dt_days: float,
    cell_degradation: bool,
    full_grid: bool,
    isru_material_rate: float = 0.0,
) -> dict:
    """Vectorized 1:1 transcription of the per-step cell pipeline (see module
    docstring for the exact source call sites and the habitability/wear
    ordering note). Mutates `cells` and `agents` in place.

    `full_grid` mirrors `update_biology_cells`'s own parameter
    (src/simulation/biology_update.py): False (the object engines' default -
    `_full_biology_update_enabled`) restricts the BIOLOGY sub-step (ice/water
    exchange, pedogenesis, vegetation growth, nutrient cycling, occupancy
    degradation) and its habitability recompute to cells passing
    `_active_biology_mask` (the vectorized `_biology_cell_can_change`);
    everything else in this function - structure effects, colony feedback,
    wear - already runs whole-grid in BOTH engines regardless of that flag
    (`apply_structure_effects`/`apply_colony_resource_feedback`/
    `apply_structure_wear` are separate calls, not gated by
    `biology_update_scope`) and their contribution is a no-op on
    unstructured cells anyway (`struct_fx` is all-zero there), so they stay
    unmasked here too.

    REQUIRED, no default (review round 3, Minor 4): `update_biology_cells`
    (the object wrapper this mirrors) defaults `full_grid` to False (the
    "active" scope) - the opposite of what this function used to default to.
    A caller that forgot to pass it here would have silently gotten
    whole-grid behavior neither object engine ever uses, instead of an error.
    Every caller must now decide and pass the value explicitly:
    `src/core/kernel.py` passes the config-derived value, like the object
    engines do; `tests/core/test_kernel_biology_equivalence.py` (Task 8),
    which drives single, already-active test cells directly, passes
    `full_grid=True` explicitly (harmless there since every cell those tests
    build is already active, so the scope choice is a no-op on their
    scenarios - but it must still be a deliberate, visible choice at each
    call site rather than an implicit fallback).

    Preconditions: `cells.occupancy` must already reflect this step's agent
    positions (the object test sets `cell.agents_present` the same way before
    calling `update_biology`); `cells.struct_fx` must already be current for
    `cells.struct_count`/`struct_integrity` (true right after
    `CellArrays.from_world` or any `recompute_struct_fx` call).

    Returns `{"events": list[dict], "knowledge_gain": float}`.
    """
    dt = float(dt_days)

    temp_c = float(planetary_state.get("mean_temperature_c", -63.0))
    pressure = float(planetary_state.get("pressure_pa", 600.0))
    liquid_stability = float(planetary_state.get("liquid_water_stability", 0.0))
    shielding_index = float(planetary_state.get("radiation_shielding_index", 0.0))

    if full_grid:
        active_mask = np.ones((cells.H, cells.W), dtype=bool)
        ys_active = xs_active = None
    else:
        melt_active = liquid_stability > 0.0 and temp_c > -10.0
        active_mask = _active_biology_mask(cells, melt_active)
        ys_active, xs_active = np.nonzero(active_mask)

    # Sotto lo scopo "active" ogni formula di questo blocco veniva valutata su
    # tutte le celle e poi buttata da `np.where(active_mask, ...)`. Sulla
    # configurazione reale le celle attive sono 1-5 su 64.800 (misurato su 120
    # passi), cioe' ~24.000x di aritmetica inutile: il blocco costava il 22,6%
    # del passo. Con `ys_active`/`xs_active` le stesse formule girano solo dove
    # il risultato verra' scritto.
    #
    # E' lo stesso idioma che `recompute_habitability` usa gia' qui sotto, e per
    # la stessa ragione: le coordinate restano GLOBALI, quindi non serve
    # tradurre indici e nessun evento puo' finire su una cella sbagliata.
    #
    # E' lecito perche' nessuna formula di questo modulo legge una cella diversa
    # da quella che scrive: non esiste diffusione ne' alcun termine di vicinato
    # (verificato: nessun `np.roll` ne' indicizzazione sfalsata nel file). Il
    # valore di una cella non dipende quindi da quali altre celle stiano
    # nell'array.
    restricted = _BIOLOGY_SUBSET_ENABLED and ys_active is not None

    def f(name: str):
        """Il campo, intero o ristretto alle sole celle attive."""
        arr = getattr(cells, name)
        return arr[ys_active, xs_active] if restricted else arr

    def commit(name: str, values) -> None:
        """Scrive dove il risultato conta, con la stessa semantica dei due percorsi.

        Nel percorso ristretto le celle selezionate sono tutte attive per
        costruzione, quindi la scelta con `np.where` non serve piu': e' gia'
        stata fatta scegliendo gli indici.
        """
        target = getattr(cells, name)
        if restricted:
            target[ys_active, xs_active] = values
        else:
            target[...] = np.where(active_mask, values, target)

    def put(name: str, values) -> None:
        """Scrittura NON filtrata da ``active_mask``.

        Serve ai passaggi che hanno gia' una maschera propria (``fert_mask``) e
        che nel percorso storico scrivevano su tutta la griglia: dove la loro
        maschera e' falsa riscrivono il valore corrente, quindi restringere non
        cambia nulla.
        """
        target = getattr(cells, name)
        if restricted:
            target[ys_active, xs_active] = values
        else:
            target[...] = values

    def add_res(res_key: str, values) -> None:
        """Accumula su una colonna di ``cell_res``, ristretta o intera."""
        if restricted:
            cells.cell_res[ys_active, xs_active, C.R[res_key]] += values
        else:
            cells.cell_res[..., C.R[res_key]] += values

    # ---- 1. Biology (cell.py:69-153) -------------------------------------

    # 0. Ice <-> liquid water exchange (cell.py:81-93).
    exchange_fraction = min(1.0, dt / 365.25)
    water_ice = f("water_ice")
    liquid_water = f("liquid_water")
    effective_temp_c = temp_c + f("temp_mod") + f("struct_fx")[..., C.E["temperature"]]
    melt_mask = (liquid_stability > 0.0) & (effective_temp_c > 0.0) & (water_ice > 0.0)
    melt_factor = np.minimum(
        0.5, liquid_stability * 0.05 * exchange_fraction * np.maximum(0.0, effective_temp_c) / 10.0
    )
    melted = water_ice * melt_factor
    refreeze_mask = (~melt_mask) & (liquid_water > 0.0) & ((effective_temp_c < -5.0) | (liquid_stability <= 0.0))
    refreeze_factor = min(1.0, 0.5 * exchange_fraction)
    refrozen = liquid_water * refreeze_factor

    new_water_ice = water_ice + np.where(melt_mask, -melted, 0.0) + np.where(refreeze_mask, refrozen, 0.0)
    new_liquid_water = liquid_water + np.where(melt_mask, melted, 0.0) + np.where(refreeze_mask, -refrozen, 0.0)
    commit("water_ice", new_water_ice)
    commit("liquid_water", new_liquid_water)

    # 1. Pedogenesis (cell.py:95-101). weathering_rate's base term depends
    # only on planetary temp_c (not per-cell); the vegetation multiplier uses
    # the PRE-growth vegetation array.
    vegetation = f("vegetation")
    weathering_base = 0.0001 * dt * max(0.0, (temp_c + 20.0) / 40.0) * (liquid_stability + 0.1)
    weathering_rate = weathering_base * np.where(vegetation > 0.1, 1.0 + 0.5 * vegetation, 1.0)
    new_proto_soil = np.minimum(1.0, f("proto_soil") + weathering_rate)
    commit("proto_soil", new_proto_soil)

    # 2. Vegetation growth, exact exponential solution (cell.py:103-131).
    f_temp = max(0.0, 1.0 - abs(temp_c - 15.0) / 35.0) if temp_c > -10.0 else 0.0
    f_press = min(1.0, pressure / 40000.0) if pressure > 6000.0 else 0.0
    # Post-scambio: `commit` ha gia' scritto, quindi si rileggono i campi
    # aggiornati esattamente come faceva il percorso a griglia intera.
    f_water = np.minimum(1.0, (f("liquid_water") + f("water_ice") * 0.2) / 5.0)
    f_soil = 0.1 + 0.9 * f("proto_soil")
    nut_n = f("nut_n")
    nut_p = f("nut_p")
    nut_c = f("nut_c")
    n_ratio = nut_n / 10.0
    p_ratio = nut_p / 2.0
    c_ratio = nut_c / 5.0
    f_nutrients = np.minimum(np.minimum(n_ratio, p_ratio), np.minimum(c_ratio, 1.0))

    radiation_eff = f("radiation") * (1.0 - shielding_index)
    growth_potential = f_temp * f_press * f_water * f_soil * (1.0 - radiation_eff * 0.5)
    growth_per_day = 0.1 * np.maximum(0.0, growth_potential) * np.maximum(0.0, f_nutrients)
    decay_per_day = 0.02 * max(0.0, (temp_c + 10.0) / 30.0)

    # `vegetation` e' gia' il valore PRE-crescita letto sopra per la pedogenesi:
    # nulla lo ha scritto nel frattempo, e nel percorso ristretto e' gia' una
    # copia prodotta dall'indicizzazione.
    before = vegetation if restricted else vegetation.copy()
    if decay_per_day > 0.0:
        decay_factor = math.exp(-decay_per_day * dt)
        equilibrium = growth_per_day / decay_per_day
        new_veg = equilibrium + (before - equilibrium) * decay_factor
        decayed = before * (1.0 - decay_factor)
    else:
        new_veg = before + growth_per_day * dt
        decayed = np.zeros_like(before)
    new_veg = np.clip(new_veg, 0.0, VEGETATION_CARRYING_CAPACITY)
    grown = np.maximum(0.0, new_veg - before + decayed)
    commit("vegetation", new_veg)

    new_organic = f("organic") + decayed * 0.5
    new_nut_n = np.maximum(0.0, nut_n + decayed * 0.1 - grown * 0.1)
    new_nut_p = np.maximum(0.0, nut_p + decayed * 0.02 - grown * 0.02)
    new_nut_c = np.maximum(0.0, nut_c + decayed * 0.4 - grown * 0.4)
    commit("organic", new_organic)
    commit("nut_n", new_nut_n)
    commit("nut_p", new_nut_p)
    commit("nut_c", new_nut_c)

    # 3. Degradation from occupancy (cell.py:139-151), gated by cell_degradation
    # AND (like the rest of this sub-step, cell.py:69-153 is the whole body of
    # `Cell.update_biology`, only called for active cells under the "active"
    # scope) by `active_mask`.
    years = dt / 365.25
    passo = dt / GIORNI_PER_PASSO_RIFERIMENTO
    struct_count = f("struct_count")
    unsupported_occupants = overcrowding_excess_from_housing(
        f("occupancy"),
        struct_count[..., C.S[StructureType.SHELTER]],
        struct_count[..., C.S[StructureType.HABITAT]],
        struct_count[..., C.S[StructureType.INFIRMARY]],
    )
    pollution = f("pollution")
    if not cell_degradation:
        new_pollution = np.zeros_like(pollution)
    else:
        mask_occ = unsupported_occupants > 0
        ceiling = np.minimum(
            POLLUTION_DEGRADATION_CEILING, 0.05 * unsupported_occupants
        )
        below = mask_occ & (pollution < ceiling)
        increased = np.minimum(
            ceiling, pollution + unsupported_occupants * 0.05 * years
        )
        occ_result = np.where(below, increased, pollution)
        recovered = np.maximum(0.0, pollution - 0.02 * years)
        new_pollution = np.where(mask_occ, occ_result, recovered)
    commit("pollution", new_pollution)

    # ---- 2. Structure effects (step_effects.py:33-55) ---------------------
    # Questo blocco non era filtrato da `active_mask`, e non serviva: `struct_fx`
    # e' nullo dove non ci sono strutture, quindi altrove sommava zero. E ogni
    # cella con strutture e' attiva per costruzione (e' il primo ramo di
    # `_active_biology_mask`), quindi restringere alle attive non toglie nulla.
    struct_fx = f("struct_fx")  # still pre-wear/valid: nothing above touched struct_count/integrity
    # **L'energia solare vede la polvere sopra la cella.** `struct_fx` nasce da
    # una matrice di effetti costanti (`STRUCT_FX_M`), quindi senza questo
    # fattore un pannello renderebbe uguale sotto un cielo limpido e dentro una
    # tempesta globale — e una tempesta di polvere, che e' il modo in cui Marte
    # uccide davvero un impianto solare, resterebbe un fatto cosmetico.
    #
    # Il fattore e' RELATIVO alla polvere tipica, non assoluto: la colonia e'
    # dimensionata su quel cielo li' (margine di potenza ~1,5 negli audit), e
    # una resa assoluta ricalibrerebbe l'intero bilancio energetico per poi
    # rimetterlo dov'era. Cosi' invece la baseline resta quella che e', e cio'
    # che si aggiunge e' soltanto la PERDITA quando la polvere sale sopra il
    # normale. La profondita' viene dal MCD: 57,8 W/m2 contro 527 in
    # climatologia, cioe' l'11% — vedi docs/benchmarks/2026-08-25-mcd-e-eventi.md.
    dust_now = f("dust")
    solar_yield = 1.0 - SOLAR_DUST_LOSS * np.clip(
        (dust_now - TYPICAL_DUST) / (1.0 - TYPICAL_DUST), 0.0, 1.0
    )
    add_res("energy", struct_fx[..., C.E["energy"]] * RESA_ENERGIA * solar_yield)

    # **Il carico elettrico si paga (2026-08-25).** Trascrizione del blocco in
    # `step_effects.apply_structure_effects`, stesse costanti. Si calcola sulla
    # griglia INTERA anche quando la biologia e' ristretta: il carico e' zero
    # dove non ci sono strutture, e ogni cella con strutture e' attiva per
    # costruzione (`_active_biology_mask` include `struct_count.sum(-1) > 0`),
    # quindi le due forme coincidono. La copertura serve piu' avanti al blocco
    # di cibo e acqua, che gira sulla griglia intera: tenerla intera evita di
    # doverla riportare da un dominio all'altro.
    carico_elettrico = cells.struct_count @ _PESI_CARICO
    energia_disponibile = cells.cell_res[..., C.R["energy"]]
    energia_usata = np.minimum(energia_disponibile, carico_elettrico)
    copertura = np.divide(
        energia_usata,
        carico_elettrico,
        out=np.ones_like(carico_elettrico),
        where=carico_elettrico > 0.0,
    )
    cells.cell_res[..., C.R["energy"]] = np.maximum(0.0, energia_disponibile - energia_usata)
    # **Copertura vera per il governo (2026-09-26), solo se richiesta.** Il
    # vocabolario dei governatori legge `power_coverage` dalla giacenza avanzata
    # dopo il consumo (~0 anche a corrente coperta); con l'interruttore acceso
    # si conserva QUESTA copertura, che e' quella che governa la produzione.
    # Spento, nessun array nuovo: il digest delle celle resta identico.
    if getattr(cells, "copertura_vera_attiva", False):
        cells.copertura_elettrica = np.array(copertura, dtype=np.float64, copy=True)
    copertura_locale = copertura[ys_active, xs_active] if restricted else copertura

    add_res("oxygen", struct_fx[..., C.E["oxygen"]] * RESA_OSSIGENO * copertura_locale)
    add_res("biomass", struct_fx[..., C.E["biomass"]] * 0.02)
    biomass_effect = struct_fx[..., C.E["biomass"]]
    fert_mask = biomass_effect > 0.0
    veg_now = f("vegetation")
    nut_n_now = f("nut_n")
    nut_p_now = f("nut_p")
    nut_c_now = f("nut_c")
    put("vegetation", np.where(fert_mask, np.minimum(10.0, veg_now + biomass_effect * 0.01), veg_now))
    put("nut_n", np.where(fert_mask, np.minimum(20.0, nut_n_now + biomass_effect * 0.10), nut_n_now))
    put("nut_p", np.where(fert_mask, np.minimum(4.0, nut_p_now + biomass_effect * 0.02), nut_p_now))
    put("nut_c", np.where(fert_mask, np.minimum(10.0, nut_c_now + biomass_effect * 0.05), nut_c_now))

    # Habitability recompute - see the module docstring's ORDERING NOTE for why
    # this runs here (using pre-wear struct_fx) rather than after wear.
    # Gated by `active_mask` via the existing partial-recompute path
    # (`ys`/`xs`, otherwise used by WorldView.add_structure): under the
    # "active" scope a bare/frozen cell never gets `cell.update_biology`
    # called on the object side (cell.py:153's internal recompute never
    # runs), and - having no structures either, since `_biology_cell_can_change`
    # already treats any structured cell as active - it never gets
    # `apply_structure_effects`'s second recompute (step_effects.py:56)
    # either. Every structured cell is always a member of `active_mask` (the
    # object predicate's first branch), so this never skips a cell the
    # structure-effects pass would have recomputed.
    if full_grid:
        recompute_habitability(cells, planetary_state, cell_degradation)
    elif ys_active.size:
        # `ys_active`/`xs_active` sono ora calcolati una volta sola in cima:
        # erano gia' il modo in cui questo passaggio restringeva se stesso, ed e'
        # l'idioma che il resto del blocco ha adottato.
        recompute_habitability(cells, planetary_state, cell_degradation, ys=ys_active, xs=xs_active)

    # ---- 3. Colony feedback (step_effects.py:63-96) ------------------------
    # NON ristretto, deliberatamente. `eff2_agg @ C.STRUCT_FX_M` sarebbe il
    # candidato piu' ghiotto del blocco (~7,8 milioni di moltiplicazioni per
    # passo, nulle fuori dalle celle con strutture), ma il prodotto matriciale
    # e' l'unica operazione qui il cui RISULTATO dipende dalla forma: BLAS
    # sceglie un blocking diverso per [180,360,10] e per [K,10], quindi somma i
    # dieci termini in un ordine diverso e l'ultimo bit cambia. La parita' di
    # questo progetto e' bit-exact, non approssimata -- misurato: restringerlo
    # faceva divergere `cells.struct_fx` al primo passo, su una sola cella e
    # oltre l'ottava cifra. Restringere qui costerebbe la garanzia, non il
    # comportamento; il resto del blocco resta ristretto.
    count_f = cells.struct_count.astype(np.float64)
    mean_int = np.divide(cells.struct_integrity, count_f, out=np.zeros_like(cells.struct_integrity), where=count_f > 0)
    eff_adj = np.where(mean_int > 0.4, mean_int, mean_int * 0.5)
    # squared-efficiency aggregate: apply_colony_resource_feedback multiplies
    # `structure.local_effect.get(key)` (already efficiency-weighted once) by
    # ANOTHER `structure.efficiency` factor (step_effects.py:74-79) - so the
    # per-structure contribution is base_value * efficiency**2, not efficiency.
    eff2_agg = (eff_adj ** 2) * count_f
    eff2_fx = eff2_agg @ C.STRUCT_FX_M  # [H,W,NE]

    tool_factor = 1.0 + np.minimum(0.75, cells.cell_res[..., C.R["tools"]] * 0.05)
    cells.cell_res[..., C.R["food"]] += eff2_fx[..., C.E["food"]] * tool_factor * RESA_CIBO * copertura
    if isru_material_rate > 0.0:
        # **ISRU: i depositi lavorano il REGOLITO della cella in materiale da
        # costruzione.** Il regolito non e' un giacimento: e' la superficie di
        # Marte, presente in ogni cella. Per questo non compare come scorta e
        # non c'e' nessun minimo da rispettare — la conversione e' limitata dal
        # PARCO IMPIANTI, non dal deposito minerario sotto i piedi.
        #
        # **Perche' serve.** Senza conversione il materiale da costruzione e'
        # una dotazione iniziale non rinnovabile, e la colonia non si limita a
        # smettere di crescere: MUORE. Misurato il 2026-08-31 (seme 9, 50
        # coloni): il materiale tocca 3,1 unita' al passo 250 e li' resta; da
        # quel momento nessuno puo' piu' pagare `costo_manutenzione`, che si
        # salda in materiale; l'integrita' media del parco scende sotto 0,4,
        # dove `Structure.efficiency` la DIMEZZA; supporto vitale e serre
        # crollano e i morti passano da 1 ogni 50 passi a 33 (passi 350-400).
        # A 450 passi la popolazione e' 23 su 73. Non e' un altopiano: e' uno
        # stato assorbente letale.
        #
        # **Perche' il regolito e non i minerali (2026-08-31).** Legata ai
        # minerali della cella, questa conversione era INERTE nella run di
        # default: il sito colonia scelto da `colony_site` e' la prima piana di
        # regolito in fascia equatoriale e ha `minerals = 0,00` (misurato al
        # passo 1, seme 9), e solo il 6,4% della mappa ha un giacimento. La
        # regola cambiava dunque significato da cella a cella — la colonia
        # madre poteva prosperare o morire secondo un sorteggio geografico, e
        # un avamposto fondato lontano da un centro minerario non poteva
        # rinnovare NULLA. Sul regolito la regola e' la stessa ovunque: cio'
        # che un colono puo' fare nella cella madre puo' farlo identico in
        # qualunque cella colonizzi, che era il requisito.
        #
        # **Dove resta la difficolta'.** Non nella geografia del materiale ma
        # nel capitale: `eff2_fx[storage]` pesa i depositi per l'efficienza AL
        # QUADRATO, quindi un parco lasciato degradare produce quasi nulla
        # proprio quando servirebbe per ripararlo. I minerali restano
        # geografici e ogni struttura ne costa 1-3: la geografia non decide
        # piu' la sopravvivenza, decide la CRESCITA.
        #
        # **Due prodotti, una sola resa (2026-08-31).** L'impianto rende
        # materiale E minerali, perche' estrarre metallo dal regolito e' cio'
        # che «ISRU» significa. La resa TOTALE non cambia di una virgola: e'
        # la stessa `domanda` di prima, ripartita fra i due beni secondo
        # `DOMANDA_MINERALI_SU_MATERIALE`, cioe' secondo `BUILD_COSTS`. La
        # colonia non diventa piu' ricca: riceve la miscela che consuma.
        #
        # **Perche' serviva.** Con i soli minerali della dotazione iniziale il
        # totale delle costruzioni di una run e' fissato in partenza: ~117
        # unita' a 50 coloni, ~1,8 per struttura, cioe' ~65 strutture e mai una
        # di piu'. Misurato sul seme 9: il parco va da 69 a 137 e li' si ferma
        # per sempre al passo ~225, quando i minerali toccano lo zero. Nessuna
        # run lunga poteva mostrare alcunche' dopo quel punto, e nessun
        # avamposto poteva crescere oltre la griglia che il kit gli paga.
        #
        # **Perche' la geografia conta ancora.** Questa resa e' limitata dai
        # depositi, cioe' da capitale che va costruito e mantenuto; una cella
        # con giacimento si raccoglie con `COLLECT_MINERALS` senza alcun
        # impianto. Il minerale dal regolito e' il rivolo che tiene viva la
        # frontiera, il giacimento e' la vena che la fa correre: la geografia
        # non decide piu' la sopravvivenza, decide la velocita'.
        quota_minerali = DOMANDA_MINERALI_SU_MATERIALE / (1.0 + DOMANDA_MINERALI_SU_MATERIALE)
        resa = eff2_fx[..., C.E["storage"]] * isru_material_rate * tool_factor
        cells.cell_res[..., C.R["construction_material"]] += resa * (1.0 - quota_minerali)
        cells.cell_res[..., C.R["minerals"]] += resa * quota_minerali
    # Gli attrezzi escono dal materiale da costruzione, e si fermano dove non
    # ce n'e' piu' (trascrizione del blocco in `apply_colony_resource_feedback`).
    attrezzi = eff2_fx[..., C.E["knowledge"]] * RESA_ATTREZZI * tool_factor * passo
    attrezzi = np.minimum(
        attrezzi, np.maximum(0.0, cells.cell_res[..., C.R["construction_material"]])
    )
    cells.cell_res[..., C.R["construction_material"]] -= attrezzi
    cells.cell_res[..., C.R["tools"]] += attrezzi
    # **La conoscenza si accumula, e prima non lo faceva da nessuna parte.**
    # I laboratori di ricerca dichiarano `{"knowledge": 1.0}` fra i propri effetti
    # e quell'effetto alimentava i soli `tools`, mentre `scientific_knowledge` —
    # la metrica che sblocca `tech_tier` alle soglie 50/150/400 — arrivava
    # esclusivamente dal layer planetario (`ConoscenzaScientifica`), che nelle run
    # di questo progetto e' disattivato. Il risultato era un'incoerenza fra due
    # parti del modello: si potevano costruire trentanove laboratori e vedere la
    # conoscenza scientifica ferma a zero e il livello tecnologico mai avanzare.
    #
    # Accumularla qui la lega a cio' che la colonia costruisce davvero. Quando il
    # layer planetario e' attivo continua a scrivere il proprio valore, che ha
    # la precedenza: questo termine copre il caso in cui non c'e'.
    cells.cell_res[..., C.R["knowledge"]] += eff2_fx[..., C.E["knowledge"]] * RESA_CONOSCENZA * passo
    cells.cell_res[..., C.R["water"]] += eff2_fx[..., C.E["water"]] * tool_factor * RESA_ACQUA * copertura
    cells.cell_res[..., C.R["med_kits"]] += eff2_fx[..., C.E["healing_bonus"]] * tool_factor * RESA_KIT_MEDICI * copertura

    # Tetto di magazzino, applicato una volta sola dopo tutte le aggiunte del
    # passo — stesso punto del motore a oggetti (`apply_colony_resource_feedback`).
    # `struct_fx` e non `eff2_fx`: la capienza di un deposito non dipende
    # dall'efficienza al quadrato, un magazzino tiene cio' che tiene.
    # **Solo dove ci sono strutture**, come il motore a oggetti, che itera su
    # `structure_cells`. Una cella spoglia non ha magazzino e non va svuotata:
    # un avamposto in costruzione riceve scorte prima di avere qualunque
    # edificio, e un tetto pari a zero le cancellerebbe. La differenza non
    # sarebbe apparsa in nessun test — nessuno mette risorse in una cella
    # spoglia — e sarebbe divergenza silenziosa fra i due motori.
    strutture_per_cella = cells.struct_count.sum(axis=-1, dtype=np.float64)
    con_strutture = strutture_per_cella > 0.0
    tetto = np.where(
        con_strutture,
        np.maximum(
            cells.occupancy.astype(np.float64) * SCORTA_PRO_CAPITE,
            strutture_per_cella * CAPIENZA_PER_STRUTTURA,
        )
        + cells.struct_fx[..., C.E["storage"]] * CAPIENZA_PER_DEPOSITO,
        np.inf,
    )
    def _applica_tetto():
        # `tetto` non dipende da `cell_res`: occupanti, strutture e depositi non
        # cambiano fra i due punti, quindi spostare SOLO il taglio e' esatto.
        for risorsa in RISORSE_A_TETTO:
            i_res = C.R[risorsa]
            cells.cell_res[..., i_res] = np.minimum(cells.cell_res[..., i_res], tetto)

    if not TETTO_DOPO_PRELIEVI:
        _applica_tetto()

    # ALIVE-ONLY ASSUMPTION: filtered to `agents.alive_rows()` here (and again
    # at the tool_stock sum below and in `elig_rows`/reserve draws), whereas
    # the object model's `apply_colony_resource_feedback` (step_effects.py:83)
    # iterates every agent in the dict unconditionally. This is safe ONLY
    # because dead agents are expected to be removed from the live
    # dict/AgentArrays promptly (not left lingering with `alive=False` rows
    # still holding a position) - if that ever changes, this alive-only filter
    # would start silently diverging from the object model wherever a dead
    # agent's cell still matters (it currently never does, by convention).
    rows = agents.alive_rows()
    per_agent_food_draw = PRELIEVO_CIBO_PERSONALE * passo
    per_agent_material_draw = PRELIEVO_MATERIALE_PERSONALE * passo
    if rows.size:
        agents.inv[rows, C.R["food"]] = np.maximum(0.0, agents.inv[rows, C.R["food"]] - per_agent_food_draw)
        agents.inv[rows, C.R["construction_material"]] = np.maximum(
            0.0, agents.inv[rows, C.R["construction_material"]] - per_agent_material_draw
        )

        ay = agents.y[rows].astype(np.intp)
        ax = agents.x[rows].astype(np.intp)
        # Somma solo nelle celle degli agenti invece che su tutta la mappa: il
        # predicato serviva comunque solo li', e la riduzione su [H,W,NS] era
        # l'unica parte cara di questo passaggio.
        elig_mask = cells.struct_count[ay, ax].sum(axis=-1) > 0
        elig_rows = rows[elig_mask]
        ey = ay[elig_mask]
        ex = ax[elig_mask]
        if elig_rows.size:
            _serialized_reserve_draw(
                cells.cell_res, C.R["energy"], agents.inv, C.R["energy"],
                elig_rows, ey, ex, AGENT_ENERGY_RESERVE_CAP, AGENT_REFILL_PER_STEP,
            )
            _serialized_reserve_draw(
                cells.cell_res, C.R["oxygen"], agents.inv, C.R["oxygen"],
                elig_rows, ey, ex, AGENT_OXYGEN_RESERVE_CAP, AGENT_REFILL_PER_STEP,
            )

    if TETTO_DOPO_PRELIEVI:
        # Ordine del motore a oggetti: il colono riempie la muta da cio' che la
        # cella ha in mano, e solo l'eccedenza oltre la capienza si disperde.
        _applica_tetto()

    # Somma equivalente: fuori dalle celle attive `struct_count` e' zero, perche'
    # ogni cella con strutture e' attiva per il primo ramo di `_active_biology_mask`.
    research_counts = f("struct_count")
    research_assets = int(
        research_counts[..., C.S[StructureType.RESEARCH_LAB]].sum()
        + research_counts[..., C.S[StructureType.WEATHER_STATION]].sum()
    )
    tool_stock = float(agents.inv[rows, C.R["tools"]].sum()) if rows.size else 0.0
    if research_assets <= 0 and tool_stock <= 0:
        knowledge_gain = 0.0
    else:
        knowledge_gain = (research_assets * 0.5 + tool_stock * 0.03) * (dt / 365.0)

    # ---- 4. Structure wear (step_effects.py:113-147) -----------------------
    wear_years = float(dt) * SCALA_USURA_PER_GIORNO
    wear_reduction = float(planetary_state.get("wear_reduction", 0.0))
    wear_multiplier = 1.0 - wear_reduction
    base_wear = 0.004 * wear_years
    dust_wear = f("dust") * 0.010 * wear_years
    wind_speed = float(planetary_state.get("wind_speed_m_s", 5.0))
    wind_wear = max(0.0, (wind_speed - 10.0) ** 2 / 400.0) * 0.006 * wear_years
    rad_wear = f("radiation") * 0.003 * wear_years
    total_wear = (base_wear + dust_wear + wind_wear + rad_wear) * wear_multiplier

    # Dove non ci sono strutture `count_f` e' zero e l'integrita' e' gia' zero,
    # quindi la sottrazione era comunque un'operazione a vuoto su tutta la mappa.
    count_f = f("struct_count").astype(np.float64)
    new_integrity = np.maximum(0.0, f("struct_integrity") - total_wear[..., None] * count_f)
    put("struct_integrity", new_integrity)
    # struct_integrity mutated in BULK, bypassing every StructureView setter:
    # any cached StructureView list now holds stale `_integrity_cache` values,
    # so drop the whole cache (rebuilt lazily on the next `cell.structures`).
    cells.clear_structure_cache()

    post_mean_int = np.divide(
        new_integrity, count_f, out=np.zeros_like(new_integrity), where=count_f > 0
    )
    warning_mask = (count_f > 0) & (post_mean_int < 0.2)
    events: list[dict] = []
    if restricted:
        # `warning_mask` e' [K,NS]: `nonzero` da' (indice-nella-selezione, tipo),
        # e `ys_active`/`xs_active` lo riportano alle coordinate della mappa. Il
        # `lexsort` sotto ordina poi su quelle coordinate globali, quindi la
        # sequenza degli eventi -- e l'assegnazione degli `event_id` in replay --
        # e' la stessa che produce il percorso a griglia intera.
        ks_e, ts_e = np.nonzero(warning_mask)
        ys_e = ys_active[ks_e]
        xs_e = xs_active[ks_e]
        warned_mean = post_mean_int[ks_e, ts_e]
    else:
        ys_e, xs_e, ts_e = np.nonzero(warning_mask)
        warned_mean = post_mean_int[ys_e, xs_e, ts_e]
    # EVENT ORDERING: `np.nonzero` on a (H,W,NS) array yields indices in
    # row-major order - y outermost, x, then structure type innermost - while
    # the object model's `structure_cells` (step_effects.py:26) iterates
    # `sorted(positions)` on (x, y) tuples, i.e. x outermost. Same set of
    # events, different sequence, which would show up in `event_id`
    # assignment on replay. `np.lexsort` with x as the last (primary) key
    # reproduces the object model's cell order exactly; `t` is an arbitrary
    # but stable tie-break for multiple warned structure types at one cell
    # (the aggregate model has no finer-grained instance order to match there
    # - see the module docstring's AGGREGATION SACRIFICES section).
    order = np.lexsort((ts_e, ys_e, xs_e))
    for idx in order.tolist():
        y, x, t = int(ys_e[idx]), int(xs_e[idx]), int(ts_e[idx])
        st = C.STRUCTURES[t]
        events.append({
            "type": "structure_warning",
            "message": f"{st.value} at ({x},{y}) integrity is critical ({warned_mean[idx] * 100:.1f}%)",
            "x": x,
            "y": y,
        })

    # struct_fx now reflects post-wear integrities for downstream consumers
    # (kernel_vitals.py's oxygen/food/water/habitability support terms, the
    # next step's biology/structure-effects pass, etc.) - the object model's
    # `structure.efficiency` is a live property, always current.
    # Anche questo resta su griglia intera, e per la stessa ragione: il suo corpo
    # termina con `eff @ C.STRUCT_FX_M`. La variante `ys`/`xs` esiste ed e'
    # corretta per i ricalcoli puntuali dopo una costruzione, ma usarla qui
    # cambierebbe la forma dell'operando del prodotto e con essa l'ultimo bit.
    cells.recompute_struct_fx()

    return {"events": events, "knowledge_gain": knowledge_gain}
